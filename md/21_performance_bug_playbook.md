# 第 21 章 — Performance Bug：如何看 Profile 與 Fix DRAM Write

> 上一章：[第 20 章 — Junior Exercises 與 Roadmap](20_junior_exercises_roadmap.md)

本章你會學到什麼：

- 如何從 `.profile.csv` 找 performance bug。
- 怎麼判斷問題是 tile fuse、microblock fuse、還是 DRAM write 沒省掉。
- 如何看 `dram_r / dram_w / sram_r / sram_w / tiles_h / tiles_oc`。
- 如何用小 model reproducer 驗證修法。
- 如何把「中間層 writeback」改成 on-chip handoff 或 metadata-only boundary。

---

## 21.1 Performance bug 不是只有 cycles 變大

Junior 一開始看 performance，常常只看：

```text
sim time: 5.088 ms
```

但在 MDLA7 裡，cycles 只是結果。真正要看的原因通常在 per-layer profile：

```text
id,op,in_h,in_w,in_c,out_h,out_w,out_c,...,tiles_h,tiles_oc,...,dram_r,dram_w,sram_r,sram_w
```

Performance bug 常見不是「某個 engine 算太慢」，而是：

| 現象 | 可能 root cause |
|---|---|
| `dram_w` 很大 | 中間 tensor 被寫回 DRAM |
| `dram_r` 很大 | 下一層又從 DRAM 讀回中間 tensor |
| `tiles_h` 很大 | 單層 input/output 放不進 L1，必須 H-tiling |
| `tiles_oc` 很大 | OC slice 太大，weight/output 要分 channel tile |
| `conv_util_pct` 很低 | UDMA / store / dependency 等待吃掉時間 |
| 很多 1-byte `dram_w` | writeback 已被 suppress，只剩 scheduling barrier |

所以看 performance 的第一步不是猜，而是把 profile 做成表。

---

## 21.2 快速找出 DRAM write 熱點

先跑單一 model，保留 profile：

```bash
./batch/run_ethz_v6.py --filter imdn_quant
```

然後看最大 `dram_w` layer：

```bash
python3 - <<'PY'
import csv
from pathlib import Path

csv_path = Path("batch/output/imdn_quant.profile.csv")
rows = list(csv.DictReader(csv_path.open()))

for r in sorted(rows, key=lambda x: int(x["dram_w"]), reverse=True)[:20]:
    print(
        f"L{int(r['id']):03d} {r['op'].strip():8s} "
        f"{r['in_h']}x{r['in_w']}x{r['in_c']} -> "
        f"{r['out_h']}x{r['out_w']}x{r['out_c']} "
        f"tiles={r['tiles_h']}x{r['tiles_oc']} "
        f"dram_w={int(r['dram_w'])/1024/1024:.2f} MB"
    )
PY
```

如果你看到很多中間層有 8 MB、16 MB、24 MB write，先不要急著改 engine。要先問：

```text
這個 output 真的需要落 DRAM 嗎？
還是只是 simulator 為了 per-layer verification 寫出去？
```

---

## 21.3 判斷 final output 與 intermediate output

不是所有 `dram_w` 都該省。

| 類型 | 是否可省 | 原因 |
|---|---|---|
| 最後輸出 | 通常不可省 | host / checker 要看到 final tensor |
| multi-tile conv tile output | 通常不可完全省 | full tensor 分散在 DRAM，下一層不能直接整顆從 L1 讀 |
| single-tile producer output | 常可省 | output 可留在 L1 給下一層 |
| concat 前 branch output | 常可省 | concat 可能已是 logical / metadata-only |
| reshape / gather copy | 視情況 | 若 downstream input 已由 compiler pre-materialize，可省 |
| softmax / gelu / h_swish | 視情況 | 若下一層可接 L1 output，可做 source-fusion |

在這份 simulator 裡，有一個很重要的設計：

```text
compile_model.py 會替每個 compiled layer 產生 synthetic input。
所以很多中間層的 DRAM writeback 只是 verification boundary，
不是下一層真正必須讀取的資料來源。
```

這表示有些 `dram_w` 可以安全省掉，但要留下 dependency barrier，避免排程 race。

---

## 21.4 Tile fuse：省掉 producer store 與 consumer load

Tile fuse 的核心想法：

```text
Producer output stays in L1
Consumer reads from producer's L1_OUT
Producer skips UDMA_W
Consumer skips UDMA_R
```

典型條件：

| 條件 | 說明 |
|---|---|
| shape match | producer `out_h/out_w/out_c` 等於 consumer `in_h/in_w/in_c` |
| dtype match | INT8 / INT16 / FP16 storage width 要一致 |
| single tile | producer full output 必須能完整留在 L1 |
| L1 layout 不 overlap | input、weight、params、output 區域不能互相覆蓋 |
| dependency tag 正確 | consumer 必須 wait producer compute done，不是 wait skipped store |

在 [test_model.cpp](/Volumes/4T_OFFICE/_Codex/MDLA7_Codex/systemc/src/test_model.cpp) 裡，這類狀態通常包含：

```cpp
fuse_prev_l1_out_addr
fuse_prev_l1_out_size
fuse_prev_done_tag
fuse_prev_out_h
fuse_prev_out_w
fuse_prev_out_c
fuse_prev_dtype
fuse_prev_single_tile
```

Tile fuse 常見 bug：

| Bug | Profile 現象 |
|---|---|
| producer store 沒 defer | producer `dram_w = full output size` |
| consumer 沒 fuse | consumer `dram_r = full input size` |
| single tile 判斷太保守 | 明明可留 L1，卻落 DRAM |
| pending store 沒 drop | 下一層已 fused，但前一層仍 writeback |
| done tag 用錯 | cycles 不穩、FAIL、或 Gantt dependency 看起來怪 |

---

## 21.5 Pending store：先不要急著寫 DRAM

一個好用的技巧是 pending store：

```text
producer 完成後，先把 UDMA_W descriptor 放進 pending
看到下一層後再決定：
  - 下一層能 fuse：drop pending store
  - 下一層不能 fuse：flush pending store
```

概念流程：

```text
Layer i compute done
  -> create pending udma_w

Layer i+1 starts
  -> if fused with layer i:
       skip pending store
     else:
       emit pending store before layer i+1 body
```

這可以避免 producer 一完成就寫 DRAM，給下一層 fusion 一個機會。

看 profile 時：

| Profile | 意義 |
|---|---|
| `dram_w = output bytes` | store 沒省 |
| `dram_w = 1` | full store 被省，只留 1-byte barrier |
| `streamed = true` in JSON | 此層 writeback 被視為 streamed / skipped |
| layer log 顯示 `FUSED` | output 留在 L1，該層不做 DRAM readback verify |

---

## 21.6 Microblock fuse：不是只省 bytes，也省等待

Tile fuse 是 layer-level。Microblock fuse 是更細的 tile / block-level overlap。

常見在 binary EWE：

```text
ADD / MUL / SUB
```

一顆大 tensor 如果放不進 2 MB L1，就要切 microblock：

```text
load A tile 0
load B tile 0
compute tile 0
store tile 0

同時 load tile 1 / compute tile 0 / store tile -1
```

這就是 wavefront：

```text
UDMA_R(tile+1) overlaps EWE(tile) overlaps UDMA_W(tile-1)
```

在 profile 裡要看：

| 欄位 | 怎麼判斷 |
|---|---|
| `dram_r` | 是否每個 tile 都重讀 A/B |
| `dram_w` | intermediate tile output 是否被 suppress |
| `sram_r/sram_w` | L1 traffic 是否符合 2-input + 1-output |
| Gantt | UDMA_R、EWE、UDMA_W 是否形成 staggered wavefront |

Microblock fuse 的 bug 常見是：

| Bug | 現象 |
|---|---|
| tile slot reuse 太早 | intermittent FAIL |
| barrier tag 不夠 | consumer race |
| final tile 標記錯 | layer done time 太早 |
| suppress store 後沒有 barrier | 下一層可能提早開始 |
| tile size 太大 | L1 overflow 或 fallback |

---

## 21.7 省 DRAM write：GraphMeta 是重要線索

`compile_model.py` 會在 `program.bin` 裡寫入 GraphMeta：

```text
input0_tensor
input1_tensor
output_tensor
producer0_layer
producer1_layer
first_consumer_layer
last_consumer_layer
consumer_count
```

這讓 simulator 可以知道：

```text
這個 compiled layer 的真實 TFLite output 後面還有沒有 consumer？
```

如果有 consumer，而且目前 simulator 會替 consumer pre-load synthetic input，那 producer 的 DRAM write 很可能只是中間 verification boundary。

修法方向：

```cpp
if (suppressible && G.consumer_count > 0 && G.last_consumer_layer > int32_t(k)) {
    producer_no_store[k] = true;
}
```

`suppressible` 不能亂加。每個 op path 必須真的支援 suppress store。

目前比較安全的類別：

| Op | 為什麼相對安全 |
|---|---|
| CONV / DWCONV / FC | path 已有 `suppress_producer_store` |
| ADD / MUL / SUB | EWE path 已有 barrier / streamed handling |
| AVG_POOL / MAX_POOL | pool path 已支援 deferred store |
| D2SPACE | UDMA d2s path 可 skip write 或留下 barrier |

比較需要小心的類別：

| Op | 風險 |
|---|---|
| RESHAPE | 現在多是 DRAM copy，若省掉要確定 layout / consumer |
| GATHER | index semantics，不一定只是 layout boundary |
| SOFTMAX | row-wise tiled path，要補 L1-resident handoff |
| GELU / HARD_SWISH | unary path 要補 pending / source-fusion |
| CONCAT | logical concat 可省，但 branch producer 判斷要精準 |

---

## 21.8 Case Study：ETHZ_v6 `imdn_quant`

修之前，`imdn_quant` profile 會看到：

```text
L001 conv  512x512x32 -> 512x512x32  dram_w = 8 MB
L002 conv  512x512x24 -> 512x512x32  dram_w = 8 MB
L003 conv  512x512x24 -> 512x512x32  dram_w = 8 MB
L004 conv  512x512x24 -> 512x512x8   dram_w = 2 MB
L005 concat                              dram_w = 1 byte
```

`L005 concat` 已經是 metadata-only，但 concat 前的 branch producer 還是把 output 寫回 DRAM。這表示：

```text
concat 自己省了，但是 concat input producer 沒省。
```

用 GraphMeta 檢查會看到這些 branch conv 的 output 後面仍有 consumer：

```text
consumer_count > 0
last_consumer_layer > producer_layer
```

修完後：

```text
imdn_quant:
  middle dram_w: ~104 MB -> ~0 MB
  total dram_w : ~107 MB -> ~3 MB
  sim time     : 6.656 ms -> 5.088 ms
```

剩下的 3 MB 是最後 `D2SPACE` output，這是 final output，不應該省。

---

## 21.9 Case Study：ETHZ_v6 `imdn_float`

同一個修法也適用 FP16 storage path。

修完後：

```text
imdn_float:
  middle dram_w: ~208 MB -> ~0 MB
  total dram_w : only final output ~6 MB
```

這說明省 DRAM write 不是 quant-only optimization，而是 memory scheduling / graph boundary optimization。

---

## 21.10 什麼時候不要省 DRAM write

Performance optimization 最危險的地方是：省掉看似中間的 store，但其實後面真的需要。

不要省的情況：

| 情況 | 原因 |
|---|---|
| final output | host / verification 要讀 |
| graph output tensor | external visible boundary |
| multi-tile producer full tensor 不在 L1 | 下一層無法整顆 source-fuse |
| consumer 需要不同 layout | L1 bytes 不等於 consumer input bytes |
| op path 沒有 barrier | 省 store 可能讓 done time 太早 |
| graph metadata 不可靠 | producer/consumer 可能被 unsupported op 隱藏 |

省 store 前要回答三個問題：

```text
1. 下一層的 input bytes 從哪裡來？
2. 如果不寫 DRAM，誰提供 dependency done tag？
3. 這層還需要 per-layer verification 嗎？
```

答不出來就先不要省。

---

## 21.11 Fix performance bug 的標準流程

建議 junior 照這個流程做：

```text
1. 找 reproducer
2. 讀 profile CSV
3. 找 top dram_w / dram_r layer
4. 分類：final output / intermediate / multi-tile / logical boundary
5. 找 source code path
6. 加最小修法
7. rebuild
8. rerun target model
9. rerun neighboring model
10. 比較 before/after profile
```

對應 command：

```bash
make -C systemc -s
./batch/run_ethz_v6.py --filter imdn_quant
./batch/run_ethz_v6.py --filter imdn_float
./batch/run_ethz_v6.py --filter resnet_quant
```

比較 profile：

```bash
python3 - <<'PY'
import csv

for m in ["imdn_quant", "imdn_float", "resnet_quant"]:
    rows = list(csv.DictReader(open(f"batch/output/{m}.profile.csv")))
    mid = sum(int(r["dram_w"]) for r in rows[:-1])
    total = sum(int(r["dram_w"]) for r in rows)
    print(f"{m:14s} mid_w={mid/1024/1024:.6f} MB total_w={total/1024/1024:.6f} MB")
PY
```

---

## 21.12 看 Gantt：確認不是假省

省掉 `dram_w` 後還要看 Gantt。

要確認：

| Gantt 現象 | 正確期待 |
|---|---|
| producer 沒有大段 UDMA_W | 中間 write 已省 |
| 仍有小 barrier | dependency 還在 |
| consumer 沒有不合理提前 | tag dependency 正確 |
| UDMA_R/EWE/UDMA_W 有 overlap | microblock wavefront 正常 |
| layer done time 沒變 0 | profile accounting 正確 |

如果 profile 變好但 Gantt 有 race 味道，不能收工。

---

## 21.13 如果 UDMA_R 還是 dominate：Activation Compression

前面的 tile fuse、microblock fuse、DRAM write suppress，主要是在省中間 boundary 的 read/write。可是有些模型修完後，profile 仍然會像這樣：

```text
per-engine busy:
   udma_r:  1974579 cyc  (82.1 %)
     conv:   665035 cyc  (27.7 %)
  requant:   995813 cyc  (41.4 %)
```

這代表主要瓶頸還在「從 DRAM 把 activation tile 搬進 L1」。這時候只增加 CONV MAC 或 EWE lanes，幫助會有限，因為 compute engine 還是在等資料。

一個合理的下一級硬體解法是加 **ACTC（Activation Compression / Decompression）**：

```text
DRAM compressed ACT
    -> UDMA_R + ACT_DECOMP
    -> L1 normal NHWC tile
    -> CONV / EWE / POOL / REQUANT existing path
```

第一版建議只做：

```text
DRAM compressed, L1 decompressed
```

不要一開始就讓 CONV 直接讀 compressed L1。原因是 CONV 需要 3x3 window、halo、stride、padding，input address 必須像一般 NHWC tile 一樣連續。先在 UDMA_R path decompress 到 L1，可以讓既有 CONV/EWE/POOL 完全不用改，風險最低。

### 21.13.1 ACTC 放在哪裡

推薦資料路徑：

```text
UDMA_R normal:
  DRAM raw bytes -> L1 raw bytes

UDMA_R + ACT_DECOMP:
  DRAM compressed block stream -> ACT_DECOMP -> L1 raw NHWC tile

UDMA_W + ACT_COMP:
  L1 raw NHWC tile -> ACT_COMP -> DRAM compressed block stream
```

也就是 ACTC 是 memory path resource，不是 CONV engine 的一部分。

硬體 block 可以想成：

| Block | 功能 |
|---|---|
| `ACT_DECOMP` | DRAM compressed activation 解回 L1 raw tile |
| `ACT_COMP` | L1 raw activation 壓成 DRAM compressed format |
| block metadata reader | 讀每個 compressed block 的 offset / size / raw fallback flag |
| raw fallback path | 壓不下去時直接搬 raw bytes |

### 21.13.2 Compression format 要保守

第一版不要追求最強壓縮率，要追求硬體簡單、latency 可估、worst case 安全。

建議 block granularity：

| 欄位 | 建議 |
|---|---|
| block size | 64B 或 128B raw block |
| granularity | row-major NHWC，盡量不要跨太多 row |
| dtype | INT8 / INT16 / FP16 都以 storage byte stream 處理 |
| metadata | per-block offset + compressed length + raw flag |
| fallback | compressed size >= raw size 時存 raw |

可先支援的 lossless scheme：

| Scheme | 適合資料 | 硬體成本 |
|---|---|---|
| zero-run / repeated-value RLE | activation sparse 或大量相同值 | 低 |
| base-delta | 鄰近 activation 數值變化小 | 中 |
| small dictionary | 小 block 內常見 byte pattern | 中 |
| raw block | 壓縮無效時 fallback | 低 |

最重要的是 raw fallback。沒有 raw fallback，worst case 可能變大，compiler 和 DRAM allocator 會很難保證空間。

### 21.13.3 Performance 怎麼估

先把 DRAM read 拆成：

```text
total_udma_r = act_read + weight_read + params_read + layout_read
```

ACT compression 只會改善 `act_read`，不會改善 weight/params。

粗估：

```text
effective_act_read = act_read / compression_ratio
effective_udma_r   = effective_act_read + weight_read + params_read + metadata_read
```

例如一個 model：

```text
DRAM total read = 74 MB
其中 activation read = 50 MB
weight + params = 24 MB
ACT compression ratio = 2.0
metadata overhead = 1 MB
```

那新的 DRAM read 近似：

```text
50 / 2 + 24 + 1 = 50 MB
```

這不代表 latency 會直接從 74/50 等比例下降，因為 bottleneck 可能轉移到 Requant、CONV、EWE、或 ACT_DECOMP 自己。但如果原本 `udma_r` 是 80% 以上，通常會有感。

### 21.13.4 Cycle model 要加哪些東西

Simulator 不應該只把 bytes 乘上一個 compression ratio。比較正確的建模要包含：

| 成本 | 說明 |
|---|---|
| compressed DRAM read bytes | 真的少從 DRAM 讀 |
| metadata read bytes | offset table / block header 也要讀 |
| decompress cycles | ACT_DECOMP lanes / bytes per cycle |
| L1 write raw bytes | 解壓後寫入 L1 的 bytes 不變 |
| descriptor startup | 新 UDMA mode 或 ACTC descriptor 仍有 decode cost |
| fallback ratio | 部分 block 可能 raw，不會壓縮 |

第一版可新增兩個 UDMA mode：

```text
UM_ACT_DECOMP_COPY
UM_ACT_COMP_COPY
```

descriptor body 可以沿用 `src_addr / dst_addr / length`，再把 `idx_table_addr` 指向 compressed block table。

### 21.13.5 什麼時候 ACT compression 幫助小

ACT compression 不是萬靈丹。

| 情況 | 為什麼幫助小 |
|---|---|
| weight read dominate | ACT 不是主要 DRAM traffic |
| Requant dominate | UDMA_R 降低後瓶頸轉到 Requant |
| activation entropy 高 | 壓縮率接近 1 |
| metadata 太碎 | block table overhead 吃掉收益 |
| L1 write dominate | L1 raw write bytes 沒變 |
| latency-critical 小 tensor | descriptor / metadata overhead 可能比省 bytes 大 |

所以 ACTC patch 必須用 profile 驗證：

```text
before:
  udma_r busy
  dram_r bytes
  top act-read layers

after:
  effective compressed bytes
  ACT_DECOMP busy
  total sim time
  correctness PASS
```

### 21.13.6 對 junior 的判斷口訣

看到 `UDMA_R dominate` 時，先照順序問：

```text
1. 是不是重複讀 weight？先做 persistent weight。
2. 是不是 branch 共用 input？先做 fanout input tile reuse。
3. 是不是 H tile halo 重複讀？需要 rolling halo / L1 rotate。
4. ACT read 還是很大？才考慮 ACT compression。
```

ACT compression 是硬體資源，不是 scheduling 小修。它適合放在 architecture roadmap，並用 simulator 先做趨勢實驗。

---

## 21.14 常見誤解

| 誤解 | 正確理解 |
|---|---|
| `dram_w` 越少一定越正確 | 可能只是錯誤 skip final output |
| `dram_w=1` 是 bug | 可能是 deliberate 1-byte barrier |
| concat 已經 metadata-only 就沒事 | concat 的 branch producer 也可能還在寫 DRAM |
| multi-tile conv 一定能 fuse | full tensor 不在 L1，通常不能直接 layer fuse |
| `ok` 代表 performance model 正確 | `ok` 只代表 byte check pass，不代表 memory schedule 最佳 |
| 只看 total ms 就能 debug | 要看 per-layer `dram_r/w` 和 Gantt |
| 加 ACT compression 一定會變快 | 只有 ACT DRAM read 是瓶頸且壓縮率夠好時才會明顯 |

---

## 21.15 本章小結

Performance debug 的核心不是「看到慢就調參」，而是建立一條證據鏈：

```text
profile hotspot
  -> layer shape / op / dtype
  -> memory traffic classification
  -> source code scheduling path
  -> minimal fix
  -> before/after profile
  -> neighboring regression
```

本章最重要的三句話：

```text
Tile fuse 省 layer boundary。
Microblock fuse 省 tile boundary 等待。
DRAM write suppress 省中間 verification boundary。
ACT compression 省 DRAM activation read bandwidth。
```

真正好的 performance patch，應該同時做到：

- correctness 還是 `ok`。
- `dram_w` 明確下降。
- Gantt dependency 合理。
- 對鄰近 model 沒 regression。
- 你能說清楚哪些 write 省了，哪些 write 不該省。

> 下一步：把本章流程用在 `pynet_v2_*`、`sam_float`、`sd_*`，逐一分類剩下的 `reshape / softmax / gelu / h_swish` 中間 write。
