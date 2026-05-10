# MDLA7 Handoff

日期時間：2026-05-10 11:35:34 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 microblock commit：`d51a330 Strengthen microblock paths 7-10`
- 前一版 microblock tail commit：`a8dd50f Expand microblock fused pipeline tails`
- Gantt task-meta commit：`a490aab Add task-meta microblock Gantt timeline`
- Fast-only runner commit：`09d97cb Implement fused microblock fast-only runners`
- 工作樹仍有使用者/文件/profile 類未提交修改；不要混進 microblock code commit。

## 已完成

- Profile HTML 有兩張 Gantt：
  - original engine timeline
  - microblock stage timeline
- `task_meta` 已從 Command Engine trace 到 profile JSON，Gantt 用 descriptor metadata 判斷 microblock。
- `run_*.py` 已統一支援 `--fast-only`。
- Microblock engine path 狀態：

| Path | Microblock engine pipeline | 狀態 |
|---|---|---|
| 1 | `CONV/DW/FC -> Requant -> store/forward` | Done |
| 2 | `CONV -> Requant -> EWE ADD/MUL/SUB -> store/forward` | Done |
| 3 | `CONV -> Requant -> D2SPACE` | Done |
| 4 | `Binary EWE chain -> store/forward` | Done |
| 5 | `Binary EWE chain -> D2SPACE` | Done |
| 6 | `CONV -> Requant -> EWE ADD -> unary EWE/ReLU -> store/forward` | Done |
| 7 | `CONV -> Requant -> EWE ADD -> pool/tnps consumer` | Done；`pool_tail` corpus reach 還需要自然模型覆蓋 |
| 8 | `POOL -> unary/EWE/TNPS consumer` | Done basic path |
| 9 | `TNPS/layout -> CONV` / `layout producer -> compute consumer` | Partial；`sslice -> conv` 還需要真正 L1 slice-feed 泛化 |
| 10 | `producer -> multiple consumers / concat fanout` | Done；包含 direct fanout、concat pointwise、Requant direct packed concat |
| 11 | `UDMA_R activation preload -> CONV/Requant` head/tail streaming | Done；3x3 和 1x1 都已套用 |
| 12 | `Requant -> packed concat L1 -> pointwise CONV` | Done；不再需要 TNPS pack descriptor |
| 13 | `UDMA_W store as microblock tail` | Done |
| 14 | `UDMA_R/UDMA_W as engine lanes in scheduling/timeline` | Done |
- 已補強：
  - Binary EWE chain -> unary EWE tail
  - CONV/Requant -> EWE -> real-window/global POOL tail
  - POOL producer microblock tail to binary EWE / unary EWE / D2S(TNPS)
  - TRANSPOSE/PACK/UNPACK/SPLIT GraphMeta layout handoff no-store
  - conv fanout generalized to safe CONV/DW/FC row-tiled subset
  - tiled POOL / unary EWE descriptor metadata
  - POOL layer cycle accounting
  - RESHAPE/SQUEEZE/EXPAND_DIMS L1 view passthrough safe subset
  - Requant strided output mode for direct packed concat L1 writes
  - UDMA activation streaming preload：用 head-ready / tail-busy descriptor
    降低 tiled Conv bubble。延伸到 1x1 pointwise Conv 後，
    `xlsr_quant` fast 從 0.698 ms 改善到 0.497 ms。

## Microblock Path 策略

未來模型一定還會出現更多 layer 組合，但實作不應該一直用
Path 15、Path 16 這種方式加 special-case。比較正確的方向是把
engine path 收斂成可重用的 microblock pattern：

| Pattern | 覆蓋範圍 | 狀態 |
|---|---|---|
| Producer compute | `CONV/DW/FC/EWE/POOL/TNPS -> L1 tile` | 大多已實作 |
| Consumer tail | `producer -> EWE/POOL/TNPS/store` | 已實作，部分已泛化 |
| Layout bridge | `sslice/concat/d2space/transpose -> compute` | Partial；下一個主要整理目標 |
| Fanout/live-range | `one producer -> multiple consumers` | Partial；safe subset 已實作 |
| UDMA as engine | `udma_r/udma_w preload/store` 參與 pipeline | Done |
| Streaming preload | `udma_r head/tail -> compute overlap` | Done |

已知未來可能遇到的 layer 組合，應該映射到這些 pattern，而不是再增加新的
hardcoded path：

| 未來可能的 path 形狀 | 應落到的實作 pattern |
|---|---|
| `CONV -> activation -> POOL -> store` | Consumer tail |
| `CONV -> Requant -> CONCAT -> CONV` | Packed concat layout bridge |
| `sslice -> CONV` | Layout bridge with L1 slice-feed |
| `transpose/layout -> FC/CONV` | Layout bridge |
| `producer -> two CONV + one EWE` | Fanout/live-range |
| `POOL -> CONV` | Producer/consumer L1 handoff |
| `TNPS pack -> CONV/EWE` | Layout bridge |
| `Requant output -> multi-consumer live range` | Fanout/live-range |

## 最近驗證

```bash
make -C systemc -s
git diff --check
./batch/run_model.py --model model/MLPerf_Tiny/vww_96_int8.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
```

結果：

- `vww_96_int8`: `31/31 PASS`
- `llama2_quant_L35_L74`: `43/43 PASS`
- `mobilenet_v3_quant`: `123/123 PASS`
- `gpt2_quant_L24_L63`: `42/42 PASS`
- `path7_pool_tail_synth`: `3/3 PASS`，`CONV -> ADD -> MAX_POOL`
  觸發 Path 7 pool_tail，三層皆 `tiles=12x1`

## GPT2 L2 結論

`MDLA7 profile - gpt2_quant_L24_L63.tflite` 的 L2 現在有 FC OC-slice microblock。

- L2 是 `FC in=1x1x768 out=1x1x768`
- `tiles=1x3`，以 256 OC 為預設 slice，3 個 microblocks
- profile JSON 已有 `task_meta` layer 2：
  - `udma_r`: input preload + 3 個 weight-slice loads
  - `conv`: 3 個 FC compute microblocks
  - `requant`: 3 個 requant microblocks
- `gpt2_quant_L24_L63`: `42/42 PASS`

實作邊界：

- 這版是 `OC slice x full-K`，也就是每個 microblock 載一段 weight rows，
  input vector / params 留在 L1。
- 還不是 partial-K accumulation；真的 K-slice 需要 CONV engine 增加 psum
  buffer / accumulate descriptor 協定。

## Next To-do

目前下一步若要再加強，可做：

0. Path 7 POOL tail corpus coverage
   `try_stream_conv_ewe()` 裡已有 `pool_tail` implementation，但掃描目前
   `batch/output/**/*.profile.json` 共 614 個 profile 後，沒有任何
   `CONV/DW/FC -> binary EWE -> POOL` layer-order candidate；
   `profiles_with_stream_pool_meta=11` 都是 standalone POOL streaming/tiling
   path（例如 `unet_*`、`inception_v3_*`），不是 Path 7 pool_tail。
   已補 `systemc/scripts/gen_path7_pool_tail_synth.py` 產生
   `CONV -> ADD -> MAX_POOL` synthetic MDL7 program，tensor 大到不能整層塞進 L1；
   驗證 `3/3 PASS`，profile `task_meta` 顯示 pool layer 2 有 mb stream flags。
   Real corpus 仍需要等自然模型/切片覆蓋這個 topology。

1. `sslice/layout -> CONV` true L1 slice-feed
   L8/L16 已靠 UDMA head/tail streaming 大幅減少 bubble，但仍從 DRAM reload
   sliced inputs。真正結構性解法是讓 upstream producer tile / packed concat
   output 的 live range 留在 L1，layout bridge 直接餵後續 Conv。

2. True TNPS tiled kernels
   目前 `TRANSPOSE/PACK/UNPACK/SPLIT` 對 GraphMeta-confirmed intermediate
   boundary 會做 no-store handoff；任意 layout permute 的真正 tile kernel
   仍是後續工作。

3. Multi-source concat / packed layout bridge cleanup
   目前 concat pointwise path 已可用 Requant direct packed L1 output；
   任意 concat source mix 仍需要更通用的 layout bridge / ownership model。

4. FC partial-K accumulation
   GPT2 L2 已有 FC OC-slice microblock；若要做到真正 K-slice，CONV engine
   需要支援多段 partial sum accumulate，再由最後一段觸發 Requant。

5. Wider fanout ownership model
   目前 fanout 放寬到 safe CONV/DW/FC subset；任意 producer multiple-consumer 仍需更完整的 live-range / slot ownership。

## 快速命令

```bash
make -C systemc -s
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
```

Profile entry：

```text
batch/profile_hotspot.html
batch/profile_ethz_v6.html
batch/output/<stem>.html
```
