# 第 15 章 — TileCommand / Microblock Wavefront Scheduler

> 上一章：[第 14 章 — Tiling、Fusion、Pending Store、L1-Resident Handoff](14_tiling_fusion_handoff.md)

本章你會學到什麼：

- `TileCommand` / `Microblock` 在目前 source 中的定位。
- wavefront scheduler 如何把大 tensor 拆成 load / compute / store microblocks。
- `DF_STREAM`、`stream_slot`、`microblock_id`、`stream_meta_flags` 如何配合 Command Engine。
- binary EWE streaming 和 CONV-D2S-EWE streaming 的基本結構。
- 這套設計和 handoff.md 提到的 cleaner architecture 的關係。

---

## 15.1 背景

handoff 裡提到下一步希望有更乾淨架構：

```text
Host/compiler emits coarse TileCommand
Command Engine expands into block-level UDMA/compute/store work
```

目前 source 裡已經有部分雛形在 [`test_model.cpp`](../systemc/src/test_model.cpp)：

```cpp
struct TileCommand { ... };
struct Microblock { ... };
emit_binary_ewe_wavefront(...)
```

它還不是完整 Command Engine 內部 tile expander，但已經把 microblock streaming 的概念做出來。

---

## 15.2 TileCommand

`TileCommand` 代表比 descriptor 更高階的工作：

```text
這一層 / 這個 fused pattern
要用哪些 L1 buffer
tile size 多大
是否 suppress store
```

重要欄位：

| 欄位 | 說明 |
|---|---|
| `kind` | `BINARY_EWE` 或 `CONV_D2S_EWE` |
| `layer_idx` / `layer_end` | 涵蓋 layer range |
| `layer` | LayerMeta copy |
| `params_l1` | params buffer in L1 |
| `tile_elems` / `tile_rows` | microblock size |
| `elem_size` | 1 or 2 bytes |
| `h_tiled` | 是否 row-based tiling |
| `suppress_store` | 是否不 store final output |
| `in_a_l1` / `in_b_l1` / `out_l1` | ping-pong buffers |

---

## 15.3 Microblock

`Microblock` 是實際 stream unit：

```cpp
struct Microblock {
    uint16_t id;
    uint8_t slot;
    uint64_t elem_off;
    uint32_t rows;
    uint32_t elems;
    uint32_t bytes;
};
```

解讀：

| 欄位 | 用途 |
|---|---|
| `id` | ordering / tie-break |
| `slot` | ping-pong slot |
| `elem_off` | global element offset |
| `rows` | row tile count |
| `elems` | element count |
| `bytes` | transfer bytes |

Microblock 是 descriptor stream metadata 的來源。

---

## 15.4 mark_stream

`mark_stream()` 將 descriptor 標記成 stream work：

```cpp
d.hdr.flags |= DF_STREAM;
d.hdr.layer_id = layer_idx;
d.hdr.stream_slot = mb.slot;
d.hdr.microblock_id = mb.id;
d.hdr.stream_meta_flags = meta_flags;
```

`meta_flags` 包含：

| flag | 說明 |
|---|---|
| `SMF_LOAD_A` | load input A |
| `SMF_LOAD_B` | load input B / weight / params |
| `SMF_COMPUTE` | compute descriptor |
| `SMF_STORE` | store output |
| `SMF_FINAL_TILE` | final microblock |

Command Engine 的 stream priority 會利用這些資訊。

---

## 15.5 Binary EWE wavefront

`emit_binary_ewe_wavefront()` 對大 element-wise tensor 做：

```text
for each microblock:
    UDMA load A
    UDMA load B
    EWE compute
    optional UDMA store
```

每個 microblock 用 tags 串起：

```text
A load tag
B load tag
EWE waits A/B
store waits EWE
slot_free waits store
```

如果 suppress store，最後 done tag 可能是 EWE tag。

---

## 15.6 Slot free tag

每個 ping-pong slot 有 `slot_free_tag[2]`：

```text
下一次使用同一 slot 前，要等上一次 store / compute 完成
```

這保護 L1 buffer reuse：

```text
slot0 load new tile
must wait slot0 old store or compute done
```

沒有 slot_free tag，stream scheduler 很容易覆蓋 live data。

---

## 15.7 Wavefront overlap

理想 timeline：

```text
mb0: load A/B -> compute -> store
mb1:          load A/B -> compute -> store
mb2:                   load A/B -> compute -> store
```

Command Engine priority 讓：

- EWE compute 優先啟動。
- UDMA read 早於 UDMA write。
- store 背景化。

這就是 microblock wavefront 的價值。

---

## 15.8 CONV-D2S-EWE streaming

`TileCommand::CONV_D2S_EWE` 代表 fused pattern：

```text
CONV
  -> Requant
  -> DEPTH_TO_SPACE
  -> ADD / EWE
```

這種 pattern 用 microblock 表示更複雜的 tail：

- load input / weight / params。
- compute conv / requant。
- D2S transform。
- load ADD branch / params。
- EWE compute。
- store final output。

它比 plain conv chain 更接近 real pipeline，但 correctness 條件也更多。

---

## 15.8.1 CONV-EWE microblock streaming

`test_model.cpp` 也有保守的 `CONV -> EWE(ADD/MUL/SUB)` streaming path。它不是把
EWE 接進 CONV/Requant 的硬體 chain，而是在 row microblock 上做 handoff：

```text
load CONV input tile
CONV -> Requant writes tile to L1
load EWE input-B tile / params
EWE consumes CONV tile directly
optional store EWE output
```

啟用條件刻意保守：

- CONV output shape / dtype 必須等於下一層 EWE input/output。
- CONV intermediate store 必須是 producer-no-store boundary。
- single-tile CONV 留給原本 L1-resident fused path；新路徑只處理需要 H-tiling
  的大型 residual。
- EWE output 若還是 producer，store 可以 suppress；slot reuse 等 EWE done tag。

這主要服務 Deeplab / large residual 類 pattern，避免 `CONV store -> EWE reload`
在大 activation 上造成 DRAM 與 L1Mesh hotspot。

## 15.8.2 CONV-CONV microblock streaming

`try_stream_conv_chain()` 現在把 Conv chain 分成兩種安全等級：

- `CONV(1x1) -> CONV(1x1) -> ...` linear pointwise chain 可以用 generic
  microblock streaming，只要每個 intermediate 是 direct single-consumer
  boundary。
- `CONV(3x3 SAME) -> ... -> DEPTH_TO_SPACE -> EWE` 保留既有的 deep stream
  path，會為下游 spatial layer 擴張 halo rows。

Plain `CONV(3x3 SAME) -> CONV(3x3 SAME)` chain 仍先走保守 per-layer tiler。
U-Net 類 tiled pair 會立刻暴露 seam correctness 風險；要開這條路徑，需要
更明確的 line-buffer / halo ownership，而不是只靠 row-range expansion。

## 15.8.3 Binary EWE microblock chain

`try_stream_binary_ewe_chain()` 針對線性的 `ADD/MUL/SUB -> ADD/MUL/SUB`
chain 做真正的 L1 handoff。條件刻意保守：

- INT8 binary EWE。
- 每層 shape / dtype 相同。
- `GraphMeta` 確認 producer output 是下一層的 `input0`，避免 ADD/SUB/MUL
  量化參數因 input0/input1 對調而失真。
- 中間 producer 允許 `no-store`。
- chain 尾端若直接接 `SOFTMAX`，會退回原本的 per-layer EWE wavefront；
  這條 path 能把 attention matrix 以 contiguous L1 tensor 留給 softmax，
  避免 softmax 重新從 DRAM 讀整張 matrix。

每個 microblock 會先載入第一層 input-A tile，接著對同一個 tile 依序跑
多個 EWE stage。中間結果只在兩個 L1 data buffer ping-pong，不寫回 DRAM；
每層自己的 input-B tile 和 48B params 仍從 DRAM 載入。Profile 的 `flow`
欄位會把這些 stage 標成同一個 flow，例如 sd decoder slice 的
`L9 -> L10 -> L11`。

同時，Hotspot 裡 `RESHAPE/MATERIALIZE` 若只是中間 graph boundary 且大小不變，
會被當成 metadata/reference checkpoint 跳過 DRAM copy；下游 layer 的 synthetic
input 已由 compiler materialize，所以 functional correctness 仍由後續可驗證 layer
錨定。

同樣的 rule 也套到中間 `CONV/DWCONV/FC` 與 unary EWE (`GELU/HARD_SWISH`)
producer：GraphMeta 顯示後面還有 consumer 時，不再把這個 transient activation
寫回 DRAM；若現有 fused/tiled path 能保留 L1 layout，就走真 on-chip handoff，
否則 consumer 仍使用 compiler 已 materialize 的 synthetic input。

## 15.8.4 Path 7：CONV/Requant -> EWE -> POOL/TNPS consumer

`try_stream_conv_ewe()` 現在不只支援 binary/unary/D2S tail，也支援 POOL tail。
POOL 不再只限 `1x1 / stride1 / no-pad` safe subset；scheduler 會用 consumer
POOL 的 output row 切 microblock，再反推這個 POOL window 需要的 producer rows：

```text
producer rows = output rows * pool_stride + pool_kernel - 1
conv input rows = producer rows * conv_stride + conv_kernel - 1
```

因此一個 microblock 內可以是：

```text
UDMA load CONV input rows
CONV + Requant -> L1 data0
UDMA load EWE input-B rows
EWE ADD/MUL/SUB -> L1 data1
POOL real/global window -> L1 data0
optional store or forward
```

這條 path 的限制仍然保守：

- dtype / channel 必須一致。
- GraphMeta 若存在，producer 必須是 direct single consumer。
- POOL padding 必須能放進 descriptor 的 3-bit pad field。
- 任意 overlapping multi-consumer POOL live range 還不走這條。

## 15.8.5 Path 8：POOL -> EWE/TNPS consumer

新增 `try_stream_pool_consumer()`，讓大型 POOL producer 可以直接接後續 consumer，
不必先完整寫回 DRAM。支援的 tail：

| Tail | Microblock 行為 |
|---|---|
| `POOL -> ADD/MUL/SUB` | POOL output tile 當 EWE input-A，載入 input-B tile 後 compute |
| `POOL -> GELU/HARD_SWISH` | POOL output tile 直接進 unary EWE |
| `POOL -> D2SPACE` | POOL output tile 交給 TNPS D2S |

這條 path 仍只開 INT8 safe subset，並要求 GraphMeta direct producer-consumer。
如果 POOL output 本身可以完整留在 L1，舊的 layer-level fuse 仍可用；這條 path
主要服務 output / input 太大、需要 row microblock 的情況。

## 15.8.6 Path 9：TNPS/layout -> compute consumer

Layout op 現在分兩層看：

- 真 kernel / materialized layout：`TRANSPOSE`、`SLICE`、`STRIDED_SLICE`、
  `D2SPACE/S2D` 等仍可走 TNPS descriptor。
- GraphMeta-confirmed intermediate handoff：`TRANSPOSE/PACK/UNPACK/SPLIT`
  若只是 producer 到後續 compute 的 transient boundary，scheduler 會 suppress
  DRAM checkpoint，讓 profile/cycle 反映 layout producer -> compute consumer 的
  on-chip handoff intent。

這不是宣稱任意 transpose 都已經 tile-kernel 化。它的安全假設是 compiler 已經為
consumer materialize synthetic input bytes；functional correctness 由後續 layer
verify 錨定。真正的任意 permute tile kernel 仍是後續工作。

## 15.8.7 Path 10：producer fanout

`try_stream_conv_fanout()` 從原本的 `CONV` special-case 放寬到 safe
`CONV/DWCONV/FC` subset。多個 consecutive branch 若共用同一 logical input，
且後面接 concat-like boundary，可以共用 input microblock：

```text
UDMA load shared input tile
branch0 CONV/DW/FC + Requant
branch1 CONV/DW/FC + Requant
...
logical CONCAT / fanout boundary
```

目前仍是保守 fanout framework：

- branch 的 spatial shape / dtype / kernel / stride / pad 要匹配。
- DW 只接受 depthwise-safe shape。
- FC 只接受 `1x1` output subset。
- 通用 producer -> multiple arbitrary consumers 還需要更完整的 live-range /
  slot ownership model。

## 15.8.8 目前 10 path 狀態

| Path | Pipeline | 狀態 |
|---|---|---|
| 1 | `CONV/DW/FC -> Requant -> store/forward` | 可用 |
| 2 | `CONV -> Requant -> EWE ADD/MUL/SUB -> store/forward` | 可用 |
| 3 | `CONV -> Requant -> D2SPACE` | 可用 |
| 4 | `ADD/MUL/SUB -> ADD/MUL/SUB -> store/forward` | 可用 |
| 5 | `ADD/MUL/SUB chain -> D2SPACE` | 可用 |
| 6 | `CONV/Requant -> EWE -> unary EWE` | 可用 |
| 7 | `CONV/Requant -> EWE -> POOL/TNPS consumer` | 可用，real-window POOL row microblock |
| 8 | `POOL -> unary/binary EWE/TNPS consumer` | 可用，INT8 safe subset |
| 9 | `TNPS/layout -> compute consumer` | 可用於 GraphMeta handoff no-store；true arbitrary layout tile kernel 待補 |
| 10 | `producer tile -> multiple consumers / concat fanout` | 可用於 safe CONV/DW/FC fanout subset |

---

## 15.8.9 FC OC-slice microblock

`1x1xK -> 1x1xOC` 的 safe FC subset 現在可以用 output-channel slicing
產生 microblock。這條 path 不是 pseudo metadata；它真的把 weight matrix
按 OC rows 切成多段：

```text
load FC params once
load input vector once
mb0: UDMA_R weight OC[0:256]   -> FC full-K -> Requant OC[0:256]
mb1: UDMA_R weight OC[256:512] -> FC full-K -> Requant OC[256:512]
mb2: UDMA_R weight OC[512:768] -> FC full-K -> Requant OC[512:768]
```

每個 microblock 的 `Requant.oc_start` 指向原 layer 的 OC offset，output
slice 寫回完整 output tensor 的對應位置。若 producer store 被 suppress，
完整 output 仍留在 L1；否則每個 OC slice 以 UDMA_W store drain。

目前邊界：

- 只吃 `OK_FC`、`in/out H=W=1`、`group=1`、非 FP dtype。
- 預設 `tile_oc=256`，不足時按 16-channel alignment 往下縮。
- 這是 `OC slice x full-K`；真正 partial-K accumulation 需要 CONV engine
  增加 psum buffer / accumulate descriptor 協定。

---

## 15.9 Stream metadata 和 HTML profile

`layer_id`、`microblock_id`、`stream_slot` 不只是 scheduling，也能讓 profile 更可讀：

```text
L22 slot=1 mb=4 load_b
L22 slot=1 mb=4 compute
L22 slot=1 mb=4 store
```

這對 debug overlap 很有價值。

---

## 15.10 Cleaner architecture 的方向

目前 TileCommand expansion 在 `test_model.cpp`，比較像 test harness scheduler。handoff 希望未來改成：

```text
Host/compiler emits coarse TileCommand
Command Engine internally expands microblocks
```

好處：

| 好處 | 說明 |
|---|---|
| abstraction 清楚 | compiler 不必手排每個 low-level descriptor |
| Command Engine 更像 hardware scheduler | microblock scheduling 集中 |
| easier tuning | priority / overlap policy 在一處 |
| profile 更一致 | TileCommand 作為 layer/tile identity |

這是 architecture roadmap，不是目前已完全完成的設計。

---

## 15.11 Debug checklist

| 問題 | 檢查 |
|---|---|
| slot overwrite | `slot_free_tag` |
| stream descriptor 不越序 | 是否有 `DF_STREAM` |
| compute 太晚 | priority / wait tags |
| store 擋 load | UDMA direction / priority |
| final tile 卡住 | `SMF_FINAL_TILE` / tail barrier |
| profile label 錯 | `layer_id` / `microblock_id` |

---

## 15.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| TileCommand 已是最終硬體 command format | 目前是 source 裡的 scheduler abstraction 雛形 |
| Microblock 不需要 dependency tag | 每個 load/compute/store 都靠 tag 保護 |
| ping-pong slot 自動安全 | slot reuse 要等 slot_free tag |
| stream priority 可任意調 | 調錯會造成 lifetime hazard |
| suppress store 只看 current microblock | 還要看 layer consumer / final output |

---

## 15.13 本章小結

Microblock wavefront 是把 coarse tensor work 變成可 overlap 的小工作：

```text
TileCommand -> Microblocks -> stream descriptors -> Command Engine scheduling
```

這章也是理解 future architecture 的入口。若要繼續優化 performance，最可能動到：

1. TileCommand schema。
2. microblock size。
3. stream priority。
4. slot lifetime。
5. Command Engine expansion boundary。

> 下一章 → [第 16 章 — Cycle Model 與 Cycle Accuracy](16_cycle_accuracy.md)
