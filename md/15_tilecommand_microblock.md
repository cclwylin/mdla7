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
