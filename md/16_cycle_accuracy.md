# 第 16 章 — Cycle Model 與 Cycle Accuracy

> 上一章：[第 15 章 — TileCommand / Microblock Wavefront Scheduler](15_tilecommand_microblock.md)

本章你會學到什麼：

- MDLA7 simulator 的 cycle 來源。
- CONV bit-mult model、Requant lanes、EWE / POOL cycles 如何估。
- Memory latency 如何和 compute overlap。
- wall time、busy time、utilization 的差異。
- cycle accuracy 目前能相信什麼，不能相信什麼。
- junior 如何 debug cycle regression。

---

## 16.1 Cycle accuracy 的層級

先分清楚三種「準」：

| 層級 | 問題 |
|---|---|
| functional accuracy | output bytes 對不對 |
| performance model | 大致瓶頸和趨勢對不對 |
| RTL cycle accuracy | 每 cycle event 和硬體 RTL 對齊 |

目前 MDLA7 SystemC 屬於：

```text
functional simulator + first-order cycle model
```

它不是 RTL cycle-accurate model，但會捕捉：

- compute throughput
- tile fill overhead
- memory bandwidth
- DRAM row miss / refresh
- dependency scheduling / overlap
- per-engine busy timeline

---

## 16.2 Simulation time unit

source 裡常看到：

```cpp
wait(cycles, sc_core::SC_NS);
```

在這份 simulator 中，`SC_NS` 被當作 abstract cycle unit。外層報告用：

```text
cycles @ 1.9 GHz -> ms
```

換算：

```text
ms = cycles / 1.9e6
```

不要把 `SC_NS` 當真實 nanosecond。它是 SystemC time carrier。

---

## 16.3 CONV bit-mult model

CONV cycles：

```text
cycles = ceil(MAC_total * a_bits * b_bits / 1,048,576) + 64
```

其中：

```text
MAC_total = Kh * Kw * in_per_group * OH * OW * OC
```

`a_bits` / `b_bits` 依 dtype：

| dtype | bits |
|---|---|
| INT8x8 | 8 × 8 |
| INT16x8 | 16 × 8 |
| INT16x16 | 16 × 16 |
| FP16 | 16 × 16 |

`+64` 是 tile fill。每個 CONV descriptor 都付一次。

---

## 16.4 Tiling 對 CONV cycles 的影響

假設一層不切 tile：

```text
cycles = compute + 64
```

切成 8 個 tile：

```text
cycles = sum(tile compute) + 8 * 64
```

MAC_total 大致相同，但 fill overhead 增加。OH tiling 還可能增加 input halo memory traffic。

因此：

```text
tile 太大：L1 放不下
tile 太小：fill / UDMA startup / halo overhead 變大
```

cycle tuning 就是在這兩者之間找平衡。

---

## 16.5 Requant cycle model

Requant lanes：

```cpp
LANES = 512
```

cycle：

```text
ceil(output_elements / 512)
```

它代表 CONV / EWE 共用的 quantize-pack / clamp resource。Functional path 仍用 chain FIFO，但 timing 上假設硬體有 512 elem/cycle 吞吐。

---

## 16.6 EWE cycle model

EWE lanes 依 dtype：

| dtype | lanes |
|---|---:|
| INT8 | 64 |
| INT16 | 32 |
| FP | 32 |

| op | cycle |
|---|---|
| ADD / MUL / SUB | `ceil(elems / lanes)` |
| HARD_SWISH / GELU | `ceil(elems / lanes)` |
| SOFTMAX | `3 * ceil(elems / lanes)` |

softmax 是三 pass：max / exp-sum / divide。

---

## 16.7 POOL cycle model

POOL cycle：

```text
out_elems = OH * OW * OC
per_lane = ceil(out_elems / lanes)
cycles = per_lane * max(Kh * Kw, 1)
```

lanes 跟 EWE 相同：INT8=64、INT16=32、FP=32。

global average pool 的 `Kh*Kw` 可能很大，所以 tail op 不一定便宜。

---

## 16.8 UDMA and memory cycles

UDMA 本身：

```text
16 cycles startup per descriptor
```

Memory：

```text
L1 sequential peak ~= 256 B/cycle * SRAM ratio
DRAM ~= 48 B/cycle + row miss + refresh
```

UDMA descriptor time 大致：

```text
read source latency + write destination latency + 16
```

若 source 是 DRAM、destination 是 L1：

```text
DRAM read dominates for large transfer
```

---

## 16.9 Overlap model

CONV/Requant 使用補差方式：

```cpp
elapsed = sc_time_stamp() - t_begin;
if (compute_cycles > elapsed)
    wait(compute_cycles - elapsed);
```

意思是：

```text
engine time = max(memory elapsed, compute formula)
```

這模擬 operand streaming 和 compute overlap。

其他 engine 的 overlap 模型不完全相同，讀 cycle 時要回 source 確認。

---

## 16.10 Wall time vs busy time

Profile 裡常有：

| 指標 | 意義 |
|---|---|
| wall time | 整個 simulation 從 start 到 last activity |
| engine busy time | 某 engine 正在處理 descriptor 的累積時間 |
| utilization | busy time / wall time |
| layer cycles | layer done tag fire time difference |

高 busy time 不一定壞。如果多 engine overlap 好，wall time 仍可能低。

低 busy time 也不一定好。如果 wall time 高但 engines idle，可能 dependency 太保守。

---

## 16.11 Ideal cycle

HTML profile 會顯示 ideal cycle / cumulative ideal cycle。這通常是用 layer compute estimate 做比較。

用法：

```text
actual cycles >> ideal cycles
```

可能原因：

- memory dominated
- UDMA overhead
- dependency serialization
- tile fill overhead
- unsupported fusion / handoff

ideal 不等於目標硬體保證，只是 sanity baseline。

---

## 16.12 Cycle regression debug

步驟：

1. 找哪個 model ms 變了。
2. 看 summary cycles。
3. 看 profile layer table，找 cycles_layer 增加最多的 layer。
4. 看 engine timeline，是 CONV、UDMA、EWE、POOL 哪個 lane 變長。
5. 手算該 layer 的 compute / memory first-order estimate。
6. 看 tiles_h / tiles_oc 是否變多。
7. 看 streamed / handoff 是否從 true 變 false。

---

## 16.13 常見誤解

| 誤解 | 正確理解 |
|---|---|
| PASS 就代表 cycle 準 | PASS 只代表 functional reference match |
| cycle model 是 RTL 級 | 目前是 first-order performance model |
| CONV cycles 只看 MAC | tile fill、memory overlap、tiling 都影響 |
| UDMA 只看 bytes | descriptor count、row miss、refresh 也影響 |
| utilization 越高一定越好 | 要一起看 wall time 和 overlap |

---

## 16.14 本章小結

Cycle model 的主線：

```text
compute formula
memory latency
descriptor dependency
overlap
profile reporting
```

Cycle debug 不要只看一個數字。要從 model summary 下鑽到 layer，再下鑽到 engine timeline 和 descriptor / tiling decision。

> 下一章 → [第 17 章 — Functional Verification 與 SystemC Function Coverage](17_verification_coverage.md)
