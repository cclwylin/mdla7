# 第 14 章 — Tiling、Fusion、Pending Store、L1-Resident Handoff

> 上一章：[第 13 章 — EWE / POOL / SOFTMAX / D2SPACE](13_ewe_pool_softmax_d2space.md)

本章你會學到什麼：

- 為什麼 3 MB L1Mesh 需要 tiling。
- OH tiling、OC tiling、ping-pong allocation 的基本策略。
- L1-resident handoff 如何減少 DRAM traffic。
- pending store 是什麼，為什麼不能亂 drop。
- store barrier 如何修 correctness bug。
- fusion 和 correctness 之間的 tradeoff。

---

## 14.1 為什麼需要 tiling

L1Mesh 只有 3 MB。一個 layer 可能需要：

```text
input tile
weight tile
output tile
requant params
correction map
scratch
double buffer
```

如果整層塞不下，就必須切 tile。

常見 tiling axes：

| axis | 用途 |
|---|---|
| OH | 降低 input/output activation footprint |
| OC | 降低 weight/output channel footprint |
| element microblock | element-wise 大 tensor streaming |

---

## 14.2 OH tiling

OH tiling 把 output height 切段：

```text
OH = 224
tile_oh = 32
tiles_h = 7
```

每個 tile 需要 input halo。scheduler 要算：

- input row start
- input row count
- local padding
- output dram offset
- correction map offset

OH tiling 常見 bug 是 tile 邊界 wrong。

---

## 14.3 OC tiling

OC tiling 把 output channel 切段：

```text
OC = 1024
tile_oc = 256
tiles_oc = 4
```

每個 tile 需要：

- weight slice
- params slice via `oc_start`
- output channel slice

OC tiling 常見 bug 是 params offset 或 weight offset 錯。

---

## 14.4 Tile fill overhead

每個 CONV descriptor 都付：

```text
+64 cycles tile fill
```

所以 tiling 越碎：

```text
total fill = num_conv_descriptors * 64
```

這是 performance tradeoff：

| tile 大 | tile 小 |
|---|---|
| fill overhead 少 | L1 容易放下 |
| reuse 好 | overlap 可能更細 |
| 可能爆 L1 | UDMA / fill overhead 多 |

---

## 14.5 L1-resident handoff

若 layer N output 直接留在 L1，layer N+1 直接使用：

```text
skip N store to DRAM
skip N+1 load from DRAM
```

優點：

- 降低 DRAM bandwidth。
- 降低 UDMA descriptor count。
- 可能降低 wall time。

要求：

- shape / dtype match。
- producer output buffer 未被覆蓋。
- consumer 在 correct wait tag 後開始。
- multi-consumer branch 要更保守。

---

## 14.6 pending store

pending store 是：

```text
producer layer 的 store descriptor 先不要發
等確認下一層是否能 fuse / handoff 再決定
```

若下一層成功 fuse：

```text
可以 drop store 或延後 store
```

若下一層不能 fuse：

```text
必須 flush pending store
```

這是 performance optimization，但也是 correctness 風險點。

---

## 14.7 為什麼 pending store 不能亂丟

如果 output tensor 之後還有 consumer：

```text
branch / concat / later layer / final output
```

drop store 可能讓後面找不到正確 DRAM bytes。

GraphMeta 的 `last_consumer_layer`、`consumer_count` 可以幫助判斷：

| 情況 | store policy |
|---|---|
| single direct consumer and fused | 可 suppress |
| multi-consumer | 保守 store |
| final output | 必須 store |
| shape/dtype mismatch | 必須 store |

---

## 14.8 ping-pong L1 allocation

融合 chain 可能讓多層 output 都留在 L1。若固定用同一個 `L1_OUT`，很容易覆蓋 live input。

程式裡有 `chain_alt`：

```text
try low address
try high address
toggle after successful fused layer
reset on chain break
```

目的：

```text
讓 live input 和 new output 避開，降低 3 MB L1Mesh 壓力。
```

---

## 14.9 direct producer-consumer boundary

Compiler v3 GraphMeta 提供 tensor-level relation。C++ 可以判斷：

```text
layer i output tensor
is consumed by layer i+1 input0?
dtype/shape match?
consumer count safe?
```

這比只看 layer index 更可靠，因為 TFLite graph 中可能有 skipped / elided ops。

---

## 14.10 store barrier

store barrier 是小型 UDMA store，例如 1 byte，重點是 tag ordering：

```cpp
make_store_barrier(L1_OUT, L.dram_out, barrier_tag, wait_tag)
```

它常帶：

```text
DF_STREAM | DF_STREAM_TAIL
SMF_STORE | SMF_FINAL_TILE
```

用途：

- 建立「某個 producer store / tail 已完成」的 waitable event。
- 防止 L1 buffer 太早被覆蓋。
- 讓 stream scheduler 能在 tail wait 時允許 safe prefetch。

---

## 14.11 CONCAT 與 handoff

CONCAT / branch 是 handoff 高風險場景：

```text
branch A output
branch B output
concat consumes both
```

若其中一個 branch output 被 suppress store，但 concat path 從 DRAM materialized reference 讀，就會 mismatch。

所以 concat 相關 path 常需要：

- source lifetime 檢查。
- conservative fallback。
- barrier。
- logical concat 和 materialized concat 分清楚。

---

## 14.12 channel-changing tail

Plain conv-chain streaming 假設 channel shape 穩定較容易。若 tail 會改 channel：

```text
CONV 32 channels
DEPTH_TO_SPACE -> 8 channels
ADD -> output
```

這會破壞簡單 chain assumption。scheduler 需要顯式 D2S+ADD path 或 fallback conservative tiling。

這也是為什麼某些 performance optimization 要先守 correctness。

---

## 14.13 Debug checklist

| 症狀 | 檢查 |
|---|---|
| single tile PASS, multi tile FAIL | L1 lifetime / store barrier |
| branch model FAIL | consumer_count / pending store |
| concat FAIL | source materialization / axis / barrier |
| final output missing | final store 被 suppress |
| performance 變慢 | handoff 失效或 tiling 變碎 |
| stream only FAIL | DF_STREAM priority / tail wait |

---

## 14.14 常見誤解

| 誤解 | 正確理解 |
|---|---|
| fusion 一定安全 | fusion 要滿足 lifetime 和 consumer 條件 |
| suppress store 只影響 performance | 錯 suppress 會造成 functional mismatch |
| pending store 可以一直留著 | chain break 時要 flush |
| L1 有 allocator 會自動防 overwrite | 主要靠 scheduler / descriptor dependency |
| branch output 只有下一層會用 | GraphMeta 要確認 consumer_count |

---

## 14.15 本章小結

Tiling 和 fusion 是 MDLA7 performance 的核心，但 correctness 更重要：

```text
tile to fit L1
handoff to reduce DRAM
barrier to protect lifetime
fallback when shape/lifetime unsafe
```

Debug 時要同時看：

1. L1 address range。
2. producer / consumer relation。
3. pending store state。
4. descriptor wait tags。
5. stream flags。

> 下一章 → [第 15 章 — TileCommand / Microblock Wavefront Scheduler](15_tilecommand_microblock.md)
