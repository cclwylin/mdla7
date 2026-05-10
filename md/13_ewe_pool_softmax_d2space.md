# 第 13 章 — EWE / POOL / SOFTMAX / D2SPACE

> 上一章：[第 12 章 — UDMA、DRAM Model、L1Manager Module Design](12_udma_dram_l1manager.md)

本章你會學到什麼：

- EWE engine 如何支援 binary、unary、softmax。
- POOL engine 如何支援 max / avg / global。
- DEPTH_TO_SPACE 何時由 Requant final-store、TNPS、或 legacy UDMA fallback 執行。
- 這些 non-CONV op 如何和 L1 handoff / streaming scheduler 互動。
- 常見模型 tail op 的 debug 方法。

---

## 13.1 Non-CONV op 的角色

真實模型不只有 convolution。常見 non-CONV：

| 類型 | 例子 |
|---|---|
| element-wise | ADD、MUL、SUB |
| activation | HARD_SWISH、GELU |
| normalization-like tail | SOFTMAX |
| spatial reduction | AVG_POOL、MAX_POOL、MEAN |
| layout transform | RESHAPE、CONCAT、GATHER、DEPTH_TO_SPACE |

MDLA7 simulator 把其中一部分放在 EWE / POOL；layout transform 主路徑由 TNPS 處理，`CONV -> final DEPTH_TO_SPACE` 則可直接併入 Requant final-store。

---

## 13.2 EWE binary op

EWE binary descriptor：

```text
in_a_addr
in_b_addr
out_addr
h,w,c
lut_addr
subtype = ADD/MUL/SUB
```

INT path 使用 quant params；FP path 使用 FP16 storage + FP32 compute。

Dependency：

```text
EWE waits input A producer
EWE waits input B / params load
store waits EWE done
```

branch model 常見 ADD residual，所以 input A / B 的 producer layer 很重要。

---

## 13.3 Binary EWE streaming

`test_model.cpp` 有 `TileCommand::BINARY_EWE` 和 `emit_binary_ewe_wavefront()`，用 microblocks 做：

```text
load A tile
load B tile
EWE compute
store output
```

兩個 ping-pong slot 可以 overlap：

```text
slot0 compute while slot1 load
slot1 compute while slot0 store
```

這對大 element-wise tensor 很重要，否則整層 load/compute/store 會太串行。

---

## 13.4 Unary EWE

HARD_SWISH / GELU 使用 single input：

```text
in_a_addr
in_b_addr = 0
params = clamp sentinel
```

FP path implemented：

| op | formula |
|---|---|
| HARD_SWISH | `x * relu6(x+3) / 6` |
| GELU | tanh approximation |

INT unary support 目前較有限。若 INT model 有 unsupported unary，compiler / scheduler 可能 skip 或 chain-preserve。

---

## 13.5 Softmax

Softmax 也是 EWE subtype：

```text
subtype = ES_SOFTMAX
```

FP softmax：

```text
max reduce
exp / sum
divide
FP16 output
```

INT softmax：

```text
softmax_int8 LUT path
```

Softmax 的 cycle 是三 pass。對 transformer attention 大 tensor，softmax 可能是 visible bottleneck。

---

## 13.6 POOL

POOL descriptor：

```text
input shape
output shape
mode
kernel
stride
padding
count_include_pad
```

MAX pool：

```text
output = max(valid window)
```

AVG / GLOBAL：

```text
output = rounded or FP average(valid window or full window)
```

MEAN 在 compiler 裡可 route via AVG_POOL 類 path。

---

## 13.7 Pool tiling

Large pool 也可能 height tiled。注意：

- global pool kernel 可能用 `255` sentinel。
- tile split 不能破壞 reduction window。
- avg divisor 要和 `count_include_pad` 一致。

若 pool output off-by-one，先看 rounding；若整片錯，先看 kernel / stride / sentinel。

POOL 現在也可以當 microblock producer / consumer tail：

| Pattern | Scheduler 行為 |
|---|---|
| `CONV/Requant -> EWE -> POOL` | 以 POOL output row 切 microblock，反推 EWE / CONV 需要的 producer rows |
| `POOL -> ADD/MUL/SUB` | POOL output tile 直接作為 binary EWE input-A |
| `POOL -> GELU/HARD_SWISH` | POOL output tile 直接作為 unary EWE input |
| `POOL -> D2SPACE` | POOL output tile 交給 TNPS D2S |

這些 path 不代表 POOL 硬接在 CONV/Requant datapath 後面；POOL 仍是獨立 engine。
差別在於 Command Engine 用 dependency tag 和 L1 slot ownership，把 producer
tile 直接交給 consumer，省掉中間 DRAM checkpoint。

---

## 13.8 DEPTH_TO_SPACE

DEPTH_TO_SPACE 是 NHWC pixel-shuffle layout transform。現在的分工是：

| Pattern | 執行位置 |
|---|---|
| `CONV/FC/DWCONV -> DEPTH_TO_SPACE` 且 D2SPACE 是 final output | Requant final-store D2SPACE swizzle |
| `CONV/FC/DWCONV -> DEPTH_TO_SPACE -> consumer` | TNPS tiled streaming path |
| 非 CONV producer 或 standalone D2SPACE | TNPS `TM_DEPTH_TO_SPACE` |
| legacy / debug fallback | UDMA `UM_DEPTH_TO_SPACE` |

NHWC mapping：

```text
input [ih, iw, ic]
q  = ic / Cout
oc = ic % Cout
bh = q / block
bw = q % block
output [ih*block + bh, iw*block + bw, oc]
```

它看似 reshape，但在 memory layout 中需要 byte reorder。

Final-output special case 不需要先把 Requant output 線性寫到 L1，再由 TNPS 重排；Requant drain CONV chain 時直接用上面的 mapping 寫 final DRAM address。Profile 仍保留 D2SPACE layer 的 final DRAM W 統計，但 D2SPACE/TNPS cycles 會是 0。

---

## 13.9 D2S + ADD streaming path

handoff 裡提到 VSR-like path：

```text
CONV -> DEPTH_TO_SPACE -> ADD
```

這種 tail 比普通 conv chain 複雜：

- CONV output channel count 會因 D2S 改變。
- D2S layout transform 必須在 ADD 前完成。
- ADD 可能還需要另一個 branch input。
- store barrier 要保護 tail output。

所以 scheduler 對 channel-changing tail 會比 plain conv chain 更保守。只有當 D2SPACE 是 final output 時，才可改用 Requant final-store swizzle；若後面還有 ADD/consumer，仍要讓 TNPS 產生正確的 consumer tile layout。

---

## 13.10 CONCAT / GATHER / RESHAPE

這些 op 常走 data movement / materialized reference path：

| op | 常見實作 |
|---|---|
| RESHAPE | byte passthrough or DRAM copy |
| CONCAT | materialized concat / TNPS concat / barrier |
| GATHER | numpy materialized reference + UDMA copy |
| MATRLZ | compiler pre-materialized reference + chunked `DRAM -> L1 -> DRAM` UDMA copy |
| TRANSPOSE / PACK / UNPACK / SPLIT | TNPS/materialized layout；若 GraphMeta 確認是 intermediate handoff，可 suppress DRAM checkpoint |

要記住：

```text
不是所有 graph op 都有 dedicated engine。
有些 op 是 compiler materialize + UDMA passthrough。
```

`matrlz` 是目前用來取代 Hotspot skipped rows 的明確 fallback。它保留
graph node、timing movement、PASS/FAIL verification，但不表示 EWE/POOL/CONV
已經支援該 arithmetic。後續如果補上真 lowering，應該讓 `matrlz` 數量下降。

---

## 13.11 Debug non-CONV op

| op | debug point |
|---|---|
| ADD | input A/B dependency、params blob、broadcast |
| MUL | multiplier params、scalar broadcast |
| SUB | operand order A-B |
| HARD_SWISH | FP formula / clamp |
| GELU | tanh approximation |
| SOFTMAX | axis / 3-pass / LUT |
| AVG_POOL | divisor / rounding |
| D2S | block size / Cin=Cout*b*b |
| CONCAT | source lifetime / axis layout |
| GATHER | index table |

---

## 13.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| EWE 只是 ADD | EWE 包含 binary、unary、softmax |
| DEPTH_TO_SPACE 是 no-op reshape | NHWC 下通常要搬 bytes |
| D2SPACE 都從 TNPS 移走 | 只有 `CONV -> final D2SPACE` 併入 Requant final-store；intermediate / standalone 仍由 TNPS 做 |
| MEAN 必須有專屬 engine | 目前可 route via avg-pool-like path |
| CONCAT 一定只是 pointer alias | 多數情況需要 materialize 或 barrier |
| Layout no-store 等於 transpose kernel 完成 | 不是；GraphMeta handoff no-store 是 intermediate checkpoint suppress，任意 layout tile kernel 還是後續工作 |
| softmax cycle 很小 | 大 tensor softmax 是三 pass，可能很重 |

---

## 13.13 本章小結

Non-CONV op 是模型完整度的關鍵：

```text
EWE handles element-wise / nonlinear / softmax
POOL handles spatial reductions
TNPS handles layout movement
UDMA handles DRAM/L1 movement and fallback copies
```

Debug 時先判斷 op 是 compute 還是 layout，再追 input producer、params、output writer。

> 下一章 → [第 14 章 — Tiling、Fusion、Pending Store、L1-Resident Handoff](14_tiling_fusion_handoff.md)
