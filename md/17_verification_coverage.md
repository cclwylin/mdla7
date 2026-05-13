# 第 17 章 — Functional Verification 與 SystemC Function Coverage

> 上一章：[第 16 章 — Cycle Model 與 Cycle Accuracy](16_cycle_accuracy.md)

本章你會學到什麼：

- MDLA7 如何做 functional verification。
- embedded numpy reference 和 TFLite fidelity 的差異。
- PASS / FAIL / skipped layer 要怎麼解讀。
- SystemC function coverage 應該怎麼建立。
- junior 如何設計新 op 的測試矩陣。

---

## 17.1 Verification 的三個層級

| 層級 | 比對對象 | 用途 |
|---|---|---|
| self-consistency | SystemC output vs compiler numpy reference | daily regression 主力 |
| TFLite fidelity | simulator/reference vs real TFLite Interpreter | 確認 lowering 接近 TFLite |
| RTL/hardware fidelity | SystemC vs RTL / silicon | 未來 signoff 類工作 |

目前 notebook 主要討論前兩個。

---

## 17.2 Embedded reference

compiler 對每個 layer 算 reference bytes，寫入 `program.bin`。

test_model 跑完後：

```text
read actual output from simulated DRAM
compare with reference bytes
```

優點：

- 快。
- deterministic。
- 不需要每次啟動 TensorFlow Interpreter。
- 可測 SystemC engine 和 scheduler。

限制：

```text
如果 compiler reference 本身和 TFLite lowering 都錯，
self-consistency 仍可能 PASS。
```

所以需要 TFLite fidelity 工具補充。

---

## 17.3 INT bit-exact

INT8 / INT16 path 預期 bit-exact：

```text
actual bytes == reference bytes
```

若 mismatch，通常代表：

- descriptor wrong
- engine math wrong
- quant params wrong
- memory lifetime wrong
- byte width wrong

INT path 不應用 tolerance 掩蓋錯誤。

---

## 17.4 FP tolerance

FP path 使用 tolerance，例如 1e-3。原因在第 7 章已講過：

- FP reduction order。
- FP16 rounding。
- math library differences。
- nonlinear approximation。

FP verification 要記錄：

| 指標 | 用途 |
|---|---|
| max abs diff | 最大誤差 |
| mean abs diff | 整體漂移 |
| mismatch count above tol | PASS/FAIL 判定 |

---

## 17.5 validate_tflite.py

[`validate_tflite.py`](../systemc/scripts/validate_tflite.py) 用 real TFLite Interpreter 幫忙檢查 fidelity。

它可做：

- compile model。
- run TFLite interpreter。
- preserve intermediate tensors。
- compare layer output。

handoff 裡提醒：

```text
Layer 0 最容易 bit-exact comparable。
N>0 可能因 chained input / skipped op 而不是完全 TFLite full inference。
```

所以讀 validate result 時要分清「self-consistency」與「TFLite intermediate fidelity」。

---

## 17.6 PASS / FAIL / N-FAIL

test_model summary：

```text
summary: P/N layers PASS, F FAIL
```

regression script 可能轉成：

```text
ok
N-FAIL
compile-fail
sim-fail
timeout
```

`N-FAIL` 的意思：

```text
simulation completed，但有 N 個 layer mismatch。
```

這比 compile-fail 好，因為你有 profile、logs、fail layer 可以 debug。

ETHZ/BMM 目前還有一條更硬的規則：

```text
compile-skipped:N 不能算 clean PASS。
```

如果原始 TFLite op 沒有被 compiler 支援，它必須變成明確的 lowered layer，
或明確的 `matrlz` supported-but-not-native fallback。`matrlz` 會保留 layer /
final reference check，但不能被解讀成 native arithmetic datapath 已完成。
native-only coverage 要用：

```bash
systemc/scripts/audit_unsupported_ops.py --strict-native model/BMM model/ETHZ_v6
```

---

## 17.7 SystemC function coverage

這裡的 function coverage 不是 EDA coverage database，而是 simulator 功能覆蓋表：

| coverage axis | examples |
|---|---|
| op coverage | CONV、DWCONV、FC、ADD、MUL、SUB、POOL、SOFTMAX |
| dtype coverage | INT8、UINT8-shift、INT16x8、INT16x16、FP16 |
| shape coverage | 1x1、3x3、depthwise、large H/W、OC tiling |
| memory mode coverage | linear、strided、concat、gather、d2s |
| scheduler coverage | normal、stream、tail barrier、handoff |
| quant coverage | per-tensor、per-channel、zp_w correction、fused activation |

這些可由 regression CSV 和 profile JSON 彙整。

---

## 17.8 建立 coverage matrix

一個實用表格：

| Feature | Unit test | Real model | Status |
|---|---|---|---|
| INT8 CONV | synthetic conv | mobilenet / inception | covered |
| DWCONV | synthetic / mobilenet | mobilenet_v3 | covered |
| INT16x8 | unet_int16 | unet_int16 | covered |
| FP CONV | mobilenet_v3_float | fp models | covered |
| FP SOFTMAX | inception_v3_float | inception | covered |
| BATCH_MATMUL fallback | BMM synthetic / SAM slice | model/BMM | materialized |
| CONCAT | inception_v3_quant | inception | covered |
| GATHER | llama2_quant | transformer | covered |
| D2S | vsr / xlsr | VSR-like | covered |

這張表要隨 source 更新，不要只寫一次。

---

## 17.9 新增 op 的 verification checklist

1. compiler 支援 op extraction。
2. numpy reference。
3. params blob layout。
4. descriptor body mapping。
5. SystemC engine functional path。
6. byte width and dtype。
7. small synthetic test。
8. real model regression。
9. profile sanity。
10. TFLite fidelity spot check。

---

## 17.10 常見誤解

| 誤解 | 正確理解 |
|---|---|
| PASS 等於完全 TFLite fidelity | PASS 先表示 self-consistency |
| FP 一定要 bit-exact | FP path 合理 tolerance 更實際 |
| coverage 只看 op count | dtype、shape、scheduler、memory mode 都要看 |
| N-FAIL 沒用 | N-FAIL 是 debug 最有資訊的狀態 |
| skipped layer 不影響 downstream | skip 會 break chain，仍需理解 coverage gap |

---

## 17.11 本章小結

Verification 要同時看：

```text
SystemC vs embedded reference
compiler/reference vs TFLite
feature coverage matrix
regression status trend
```

對 junior 來說，最好的習慣是每修一個 bug，都補一筆 coverage note：它覆蓋哪個 op、dtype、shape、scheduler case。

> 下一章 → [第 18 章 — Regression Scripts 與 Profile HTML](18_regression_profile_html.md)
