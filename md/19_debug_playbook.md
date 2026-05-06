# 第 19 章 — Debug Playbook：從 N-FAIL 到 Root Cause

> 上一章：[第 18 章 — Regression Scripts 與 Profile HTML](18_regression_profile_html.md)

本章你會學到什麼：

- 面對 `N-FAIL` 時如何有系統地縮小範圍。
- compile-fail、sim-fail、functional fail、cycle regression 的不同處理法。
- 如何判斷是 compiler、descriptor、engine、memory、scheduler 哪一層。
- 常見真實 bug pattern。

---

## 19.1 第一原則

不要一看到 FAIL 就改 engine。

先分類：

```text
compile-fail?
sim crash / timeout?
N-FAIL?
cycle regression?
HTML/profile issue?
```

每一類的 root cause search path 不同。

---

## 19.2 N-FAIL triage

步驟：

1. 找 summary：幾層 PASS / FAIL。
2. 找第一個 FAIL layer。
3. 看該 layer op / dtype / shape。
4. 看前一層是否 PASS。
5. 看該 layer input 是否 chained / fused / streamed。
6. 開 HTML profile 看該 layer timeline。
7. 回 source 查 descriptor generation。

第一個 FAIL 通常比後面 FAIL 有價值，因為後面可能只是吃到壞 input。

---

## 19.3 compile-fail

常見原因：

| 原因 | 看哪裡 |
|---|---|
| unsupported op | `SUPPORTED_OPS` / skip log |
| unsupported stride | compile log |
| shape overflow | dim > 65535 |
| file/address > 4GB | budget guard |
| missing weight buffer | DEQUANTIZE producer |
| unsupported quantization | zp_w / per-channel condition |

先修 compiler，再跑 single model，不要直接進 C++。

---

## 19.4 sim-fail / timeout

常見原因：

| 原因 | 症狀 |
|---|---|
| descriptor deadlock | sc_start 到 cap，last_activity 沒進 |
| bad address | SC_REPORT_ERROR 或 crash |
| engine missing done | Command Engine wait forever |
| chain mismatch | Requant read blocking |
| DRAM too小 | out-of-bounds / crash |

查：

- Command Engine dispatch log。
- last engine done log。
- wait tags。
- layer_done_tag。
- chain psum count。

---

## 19.5 Functional fail：先判斷 writer

問：

```text
這個 layer output 是誰寫的？
```

| op | writer |
|---|---|
| CONV/DWCONV/FC | Requant |
| ADD/MUL/SUB | EWE |
| POOL | POOL |
| SOFTMAX | EWE |
| RESHAPE/CONCAT/GATHER/D2S | UDMA / materialized movement |

找錯 writer 會浪費很多時間。

---

## 19.6 Functional fail：再判斷 pattern

| pattern | 可能原因 |
|---|---|
| 全部元素錯 | dtype / address / params |
| 只有邊界錯 | padding / halo |
| channel range 錯 | OC tiling / params offset |
| 每隔 16 channel 錯 | chain lane ordering |
| off by 1 | rounding / activation clamp |
| only multi-tile | L1 lifetime / barrier |
| only stream mode | priority / slot reuse |
| only FP | tolerance / FP16 rounding |

---

## 19.7 Memory lifetime fail

懷疑條件：

- single tile pass。
- conservative mode pass。
- stream mode fail。
- fail layer 之前有 suppress store。
- branch / concat / D2S tail。

查：

```text
L1 addresses
slot_free_tag
pending store
make_store_barrier
consumer_count
last_consumer_layer
```

---

## 19.8 Quantization fail

查：

| 檢查 | 說明 |
|---|---|
| `zp_in_eff` | padding |
| `zp_out` shift | uint8 output |
| `mult/shift` | quantize multiplier |
| `bias_eff` | bias + zero-point folding |
| correction map | asymmetric `zp_w` |
| activation clamp | fused activation |
| int16_out | INT16 output range |

如果 mismatch 是 1-LSB，優先看 rounding。

---

## 19.9 Cycle regression playbook

步驟：

1. 比 regression CSV 的 ms。
2. 找 profile CSV top cycle layer。
3. 看 tiles 是否變多。
4. 看 streamed 是否變 false。
5. 看 DRAM bytes 是否變多。
6. 看 UDMA descriptor count。
7. 看 Gantt idle gaps。
8. 手算 compute / memory estimate。

不要只看總 ms，總 ms 會掩蓋是哪一層變慢。

---

## 19.10 寫 fix 時的基本原則

- 先寫最小修正。
- 保留 existing dirty worktree，不 revert unrelated changes。
- 修完跑最小 failing model。
- 再跑相鄰 regression。
- 更新 notebook / handoff 若改 architecture。
- 若是 cycle tuning，保留 correctness 優先。

---

## 19.11 常見誤解

| 誤解 | 正確理解 |
|---|---|
| 第一個想到哪就改哪 | 先分類，找 first fail |
| 多個 FAIL 要一起看 | 先看第一個 FAIL |
| N-FAIL 是壞狀態 | N-FAIL 提供最多 debug 資訊 |
| cycle regression 一定是 cycle formula | scheduler / memory / tiling 都可能 |
| fix performance 可犧牲 small mismatch | correctness 先行 |

---

## 19.12 本章小結

Debug playbook 的核心：

```text
classify -> first fail -> writer -> input/params/dependency -> source fix -> regression
```

做久了你會發現，很多難 bug 不是因為數學難，而是資料生命週期、byte width、descriptor dependency 其中一個小合約錯了。

> 下一章 → [第 20 章 — Junior Exercises 與 Roadmap](20_junior_exercises_roadmap.md)
