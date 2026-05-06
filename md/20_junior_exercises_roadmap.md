# 第 20 章 — Junior Exercises 與 Roadmap

> 上一章：[第 19 章 — Debug Playbook：從 N-FAIL 到 Root Cause](19_debug_playbook.md)

本章你會學到什麼：

- 新進工程師如何用 4 週建立 MDLA7 codebase 手感。
- 每個主題可以做哪些練習。
- 怎麼從讀 code 進到修 bug、加 coverage、做 architecture improvement。
- 目前 source 的 roadmap 和下一步建議。

---

## 20.1 你已經讀完哪些主題

這本 notebook 到這裡涵蓋：

| Part | 主題 |
|---|---|
| I | repo map、build/run |
| II | HW spec、descriptor、memory、engines |
| III | compiler、quantization、program.bin |
| IV | SystemC top、Host/CmdEng、data path |
| V | tiling、fusion、microblock、cycle |
| VI | verification、regression |
| VII | debug playbook、roadmap |

你不需要一次背完。目標是遇到問題時知道去哪裡查。

---

## 20.2 第一週：跑起來

目標：熟悉工具和 log。

練習：

```bash
cd systemc
python3 run_model.py --list
python3 run_model.py inception_v3_quant --keep-intermediate
python3 run_model.py unet_int16 --keep-intermediate
```

要能回答：

- `.bin`、`.profile.csv`、`.html` 在哪裡？
- summary 怎麼看？
- 第一個 layer log 包含哪些欄位？
- HTML Gantt 怎麼 zoom / search？

---

## 20.3 第二週：讀 compiler

目標：理解 `.tflite -> LayerMeta -> reference`。

練習：

1. 找一個 CONV layer 的 compile log。
2. 在 `compile_model.py` 找該 op path。
3. 手算 output shape。
4. 找 params blob layout。
5. 打開 `program.bin` 對應 LayerMeta 概念。

要能回答：

- `zp_in_eff` 何時非 0？
- INT16x8 如何判斷？
- FP32 source 為什麼 lowered to FP16？
- skip layer 會如何 break chain？

---

## 20.4 第三週：讀 SystemC datapath

目標：能從 descriptor 追到 engine output。

練習：

1. 找一個 CONV descriptor。
2. 追 `CommandEngine::issue()`。
3. 追 `ConvEngine::run()`。
4. 追 chain 到 `RequantEngine::run()`。
5. 看 output store UDMA。

要能回答：

- CONV 為什麼不直接寫 final output？
- Requant total elements 怎麼算？
- chain lane `oc & 0xF` 如何 drain？
- done tag 如何回到 Command Engine？

---

## 20.5 第四週：做一個小修正

建議小題：

| 題目 | 學到 |
|---|---|
| 加一個 profile CSV 欄位 | profile flow |
| 補一個 compile skip reason | compiler robustness |
| 改善 HTML label | report/debug |
| 增加一個 synthetic op case | coverage |
| 調整 microblock tile size實驗 | scheduler/cycle |

小修正流程：

```text
read source
make small patch
run target model
run neighboring regression
write note
```

---

## 20.6 建議閱讀順序回顧

如果你要重新快速複習：

1. [第 0 章](00_intro.md)
2. [第 1 章](01_build_and_run.md)
3. [第 3 章](03_descriptor_tag.md)
4. [第 6 章](06_tflite_flatbuffer.md)
5. [第 8 章](08_program_bin_reference.md)
6. [第 10 章](10_host_command_engine.md)
7. [第 11 章](11_conv_requant_datapath.md)
8. [第 14 章](14_tiling_fusion_handoff.md)
9. [第 18 章](18_regression_profile_html.md)
10. [第 19 章](19_debug_playbook.md)

這十章可以讓你快速進入實戰。

---

## 20.7 Roadmap：architecture

handoff 提到的下一個大方向：

```text
Command Engine consumes coarse TileCommand objects,
then internally schedules microblocks across UDMA / compute / store.
```

建議拆法：

| Step | 工作 |
|---|---|
| 1 | 定義 TileCommand binary/schema |
| 2 | 保留現有 descriptor path 作 fallback |
| 3 | 把 binary EWE wavefront 從 test_model 搬到 CmdEng-like expander |
| 4 | 加 stream profile identity |
| 5 | 跑 real model A/B |
| 6 | 再搬 CONV-D2S-EWE path |

原則：

```text
先不破壞 correctness，再增加 overlap。
```

---

## 20.8 Roadmap：coverage

建議建立 coverage tracking：

| Coverage | 方法 |
|---|---|
| op × dtype | regression CSV aggregate |
| descriptor mode | count op_class / UDMA mode |
| scheduler mode | streamed / non-streamed / tail |
| memory mode | DRAM bytes / SRAM bytes / D2S / concat |
| quant mode | zp_w correction / int16 / fp |

可以先從 `.profile.csv` 和 compile log parse，不需要一開始做 EDA-style coverage tool。

---

## 20.9 Roadmap：cycle accuracy

可改善方向：

- L1 read/write same-bank conflict。
- DRAM per-bank outstanding model。
- UDMA read/write channel overlap model。
- EWE/POOL memory-compute overlap consistency。
- Command Engine dispatch overhead。
- descriptor FIFO pressure。
- tile fill model per dtype / kernel。

每一項都要配 regression，避免 tuning 後失去可解釋性。

---

## 20.10 Roadmap：compiler

可改善方向：

- 更完整 high-rank shape / axis handling。
- broadcast lowering。
- more op support。
- better TFLite fidelity beyond self-consistency。
- robust model-size handling。
- clearer skip diagnostics。
- structured intermediate dump for debug。

Compiler 是最容易同時影響 correctness 和 coverage 的地方，改動要小心。

---

## 20.11 Junior 的工作習慣

建議養成：

- 每次改 code 前先跑一個 failing reproducer。
- 每次修 bug 後記下 root cause 分類。
- 把 log 裡的 layer id、op、dtype、shape 寫在筆記。
- 不確定時先做小 synthetic case。
- 不要把 performance optimization 和 correctness fix 混在一個 patch。
- 遇到 dirty worktree，不要 revert 別人的變更。

---

## 20.12 最終小結

MDLA7 codebase 的主線是：

```text
TFLite graph
  -> compiler lowering
  -> program.bin
  -> descriptor generation
  -> Command Engine scheduling
  -> SystemC engines
  -> memory hierarchy
  -> verification and profile
```

你作為新進工程師，最重要的能力不是一次懂所有細節，而是遇到一個 FAIL 時能沿著這條主線往回找。

這本 notebook 到第 20 章完成。下一步可以把它拿來做：

- 新人 onboarding。
- debug checklist。
- architecture review。
- coverage planning。
- cycle tuning baseline。

> 完結。
