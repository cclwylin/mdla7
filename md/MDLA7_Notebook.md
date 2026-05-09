# MDLA7 Source Code 讀書筆記 — 150 頁教材寫作 Plan

本文件是「MDLA7 Source Code 讀書筆記」的寫作計畫與章節索引。正式章節已先完成第 0 章到第 20 章，放在 `md/00_intro.md` 到 `md/20_junior_exercises_roadmap.md`。後續若要輸出 PDF、補圖、補公式或加深某章，可以依照本 plan 繼續擴充。

讀者設定：剛進公司的 junior engineer。假設已有基本 C/C++、Python、數位邏輯、CNN / TFLite 的入門概念，但還不熟 SystemC、NPU architecture、cycle model、descriptor scheduling。

---

## 0. 教材目標

### 0.1 讀完後要能做到什麼

讀完 150 頁後，junior 應該能做到：

| 能力 | 具體成果 |
|---|---|
| 看懂系統定位 | 說明 MDLA7 是什麼、Host / Command Engine / Engines / Memory 怎麼分工 |
| 看懂 compile flow | 追 `TFLite -> compile_model.py -> program.bin -> test_model` 的資料流 |
| 看懂 descriptor | 解釋 descriptor header、body union、dependency tag、stream metadata |
| 看懂 SystemC module | 從 `SC_MODULE`、FIFO、thread、done tag 追到 engine dispatch |
| 看懂主要 engine | CONV、REQUANT、UDMA、EWE、POOL、DRAM / L1Manager 的功能與限制 |
| 看懂 regression | 會跑單 model、pattern regression、ETHZ sweep，知道 FAIL 怎麼分類 |
| 看懂 cycle number | 會解讀 cycles、ms、per-engine busy、layer profile、Gantt |
| 能改小功能 | 能新增一個 op / 修 scheduling race / 加 profile 欄位 / 補 regression |
| 能 debug | 能從 `N-FAIL` 找 failing layer，切小模型，判斷是 math 還是 scheduling |

### 0.2 不把教材寫成什麼

- 不是 spec 的逐字翻譯。
- 不是只列檔案名稱的目錄。
- 不是只講 HW，不講 compiler / simulator / regression。
- 不是只講功能正確，不講 cycle accuracy。
- 不是只講現在成功的地方，也要講 known gaps 與模型限制。

---

## 1. 交付格式

### 1.1 檔案組織

正式章節放在 `md/`，採用 `notebook.md` 建議的 build 規格：

```text
md/
  00_intro.md
  01_build_and_run.md
  02_hw_spec.md
  03_descriptor_tag.md
  04_memory_hierarchy.md
  05_compute_engines.md
  06_tflite_flatbuffer.md
  07_quantization_fp_int16.md
  08_program_bin_reference.md
  09_system_top_wiring.md
  10_host_command_engine.md
  11_conv_requant_datapath.md
  12_udma_dram_l1manager.md
  13_ewe_pool_softmax_d2space.md
  14_tiling_fusion_handoff.md
  15_tilecommand_microblock.md
  16_cycle_accuracy.md
  17_verification_coverage.md
  18_regression_profile_html.md
  19_debug_playbook.md
  20_junior_exercises_roadmap.md
  MDLA7_Notebook.md          # 本 plan，不列入 PDF 章節

drawio/
  mdla7_top.drawio
  compile_flow.drawio
  descriptor_flow.drawio
  ...

eq/
  requant_eq.tex
  roofline_eq.tex

pdf/
  SystemC_textbook.pdf
```

`scripts/build_pdf.sh` 會合併 `md/[0-9][0-9]_*.md`，所以本 plan 檔 `MDLA7_Notebook.md` 不會自動放入正式 PDF。等正式章節開始寫，再依照 `00_*.md` 命名。

### 1.2 章節格式規則

每一章用這個格式：

```markdown
# 第 N 章 — 章名

> 上一章：[第 N-1 章 — ...](0N-1_xxx.md)

本章你會學到什麼：

- ...
- ...

## N.1 ...

...

## 小結

> 下一章 → [第 N+1 章 — ...](0N+1_xxx.md)
```

寫作風格：

| 項目 | 規則 |
|---|---|
| 語言 | 繁體中文為主，保留英文術語 |
| 語氣 | 教學筆記，不寫成規格書，不寫成流水帳 |
| 程式碼引用 | 用 markdown link，例如 `[test_model.cpp](/abs/path/systemc/src/test_model.cpp:1134)` |
| code block | 使用 fenced code block，標 `cpp` / `python` / `bash` / `text` |
| 圖 | drawio 圖放 `drawio/`，章節用 `![圖說](../drawio/xxx.drawio.png)` |
| 表格 | 優先用表格整理 module、欄位、cycle、coverage |
| 每章結尾 | 必須有小結、常見誤解、下一章預告 |

---

## 2. 150 頁章節配置

目標 150 頁左右，分成 7 個 Part、22 章。頁數是寫作預算，不必逐字精準，但總量要控制在 145 到 170 頁。

| Part | 章 | 主題 | 頁數 |
|---|---:|---|---:|
| I. 入門與地圖 | 0 | MDLA7 是什麼、怎麼讀這份 repo | 6 |
| I. 入門與地圖 | 1 | Source tree 與 build / run 基礎 | 7 |
| II. HW Spec | 2 | Top architecture：Host、Command Engine、Engine、Memory | 8 |
| II. HW Spec | 3 | Descriptor ISA 與 dependency tag | 8 |
| II. HW Spec | 4 | Memory hierarchy：DRAM、UDMA、L1Mesh | 8 |
| II. HW Spec | 5 | Compute engines：CONV、REQUANT、EWE、POOL、TNPS | 10 |
| III. Compiler | 6 | TFLite flatbuffer 與 op extraction | 8 |
| III. Compiler | 7 | Quantization / FP / INT16 compile path | 9 |
| III. Compiler | 8 | program.bin 格式與 reference generation | 7 |
| IV. SystemC Architecture | 9 | `Mdla7System` top wiring | 7 |
| IV. SystemC Architecture | 10 | Host 與 Command Engine module design | 8 |
| IV. SystemC Architecture | 11 | CONV / Requant data path | 9 |
| IV. SystemC Architecture | 12 | UDMA、DRAM model、L1Manager | 8 |
| IV. SystemC Architecture | 13 | EWE / POOL / SOFTMAX / D2SPACE | 8 |
| V. Scheduler & Performance | 14 | Tiling、fusion、pending store、L1-resident handoff | 10 |
| V. Scheduler & Performance | 15 | TileCommand / microblock wavefront scheduler | 9 |
| V. Scheduler & Performance | 16 | Cycle model 與 cycle accuracy | 10 |
| VI. Verification | 17 | Functional verification 與 TFLite fidelity | 8 |
| VI. Verification | 18 | Regression scripts 與 profile HTML | 8 |
| VII. 實戰 | 19 | Debug playbook：從 N-FAIL 到 root cause | 7 |
| VII. 實戰 | 20 | Junior exercises 與 roadmap | 7 |
| VII. 實戰 | 21 | Performance bug：profile、DRAM write、UDMA_R、ACT compression | 8 |
|  |  | **合計** | **168** |

如果必須壓到剛好 150 頁，優先縮：

- 第 5 章 engine overview：10 -> 8
- 第 15 章 microblock：9 -> 7
- 第 20 章 exercises：7 -> 5

---

## 3. 正式章節大綱

### 第 0 章 — MDLA7 是什麼、怎麼讀這份 repo

頁數目標：6

本章定位：

- 用一張 top diagram 建立全局印象。
- 先解釋 MDLA7 是 NPU simulator，不是單純 Python model runner。
- 說明「compiler + descriptor + SystemC simulator + regression」四件事是一體的。

必讀檔案：

| 檔案 | 讀法 |
|---|---|
| `handoff.md` | 看目前功能狀態、版本演進、known gaps |
| `spec/spec.md` | 看 HW architecture 與 spec |
| `batch/run_model.py` | 看使用者入口 |
| `systemc/src/test_model.cpp` | 看 simulator main flow |

本章圖：

- `drawio/mdla7_learning_map.drawio`
- `drawio/mdla7_end_to_end.drawio`

本章一定要回答：

- MDLA7 跟 MDLA6 baseline 的關係是什麼？
- 為什麼一個 `.tflite` model 會變成 descriptor stream？
- PASS 是不是等於完全 match TFLite？

### 第 1 章 — Source tree 與 build / run 基礎

頁數目標：7

本章教 junior 先跑起來，不先陷入細節。

必讀檔案：

| 檔案 | 用途 |
|---|---|
| `systemc/Makefile` | SystemC build |
| `systemc/setup.sh` | venv 與 Python dependency |
| `batch/run_model.py` | 單模型入口 |
| `batch/run_mdla6_pattern.py` | pattern regression |
| `batch/run_ethz_v6.py` | ETHZ sweep |

要放的 command：

```bash
./batch/run_model.py --list
./batch/run_model.py inception_v3_quant
./batch/run_mdla6_pattern.py --filter unet_int16 --rerun-all
make -s
```

本章要解釋輸出：

- compile log
- per-layer PASS / FAIL
- `sim time: <cycles> cycles @ 1.9 GHz (= <ms> ms)`
- output artifacts：`.bin`、`.profile.json`、`.profile.csv`、`.profile.png`、`.html`

### 第 2 章 — HW Spec Top Architecture

頁數目標：8

核心問題：

- Host 做什麼？
- Command Engine 做什麼？
- Engine 之間是平行還是串行？
- DRAM / L1 / UDMA 的位置在哪裡？

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `spec/spec.md` | §1、§2、§3A |
| `spec/mdla7.drawio` | top block diagram |
| `systemc/include/mdla7/system.h` | SystemC top wiring |

章節內容：

- MDLA7 top block
- RISC-V Host 與 NPU controller
- Descriptor-driven architecture
- 5 engines：CONV、REQUANT、EWE、POOL、TNPS
- Memory hierarchy：LPDDR5X、UDMA、L1Mesh

本章圖：

- `drawio/top_architecture.drawio`
- `drawio/control_data_flow.drawio`

### 第 3 章 — Descriptor ISA 與 Dependency Tag

頁數目標：8

這是 junior 最容易卡住的一章，要用很多表格。

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/descriptor.h` | descriptor header / body |
| `systemc/include/mdla7/command_engine.h` | wait tag / signal tag |
| `systemc/src/test_model.cpp` | `make_desc`、`make_udma`、`alloc_tag` |

章節內容：

- Descriptor = header + body union
- op class：UDMA、CONV、REQUANT、EWE、POOL
- dtype：INT8、INT16x16、INT16x8、FP16、BF16、FP8
- wait tag / signal tag 模型
- rolling tag allocator
- stream metadata：layer id、slot、microblock id、flags

要放的表：

| 欄位 | 意義 | junior 常見誤解 |
|---|---|---|
| `signal_tag` | engine done 後發出的 tag | 不是 layer id |
| `wait_tags` | descriptor dispatch 前要等待的 tag | 不是 data pointer |
| `op_class` | 決定送到哪個 engine | 不等於 TFLite op |
| `dtype` | engine compute / storage path | 不一定等於 model 全局 dtype |

### 第 4 章 — Memory Hierarchy：DRAM、UDMA、L1Mesh

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/memory.h` | DRAM model、L1Manager、L1Mesh |
| `systemc/include/mdla7/udma.h` | LINEAR、STRIDED_2D、D2SPACE |
| `systemc/src/test_model.cpp` | L1 layout、tile buffer placement |

章節內容：

- DRAM size 如何由 model metadata 決定
- LPDDR row hit / miss / bandwidth 的 cycle model
- UDMA read / write / strided copy / depth-to-space
- L1 SRAM 2 MB budget
- banked L1Mesh arbitration
- 為什麼 L1 reuse 會造成 scheduling race

本章練習：

- 找一個 model 的 layer，手算 input / weight / output byte size。
- 對照 profile 裡 DRAM r/w 是否合理。

### 第 5 章 — Compute Engines Overview

頁數目標：10

必讀檔案：

| Engine | 檔案 |
|---|---|
| CONV | `systemc/include/mdla7/conv_engine.h` |
| REQUANT | `systemc/include/mdla7/requant_engine.h`、`requant.h` |
| EWE / POOL | `systemc/include/mdla7/ewe_pool.h` |
| SOFTMAX LUT | `systemc/include/mdla7/softmax_lut.h` |
| FP helper | `systemc/include/mdla7/fp_utils.h` |

章節內容：

- CONV：read L1 input / weight，push int32 chain
- REQUANT：CONV / EWE 共用 512-lane quantize-pack resource，apply TFLite MBQM，write int8/int16/fp output
- EWE：ADD / MUL / SUB / activation / softmax
- POOL：AVG / MAX / global-ish pooling
- D2SPACE：final `CONV -> D2SPACE` 可由 Requant final-store swizzle；intermediate / standalone path 由 TNPS；UDMA 只保留 fallback

要特別教：

- CONV 不直接寫 final tensor，REQUANT 才寫 L1 output。
- INT16x8 是 activation int16、weight int8 的 hybrid path。
- FP path 的 storage 是 FP16 / BF16，但 accumulate 可能是 FP32。

### 第 6 章 — TFLite Flatbuffer 與 Op Extraction

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/scripts/compile_model.py` | flatbuffer parse、operator loop |
| `systemc/scripts/validate_tflite.py` | TFLite Interpreter cross-check |

章節內容：

- `.tflite` 是 flatbuffer graph
- tensor、operator、buffer、quantization params
- supported ops 與 skipped ops
- shape inference：TFLite static shape vs derived shape
- `compile_model.py` 如何把 TFLite op 映射成 MDLA layer metadata

要放的 flow：

```text
.tflite
  -> tensors / operators / buffers
  -> LayerMeta[]
  -> DRAM blobs
  -> reference outputs
  -> program.bin
```

### 第 7 章 — Quantization、FP、INT16 Compile Path

頁數目標：9

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/scripts/compile_model.py` | quant params、bias fold、FP branch |
| `systemc/include/mdla7/requant.h` | fixed-point requant |
| `systemc/include/mdla7/fp_utils.h` | FP conversion |

章節內容：

- TFLite affine quantization：scale、zero_point
- per-channel weight scale
- bias folding
- `multiply_by_quantized_multiplier`
- activation clamp
- INT16x16 vs INT16x8
- FP16 / BF16 / FP8 的資料表示

本章公式：

- affine quantization
- conv accumulation
- requantization

### 第 8 章 — program.bin 與 Reference Generation

頁數目標：7

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/scripts/compile_model.py` | binary layout、reference pack |
| `systemc/src/test_model.cpp` | loader、metadata parse、DRAM preload |

章節內容：

- program file header
- LayerMeta array
- graph meta / consumer info
- DRAM preload regions
- reference output bytes
- self-consistency verification 的意義與限制

常見誤解：

- PASS 不一定代表 TFLite bit-exact。
- skipped op 不一定代表 graph 不能跑，有些由 synthetic tensor / precomputed blob 接住。

### 第 9 章 — `Mdla7System` Top Wiring

頁數目標：7

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/system.h` | module instance 與 FIFO wiring |
| `systemc/include/mdla7/host.h` | descriptor source |
| `systemc/src/test_model.cpp` | construct system、load DRAM、run `sc_start` |

章節內容：

- SystemC module instance
- FIFO channels
- dtype latch side-channel
- CONV -> REQUANT 16-lane chain
- sc_start cap 與 simulation finish

本章圖：

- `drawio/systemc_top_wiring.drawio`

### 第 10 章 — Host 與 Command Engine Module Design

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/host.h` | Host descriptor feed |
| `systemc/include/mdla7/command_engine.h` | dispatch、collect、pending queue |
| `systemc/include/mdla7/descriptor.h` | descriptor definitions |

章節內容：

- Host 在 simulator 裡是簡化版
- Command Engine 的兩個 thread：dispatch / collect
- tag_done table
- in-order normal descriptor
- stream descriptor lookahead
- tail priority / allowed during tail wait
- done tag collect path

要用一個例子：

```text
UDMA_R(input) -> UDMA_R(weight) -> CONV -> REQUANT -> UDMA_W(output)
```

把每個 descriptor 的 wait / signal tag 畫出來。

### 第 11 章 — CONV / Requant Data Path

頁數目標：9

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/conv_engine.h` | integer / FP compute |
| `systemc/include/mdla7/requant_engine.h` | chain drain / output write |
| `systemc/include/mdla7/requant.h` | gemmlowp primitive |
| `systemc/src/test_model.cpp` | descriptor emission for conv-class ops |

章節內容：

- conv tile loop：OH outer、OC inner
- padding / stride / group
- input tile size 與 weight slice
- CONV writes int32 chain
- Requant drains NHWC order
- per-channel params slice
- corr map for asymmetric quant

Debug note：

- 如果 layer 只有邊界錯，先查 padding / zp_in / corr map。
- 如果 sparse mismatch，先查 tag / L1 overwrite / scheduling。

### 第 12 章 — UDMA、DRAM Model、L1Manager

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/udma.h` | UDMA modes |
| `systemc/include/mdla7/memory.h` | DRAM / L1 arbitration |
| `systemc/src/test_model.cpp` | L1 address assignment |

章節內容：

- LINEAR copy
- STRIDED_2D store for OC tiling
- DEPTH_TO_SPACE copy
- DRAM bandwidth cycle
- L1 bank conflict
- why UDMA is often bottleneck

本章要對照 profile：

- `udma_r busy`
- `udma_w busy`
- Gantt 裡 DRAM read / write 的位置

### 第 13 章 — EWE / POOL / SOFTMAX / D2SPACE

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/include/mdla7/ewe_pool.h` | EWE / POOL implementation |
| `systemc/include/mdla7/softmax_lut.h` | LUT |
| `systemc/scripts/compile_model.py` | ADD / MUL / SUB / POOL compile |
| `systemc/src/test_model.cpp` | `make_ewe_add`、`make_pool`、D2SPACE descriptor |

章節內容：

- ADD / MUL / SUB 的 TFLite math
- FP EWE path
- AVG / MAX pool
- MEAN routed as avg_pool
- SOFTMAX LUT
- D2SPACE as layout operation

### 第 14 章 — Tiling、Fusion、Pending Store、L1-Resident Handoff

頁數目標：10

這章是整份教材的核心之一。

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/src/test_model.cpp` | main scheduling logic |
| `handoff.md` | v8.14、v8.20、v8.21、v8.34+ notes |

章節內容：

- 為什麼要 tiling：2 MB L1 budget
- OH tiling、OC tiling
- single-tile fusion eligibility
- `pending` store：先延後，下一層決定 drop 或 flush
- `producer_no_store`
- `fuse_prev_*` state
- ping-pong allocator：low / high L1_OUT
- store barrier：只寫 1 byte 但提供 ordering

要放近期 bug case study：

| Model | Bug | Root cause | Fix |
|---|---|---|---|
| `inception_v3_quant` | 7-FAIL | logical CONCAT 沒有 boundary，downstream conv 太早 reuse L1 | concat 1-byte barrier |
| `unet_int16` | 2-FAIL | multi-tile suppressed conv store 沒有 completion boundary | suppressed producer store barrier |
| `yolo_v8_quant` | performance regression risk | conservative MUL graph 不該套 concat barrier | scope barrier path |

### 第 15 章 — TileCommand / Microblock Wavefront Scheduler

頁數目標：9

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/src/test_model.cpp` | `TileCommand`、`Microblock`、wavefront emit |
| `systemc/include/mdla7/descriptor.h` | stream metadata flags |
| `systemc/include/mdla7/command_engine.h` | stream issue priority |
| `handoff.md` | current scheduler work |

章節內容：

- coarse TileCommand vs micro-op descriptor
- microblock id / stream slot
- ping-pong slot reuse
- EWE wavefront：UDMA load tile+1 overlaps EWE tile
- VSR `CONV -> D2SPACE -> ADD`
- Command Engine stream lookahead
- what remains architectural vs simulator-only

本章圖：

- `drawio/microblock_wavefront.drawio`
- `drawio/stream_scheduler_timeline.drawio`

### 第 16 章 — Cycle Model 與 Cycle Accuracy

頁數目標：10

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `spec/spec.md` | frequency、bandwidth、engine model |
| `systemc/include/mdla7/conv_engine.h` | conv cycle wait |
| `systemc/include/mdla7/requant_engine.h` | requant cycle wait |
| `systemc/include/mdla7/memory.h` | DRAM / L1 cycle |
| `profile_html.md` | HTML profile format 與 ideal cycles |

章節內容：

- clock convention：1.9 GHz
- cycles -> ms
- per-engine busy
- critical path vs sum of engine work
- ideal cycle vs simulated cycle
- conv utilization
- memory-bound vs compute-bound
- current gaps in cycle accuracy

要放一個例子：

```text
sim time: 3,443,920 cycles @ 1.9 GHz (= 1.813 ms)
```

說明：

- 這不是 wall time。
- wall time 是 host simulation 執行時間。
- cycles 是 simulator 建模的硬體時間。

### 第 17 章 — Functional Verification 與 TFLite Fidelity

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `systemc/src/test_model.cpp` | compare DRAM output with reference |
| `systemc/scripts/compile_model.py` | numpy reference |
| `systemc/scripts/validate_tflite.py` | real TFLite interpreter |
| `handoff.md` | verification explanation |

章節內容：

- self-consistency PASS
- TFLite fidelity PASS
- bit-exact vs tolerance
- FP zero tolerance / FP rounding
- 1-LSB residual
- skipped layer 與 fused / streamed layer 的 verify policy

重要警語：

> `summary: 126/126 layers PASS` 代表 simulator 與內部 reference 一致，不一定代表每一層都已經和 real TFLite Interpreter bit-exact。

### 第 18 章 — Regression Scripts 與 Profile HTML

頁數目標：8

必讀檔案：

| 檔案 | 重點 |
|---|---|
| `batch/run_model.py` | one model flow、HTML report |
| `batch/run_mdla6_pattern.py` | MDLA6 pattern regression |
| `batch/run_ethz_v6.py` | ETHZ sweep |
| `batch/gen_model_profile.py` | index HTML |
| `profile_html.md` | profile format |

章節內容：

- one model regression
- `--all`
- MDLA6 pattern baseline：cx / our_ms
- cached ok rows
- `--rerun-all`
- profile JSON / CSV / PNG / HTML
- profile_mdla6_pattern.html index
- 如何判讀 `compile-fail`、`sim-fail`、`N-FAIL`

本章命令：

```bash
./batch/run_mdla6_pattern.py --filter inception_v3_quant --rerun-all
./batch/run_mdla6_pattern.py --rerun-all
./batch/run_ethz_v6.py --filter vit --limit 3
```

### 第 19 章 — Debug Playbook：從 N-FAIL 到 Root Cause

頁數目標：7

本章不是單純介紹，要寫成 junior 可以照做的 SOP。

流程：

1. 重跑單模型，確認 fail layer。
2. 看 compile log，確認 op / dtype / shape。
3. 看上一層是否 streamed / fused / skipped。
4. 切小 program，只跑 failing layer。
5. 切 subset，加入 producer / boundary。
6. 判斷 math bug 還是 scheduling bug。
7. 加最小修正。
8. 跑 canary：目標模型 + YOLO + Inception + VSR。

要放 case study：

- Inception concat boundary
- UNet INT16 suppressed-store boundary
- YOLO conservative path performance guard
- VSR D2SPACE ADD streaming

### 第 20 章 — Junior Exercises 與 Roadmap

頁數目標：7

練習設計由淺到深：

| Level | 題目 | 目標 |
|---|---|---|
| 1 | 跑 `mobilenet_v1` 並截圖 profile HTML | 熟悉工具 |
| 1 | 在 profile 多印一個 layer byte count | 熟悉 `batch/run_model.py` |
| 2 | 追一個 CONV descriptor 的 wait / signal tag | 熟悉 descriptor |
| 2 | 手算一層 conv 的 weight bytes | 熟悉 tensor layout |
| 3 | 新增一個 compile log 欄位 | 熟悉 compiler |
| 3 | 新增 regression filter | 熟悉 regression |
| 4 | 修一個小 scheduling race | 熟悉 L1 reuse |
| 4 | 加一個 op 的 profile label | 熟悉 op mapping |
| 5 | 改 cycle model 並比較 regression ms | 熟悉 cycle accuracy |

Roadmap：

- Last-LSB fidelity
- shared 512-lane Requant resource modeling
- better L1 arbitration
- generic microblock wavefront
- tiled SOFTMAX
- INT16 POOL / CONCAT support
- ACT compression / decompression：DRAM compressed、L1 raw NHWC tile

### 第 21 章 — Performance Bug：如何看 Profile 與 Fix DRAM Write

頁數目標：8

本章是 performance 實戰章，補第 14 到第 16 章之後的 debug workflow。

章節內容：

- 從 `.profile.csv` 找 top `dram_w / dram_r` layer。
- 判斷 final output、intermediate output、logical concat boundary。
- Tile fuse、microblock fuse、pending store、GraphMeta suppress store。
- Case study：`imdn_quant` / `imdn_float`。
- 當 `UDMA_R` 還是 dominate 時，評估 persistent weight、fanout input reuse、rolling halo。
- ACT compression / decompression 作為下一級硬體解法：
  - DRAM compressed、L1 decompressed。
  - UDMA_R + ACT_DECOMP。
  - UDMA_W + ACT_COMP。
  - raw fallback、metadata overhead、cycle model。

---

## 4. Source Code 讀法順序

Junior 不要從最大檔案直接硬啃。建議順序：

| 順序 | 檔案 | 讀到什麼程度 |
|---:|---|---|
| 1 | `batch/run_model.py` | 知道一個 model 怎麼 compile / simulate / report |
| 2 | `systemc/scripts/compile_model.py` | 先看 main operator loop，不急著看每個 op |
| 3 | `systemc/include/mdla7/descriptor.h` | 看 descriptor 的共同語言 |
| 4 | `systemc/include/mdla7/system.h` | 看 module 怎麼接起來 |
| 5 | `systemc/include/mdla7/command_engine.h` | 看 dispatch / dependency |
| 6 | `systemc/src/test_model.cpp` | 看 descriptor emission、tiling、fusion |
| 7 | `systemc/include/mdla7/conv_engine.h` | 看 CONV compute |
| 8 | `systemc/include/mdla7/requant_engine.h` | 看 chain -> output |
| 9 | `systemc/include/mdla7/udma.h` | 看 data movement |
| 10 | `systemc/include/mdla7/memory.h` | 看 DRAM / L1 cycle |
| 11 | `systemc/include/mdla7/ewe_pool.h` | 看 non-conv ops |
| 12 | `batch/run_mdla6_pattern.py` | 看 regression automation |

---

## 5. 圖表 Backlog

正式 150 頁教材至少需要 20 張圖。先列 backlog：

| 圖名 | 放在哪章 | 內容 |
|---|---:|---|
| `mdla7_learning_map.drawio` | 0 | junior 學習地圖 |
| `mdla7_end_to_end.drawio` | 0 | TFLite 到 PASS summary |
| `repo_tree.drawio` | 1 | repo layout |
| `top_architecture.drawio` | 2 | HW top |
| `control_data_flow.drawio` | 2 | control vs data |
| `descriptor_layout.drawio` | 3 | header + body union |
| `tag_timeline.drawio` | 3 | wait / signal tag |
| `memory_hierarchy.drawio` | 4 | DRAM / UDMA / L1 |
| `l1_layout_tiling.drawio` | 4 | L1 tile layout |
| `engine_overview.drawio` | 5 | five engines |
| `compile_flow.drawio` | 6 | TFLite parse |
| `quant_flow.drawio` | 7 | quant / requant math |
| `program_bin_layout.drawio` | 8 | binary layout |
| `systemc_top_wiring.drawio` | 9 | modules and FIFOs |
| `command_engine_dispatch.drawio` | 10 | pending queue |
| `conv_requant_chain.drawio` | 11 | 16-lane chain |
| `udma_modes.drawio` | 12 | linear / strided / d2space |
| `ewe_pool_ops.drawio` | 13 | op family |
| `fusion_pending_store.drawio` | 14 | pending store state machine |
| `microblock_wavefront.drawio` | 15 | tile pipeline |
| `cycle_accounting.drawio` | 16 | engine busy vs critical path |
| `verification_ladder.drawio` | 17 | self-consistency vs TFLite |
| `regression_flow.drawio` | 18 | scripts and outputs |
| `debug_playbook.drawio` | 19 | FAIL triage |

畫圖規則依 `notebook.md`：

- 白底。
- orthogonal edge。
- 不用斜線。
- 主要資料結構藍色，engine 紅 / 橘，控制紫色，memory 綠色。

---

## 6. Regression / Coverage 筆記要怎麼寫

### 6.1 Regression 分類

教材要固定用這四類：

| 類型 | 意義 | 例子 |
|---|---|---|
| compile-fail | Python compile 階段失敗 | unsupported op / shape issue |
| sim-fail | SystemC 程式無法完成 | L1 too small / runtime abort |
| N-FAIL | simulator 有跑完，但 N 層 compare fail | scheduling / math mismatch |
| ok | compile + simulate + compare pass | `126/126 PASS` |

### 6.2 Function coverage matrix

正式教材第 17 / 18 章要放一張 matrix：

| 功能 | INT8 | UINT8 | INT16 | FP16/BF16 | 測試模型 |
|---|---|---|---|---|---|
| CONV | yes | partial | yes | yes | MobileNet / Inception / UNet |
| DWCONV | yes | partial | yes | yes | MobileNet |
| FC | yes | partial | yes | yes | Audio / classifier |
| ADD | yes | yes | gap | yes | EfficientNet / YOLO |
| MUL / SUB | yes | yes | gap | yes | YOLO / transformer |
| POOL | yes | yes | gap for INT16 | yes | Inception / DeepLab |
| SOFTMAX | yes | yes | gap | FP path exists | Inception / transformer |
| CONCAT | logical channel concat | yes | gap | yes-ish | Inception / YOLO |
| D2SPACE | yes | yes | TNPS / Requant final-store / UDMA fallback | yes | VSR / XLSR |

這張表要從最新 code 和 regression 實測更新，不要只照 handoff。

### 6.3 Cycle accuracy matrix

| 模組 | 目前 cycle model | 準確度風險 |
|---|---|---|
| CONV | bit-mult throughput + startup | tile fill latency 仍粗 |
| REQUANT | chain drain + 512-lane shared resource throughput | shared CONV/EWE arbitration 仍簡化 |
| UDMA | bandwidth + startup + mode cost | real AXI arbitration 未完全建模 |
| DRAM | row hit / miss / bandwidth / refresh | bank scheduling 簡化 |
| L1 | bank conflict model | priority policy 簡化 |
| EWE | element throughput | LUT / activation latency 粗 |
| POOL | window-based throughput | line buffer / border behavior 粗 |

---

## 7. 寫作里程碑

### Milestone A — 先完成 30 頁入門版

範圍：

- 第 0 章
- 第 1 章
- 第 2 章
- 第 3 章

完成標準：

- junior 能跑單模型。
- junior 能畫出 top architecture。
- junior 能解釋 descriptor / tag。

### Milestone B — 完成 75 頁核心版

範圍：

- 第 0 到第 10 章

完成標準：

- junior 能從 TFLite 追到 descriptor。
- junior 能解釋 SystemC module wiring。
- junior 能看懂一個 layer 的 compile log。

### Milestone C — 完成 120 頁工程版

範圍：

- 第 0 到第 17 章

完成標準：

- junior 能解釋 CONV / REQUANT / UDMA / EWE / POOL。
- junior 能看懂 tiling / fusion。
- junior 能分辨 functional correctness 與 TFLite fidelity。

### Milestone D — 完成 150 頁正式版

範圍：

- 第 0 到第 20 章
- 20+ drawio 圖
- regression / coverage / cycle accuracy matrix
- debug playbook
- exercises

完成標準：

- 新人可用這份文件 self-study 一週。
- mentor 可用它安排 code walk-through。
- 每章都有可執行命令或可查 code reference。

---

## 8. 實作時要先查的 Commands

寫章節時常用：

```bash
# 找 source
rg -n "CommandEngine|Descriptor|make_udma|make_desc" systemc

# 跑單模型
./batch/run_model.py inception_v3_quant

# 跑 pattern
./batch/run_mdla6_pattern.py --filter unet_int16 --rerun-all

# 看 profile
open batch/output/inception_v3_quant.html

# 產 PDF
cd ..
bash scripts/build_pdf.sh
```

---

## 9. 風險與注意事項

| 風險 | 對策 |
|---|---|
| code 還在快速演進 | 每章開頭標註 commit hash / 日期 |
| regression 數字會變 | cycle 數字集中放表格，避免散落全文 |
| junior 被 `test_model.cpp` 嚇到 | 先用 flow chart，再切段講 |
| PASS 語意被誤解 | 第 17 章反覆區分 self-consistency / TFLite fidelity |
| cycle accuracy 被看成 silicon signoff | 第 16 章明確講 simulator model boundary |
| 圖太少 | 每章至少 1 張核心圖，複雜章 2 到 3 張 |

---

## 10. 下一步

建議下一個工作項目：

1. 先寫 `md/00_intro.md`，約 6 頁。
2. 同時畫 `drawio/mdla7_end_to_end.drawio`。
3. 跑一次 `./batch/run_model.py inception_v3_quant`，把 console summary 和 profile HTML 截成教材範例。
4. 再寫 `md/01_build_and_run.md`，讓新人第一天就能跑起來。

第一批章節完成後，再依 `scripts/build_pdf.sh` 產出 PDF，確認頁首、目錄、code block、圖尺寸都正常。
