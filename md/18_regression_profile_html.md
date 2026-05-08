# 第 18 章 — Regression Scripts 與 Profile HTML

> 上一章：[第 17 章 — Functional Verification 與 SystemC Function Coverage](17_verification_coverage.md)

本章你會學到什麼：

- `run_model.py` 如何串 compile、simulate、plot、HTML。
- `run_mdla6_pattern.py` 和 `run_ethz_v6.py` 的用途。
- profile JSON / CSV / PNG / HTML 各自怎麼用。
- `profile_mdla6_pattern.html` / `profile_hotspot.html` index 如何生成。
- regression status 要如何解讀。
- ETHZ_v6 裡哪些模型屬於 Transformer / attention 類 coverage。

---

## 18.1 run_model.py

主要入口：

```bash
./batch/run_model.py inception_v3_quant
```

flow：

```text
resolve model
make build/test_model
compile_model.py -> batch/output/<stem>.bin
test_model -> profile.json/csv
plot_profile.py -> profile.png
_write_html_report -> batch/output/<stem>.html
gen_model_profile.py -> profile_mdla6_pattern.html 或 profile_hotspot.html
```

`run_model.py` 也會自動 re-exec 到 venv：

```text
~/.venvs/mdla7/bin/python
```

---

## 18.2 Model resolve

`resolve_model()` 支援：

| 用法 | 說明 |
|---|---|
| exact path | 直接指定 `.tflite` |
| substring | `inception` |
| fuzzy | typo suggestion |
| `--list` | 列出 model/ 下模型 |

dtype priority 會讓 ETHZ_v6 / INT8 等 active bundles 優先。

---

## 18.3 Artefacts

per-model output：

```text
batch/output/<stem>.bin
batch/output/<stem>.profile.json
batch/output/<stem>.profile.csv
batch/output/<stem>.profile.png
batch/output/<stem>.html
```

預設成功後 `.bin` / `.profile.json` 可能被刪掉，保留 `.csv` / `.png` / `.html`。用：

```bash
./batch/run_model.py model --keep-intermediate
```

可保留中間檔方便 debug。

---

## 18.4 profile JSON

`.profile.json` 包含：

- summary
- layers
- engine busy
- task timelines

適合：

- plot Gantt。
- programmatic analysis。
- comparing cycles / utilization。

---

## 18.5 profile CSV

`.profile.csv` 是 one row per layer。

常見欄位：

```text
id, op, in_h, in_w, in_c, out_h, out_w, out_c,
tiles_h, tiles_oc, pass, cycles_layer, cycles_cum,
conv_util_pct, dram_r, dram_w, sram_r, sram_w, streamed
```

適合丟到 pandas / spreadsheet，找 top cycle layers。

---

## 18.6 profile PNG / HTML

`plot_profile.py` 產生 static Gantt PNG。

`run_model.py` 的 HTML report 更完整：

- summary chips
- interactive Gantt
- layer table
- engine utilization
- compile log
- sim log
- notes / candidates

HTML 對 debug overlap 和 labeling 很有用。

---

## 18.7 Profile Index HTML

[`gen_model_profile.py`](../batch/gen_model_profile.py) 掃 `batch/output/*.html`，
並依 runner 指定的 CSV 生成 index：

```text
batch/profile_mdla6_pattern.html
batch/profile_hotspot.html
```

`profile_mdla6_pattern.html` 會整理：

- pattern
- link
- cx
- our_ms
- conflict_ms
- mesh_ms
- ratio

用來看整個 MDLA6 baseline corpus 的 performance 排名。

`profile_hotspot.html` 是 Hotspot micro-pattern index，不顯示 `cx` /
`myms/cx`，預設用 `mesh/fast` 排序，適合看 L1Mesh / NoC timing overhead。
目前 L1Mesh mesh model 已加入 transparent NoC contention、Payload scheduling
chunk，以及 Payload R/W lane latency table。看單一
model 的 `.mesh.html` 時，`max service` 代表單一 scheduling chunk 的服務時間；
`max latency` 則包含 queue wait。

---

## 18.8 run_mdla6_pattern.py

這是 MDLA6 pattern A/B style regression：

```bash
./batch/run_mdla6_pattern.py --filter unet_int16 --rerun-all
```

特點：

- pattern list 有 MDLA6 cx baseline。
- 可 cache ok rows。
- 每個 pattern 產 HTML。
- 更新 `batch/output/mdla6_pattern_regression.csv`。

status：

| status | 意義 |
|---|---|
| ok | all pass |
| N-FAIL | sim completed but N layers fail |
| compile-fail | compiler failed |
| sim-fail | simulator failed |
| html-fail | report generation issue |

---

## 18.8b run_hotspot.py

Hotspot runner 會跑 `model/Hotspot/*.tflite` 裡切出來的 repeated
Transformer-like bottleneck slice：

```bash
./batch/run_hotspot.py --filter vit --rerun-all
./batch/run_hotspot.py --limit 3
```

如果已經 `cd batch`，要改用：

```bash
./run_hotspot.py --rerun-all
```

不要直接打 `run_hotspot.py --rerun-all`；zsh 預設不會從目前目錄找 executable。

它和 `run_mdla6_pattern.py` 一樣會跑 fast / conflict / mesh 三種 timing，
但輸出的是 Hotspot 專用報表：

```text
batch/output/hotspot_regression.csv
batch/profile_hotspot.html
```

Hotspot 沒有 MDLA6 `cx` baseline，所以報表不顯示 `cx` 欄位。

最近 L1Mesh Payload/lane-stat update 後，Hotspot 的 `mesh/fast` 已接近
`1.0`。若之後調整 edge mapping 或 NoC resource model，請用：

```bash
./batch/run_hotspot.py --rerun-all
```

再檢查 `batch/profile_hotspot.html` 以及個別
`batch/output/<stem>.mesh.html` 的 lane latency table。

Hotspot compile coverage update 後，原本 73 個 skipped compile rows 都會
lower 成 `matrlz` fallback。最新完整 sweep：

```text
fast 11/11, conflict 11/11, mesh 11/11
Compile log: 364 rows, 0 skipped
matrlz fallback rows: 73
```

`matrlz` 是 compiler materialized-reference fallback，常見於 non-spatial
MEAN、runtime FC、INT GELU、shape-prop mismatch reshape、descriptor dim
overflow。它是可驗證/可 profile 的 coverage layer，不是 dedicated arithmetic
engine。

---

## 18.9 run_ethz_v6.py

ETHZ_v6 sweep：

```bash
./batch/run_ethz_v6.py
./batch/run_ethz_v6.py --filter vit --limit 3
```

它跑 `model/ETHZ_v6` corpus，介面和 `run_hotspot.py` 一致，支援 `--filter`、
`--limit`、`--offset`、`--rerun-all`、`--list`。每個 model 會一次跑 fast /
conflict / mesh 三種 L1 timing，輸出：

```text
batch/output/ethz_v6_regression.csv
batch/profile_ethz_v6.html
```

ETHZ corpus 沒有 MDLA6 `cx` baseline，所以報表不顯示 `cx` 欄位。

用途：

- large model coverage。
- dtype / op mix coverage。
- performance trend。
- detecting compile-fail after compiler changes。

---

## 18.10 ETHZ_v6 Transformer / attention coverage

ETHZ_v6 corpus 裡有一批模型明顯屬於 Transformer、ViT、LLM、BERT 或 attention-heavy 架構。這些模型對 MDLA7 很重要，因為它們會覆蓋大量 non-CNN pattern：

- `FULLY_CONNECTED` / 1x1 matmul-like path
- `RESHAPE`
- `ADD` / `MUL` / `SUB`
- `SOFTMAX`
- `GATHER`
- `GELU`
- high-rank tensor lowering
- attention Q/K/V 類資料流

確定是 Transformer 類的模型：

| Model | 類型 | 典型 coverage |
|---|---|---|
| `gpt2_quant.tflite` | decoder-only Transformer / GPT | FC、reshape、softmax、gather |
| `llama2_quant.tflite` | decoder-only Transformer / LLM | FC、mul、add、softmax、gather |
| `mobilebert_quant.tflite` | BERT encoder | large FC count、reshape、softmax |
| `vit_b16_quant.tflite` | Vision Transformer | patch/token reshape、FC、softmax |
| `swin_float.tflite` | Swin Transformer | window attention、FC、softmax、GELU |
| `swin_quant.tflite` | Swin Transformer | quantized attention-style graph |
| `mobilevit_v2_float.tflite` | CNN + Transformer hybrid | conv/dwconv + attention blocks |
| `mobilevit_v2_quant.tflite` | CNN + Transformer hybrid | quantized hybrid coverage |
| `sam_float.tflite` | Segment Anything style Transformer | FC、softmax、GELU、attention blocks |
| `sam_quant.tflite` | Segment Anything style Transformer | quantized attention blocks |

也很可能含 attention / Transformer block 的模型：

| Model | 為什麼列入 |
|---|---|
| `sd_encoder_quant.tflite` | diffusion encoder，profile 有 reshape / add / mul / softmax |
| `sd_decoder_quant.tflite` | diffusion decoder，profile 有 reshape / add / mul / softmax |
| `sd_diffusion_quant.tflite` | diffusion core，attention-like op mix 很重 |
| `midas_v3_float.tflite` | vision model，profile 有 FC / reshape / softmax pattern |
| `midas_v3_quant.tflite` | quantized MiDaS，覆蓋同類 pattern |

如果要快速找 Transformer 類 regression，可以用檔名關鍵字先篩：

```bash
./batch/run_ethz_v6.py --filter gpt2
./batch/run_ethz_v6.py --filter llama2
./batch/run_ethz_v6.py --filter mobilebert
./batch/run_ethz_v6.py --filter vit
./batch/run_ethz_v6.py --filter swin
./batch/run_ethz_v6.py --filter sam
```

如果要用 profile 交叉確認，可以看每個 model 的 `.profile.csv`，統計 op mix：

```bash
awk -F, 'NR>1{gsub(/^ +| +$/,"",$2); c[$2]++}
         END{for(k in c) print k,c[k]}' batch/output/vit_b16_quant.profile.csv
```

典型 Transformer-like profile 會看到：

```text
fc / reshape / add / mul / sub / softmax
```

注意：`xlsr_float.tflite` / `xlsr_quant.tflite` 名稱上像 speech Transformer 家族，但目前 profile 主要呈現 `conv/concat/d2spac`，沒有明顯 `fc/softmax` attention pattern。因此在這份 textbook 裡先不列為確定 Transformer coverage。

---

## 18.11 Regression triage

看 regression 時先分三類：

| 類型 | 下一步 |
|---|---|
| compile-fail | 看 compile_model log / unsupported op / shape limit |
| sim-fail / timeout | 看 descriptor deadlock / sc_start cap / crash |
| N-FAIL | 看 first fail layer / output mismatch |

不要一開始就看整個 log。先分類狀態，再下鑽。

---

## 18.12 HTML Gantt 怎麼看

看 Gantt 時問：

- UDMA_R 是否提前餵 compute？
- CONV 是否有大空洞？
- UDMA_W 是否阻擋 read？
- EWE/POOL tail 是否拖長？
- stream descriptors 是否跨 layer overlap？
- final layer done tag 在哪裡？

Gantt 是 scheduler debug 的最佳入口。

---

## 18.13 常見誤解

| 誤解 | 正確理解 |
|---|---|
| 只要看 console PASS/FAIL | performance debug 要看 profile |
| `.profile.json` 一定存在 | 成功後可能被清掉，需 `--keep-intermediate` |
| HTML 只是美化 | HTML Gantt 是 scheduler/debug 工具 |
| ok rows 永遠不用 rerun | 改 scheduler/cycle model 時要 rerun |
| ratio 越低一定越差 | 要看 cx baseline、model class、coverage |
| `xlsr` 名稱像 Transformer 就一定算 attention coverage | 要看 profile op mix；目前主要是 conv/concat/d2spac |

---

## 18.14 本章小結

Regression flow：

```text
single model -> run_model.py
pattern sweep -> run_mdla6_pattern.py
hotspot sweep -> run_hotspot.py
corpus sweep -> run_ethz_v6.py
visual indexes -> profile_mdla6_pattern.html / profile_hotspot.html
```

有效 debug 的方法是：

```text
status -> first failing layer -> profile timeline -> source root cause
```

> 下一章 → [第 19 章 — Debug Playbook：從 N-FAIL 到 Root Cause](19_debug_playbook.md)
