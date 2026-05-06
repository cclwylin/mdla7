# 第 1 章 — Source tree 與 build / run 基礎

> 上一章：[第 0 章 — MDLA7 是什麼、怎麼讀這份 repo](00_intro.md)

本章你會學到什麼：

- MDLA7 repo 的目錄怎麼分工。
- 怎麼 build SystemC simulator。
- 怎麼跑第一個 TFLite model。
- `run_model.py` 的三個階段各自做什麼。
- output 目錄裡每個檔案代表什麼。
- 怎麼跑 MDLA6 pattern regression。
- junior 第一天應該如何判斷自己環境已經準備好。

---

## 1.1 先看 repo tree

這份 repo 的主要目錄：

```text
MDLA7_Codex/
  spec/
  systemc/
  scripts/
  md/
  notebook.md
  handoff.md
  profile_html.md
```

每個目錄的角色：

| 路徑 | 角色 |
|---|---|
| [`spec/`](../spec) | HW spec、drawio block diagram |
| [`systemc/`](../systemc) | SystemC simulator、compiler backend、C++ build |
| [`systemc/include/mdla7/`](../systemc/include/mdla7) | SystemC module implementation |
| [`systemc/src/`](../systemc/src) | simulator binary entry point |
| [`systemc/scripts/`](../systemc/scripts) | TFLite compiler、validator、plotter backend |
| [`batch/`](../batch) | 使用者 runner、regression CSV、profile HTML、batch output |
| [`scripts/`](../scripts) | 教材 PDF build scripts |
| [`md/`](.) | 這份讀書筆記的正式章節 |
| [`notebook.md`](../notebook.md) | 教材製作流程與格式參考 |
| [`handoff.md`](../handoff.md) | 目前專案狀態、版本演進、known gaps |
| [`profile_html.md`](../profile_html.md) | profile HTML 格式說明 |

新人一開始最常看的目錄是 `systemc/` 和 `batch/`：`systemc/` 放 simulator/compiler backend，`batch/` 放使用者入口、regression、profile output。

---

## 1.2 重要檔案速查表

先不用每個檔案都看懂，但要知道它們存在。

### Python pipeline

| 檔案 | 你要知道的事 |
|---|---|
| [`batch/run_model.py`](../batch/run_model.py) | 單模型入口：build、compile、simulate、report |
| [`systemc/scripts/compile_model.py`](../systemc/scripts/compile_model.py) | TFLite lowering：op extraction、reference generation、binary emit |
| [`systemc/scripts/validate_tflite.py`](../systemc/scripts/validate_tflite.py) | 用 real TFLite Interpreter 做 fidelity check |
| [`systemc/scripts/plot_profile.py`](../systemc/scripts/plot_profile.py) | 把 profile JSON 畫成 Gantt |
| [`batch/gen_model_profile.py`](../batch/gen_model_profile.py) | 產生 regression index HTML |
| [`batch/run_mdla6_pattern.py`](../batch/run_mdla6_pattern.py) | 跑 MDLA6 pattern table 對照 |
| [`batch/run_ethz_v6.py`](../batch/run_ethz_v6.py) | 跑 ETHZ v6 model sweep |

### SystemC / C++ simulator

| 檔案 | 你要知道的事 |
|---|---|
| [`systemc/src/test_model.cpp`](../systemc/src/test_model.cpp) | 主要 runtime：載入 `.bin`、產生 descriptor、跑 SystemC、verify、profile |
| [`systemc/include/mdla7/system.h`](../systemc/include/mdla7/system.h) | top module wiring |
| [`systemc/include/mdla7/descriptor.h`](../systemc/include/mdla7/descriptor.h) | descriptor、dtype、memory map |
| [`systemc/include/mdla7/command_engine.h`](../systemc/include/mdla7/command_engine.h) | dependency tag dispatch |
| [`systemc/include/mdla7/conv_engine.h`](../systemc/include/mdla7/conv_engine.h) | CONV / FC / DWCONV compute |
| [`systemc/include/mdla7/requant_engine.h`](../systemc/include/mdla7/requant_engine.h) | int32 chain -> quantized output |
| [`systemc/include/mdla7/udma.h`](../systemc/include/mdla7/udma.h) | DRAM / L1 data movement |
| [`systemc/include/mdla7/memory.h`](../systemc/include/mdla7/memory.h) | DRAM、L1Mesh、L1Manager timing |
| [`systemc/include/mdla7/ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) | EWE、POOL、SOFTMAX |

如果你只記一件事：`batch/run_model.py` 是入口，`compile_model.py` 是 compiler，`test_model.cpp` 是 simulator driver，`include/mdla7/*.h` 是硬體 module。

---

## 1.3 建置環境

專案預期的基本環境：

| 類型 | 需求 |
|---|---|
| C++ | 支援 SystemC 的 C++ compiler |
| SystemC | macOS 上通常由 Homebrew 安裝 |
| Python | 由 `setup.sh` 建立 venv |
| Python packages | numpy、tflite parser、tensorflow 或 tflite-runtime、matplotlib |

setup script：

```bash
cd systemc
./setup.sh
```

這會建立外部 Python venv，通常在：

```text
~/.venvs/mdla7
```

venv 放在 repo 外面是刻意的。`handoff.md` 提過，某些 macOS 外接磁碟可能產生 AppleDouble sidecar 檔，venv 放在 repo 裡容易讓 Python site importer 出問題。

---

## 1.4 Build SystemC simulator

在 repo root 下：

```bash
make -C systemc -s
```

主要產物是：

```text
systemc/build/test_model
```

`batch/run_model.py` 會自動先跑 `make -C systemc`，所以平常你可以直接跑 model。但 junior 第一天建議先手動跑一次 `make -C systemc -s`，確認 C++ / SystemC 環境 OK。

如果 build 失敗，先看三件事：

| 問題 | 可能原因 |
|---|---|
| 找不到 SystemC header | SystemC 沒安裝或 Makefile auto-detect path 不對 |
| C++ standard library error | macOS CLT / SDK path 問題 |
| linker 找不到 SystemC | library path 不對 |

---

## 1.5 列出可用模型

在 repo root 下：

```bash
./batch/run_model.py --list
```

如果你已經 `cd batch`，要加目前目錄前綴：

```bash
./run_model.py --list
./run_hotspot.py --rerun-all
```

不要直接打 `run_hotspot.py --rerun-all`；zsh 預設不會從目前目錄找 executable。

`run_model.py` 會做幾件方便的事：

- 如果目前不在正確 venv，會 re-exec 到 `~/.venvs/mdla7/bin/python`。
- 掃描 model 目錄。
- 顯示可以用 pattern 匹配的模型。

你不需要打完整 `.tflite` 檔名。通常打 substring 就可以：

```bash
./batch/run_model.py inception_v3_quant
./batch/run_model.py yolo_v8_quant
./batch/run_model.py unet_int16
```

如果 pattern 有多個 match，程式會列出 matched model。這對拼字錯誤或模型同名變體很有用。

---

## 1.6 跑第一個模型

建議第一個模型跑 `inception_v3_quant`：

```bash
./batch/run_model.py inception_v3_quant
```

你會看到三個階段：

```text
[step 1/3] make
[step 2/3] compile inception_v3_quant.tflite -> batch/output/inception_v3_quant.bin
[step 3/3] simulate Mdla7System
```

### Step 1：make

這階段確保 `build/test_model` 是新的。

```text
[step 1/3] make
```

如果你剛改 C++，這裡會 rebuild。若只改 Python report script，可能不會 rebuild C++。

### Step 2：compile

compile 階段會印出每一層：

```text
layer  0     conv  in=346x346x3  k=3x3  s=2x2  g=1  out=172x172x32  ready
layer 15   concat  in=40x40x256  k=1x1  s=1x1  g=1  out=40x40x256  ready
```

每欄意義：

| 欄位 | 意義 |
|---|---|
| `layer` | MDLA layer id，不一定等於原始 TFLite op index |
| `conv` / `concat` | compiler 映射後的 op kind |
| `in=HxWxC` | input shape |
| `k=HxW` | kernel size，非 conv op 可能只是 placeholder |
| `s=HxW` | stride |
| `g` | group count，depthwise 會用到 |
| `out=HxWxC` | output shape |
| `ready` | 這層被 compiler 支援並 emit 到 program |
| `skipped` | 這層被略過或由其他方式接住 |

compile 結束會產生：

```text
batch/output/inception_v3_quant.bin
```

這個 `.bin` 是 `test_model` 的輸入，不是一般 executable。

### Step 3：simulate

simulate 階段會跑 SystemC：

```text
test_model: batch/output/inception_v3_quant.bin  (126 layers, v3, ...)
DRAM sized to 320 MB
layer  0 conv ... PASS
...
summary: 126/126 layers PASS, 0 FAIL
sim time: 3443920 cycles @ 1.9 GHz (= 1.813 ms)
```

每一層會印：

| 欄位 | 意義 |
|---|---|
| `tiles=HxOC` | 這層被切成多少 OH tile / OC tile |
| `cycles_layer` | 這層 window 的 cycle |
| `cum` | cumulative cycles |
| `conv-u` | CONV utilization / occupancy |
| `DRAM r/w` | 這層 DRAM read/write bytes |
| `SRAM r/w` | 這層 L1 SRAM read/write bytes |
| `PASS` | DRAM output 與 reference 相符 |
| `STREAMED` | output 被轉發在 L1，沒有做中間 DRAM verify |
| `FUSED` | output 留在 L1 給下一層 |

---

## 1.7 output 目錄裡有什麼

跑完一個 model 後，通常會看到：

```text
batch/output/<stem>.bin
batch/output/<stem>.profile.json
batch/output/<stem>.profile.csv
batch/output/<stem>.profile.png
batch/output/<stem>.html
```

各檔案用途：

| 檔案 | 用途 |
|---|---|
| `.bin` | compiler emit 的 program binary，給 `test_model` 載入 |
| `.profile.json` | structured profile，給 script 或分析工具使用 |
| `.profile.csv` | per-layer table，適合用 spreadsheet / diff |
| `.profile.png` | Gantt timeline 圖 |
| `.html` | 單模型 self-contained report |

最適合新人看的其實是 `.html`。它把 summary、engine busy、Gantt、per-layer profile、compile log 放在一頁。

macOS 可以直接：

```bash
open batch/output/inception_v3_quant.html
```

---

## 1.8 看懂 summary

典型 summary：

```text
summary: 126/126 layers PASS, 0 FAIL (102 fused/streamed — no intermediate DRAM verify)
sim time: 3443920 cycles @ 1.9 GHz (= 1.813 ms)
DRAM total r/w: 38.56 / 1.57 MB
SRAM total r/w: 48.08 / 53.02 MB
per-engine busy:
   udma_r:  1120182 cyc
   udma_w:   105660 cyc
     conv:   507322 cyc
  requant:   356955 cyc
      ewe:       59 cyc
     pool:  2023913 cyc
```

解讀順序：

1. 先看 `PASS / FAIL`。
2. 再看 `sim time`。
3. 再看 DRAM total，判斷是不是 memory-heavy。
4. 再看 per-engine busy，找 bottleneck。
5. 最後打開 HTML，看 Gantt 和 layer table。

這裡 `pool` busy 最大，不代表整個模型所有時間都只在 pool。engine busy 是各 engine 自己忙的時間，總和可以大於 critical path；真正 model time 是 `sim time`。

---

## 1.9 STREAMED / FUSED 為什麼不逐層 verify

你會看到：

```text
STREAMED (tile output forwarded in L1)
FUSED (output stays in L1)
```

這代表中間 tensor 沒有完整寫回 DRAM。既然 DRAM 裡沒有該 layer 的完整 output，就不能用 DRAM readback 做 per-layer compare。

這不是偷懶，而是模擬硬體 optimization：

```text
Producer output stays in L1
  -> Consumer reads L1 directly
  -> Skip intermediate DRAM write
```

這會降低 DRAM bandwidth，改善 cycle。代價是中間 layer 的 per-layer verification 需要換方法。現在 report 會把這些 layer 註明為 fused / streamed，summary 也會說有多少層沒有 intermediate DRAM verify。

---

## 1.10 跑 MDLA6 pattern regression

單模型跑通後，下一步是 pattern regression。

例如只跑 `unet_int16`：

```bash
./batch/run_mdla6_pattern.py --filter unet_int16 --rerun-all
```

典型輸出：

```text
[ 1/1] unet_int16  cx=3.39  8.730 ms  ok
summary: 1/1 ran
csv:  batch/output/mdla6_pattern_regression.csv
html: batch/profile_mdla6_pattern.html
```

欄位意義：

| 欄位 | 意義 |
|---|---|
| pattern | MDLA6 pattern row 的名字 |
| cx | MDLA6 baseline CX 值 |
| ms | MDLA7 simulator 的 model time |
| ok / fail | regression result |

`--rerun-all` 代表不要沿用 cached ok row，強制重跑。正在修 bug 時要用 `--rerun-all`，否則你可能看的是舊結果。

---

## 1.11 `run_model.py` 和 `run_mdla6_pattern.py` 的差別

| 工具 | 使用時機 | 輸出 |
|---|---|---|
| `run_model.py` | 開發 / debug 單一 model | 單模型 console、profile、HTML |
| `run_mdla6_pattern.py` | 跑 MDLA6 baseline 對照表 | regression CSV、profile_mdla6_pattern index |
| `run_ethz_v6.py` | 跑 ETHZ v6 collection | fast/conflict/mesh CSV、profile_ethz_v6 index |

平常 debug 流程：

```text
run_mdla6_pattern.py shows N-FAIL
  -> run_model.py same model
  -> inspect failing layer
  -> patch
  -> run_model.py target
  -> run_model.py canaries
  -> run_mdla6_pattern.py target --rerun-all
```

常用 canaries：

| Model | 為什麼常跑 |
|---|---|
| `inception_v3_quant` | CONCAT / POOL / branch boundary 很敏感 |
| `yolo_v8_quant` | MUL-heavy graph，performance regression 很敏感 |
| `vsr_quant` | CONV -> D2SPACE -> ADD streaming path |
| `unet_int16` | INT16 tiling / skipped op boundary |

---

## 1.12 第一天健康檢查 checklist

新人第一天把下面都跑過，就算環境有基本健康度。

```bash
make -C systemc -s
./batch/run_model.py --list
./batch/run_model.py inception_v3_quant
./batch/run_mdla6_pattern.py --filter inception_v3_quant --rerun-all
```

成功標準：

| 檢查 | 成功長相 |
|---|---|
| `make -C systemc -s` | 產生或更新 `build/test_model` |
| `--list` | 能列出 model |
| `run_model.py inception_v3_quant` | summary 顯示 0 FAIL |
| pattern regression | row 結尾是 `ok` |
| output HTML | `batch/output/inception_v3_quant.html` 存在且可打開 |

如果其中一項失敗，先不要急著改 simulator。先確認：

- 你在 repo root 下。
- Python venv 正常。
- SystemC 可以 include / link。
- model 檔案存在。
- 沒有另一個 regression process 同時寫 output。

---

## 1.13 Git 工作樹注意事項

這份 repo 常常會有 generated output 被更新，例如：

```text
batch/profile_mdla6_pattern.html
batch/output/*.profile.*
```

開發時要分清楚：

| 類型 | 例子 | commit 策略 |
|---|---|---|
| source change | `systemc/src/test_model.cpp` | 要 review、測試後 commit |
| report index | `batch/profile_mdla6_pattern.html` | 若 regression 更新可一起 commit |
| output artifact | `batch/output/...` | 多半 gitignored，不一定 commit |
| notebook | `md/*.md` | 屬於教材 source，可以 commit |

在開始改 code 前，先看：

```bash
git status --short
```

如果已經有別人的修改，不要用 destructive command 清掉。先理解哪些是 source、哪些是 generated。

---

## 1.14 常見錯誤與排除

| 現象 | 可能原因 | 排除方向 |
|---|---|---|
| `./batch/run_model.py xxx` 找不到模型 | pattern 不對或 model 不存在 | 先跑 `--list` |
| compile 階段 skipped 很多 op | 該 dtype / op 尚未支援 | 看 compile log reason |
| simulator 顯示 `N-FAIL` | output mismatch | 看 failing layer、前後 layer、streamed/fused |
| simulator abort | L1 fit / unsupported layout / runtime error | 看 stderr 最後一行 |
| pattern row 沒重跑 | cache 沿用舊 ok | 加 `--rerun-all` |
| cycle 變多但 PASS | scheduling / barrier / memory traffic 改變 | 看 profile / Gantt |
| HTML 數字不是最新 | 另一個 regression process 正在跑或 cache | 檢查 process、重跑 target |

---

## 1.15 本章小練習

### 練習 1：跑第一個 model

```bash
./batch/run_model.py inception_v3_quant
```

記下：

- layer 數
- PASS / FAIL
- sim cycles
- ms
- DRAM total r/w
- 最忙的 engine

### 練習 2：打開 HTML

```bash
open batch/output/inception_v3_quant.html
```

找出：

- Per-engine busy 表
- Gantt 圖
- Per-layer profile
- Compile log

### 練習 3：看 CSV

```bash
head -5 batch/output/inception_v3_quant.profile.csv
```

回答：

- 哪些欄位描述 shape？
- 哪些欄位描述 cycle？
- 哪些欄位描述 DRAM / SRAM bytes？

### 練習 4：跑 pattern

```bash
./batch/run_mdla6_pattern.py --filter inception_v3_quant --rerun-all
```

回答：

- `cx` 是多少？
- `our_ms` 是多少？
- row status 是 `ok` 還是 fail？

---

## 1.16 小結

本章把 repo 跑起來的基本路徑建立起來：`make` build simulator，`batch/run_model.py` 跑單模型，`batch/run_mdla6_pattern.py` 跑 regression，`batch/output/*.html` 看 profile。從下一章開始，我們會回到硬體架構本身，先讀 `spec/spec.md`，把 Host、Command Engine、五個 engines、UDMA、L1、DRAM 的關係建立清楚。

> 下一章 → [第 2 章 — HW Spec Top Architecture](02_hw_spec.md)
