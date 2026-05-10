# 第 0 章 — MDLA7 是什麼、怎麼讀這份 repo

本章你會學到什麼：

- MDLA7 這份 repo 在做什麼，不只是「跑 neural network model」。
- 一個 `.tflite` model 會怎麼變成 SystemC simulator 可以執行的 descriptor stream。
- 初學者應該用哪幾個視角讀 code：HW Spec、Compiler、SystemC Architecture、Regression。
- 哪些檔案要先讀，哪些檔案可以等後面章節再深入。
- `PASS`、`FAIL`、`cycle`、`profile` 這些字在這個專案裡的精確語意。

---

## 0.1 先用一句話描述 MDLA7

MDLA7 是一個類 Edge TPU 的 neural network accelerator SystemC simulator。它不是只拿 Python 跑 numpy reference，也不是只做 TFLite parser；它把 model compile 成一串硬體風格的 descriptor，再用 SystemC module 模擬 Host、Command Engine、UDMA、CONV、Requant、EWE、POOL、DRAM、L1 SRAM 等硬體 block 的行為與 cycle。

更具體地說，這份 repo 同時包含四層東西：

| 層次 | 問題 | 主要檔案 |
|---|---|---|
| HW Spec | 硬體長什麼樣？頻寬、engine、memory hierarchy 怎麼設計？ | [`spec/spec.md`](../spec/spec.md) |
| Compiler | `.tflite` 怎麼變成 MDLA7 layer metadata、weights、reference output？ | [`systemc/scripts/compile_model.py`](../systemc/scripts/compile_model.py) |
| SystemC Simulator | descriptor 進來後，各 engine 怎麼執行？cycle 怎麼累積？ | [`systemc/src/test_model.cpp`](../systemc/src/test_model.cpp)、[`systemc/include/mdla7`](../systemc/include/mdla7) |
| Regression / Report | 怎麼知道改 code 後有沒有壞？怎麼看模型時間和 profile？ | [`batch/run_model.py`](../batch/run_model.py)、[`batch/run_mdla6_pattern.py`](../batch/run_mdla6_pattern.py) |

讀這份 code 時，不要只盯一個檔案。你要一直在這四層之間切換：spec 說硬體應該做什麼，compiler 把模型轉成硬體能吃的格式，SystemC 模擬硬體怎麼跑，regression 告訴你這樣跑出來對不對、快不快。

---

## 0.2 End-to-end flow：從 `.tflite` 到 PASS summary

整體流程可以先記成這條線：

```text
TFLite model
  -> compile_model.py
  -> batch/output/<model>.bin
  -> test_model SystemC simulation
  -> per-layer compare
  -> profile json/csv/png/html
```

每一步的責任不同：

| 階段 | 做什麼 | 產物 |
|---|---|---|
| model lookup | `batch/run_model.py` 找到符合 pattern 的 `.tflite` | model path |
| compile | parse TFLite flatbuffer、建立 layer metadata、準備 DRAM blob、算 reference | `batch/output/<stem>.bin` |
| simulate | C++/SystemC 載入 `.bin`，產生 descriptor program，跑 MDLA7 system | engine timeline、DRAM output |
| verify | 把 DRAM output 跟 `.bin` 裡的 reference 比對 | `PASS` / `FAIL` |
| report | 輸出 profile 與 HTML | `.profile.json`、`.profile.csv`、`.profile.png`、`.html` |

一個新進工程師最常見的錯誤，是把這些階段混在一起。例如看到 `FAIL` 就以為一定是 `compile_model.py` 的 reference 算錯；看到 cycle 變多就以為一定是 CONV engine 變慢。實際上，FAIL 可能來自 math、layout、dependency tag、L1 overwrite、verify policy；cycle 也可能來自 UDMA、DRAM row miss、L1 bank conflict、Command Engine scheduling、barrier。

---

## 0.3 這份 repo 的核心心智模型

先把 MDLA7 想成三條線交會：

```text
Control path:
  Host -> Command Engine -> engine config FIFO -> done tag

Data path:
  DRAM <-> UDMA <-> L1Manager / L1Mesh <-> engines

Numerical path:
  TFLite tensor -> quant params -> CONV psum -> Requant -> output tensor
```

這三條線同時存在。任何 bug 都可以先問：

1. Control path 有沒有等對 tag？
2. Data path 有沒有讀到對的 L1 / DRAM 位址？
3. Numerical path 的 scale、zero-point、bias、activation clamp 有沒有對？

例如 `inception_v3_quant` 曾經出現 CONCAT 後面的 conv fail。這不是 conv math 壞掉，而是 logical CONCAT 沒有建立 dependency boundary，downstream conv 太早 reuse L1。這就是 control path / data path 的 bug，不是 numerical path 的 bug。

又例如 int8 layer 邊界差 1 LSB，通常會先查 quantization rounding、padding zero-point、bias fold，這就比較像 numerical path 的 bug。

---

## 0.4 MDLA7 的硬體大圖

Spec 裡的系統概觀可以簡化成：

```text
Host
  -> Command Engine
  -> CONV / Requant / EWE / POOL / TNPS
  -> L1Manager / L1Mesh
  -> UDMA
  -> DRAM
```

每個 block 的角色：

| Block | 角色 | 在 code 裡看哪裡 |
|---|---|---|
| Host | 把 descriptor stream 推進 NPU | [`host.h`](../systemc/include/mdla7/host.h) |
| Command Engine | 解 descriptor、看 wait tag、dispatch 給各 engine | [`command_engine.h`](../systemc/include/mdla7/command_engine.h) |
| CONV Engine | 做 convolution / fully-connected MAC，輸出 int32 partial sum chain | [`conv_engine.h`](../systemc/include/mdla7/conv_engine.h) |
| Requant Engine | 把 int32 psum 轉成 int8 / int16 / fp output | [`requant_engine.h`](../systemc/include/mdla7/requant_engine.h) |
| EWE Engine | element-wise ADD / MUL / SUB / activation / SOFTMAX | [`ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) |
| POOL Engine | AVG / MAX pooling | [`ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) |
| TNPS | tensor transpose / slice / concat / space-depth 類 layout movement | [`tnps.h`](../systemc/include/mdla7/tnps.h) |
| UDMA | DRAM 和 L1 之間搬資料，含 linear / strided / activation codec；D2SPACE 只保留 legacy fallback | [`udma.h`](../systemc/include/mdla7/udma.h) |
| L1Manager / L1Mesh | on-chip SRAM 與 arbitration / bank timing | [`memory.h`](../systemc/include/mdla7/memory.h) |
| DRAM | LPDDR5X-style off-chip memory timing model | [`memory.h`](../systemc/include/mdla7/memory.h) |

這裡有一個很重要的設計：CONV 不直接把最終 output tensor 寫回 L1。CONV 把 partial sums 推到 128 條 chain（4096 bit/cyc），Requant Engine 再 drain chain、做 fixed-point requantization、寫出 output。若後面是 final `DEPTH_TO_SPACE`，Requant 也可以直接做 final-store address swizzle。這會在第 11 章詳細走讀。

---

## 0.5 先認識 descriptor-driven architecture

MDLA7 不是每個 layer 都呼叫一個 C++ function 然後直接算完。`test_model.cpp` 會根據 layer metadata 產生很多 descriptor，每個 descriptor 描述一個小工作：

- UDMA 讀 input tile
- UDMA 讀 weight slice
- CONV compute
- REQUANT write output tile
- UDMA write output
- EWE ADD
- POOL
- barrier

descriptor 裡有 `wait_tag` 和 `signal_tag`。Command Engine 看到 wait tag 都 ready，才 dispatch descriptor；engine 做完後會回報 signal tag。

這個設計讓 simulator 可以描述硬體 overlap：

```text
UDMA_R(tile 1) can overlap CONV(tile 0)
EWE(tile 0) can overlap UDMA_R(tile 1)
UDMA_W can drain after compute
```

所以你讀 `test_model.cpp` 時，要把它看成「host-side scheduler / compiler backend」，不是一般的 layer-by-layer interpreter。

---

## 0.6 Compiler 在這個專案裡做什麼

[`compile_model.py`](../systemc/scripts/compile_model.py) 的名字叫 compiler，但它不是傳統 compiler 那種產生 machine code。它做的是 model lowering：

| Compiler 工作 | 說明 |
|---|---|
| parse TFLite | 讀 flatbuffer，抓 operator、tensor、buffer、quantization |
| op support check | 判斷 CONV、DWCONV、ADD、MUL、POOL、CONCAT 等是否支援 |
| shape / dtype mapping | 把 TFLite tensor shape 轉成 MDLA7 `LayerMeta` |
| weight packing | 把 TFLite weight / bias / params 放到 DRAM region |
| reference generation | 用 numpy 算每層 expected output |
| binary emit | 寫出 `program.bin`，給 `test_model` 載入 |

這裡的 reference generation 是理解 verification 的關鍵。預設 PASS 是 simulator output 跟 compiler 內部 numpy reference 一致，不等於一定和 real TFLite Interpreter 完全一致。真正要比 TFLite，要看 [`validate_tflite.py`](../systemc/scripts/validate_tflite.py)。

---

## 0.7 SystemC simulator 在這個專案裡做什麼

SystemC 這層主要分成兩種 code：

| 類型 | 檔案 | 說明 |
|---|---|---|
| top-level runtime | [`test_model.cpp`](../systemc/src/test_model.cpp) | 載入 binary、產生 descriptor、跑 sim、verify、profile |
| module implementation | [`systemc/include/mdla7/*.h`](../systemc/include/mdla7) | Command Engine、Engines、Memory、Host |

`test_model.cpp` 很大，原因是它同時處理很多事：

- binary loader
- descriptor helper
- layer scheduling
- tiling decision
- fusion decision
- pending store / store suppression
- profile accounting
- verification

初學者不要一開始就從頭到尾讀 `test_model.cpp`。比較好的順序是：

1. 先看 `batch/run_model.py` 知道怎麼呼叫。
2. 再看 `descriptor.h` 知道 descriptor 長什麼樣。
3. 再看 `system.h` 知道 module 怎麼接。
4. 再回來看 `test_model.cpp` 裡某一種 op，例如 CONV。

---

## 0.8 Regression 是什麼，為什麼這麼重要

這份 repo 的開發方式很像硬體 bring-up：每改一個 scheduling、cycle、op support，都可能讓某個模型從 PASS 變 FAIL，或 cycle 大幅變慢。因此 regression 不只是「測試」，而是 daily engineering loop。

常見命令：

```bash
./batch/run_model.py inception_v3_quant
./batch/run_mdla6_pattern.py --filter unet_int16 --rerun-all
./batch/run_mdla6_pattern.py --rerun-all
./batch/run_ethz_v6.py --filter vit --limit 3
```

常見結果分類：

| 分類 | 意義 | 下一步 |
|---|---|---|
| `ok` | compile、simulate、compare 都通過 | 看 cycle / profile |
| `compile-fail` | Python compile 階段失敗 | 查 unsupported op / shape / dtype |
| `sim-fail` | C++ simulator 沒正常跑完 | 查 L1 fit、descriptor、runtime error |
| `N-FAIL` | 跑完但 N 層 output mismatch | 查 failing layer、producer、scheduler、math |

`N-FAIL` 是 debug 最常見也最有價值的訊號，因為它表示 simulator 有完整跑完，只是某些 layer 不對。這通常能切小模型、定位 root cause。

---

## 0.9 PASS 的意思：self-consistency vs TFLite fidelity

在 MDLA7 repo 裡，PASS 有兩層意思。

第一層是 self-consistency：

```text
compile_model.py numpy reference == SystemC simulator output
```

這是 `batch/run_model.py` 預設檢查。它能證明 compiler reference 與 simulator 對同一套 MDLA7 semantics 一致。

第二層是 TFLite fidelity：

```text
TFLite Interpreter output == MDLA7 compiler reference / simulator output
```

這要用 `validate_tflite.py` 額外檢查。原因是 compile reference 和 simulator 可能一起犯同一個錯，仍然 self-consistency PASS。對 hardware simulator 開發來說，self-consistency 是每日開發主線；TFLite fidelity 是 numerical correctness 更高一層的校準。

所以文件後面看到：

```text
summary: 126/126 layers PASS, 0 FAIL
```

先解讀成：

> 這 126 個 MDLA7 layer 的 simulator output 與內部 reference 一致。

不要立刻解讀成：

> 這個 model 的每一個 tensor 都已與 TFLite Interpreter bit-exact。

---

## 0.10 Cycle 的意思：硬體時間，不是 wall time

SystemC simulation 會印出兩種時間概念：

```text
sim time: 3443920 cycles @ 1.9 GHz (= 1.813 ms)
(4.05s wall)
```

這兩個不是同一件事。

| 名稱 | 意義 |
|---|---|
| sim cycles | simulator 建模出來的硬體 cycle |
| ms | 用 1.9 GHz 把 cycles 換成硬體時間 |
| wall time | 你的 Mac / Linux host 實際跑 simulator 花多久 |

如果你改了一個 scheduling policy，`sim cycles` 變多，代表模型的硬體時間變慢。`wall time` 變多，可能只是 simulator 寫得比較慢，不一定代表硬體慢。

後面第 16 章會專門談 cycle accuracy：哪些 cycle model 已經比較接近硬體，哪些還是粗估。

---

## 0.11 新人第一週建議讀法

第一週不要追求把全部 code 看完。照這個順序：

| Day | 任務 | 成果 |
|---|---|---|
| 1 | 跑 `batch/run_model.py inception_v3_quant`，打開 profile HTML | 知道工具怎麼跑 |
| 2 | 讀 `spec/spec.md` 前兩節與 `system.h` | 能畫出 top block |
| 3 | 讀 `descriptor.h` 與 `command_engine.h` | 能解釋 wait / signal tag |
| 4 | 讀 `compile_model.py` 的 operator loop | 知道 TFLite 怎麼 lowering |
| 5 | 讀 `test_model.cpp` 的 CONV case | 能追一個 layer 的 descriptor |
| 6 | 讀 `conv_engine.h` / `requant_engine.h` | 能講 CONV -> Requant chain |
| 7 | 跑一個 FAIL case 或看歷史 bug | 能開始 debug |

這個順序的重點是「先有地圖，再進細節」。如果第一天就從 `compile_model.py` 第 1 行開始讀，很容易被 TFLite flatbuffer API、quantization helper、op-specific branch 淹沒。

---

## 0.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| `batch/run_model.py` 是 runner，不是 simulator | 它是 driver，真正 simulator 是 `test_model` binary |
| `compile_model.py` 只做 TFLite parser | 它也做 reference generation、weight packing、metadata emit |
| CONV engine 直接寫 output | CONV 推 int32 chain，Requant 才寫 L1 output |
| PASS 就是 TFLite bit-exact | 預設 PASS 是 self-consistency |
| cycle 就是 host runtime | cycle 是硬體模型時間，wall time 是電腦執行時間 |
| barrier 會搬完整 tensor | 有些 barrier 只做 1 byte UDMA，用來建立 ordering |
| skipped op 一定是不支援而中斷 | 有些 skipped op 可由 synthetic tensor 或 logical boundary 接住 |

---

## 0.13 小結

本章先建立 MDLA7 的全局視角：這是一個 descriptor-driven neural network accelerator simulator，不能只從 compiler、只從 HW spec、或只從 SystemC module 單點理解。你要同時看 control path、data path、numerical path，才能判斷一個 PASS / FAIL / cycle change 到底代表什麼。

接下來第 1 章會進入最實用的第一步：把 repo 跑起來，理解 `batch/run_model.py` 的三個階段、輸出檔案、profile HTML，以及最基本的 regression 命令。

> 下一章 → [第 1 章 — Source tree 與 build / run 基礎](01_build_and_run.md)


\newpage

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
| `flow` | Profile 裡的 L1 handoff group 起點 layer id；沒有 fuse 時 `flow == id` |
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

這會降低 DRAM bandwidth，改善 cycle。代價是中間 layer 的 per-layer verification 需要換方法。現在 console 會把這些 layer 註明為 fused / streamed，Profile CSV/HTML 會用 `flow` 欄位把真正的 L1 handoff group 標在一起，summary 也會說有多少層沒有 intermediate DRAM verify。

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


\newpage

# 第 2 章 — HW Spec Top Architecture

> 上一章：[第 1 章 — Source tree 與 build / run 基礎](01_build_and_run.md)

本章你會學到什麼：

- MDLA7 的硬體 top-level block 怎麼分工。
- Host、Command Engine、Compute Engines、Memory subsystem 的關係。
- 為什麼 MDLA7 採用 descriptor-driven control flow。
- CONV、Requant、EWE、POOL、UDMA 在資料路徑上各自站在哪裡。
- Spec 裡的 peak TOPS、L1 bandwidth、DRAM bandwidth 要怎麼先粗略理解。
- SystemC top module 如何對應 HW spec。

---

## 2.1 先讀哪個 spec

硬體架構的主要入口是 [`spec/spec.md`](../spec/spec.md)。這個 spec 是從 drawio block diagram 推導出來的 SystemC modelling 起點。它不是完整 silicon signoff spec，而是一份「目前 simulator 要模擬什麼」的工程 spec。

讀 spec 時先抓三個層次：

| 層次 | 你要問的問題 |
|---|---|
| system overview | 這顆 NPU 有哪些大 block？ |
| interface / bandwidth | block 之間怎麼傳資料？每 cycle 有多少 bytes？ |
| compute / memory balance | 算力和記憶體能不能餵得上？瓶頸在哪？ |

初學者常犯的錯，是一開始就盯著 TOPS 數字。TOPS 是峰值算力，不代表 model 真的能跑到那麼快。真正的模型時間還受限於：

- DRAM bandwidth
- UDMA 搬運
- L1 SRAM 容量
- L1 bank conflict
- tiling overhead
- dependency tag scheduling
- unsupported op / skipped op
- verification writeback policy

所以本章先讀 top-level block，再談頻寬與算力。

---

## 2.2 MDLA7 的 top-level 架構

Spec 裡把 MDLA7 描述成：

```text
Host
  -> Command Engine
  -> CONV / Requant / EWE / POOL / TNPS
  -> L1Manager / L1Mesh
  -> UDMA
  -> DRAM
```

更精準地說，MDLA7 是：

| 組件 | 角色 |
|---|---|
| Host | RISC-V host / runtime；透過 AXI-Lite register map 設定 Command Engine，descriptor / command ring 放 DRAM |
| Command Engine | 中央 controller，依 dependency tag dispatch 工作；具 DRAM AXI master 可 fetch descriptor / 回寫 status |
| CONV Engine | 做 convolution / fully-connected MAC |
| Requant Engine | 把 CONV partial sum 轉成 quantized output |
| EWE Engine | element-wise op、activation、softmax 類工作 |
| POOL Engine | max / average pooling |
| TNPS Engine | tensor transpose / slice / concat / space-depth 類 data movement；standalone/intermediate D2SPACE 主路徑 |
| L1Manager | on-chip memory arbitration |
| L1Mesh | 2 MB SRAM 工作區 |
| UDMA | DRAM 和 L1 之間的 DMA；D2SPACE 只保留 legacy fallback |
| DRAM | LPDDR5X-style off-chip memory model |

這些 block 不是各自獨立跑完整 layer。它們靠 descriptor 和 dependency tag 串起來。例如一個 conv layer 可能被拆成：

```text
UDMA read input tile
UDMA read weight slice
CONV compute
Requant output tile
UDMA write output tile
```

每一步都是一個或多個 descriptor。

---

## 2.3 Control path vs data path

讀硬體 spec 時要分清楚 control path 和 data path。

### Control path

Control path 描述誰命令誰：

```text
Host AXI-Lite MMIO -> Command Engine DRAM AXI master fetch -> engine config FIFO -> engine done tag
```

在 SystemC 裡對應：

| HW 概念 | SystemC 對應 |
|---|---|
| Host 透過 AXI-Lite register map 設 command ring / doorbell | [`Host`](../systemc/include/mdla7/host.h) 直接推 descriptor |
| Command/Host DRAM AXI master fetch descriptor | simulator 以 `std::vector<Descriptor> program` 代表已 fetch 的 command buffer |
| descriptor stream | `sc_fifo<Descriptor> desc_stream` |
| Command Engine dispatch | [`CommandEngine`](../systemc/include/mdla7/command_engine.h) |
| engine config interface | `sc_fifo<DescriptorBody> *_cfg` |
| done tag | `sc_fifo<uint8_t> *_done` |

### Data path

Data path 描述資料怎麼流：

```text
DRAM <-> UDMA <-> L1Manager <-> L1Mesh <-> engines
CONV -> Requant chain
```

注意 CONV 的 output 不走 L1Manager 寫回，而是先走 CONV -> Requant chain。Requant 才把 output 寫到 L1。

這個設計很重要。它讓 CONV partial sums 不必先寫 L1 再讀回，節省大量 SRAM traffic。第 11 章會細看 CONV / Requant chain。

---

## 2.4 SystemC top wiring 對應 HW spec

SystemC top module 在 [`system.h`](../systemc/include/mdla7/system.h)。核心 class 是 `Mdla7System`。

你可以先看它有哪些 member：

```cpp
L1Mesh        l1mesh;
Dram          dram;
L1Manager     l1mgr;
Udma          udma;
ConvEngine    conv;
RequantEngine requant;
EweEngine     ewe;
PoolEngine    pool;
CommandEngine cmd;
Host          host;
```

這幾乎就是 spec top block 的 C++ 版本。

再看它的 FIFO：

```cpp
sc_core::sc_fifo<Descriptor>     desc_stream{"desc_stream", 256};
sc_core::sc_fifo<DescriptorBody> conv_cfg{"conv_cfg", 4};
sc_core::sc_fifo<DescriptorBody> requant_cfg{"requant_cfg", 4};
sc_core::sc_fifo<DescriptorBody> ewe_cfg{"ewe_cfg", 4};
sc_core::sc_fifo<DescriptorBody> pool_cfg{"pool_cfg", 4};
sc_core::sc_fifo<DescriptorBody> udma_cfg{"udma_cfg", 4};
```

這些 FIFO 是 Command Engine 把工作派給 engines 的 config path。

done tag FIFO：

```cpp
sc_core::sc_fifo<uint8_t> conv_done{"conv_done", 4};
sc_core::sc_fifo<uint8_t> requant_done{"requant_done", 4};
sc_core::sc_fifo<uint8_t> ewe_done{"ewe_done", 4};
sc_core::sc_fifo<uint8_t> pool_done{"pool_done", 4};
sc_core::sc_fifo<uint8_t> udma_done{"udma_done", 4};
```

這些 FIFO 是 engines 做完後回報 Command Engine 的路徑。

---

## 2.5 CONV -> Requant chain 是特殊路徑

`system.h` 裡有一段：

```cpp
std::array<std::unique_ptr<sc_core::sc_fifo<int32_t>>, 16> chain;
```

初始化時：

```cpp
conv.chain_out[i]    = chain[i].get();
requant.chain_in[i]  = chain[i].get();
```

這表示 CONV Engine 和 Requant Engine 中間有 128 條 int32 FIFO，也就是 4096 bit/cyc。這不是一般 descriptor config FIFO，而是 data path。

為什麼要這樣設計？

CONV 做完 MAC 後產生的是 int32 partial sum。真正要寫回 activation tensor 前，還要做：

```text
psum + bias_eff
  -> multiply by quantized multiplier
  -> shift
  -> add output zero_point
  -> activation clamp
  -> int8 / int16 / fp storage
```

這些工作屬於 Requant Engine。如果 CONV 先把 int32 psum 全寫到 L1，會非常浪費 SRAM bandwidth。用 chain 直接送給 Requant，是更接近硬體 accelerator 的設計。

---

## 2.6 Descriptor-driven control 為什麼適合 NPU

MDLA7 的 Host 不直接呼叫 engine function。Host 產生 descriptor stream，Command Engine 根據 tag dispatch。

這樣做有幾個好處：

| 好處 | 說明 |
|---|---|
| decouple Host / Engine | Host 只要下 descriptor，不必逐 cycle 控制 engine |
| 支援 overlap | UDMA、CONV、Requant、EWE 可以用 tag 表達相依性 |
| 支援 tiling | 一個 layer 可以拆成多個 tile descriptor |
| 支援 microblock | 一個 tile 還能拆成更小 microblock wavefront |
| 容易 profile | 每個 descriptor 可帶 layer id / microblock metadata |

缺點是 scheduler 變複雜。你必須非常清楚：

- 哪個 descriptor 會寫 L1？
- 哪個 descriptor 會讀 L1？
- 哪些 descriptor 可以 overlap？
- 哪些 descriptor 必須建立 barrier？

近期幾個 bug 都是這類問題：

| 模型 | 問題 |
|---|---|
| `inception_v3_quant` | logical CONCAT 沒有 boundary，downstream conv 太早 reuse L1 |
| `unet_int16` | multi-tile suppressed conv store 沒有 completion boundary |
| `yolo_v8_quant` | MUL-heavy graph 的 conservative path 不能套錯 barrier |

這些不是硬體算術錯，而是 descriptor scheduling / memory lifetime 錯。

---

## 2.7 Engine 分工

### CONV Engine

CONV Engine 負責：

- CONV_2D
- DEPTHWISE_CONV_2D
- FULLY_CONNECTED mapped as 1x1 conv
- INT8 / INT16 / FP compute path

它從 L1 讀：

- activation tile
- weight slice

它輸出：

- int32 partial sums 到 Requant chain

它不直接輸出 final activation tensor。

### Requant Engine

Requant Engine 負責：

- CONV / EWE 共用 quantize-pack / clamp resource
- timing throughput 512 elem/cycle
- drain CONV int32 chain functional path
- per-channel multiplier / shift
- bias_eff
- activation min / max clamp
- int8 / int16 output write
- FP path 的 output conversion

它從 L1 讀 params，從 chain 或 element-wise datapath 取得 psum/value，最後寫 L1 output。

### EWE Engine

EWE 是 element-wise engine，負責：

- ADD
- MUL
- SUB
- HARD_SWISH
- GELU
- SOFTMAX LUT path

EWE 通常讀一個或兩個 tensor，寫一個 tensor。大型 tensor 會需要 tiling 或 microblock wavefront，否則 L1 和 UDMA 會成為 bottleneck。

### POOL Engine

POOL 負責：

- AVG_POOL
- MAX_POOL
- global-ish pooling
- MEAN routed through avg pool path

POOL 常見瓶頸不是 MAC，而是 window access、L1 traffic、border behavior。

### UDMA

UDMA 負責 data movement：

- DRAM -> L1
- L1 -> DRAM
- strided 2D copy
- depth-to-space layout copy

很多模型的 bottleneck 都是 UDMA read，不是 CONV。

---

## 2.8 Memory hierarchy

MDLA7 有兩層記憶體：

| 層 | 容量 / 性質 | 用途 |
|---|---|---|
| DRAM | spec 是 4 GB LPDDR5X；simulator 依模型動態配置 | 放 model weights、inputs、outputs、intermediate tensors |
| L1Mesh SRAM | 2 MB on-chip SRAM | 放目前 tile 的 input、weight、output、params |

2 MB L1 聽起來不少，但對大模型很快就不夠。例如：

```text
1024 x 1024 x 16 x 2 bytes = 32 MB
```

這種 tensor 完全不能整個塞進 L1，所以一定要 tiling。

常見 tiling 維度：

| 維度 | 用途 |
|---|---|
| OH tiling | 切 output height，降低 activation tile |
| OC tiling | 切 output channels，降低 weight slice / output tile |
| microblock | 在 EWE / streaming path 中切更小 block，增加 overlap |

這也是為什麼 `test_model.cpp` 很多 code 都在算 L1 address、tile size、是否 fit 2 MB。

---

## 2.9 DRAM bandwidth 與 cycle time

Spec 使用：

| 參數 | 值 |
|---|---|
| clock | 1.9 GHz |
| DRAM | LPDDR5X-10667 |
| peak bandwidth | 約 85.3 GB/s |
| simulator conversion | cycles / 1.9e6 = ms |

所以看到：

```text
sim time: 16587698 cycles @ 1.9 GHz (= 8.730 ms)
```

換算就是：

```text
16,587,698 cycles / 1.9 GHz = 8.730 ms
```

注意 peak bandwidth 不代表每個 model 都能用滿。實際還會被：

- UDMA startup
- row hit / row miss
- refresh
- dependency wait
- L1 bank arbitration
- store barrier
- tile granularity

影響。

---

## 2.10 算力：bit-mult invariant

Spec 裡有一個重要觀念：CONV array 是 bit-decomposable。

它用 fixed bit-mult capacity 來支援不同 dtype：

| Data type | MAC / cycle | 相對 INT8x8 |
|---|---:|---:|
| INT8 x 4 | 32,768 | 2.00 |
| INT8 x 8 | 16,384 | 1.00 |
| INT16 x 8 | 8,192 | 0.50 |
| INT16 x 16 | 4,096 | 0.25 |
| FP8 / FP16 / BFP16 | 4,096 | 0.25 |

核心 invariant：

```text
MAC_count * lhs_bits * rhs_bits = constant bit-mult per cycle
```

這對 cycle model 很重要。不同 dtype 不是只改 storage bytes，也會改 compute throughput。

例如 INT16x16 比 INT8x8 每個 MAC 需要更多 bit-level multiplier resource，所以 MAC/cycle 較少。這也是為什麼 INT16 model 即使 layer 數不多，cycle 可能很高。

---

## 2.11 Spec 和 simulator 哪裡一致，哪裡只是近似

讀這份專案時要分清楚三種狀態：

| 狀態 | 意義 |
|---|---|
| implemented | simulator 已有對應行為 |
| modelled approximately | simulator 有 cycle / behavior model，但比真硬體簡化 |
| spec proposal / TBD | spec 有規劃，但 simulator 未完整支援或仍待確認 |

例子：

| 項目 | 狀態 |
|---|---|
| Descriptor 64 bytes | implemented |
| Command Engine wait / signal tag | implemented |
| CONV -> Requant 128-lane chain (4096 bit/cyc) | implemented conceptually |
| L1 2 MB budget | implemented |
| DRAM dynamic sizing | implemented in simulator，spec 上仍是 4 GB |
| TNPS Engine | spec has block，simulator 重點仍不在完整 TNPS |
| real RISC-V host | currently host stub |
| real AXI protocol | simulator 用簡化 timing model |
| real silicon power | not implemented |

這不代表 simulator 沒價值。SystemC performance model 的目標是抓 architecture trend、scheduling bottleneck、functional coverage，不是取代 RTL signoff。

---

## 2.12 從 spec 走到 code 的對照表

| Spec 概念 | Code |
|---|---|
| Host | [`host.h`](../systemc/include/mdla7/host.h) |
| Command Engine | [`command_engine.h`](../systemc/include/mdla7/command_engine.h) |
| Descriptor | [`descriptor.h`](../systemc/include/mdla7/descriptor.h) |
| CONV Engine | [`conv_engine.h`](../systemc/include/mdla7/conv_engine.h) |
| Requant Engine | [`requant_engine.h`](../systemc/include/mdla7/requant_engine.h) |
| EWE / POOL | [`ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) |
| UDMA | [`udma.h`](../systemc/include/mdla7/udma.h) |
| DRAM / L1 | [`memory.h`](../systemc/include/mdla7/memory.h) |
| Top wiring | [`system.h`](../systemc/include/mdla7/system.h) |
| Runtime scheduling | [`test_model.cpp`](../systemc/src/test_model.cpp) |

這張表很重要。以後看到 spec 裡提到某個 block，就回來查這張表，看 code 在哪裡。

---

## 2.13 Junior 應該先懂的五個詞

| 詞 | 簡短定義 |
|---|---|
| descriptor | 一個硬體工作單位的 config packet |
| dependency tag | descriptor 之間的完成 / 等待關係 |
| tile | 大 tensor 放不進 L1 時切出來的一塊 |
| fusion | producer output 留在 L1 給 consumer 用，避免 DRAM round trip |
| cycle model | simulator 對硬體時間的估算 |

如果你能用自己的話解釋這五個詞，就已經能開始讀 `test_model.cpp` 的 scheduling code。

---

## 2.14 本章小練習

### 練習 1：找 top module

打開：

```bash
sed -n '1,160p' systemc/include/mdla7/system.h
```

回答：

- `Mdla7System` 建了哪些 module？
- 哪些 FIFO 是 config path？
- 哪些 FIFO 是 done tag path？
- CONV 和 Requant 之間有幾條 chain？

### 練習 2：找 descriptor op class

打開：

```bash
sed -n '1,120p' systemc/include/mdla7/descriptor.h
```

回答：

- `OpClass` 有幾種？
- `DType` 有哪些？
- `DF_STREAM` 和 `DF_STREAM_TAIL` 大概是做什麼？

### 練習 3：手算 cycle -> ms

如果 simulator 印：

```text
sim time: 4,394,946 cycles @ 1.9 GHz
```

手算：

```text
4,394,946 / 1,900,000 = 2.313 ms
```

確認你知道這是硬體模型時間，不是 wall time。

---

## 2.15 常見誤解

| 誤解 | 正確理解 |
|---|---|
| Host 就是現在的 Python script | Python script 是 driver；SystemC 裡還有 Host stub |
| Command Engine 做 compute | Command Engine 只 dispatch，不做 tensor math |
| CONV output 直接寫 L1 | CONV 先推 chain，Requant 才寫 L1 |
| UDMA 是 compute engine | UDMA 是 data movement engine |
| 2 MB L1 可以放大 tensor | 大圖像 activation 常常數十 MB，一定要 tile |
| TOPS 決定模型速度 | 真正速度常被 memory / scheduling 限制 |
| Spec 的每個 block 都完整實作 | 有些是 proposal / TBD，有些是 simulator approximation |

---

## 2.16 小結

本章把 MDLA7 的硬體大圖接到 SystemC top wiring：Host 推 descriptor，Command Engine 依 tag dispatch，各 engine 透過 L1Manager / UDMA / DRAM 搬資料，CONV 和 Requant 之間有特殊 int32 chain。你現在應該能把 `spec/spec.md` 的 top-level block 對應到 `systemc/include/mdla7/*.h` 的實作檔。

下一章會深入 descriptor 本身：64-byte descriptor 怎麼切 header / body，`wait_tag` 和 `signal_tag` 怎麼表達 dependency，stream metadata 又怎麼支援 microblock scheduling。

> 下一章 → [第 3 章 — Descriptor ISA 與 Dependency Tag](03_descriptor_tag.md)


\newpage

# 第 3 章 — Descriptor ISA 與 Dependency Tag

> 上一章：[第 2 章 — HW Spec Top Architecture](02_hw_spec.md)

本章你會學到什麼：

- MDLA7 為什麼使用 descriptor-driven control。
- 一個 64-byte descriptor 如何拆成 common header 與 op-specific body。
- `signal_tag` 與 `wait_tags` 怎麼形成 dependency graph。
- Command Engine 如何 dispatch 到 CONV、Requant、EWE、POOL、UDMA。
- stream descriptor、lookahead、tail barrier 是為了處理什麼 performance 問題。
- junior 在 debug descriptor / tag 問題時該從哪裡下手。

---

## 3.1 先建立一個直覺

NPU simulator 不會像 CPU 一樣一行一行執行 C code。它比較像硬體 command processor：

```text
Host / compiler 產生 descriptor stream
        |
        v
Command Engine 檢查 wait_tags
        |
        +--> CONV Engine
        +--> Requant Engine
        +--> EWE Engine
        +--> POOL Engine
        +--> UDMA
```

Descriptor 是「工作單」。每張工作單都說：

| 問題 | Descriptor 欄位 |
|---|---|
| 要哪個 engine 做事？ | `op_class_subtype` |
| 資料型態是 INT8、INT16 還是 FP？ | `dtype` |
| 做完要發哪個完成訊號？ | `signal_tag` |
| 開始前要等哪些完成訊號？ | `wait_count`、`wait_tags` |
| 這個工作屬於哪一層、哪個 microblock？ | `layer_id`、`microblock_id` |
| 實際參數在哪裡？ | `body` union |

對 junior 來說，最重要的不是記欄位順序，而是理解這句話：

```text
Descriptor = engine command + dependency contract + profiling hint
```

其中 dependency contract 就是 `wait_tags` 與 `signal_tag`。

---

## 3.2 必讀檔案

本章對照這幾個檔案讀：

| 檔案 | 讀法 |
|---|---|
| [`descriptor.h`](../systemc/include/mdla7/descriptor.h) | 看 descriptor binary layout 與 enum 定義 |
| [`command_engine.h`](../systemc/include/mdla7/command_engine.h) | 看 dependency tag 如何被檢查、dispatch、完成 |
| [`test_model.cpp`](../systemc/src/test_model.cpp) | 看 compiler / test harness 如何產生 descriptor |
| [`conv_engine.h`](../systemc/include/mdla7/conv_engine.h) | 看 CONV descriptor body 如何被消費 |
| [`requant_engine.h`](../systemc/include/mdla7/requant_engine.h) | 看 Requant descriptor body 如何被消費 |
| [`ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) | 看 EWE / POOL descriptor body 如何被消費 |
| [`udma.h`](../systemc/include/mdla7/udma.h) | 看 UDMA descriptor body 如何被消費 |

讀 code 的順序建議：

1. 先讀 `DescriptorHeader`。
2. 再讀 `CommandEngine::dispatch()`。
3. 再讀 `CommandEngine::issue()`。
4. 最後讀各 engine 的 `run()`。

這樣你會先知道「工作如何被發出去」，再看「工作被收到後如何執行」。

---

## 3.3 Descriptor 的 64-byte 格式

MDLA7 descriptor 固定是 64 bytes：

```text
Descriptor
  Header: 16 bytes
  Body  : 48 bytes
```

source 裡用 `static_assert` 保證這些大小：

```cpp
static_assert(sizeof(DescriptorHeader) == 16, "header must be 16 bytes");
static_assert(sizeof(DescriptorBody) == 48, "body union must be 48 bytes");
static_assert(sizeof(Descriptor) == 64, "Descriptor must be 64 bytes");
```

為什麼要固定 64 bytes？

| 原因 | 說明 |
|---|---|
| 硬體 decode 簡單 | command FIFO 每筆固定長度，decoder 不必處理 variable length |
| host 產生容易 | compiler 只要填 struct，不需要複雜 packetizer |
| alignment 好處 | 64 bytes 對 cache line / burst / ring buffer 都友善 |
| body 可擴充 | header 固定，body 用 union 給不同 engine |

你可以把它想成硬體版的 function call：

```text
op_class = 呼叫哪個 function
body     = function arguments
wait     = 呼叫前要等的 event
signal   = function return 時要 set 的 event
```

---

## 3.4 Common Header 欄位

`DescriptorHeader` 是所有 op 共用的 16 bytes。

| 欄位 | 型態 | 意義 |
|---|---:|---|
| `op_class_subtype` | `uint8_t` | low 4 bits 是 op class，high 4 bits 是 subtype |
| `flags` | `uint8_t` | trace、chain、stream、tail 等 bit flags |
| `dtype` | `uint8_t` | INT8 / INT16 / FP dtype |
| `signal_tag` | `uint8_t` | 此 descriptor 完成時要 set 的 tag；0 表示不 signal |
| `wait_count` | `uint8_t` | 有幾個 dependency tag 要等待，最多 4 個 |
| `wait_tags[4]` | `uint8_t[4]` | dependency tag 清單 |
| `layer_id` | `uint16_t` | profiler / debug 用的 layer index |
| `microblock_id` | `uint16_t` | stream scheduler 的 microblock 順序 |
| `stream_slot` | `uint8_t` | ping-pong / slot hint |
| `stream_meta_flags` | `uint8_t` | load、compute、store、final tile 等 metadata |

`op_class_subtype` 的解碼方式在 header method：

```cpp
OpClass op_class() const {
    return static_cast<OpClass>(op_class_subtype & 0xF);
}

uint8_t op_subtype() const {
    return (op_class_subtype >> 4) & 0xF;
}
```

目前主要的 op class：

| enum | 數值 | Engine |
|---|---:|---|
| `OC_CONV` | 0 | CONV Engine |
| `OC_REQUANT` | 1 | Requant Engine |
| `OC_EWE` | 2 | Element-wise / activation / softmax |
| `OC_POOL` | 3 | Pool Engine |
| `OC_UDMA` | 4 | DMA |

初學時可以先忽略 `op_subtype()`，因為 EWE subtype 目前主要放在 `EweBody.subtype`。真正決定 dispatch 到哪個 engine 的是 `op_class()`。

---

## 3.5 DType：功能正確與 timing 都會用到

`dtype` 在 header 裡，看起來只是 metadata，但實際上會影響兩件事：

| 影響 | 說明 |
|---|---|
| functional path | INT8、INT16、FP 使用不同讀取型別與運算流程 |
| cycle model | CONV 的 bit-mult cycle 會根據 activation bits 和 weight bits 改變 |

目前 enum 包含：

| DType | 數值 | 大意 |
|---|---:|---|
| `DT_INT8x4` | 0 | 8-bit activation × 4-bit weight |
| `DT_INT8x8` | 1 | 8-bit activation × 8-bit weight |
| `DT_INT16x4` | 2 | 16-bit activation × 4-bit weight |
| `DT_INT16x8` | 3 | 16-bit activation × 8-bit weight |
| `DT_INT16x16` | 4 | 16-bit activation × 16-bit weight |
| `DT_FP8` | 8 | E4M3-style FP8 |
| `DT_FP16` | 9 | FP16 |
| `DT_BFP16` | 10 | BFP16 |

Command Engine dispatch 時只把 `DescriptorBody` 寫進 engine FIFO。body 本身沒有 `dtype` 欄位，所以目前 simulator 用 side-channel latch：

```text
CommandEngine sees descriptor header dtype
        |
        +--> writes last_dtype into target engine
        |
        +--> writes body to target engine FIFO
```

這段在 [`command_engine.h`](../systemc/include/mdla7/command_engine.h) 的 `issue()`：

```cpp
if (conv_dtype_latch) *conv_dtype_latch = d.hdr.dtype;
conv_cfg_out.write(d.body);
```

這是一個 simulator-friendly 的寫法。真實硬體通常會把 dtype 和 body 一起放在 command payload，或在 decoder 裡保留完整 header。

常見 bug：

| 現象 | 可能原因 |
|---|---|
| INT16 model output 全錯 | dtype 沒有正確 latch 到 CONV / Requant |
| FP layer 走到 INT path | dtype enum 或 `is_fp_dtype()` 沒接上 |
| Requant output byte width 錯 | `DT_INT16x8` / `DT_INT16x16` 沒被當成 int16 output |

---

## 3.6 Dependency tag 的核心語意

Dependency tag 是 8-bit ID，共 0 到 255。

在目前 simulator 裡：

- `tag_done[t] = true` 表示 tag `t` 已經完成。
- `tag_done[t] = false` 表示有工作還沒完成。
- `signal_tag = 0` 表示此 descriptor 完成時不發 tag。
- `wait_count = 0` 表示此 descriptor 可以立刻 issue。
- `wait_tags[]` 是 AND dependency，也就是所有 wait tag 都 done 才能 issue。

`waits_ready()` 的邏輯很直接：

```cpp
for (int w = 0; w < d.hdr.wait_count; ++w) {
    uint8_t tg = d.hdr.wait_tags[w];
    if (!tag_done[tg]) return false;
}
return true;
```

用圖表示：

```text
Descriptor A
  signal_tag = 7

Descriptor B
  wait_tags = [7]
  signal_tag = 8

執行順序：
  A issue -> A done -> tag 7 done -> B eligible -> B issue
```

如果有多個 wait tags：

```text
Descriptor C
  wait_tags = [3, 9, 12]

C 必須等 tag 3、9、12 都完成
```

這是 hardware scheduler 最常見的 primitive：用小小的 tag table 表示一個 DAG。

---

## 3.7 為什麼 tag_done 初始是 true

Command Engine constructor 裡：

```cpp
for (int i = 0; i < 256; ++i) tag_done[i] = true;
```

這代表「沒有被 active descriptor 佔用的 tag，都視為已完成」。

為什麼不是 false？

因為 descriptor 可能 wait 一個前面沒有出現過的 tag。若全部初始 false，任何誤填 wait tag 都會讓 simulation 卡死。初始 true 的語意是：

```text
只有被某個尚未完成的 descriptor reserve 的 tag，才是 not done。
```

這也帶來一個責任：當 descriptor 宣告 `signal_tag` 時，Command Engine 必須在正確時間把該 tag 標成 false，避免後面的 descriptor 太早通過。

---

## 3.8 Normal descriptor 的 tag reservation

Normal descriptor 是沒有 `DF_STREAM` flag 的 descriptor。它們保持 in-order issue。

對 normal descriptor，Command Engine 在 `issue()` 時才 reserve signal tag：

```cpp
if (!(d.hdr.flags & DF_STREAM) && d.hdr.signal_tag)
    tag_done[d.hdr.signal_tag] = false;
```

原因是 normal descriptor 不會越過前面的 normal descriptor，所以它只需要在真的發給 engine 前標記 pending。

這可以避免 8-bit tag wrap 的 hazard：

```text
tag 5 used by old descriptor
tag 5 later reused by new descriptor

如果新 descriptor 還只是在 pending queue 裡就把 tag 5 設 false，
可能會擾亂舊 descriptor 完成時的狀態。
```

因為 tag 只有 8 bits，compiler 必須小心不要讓同一個 tag 在太短距離內重用。Command Engine 也用 `LOOKAHEAD_LIMIT = 64` 控制 stream lookahead，不讓 pending window 大到超過合理 wrap distance。

---

## 3.9 Stream descriptor 的 tag reservation

Stream descriptor 有 `DF_STREAM` flag。它們可以在 lookahead window 裡被越序 issue。

對 stream descriptor，Command Engine 在 descriptor 進入 pending window 時就 reserve signal tag：

```cpp
if ((d.hdr.flags & DF_STREAM) && d.hdr.signal_tag)
    tag_done[d.hdr.signal_tag] = false;
```

為什麼 stream 要提早 reserve？

因為 stream descriptor 可以被越序。假設有一個較晚讀進 pending 的 stream descriptor `B`，後面某個 descriptor `C` 可能 wait `B.signal_tag`。如果 `B` 的 tag 還沒有被標成 false，`C` 可能誤以為 B 已完成，造成資料 hazard。

這裡的關鍵是：

| descriptor 類型 | issue 順序 | signal tag 何時 reserve |
|---|---|---|
| normal | in-order | issue 時 |
| stream | 可越序 | 進入 lookahead pending 時 |

這個差異很重要。讀 scheduler bug 時，先確認 descriptor 有沒有 `DF_STREAM`。

---

## 3.10 Dispatch loop 的 mental model

`CommandEngine::dispatch()` 大致分成三段：

```text
1. 從 desc_in 拉 descriptor 到 pending deque
2. 在 pending 裡挑一個 waits_ready 的最佳 descriptor
3. issue 後從 pending 移除
```

簡化 pseudo code：

```cpp
while (true) {
    fill_pending_until_lookahead_limit();

    best = find_ready_descriptor();

    if (best) {
        issue(best);
        remove(best);
        continue;
    }

    wait(new_descriptor_or_tag_changed);
}
```

這裡有兩個 SystemC event 很重要：

| event | 何時發生 |
|---|---|
| `desc_in.data_written_event()` | Host / upstream 寫入新 descriptor |
| `tag_changed` | 某個 engine done，Command Engine set tag_done true |

如果 pending 是空的，Command Engine 只等新 descriptor。

如果 pending 不空但都還不能 issue，Command Engine 同時等：

```text
新 descriptor 進來
或
某個 wait tag 完成
```

這就是 event-driven simulator 的味道：沒有工作時不空轉，時間由 engine 的 wait 與 memory latency 推進。

---

## 3.11 Normal in-order 與 stream bypass

Dispatch loop 有一個很重要的規則：

```text
Only descriptors explicitly marked as stream-pipeline work may be bypassed.
Normal descriptors keep in-order issue.
```

原因是 normal schedule 很可能重用固定 L1 address。如果讓 normal descriptor 隨便越序，就可能出現：

```text
store old tile 還沒完成
load new tile 覆蓋同一塊 L1
compute 讀到錯資料
```

所以 normal descriptor 的規則保守：

```text
pending front 如果 ready，就 issue
pending front 如果不 ready，後面的 normal descriptor 不看
```

stream descriptor 則不同。stream schedule 會透過 `layer_id`、`microblock_id`、`stream_slot`、`stream_meta_flags` 提供更多結構化資訊，讓 Command Engine 可以更積極 overlap：

```text
load tile N+1
compute tile N
store tile N-1
```

這就是 tile pipeline 的基本形狀。

---

## 3.12 Stream issue priority

stream descriptor ready 之後，Command Engine 用 priority 選誰先 issue。

目前 priority 概念如下：

| 工作 | priority 越小越優先 | 原因 |
|---|---:|---|
| `DF_STREAM_TAIL` | 0 | tail / barrier 要盡快清掉，避免整個 pipeline 收尾卡住 |
| EWE | 10 | element-wise compute 越早開始，越能和後續 UDMA overlap |
| UDMA read | 20 | DRAM 到 L1 的 load 會餵 compute |
| CONV | 30 | 主 compute |
| Requant / POOL | 40 | 通常跟 compute chain 或 tensor output 有關 |
| UDMA write | 60 | store 可以背景化，不能阻擋 younger load |

實作中還把 `microblock_id` 加進 tie-breaker：

```cpp
return base * 4096 + int(d.hdr.microblock_id);
```

這代表同一類 priority 裡，較小 microblock 大致先跑。

為什麼 UDMA write priority 比 UDMA read 低？

因為 output store 通常不是下一個 compute 的 immediate input。若 store 佔住 scheduling front，可能讓下一個 tile 的 input / weight load 太晚開始，降低 overlap。

---

## 3.13 Tail barrier 與 allowed_during_tail_wait

`DF_STREAM_TAIL` 用來標記 tail / barrier 類工作。

常見例子是：

```text
某些 L1-resident handoff 需要在真正覆蓋 buffer 前補一個 store barrier
```

當有 tail descriptor 在 pending 裡但還沒 ready，Command Engine 不是完全停住。它允許某些 younger work 先做：

| 允許類型 | 原因 |
|---|---|
| UDMA read | 可以先把後續 tile 的 input / weight 搬進 L1 |
| CONV / Requant 且有 `SMF_COMPUTE` | 允許淺層 compute front overlap |

這段邏輯在 `allowed_during_tail_wait()`：

```cpp
if (d.hdr.op_class() == OC_UDMA && d.body.udma.direction == 0)
    return true;

if ((d.hdr.op_class() == OC_CONV || d.hdr.op_class() == OC_REQUANT)
    && (d.hdr.stream_meta_flags & SMF_COMPUTE))
    return true;
```

這是 performance 和 correctness 的折衷：

```text
不要讓 tail 等待把整個 pipe 停死，
但也不要放任所有 younger store / arbitrary work 越過安全邊界。
```

---

## 3.14 完成訊號如何回到 Command Engine

每個 engine 做完工作後會寫 `done_tag_out`：

```cpp
done_tag_out.write(0);
```

payload 目前固定是 0，真正的 tag 不是 engine 回傳的，而是 Command Engine 自己用 per-engine queue 對應。

`issue()` 時：

```cpp
pending_tags[d.hdr.op_class()].push(d.hdr.signal_tag);
```

`collect()` 收到某個 engine done 後：

```cpp
uint8_t t = pending_tags[cls].front();
pending_tags[cls].pop();
tag_done[t] = true;
tag_changed.notify();
```

這代表 simulator 假設：

```text
同一個 engine 內部完成順序與 Command Engine issue 到該 engine 的順序一致。
```

目前每個 engine 都是一個 `SC_THREAD(run)`，每次 `cfg_in.read()` 後執行到 done，所以這個假設成立。

如果未來某個 engine 內部支援 out-of-order completion，就不能只用 FIFO pairing。那時 done payload 必須帶回 task ID 或 signal tag。

---

## 3.15 Descriptor body union

Header 決定 dispatch，body 決定 engine 具體要做什麼。

`DescriptorBody` 是 union：

```cpp
union DescriptorBody {
    ConvBody    conv;
    RequantBody requant;
    EweBody     ewe;
    PoolBody    pool;
    UdmaBody    udma;
    uint8_t     raw[48];
};
```

各 body 固定 48 bytes：

| Body | 用途 |
|---|---|
| `ConvBody` | input / weight address、shape、kernel、stride、pad、group、params address |
| `RequantBody` | chain output shape、output address、requant params、correction map |
| `EweBody` | element-wise / softmax / activation input-output address、shape、LUT / params |
| `PoolBody` | pooling input-output shape、kernel、stride、padding、mode |
| `UdmaBody` | copy mode、direction、address、length、stride、slice metadata |

body 是 union，所以你一定要根據 `op_class()` 解讀它。用錯 body 讀欄位，數值會看起來很怪，debug 時不要被騙。

---

## 3.16 ConvBody 導讀

`ConvBody` 的重點欄位：

| 欄位 | 意義 |
|---|---|
| `in_addr` | input tensor 在 L1 / DRAM address space 的位址 |
| `wgt_addr` | weight tensor 位址 |
| `out_addr` | 保留 / trace 用；目前 CONV functional output 走 chain 到 Requant |
| `in_h`、`in_w`、`in_c` | input shape |
| `out_c` | output channel count |
| `k_h`、`k_w` | kernel size |
| `stride_dilation` | stride / dilation packed field |
| `pad_tb`、`pad_lr` | top/bottom/left/right padding |
| `group` | group convolution / depthwise support |
| `cluster_mask` | active cluster hint |
| `in_pad_value` | padding value，量化模型通常是 `zp_in` |
| `bias_addr` | bias 位址；目前實際 bias 多在 Requant params fold |
| `scale_lut_addr` | requant / scale parameter blob 位址 |
| `scale_count` | per-channel scale 數量 |

`stride_dilation` 是 packed encoding：

```text
[1:0] = stride_h
[3:2] = stride_w
[5:4] = dilation_h
[7:6] = dilation_w
```

目前 `decode_stride()` 使用 log2 encoding：

| encoding | stride |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |

讀 CONV descriptor 時，請先自己算一次 output shape：

```text
out_h = (in_h + pad_t + pad_b - k_h) / stride_h + 1
out_w = (in_w + pad_l + pad_r - k_w) / stride_w + 1
```

如果 log 裡 output shape 和你期待不同，第一個檢查 `pad_tb`、`pad_lr`、`stride_dilation`。

---

## 3.17 RequantBody 導讀

Requant 是 CONV 後面很重要的 sink，也是 CONV / EWE 共用的 quantize-pack / clamp resource。CONV 把 int32 / FP32 partial sum 推進 128-lane functional chain（4096 bit/cyc），Requant timing model 以 512 elem/cycle 估算後段 MBQM / clamp / pack，把它轉成 output tensor。

`RequantBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `out_addr` | requant 後 tensor 寫到哪裡 |
| `n`、`h`、`w`、`c` | 這次 dispatch 要處理的 output shape |
| `scale_lut_addr` | requant parameter blob |
| `scale_count` | full layer 的 channel count |
| `oc_start` | 這個 OC tile 從第幾個 output channel 開始 |
| `out_w_layer` | full layer output width，correction map index 用 |
| `oh_start` | 此 tile 的 global output row offset |
| `corr_addr` | asymmetric weight zero-point correction map |
| `corr_per_oc` | correction map 是否 per output channel |

對 junior 來說，Requant 最容易混淆的是：

```text
c         = this dispatch 的 channel count
scale_count = full layer 的 channel count
oc_start     = this dispatch 在 full layer channel 裡的 offset
```

這三個值在 OC tiling 時不一樣。若 `oc_start` 錯，通常會看到某些 channel 整片 wrong，而不是隨機 noise。

---

## 3.18 EweBody 導讀

EWE 是 element-wise engine。它支援：

| subtype | 功能 |
|---|---|
| `ES_SOFTMAX` | softmax |
| `ES_ADD` | element-wise add |
| `ES_MUL` | element-wise multiply |
| `ES_SUB` | element-wise subtract |
| `ES_HARD_SWISH` | unary hard swish |
| `ES_GELU` | unary gelu |

`EweBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `in_a_addr` | 第一個 input |
| `in_b_addr` | 第二個 input；unary op 時可為 0 |
| `out_addr` | output 位址 |
| `n`、`h`、`w`、`c` | tensor shape |
| `broadcast_axes` | broadcast metadata，目前支援程度要看 compiler path |
| `reduce_axes` | reduce metadata |
| `scalar_imm` | scalar immediate |
| `lut_addr` | params / LUT 位址 |
| `subtype` | EWE subtype |

EWE 的 cycle model 依 dtype 選 lanes：

| dtype | EWE lanes |
|---|---:|
| INT8 | 64 |
| INT16 | 32 |
| FP | 32 |

binary op 通常：

```text
cycles ~= ceil(elements / lanes)
```

softmax 是三 pass：

```text
cycles ~= 3 * ceil(elements / lanes)
```

功能上，FP path 使用 FP16 storage、FP32 compute；INT path 使用量化參數 blob。

---

## 3.19 PoolBody 導讀

POOL 支援 max、average、global：

| enum | 功能 |
|---|---|
| `PM_MAX` | max pool |
| `PM_AVG` | average pool |
| `PM_GLOBAL` | global average pool 類型 |

`PoolBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `in_addr`、`out_addr` | input / output 位址 |
| `in_n`、`in_h`、`in_w`、`in_c` | input shape |
| `out_n`、`out_h`、`out_w`、`out_c` | output shape |
| `mode` | max / avg / global |
| `k_h`、`k_w` | kernel size |
| `stride` | stride_h / stride_w packed field |
| `pad_tb`、`pad_lr` | padding |
| `count_include_pad` | avg pool 是否把 padding 算進 divisor |

POOL 的 stride decode 也支援 1、2、4、8。這對某些模型的 global / large stride pooling 很重要。

POOL 的 cycle model 跟 EWE 使用同一套 dtype lanes：

| dtype | POOL lanes |
|---|---:|
| INT8 | 64 |
| INT16 | 32 |
| FP | 32 |

```text
cycles ~= ceil(out_elements / lanes) * max(k_h * k_w, 1)
```

---

## 3.20 UdmaBody 導讀

UDMA descriptor 用來搬資料或做部分 data layout transform。

目前 modes：

| mode | 功能 |
|---|---|
| `UM_LINEAR_COPY` | 連續 memory copy |
| `UM_STRIDED_2D` | 逐 row copy，source / destination stride 可不同 |
| `UM_INDEXED_GATHER` | 根據 index table gather |
| `UM_SCATTER_CONCAT` | concat 多個 source 到連續 destination |
| `UM_STRIDED_SLICE` | 2D slice |
| `UM_DEPTH_TO_SPACE` | NHWC depth-to-space transform |

`UdmaBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `mode` | UDMA mode |
| `direction` | 0 = DRAM 到 L1，1 = L1 到 DRAM |
| `src_addr`、`dst_addr` | source / destination |
| `length` | copy bytes；某些 mode 表示每 row bytes 或 element bytes |
| `src_stride`、`dst_stride` | stride bytes |
| `num_chunks` | rows / chunks / input rows |
| `idx_table_addr` | gather / concat metadata table |
| `slice_begin`、`slice_end` | slice 或 depth-to-space metadata |

UDMA descriptor 最常見的 debug 問題是 address space：

| address range | 意義 |
|---|---|
| `0x00000000` 到 `0x001FFFFF` | L1Mesh 2 MB |
| `0x10000000` 以上 | DRAM |

如果 direction 是 DRAM 到 L1，通常期待：

```text
src_addr in DRAM
dst_addr in L1
```

如果 direction 是 L1 到 DRAM，通常期待：

```text
src_addr in L1
dst_addr in DRAM
```

但 UDMA implementation 其實透過 `L1Manager` 讀寫，source / destination 只要 address space 合法就能工作。direction 主要用於 trace / profiling 分 lane。

---

## 3.21 Descriptor 產生：test_model.cpp 的 helper

Descriptor 不是手寫的。現在主要由 [`test_model.cpp`](../systemc/src/test_model.cpp) 的 helper 產生。

常見 helper：

| helper | 用途 |
|---|---|
| `alloc_tag()` | 分配新的 dependency tag |
| `make_desc()` | 建立 common header |
| `make_udma()` | 建立 UDMA descriptor |
| `make_pool()` | 建立 POOL descriptor |
| `make_softmax()` | 建立 EWE softmax descriptor |
| `make_ewe_add()` | 建立 EWE ADD descriptor |
| `make_ewe_unary()` | 建立 unary activation descriptor |
| `mark_stream()` | 把 descriptor 標成 stream work 並填 stream metadata |
| `make_store_barrier()` | 建立小型 UDMA store barrier |

`make_desc()` 的概念：

```text
填 op_class
填 dtype
填 signal_tag
填 wait_count / wait_tags
其他 header metadata 之後再補
```

`alloc_tag()` 通常會遞增 tag ID。因為 tag 是 8-bit，compiler 需要避免 live range 太長導致 wrap hazard。

讀 program generation 時請畫出：

```text
descriptor index
op class
wait tags
signal tag
L1 / DRAM address
```

很多 functional fail 都可以用這張表找到第一個可疑點。

---

## 3.22 一個典型 CONV layer 的 descriptor chain

一個普通 convolution layer 可能長這樣：

```text
UDMA read input
  signal tag = 11

UDMA read weight
  signal tag = 12

CONV
  wait tags  = [11, 12]
  signal tag = 13

REQUANT
  wait tags  = [13]
  signal tag = 14

UDMA write output
  wait tags  = [14]
  signal tag = 15
```

用 dependency DAG 表示：

```text
input load  ----\
                 +--> CONV --> Requant --> store
weight load ----/
```

如果 layer 有 tiling，就會有多組類似 chain：

```text
tile 0: load -> conv -> requant -> store
tile 1: load -> conv -> requant -> store
tile 2: load -> conv -> requant -> store
```

stream scheduler 的目標是讓它變成 overlap：

```text
time --->

UDMA read:   tile0 load   tile1 load   tile2 load
CONV:                    tile0 conv   tile1 conv   tile2 conv
Requant:                              tile0 req    tile1 req
UDMA write:                                      tile0 store  tile1 store
```

這種 overlap 成功與否，會反映在 profile Gantt 和 per-engine busy time。

---

## 3.23 L1-resident handoff 與 store barrier

有些 layer 不一定每次都把 output 立刻 store 回 DRAM。如果 producer output 可以直接留在 L1 給 consumer 用，稱為 L1-resident handoff。

優點：

| 優點 | 說明 |
|---|---|
| 少一次 DRAM write | output 不必先寫回 off-chip |
| 少一次 DRAM read | 下一層 input 不必再從 off-chip 讀回 |
| performance 好 | DRAM bandwidth pressure 降低 |

風險：

```text
如果 buffer 被下一個 tile 或下一層覆蓋，
而某個 consumer / store 還沒有讀完，
就會發生資料被提早覆蓋。
```

所以 code 裡有 `make_store_barrier()` 這類 helper，用一個很小的 UDMA store descriptor 當作 ordering barrier。它不一定代表真正大量資料搬運，有時更像「讓 dependency graph 有一個可等待的完成點」。

debug 這類問題時，要找：

| 要看什麼 | 為什麼 |
|---|---|
| `prev_store` tag | 上一個 store 是否真的被等待 |
| `suppress_producer_store` | 是否開啟 L1-resident handoff |
| `single_tile_layer` | 單 tile 與多 tile hazard 不同 |
| `make_store_barrier()` | 是否補了保護性 ordering |

---

## 3.24 Descriptor log 怎麼看

Command Engine dispatch 時會印：

```text
[CmdEng] dispatch op_class=... layer_id=... signal_tag=... wait_count=...
```

stream descriptor 還會印：

```text
slot=... mb=... smeta=0x...
```

engine 完成時會印：

```text
[CmdEng] engine 4 done; tag 12 set
```

讀 log 的方法：

1. 先找 fail layer 的 `layer_id`。
2. 找該 layer 的第一個 dispatch。
3. 把每個 descriptor 的 `signal_tag` 記下來。
4. 檢查等待的 tag 是否真的在前面完成。
5. 如果 pending 卡住，找最後一個沒有 set 的 tag。
6. 如果 output wrong，檢查 store 是否太早或太晚。

一個簡單 trace 表：

| 順序 | op | wait | signal | 解讀 |
|---:|---|---|---|---|
| 0 | UDMA read input | none | 10 | input loaded |
| 1 | UDMA read weight | none | 11 | weight loaded |
| 2 | CONV | 10, 11 | 12 | compute after both loads |
| 3 | Requant | 12 | 13 | output after conv |
| 4 | UDMA store | 13 | 14 | store after requant |

如果看到：

```text
Requant wait tag 不包含 CONV signal tag
```

那通常就是嚴重 correctness bug。

---

## 3.25 Debug：descriptor / tag 問題常見症狀

| 症狀 | 可能原因 |
|---|---|
| simulation 卡住 | 某個 wait tag 永遠沒有被 signal |
| output 大量 wrong | consumer 太早 issue，讀到未完成資料 |
| 只有多 tile model wrong | store / L1 overwrite ordering 問題 |
| stream mode 才 wrong | `DF_STREAM` reservation 或 priority 造成 hazard |
| 長模型後段 wrong | 8-bit tag wrap / reuse 太近 |
| INT16 model wrong | dtype latch 或 output byte width |
| FP model wrong | FP path 沒進對 subtype 或 params blob layout 錯 |

排查順序：

1. 確認 descriptor count 是否符合預期。
2. 確認每個 wait tag 都有 upstream signal。
3. 確認 signal tag 沒有在 live range 內重用。
4. 確認 UDMA direction 與 address range。
5. 確認 `layer_id` 和 fail layer 對得上。
6. 若只有 stream fail，檢查 `stream_meta_flags`。
7. 若只有 particular dtype fail，檢查 `dtype` latch。

---

## 3.26 Cycle accuracy 與 descriptor 的關係

Command Engine 目前主要是 functional scheduler，不太加入額外 dispatch cycle cost。simulation time 主要來自：

| 來源 | 在哪裡 wait |
|---|---|
| CONV compute cycles | `ConvEngine::run()` |
| Requant lane cycles | `RequantEngine::run()` |
| EWE lane cycles | `EweEngine::run()` |
| POOL lane cycles | `PoolEngine::run()` |
| L1 bank latency | `L1Mesh::read()` / `write()` |
| DRAM bandwidth / row miss / refresh | `Dram::read()` / `write()` |
| UDMA decode startup | `Udma::wait_bytes()` |

Descriptor 影響 cycle accuracy 的地方：

| Descriptor 設計 | Cycle 影響 |
|---|---|
| tiling 拆得越細 | CONV tile fill latency 會重複付 |
| stream scheduling | 可增加 overlap，降低 wall time |
| UDMA direction | read / write profile lane 分開 |
| wait tags 太保守 | engine idle 增加 |
| wait tags 太鬆 | functional wrong，cycle 再漂亮也沒用 |

所以 performance tuning 不是只改 engine cycles。你也要看 descriptor DAG 是否讓 engine 有機會 overlap。

---

## 3.27 Junior 練習：手畫一個 descriptor DAG

找一個小模型或小 layer，從 log 裡抽出 5 到 10 個 descriptor，做一張表：

| idx | op_class | layer_id | wait_tags | signal_tag | address 摘要 |
|---:|---|---:|---|---:|---|
| 0 | UDMA | 3 | none | 21 | DRAM input -> L1 |
| 1 | UDMA | 3 | none | 22 | DRAM weight -> L1 |
| 2 | CONV | 3 | 21, 22 | 23 | L1 input / weight |
| 3 | REQUANT | 3 | 23 | 24 | chain -> L1 output |
| 4 | UDMA | 3 | 24 | 25 | L1 output -> DRAM |

然後問自己：

- 哪些工作理論上可以平行？
- 哪個 wait tag 讓 CONV 開始？
- 哪個 wait tag 讓 store 開始？
- 如果 store 被延後，下一層會不會受影響？
- 如果 input load 被延後，CONV 會 idle 嗎？

這個練習比直接讀 1000 行 scheduler code 更有效。

---

## 3.28 常見誤解

| 誤解 | 正確理解 |
|---|---|
| tag 是 engine 回傳的 | 目前 engine 只回 done pulse，tag 由 Command Engine 的 pending queue 對應 |
| `signal_tag = 0` 是 tag 0 完成 | 在此格式中 0 表示 no signal |
| 所有 descriptor 都可以越序 | 只有 `DF_STREAM` descriptor 可 bypass |
| `wait_count = 0` 表示等 tag 0 | 它表示沒有 dependency |
| body 裡會帶 dtype | dtype 在 header，engine 用 side-channel latch |
| UDMA direction 決定 address 合法性 | address 合法性由 L1 / DRAM range 決定；direction 主要是語意與 profiling |
| cycle fail 一定是 engine cycle formula | descriptor DAG 太保守或太鬆也會造成 timing / correctness 問題 |

---

## 3.29 本章小結

Descriptor 是 MDLA7 最核心的控制介面。它把 compiler、scheduler、SystemC module、profile 全部接在一起。

本章要記住三件事：

1. `op_class` 決定送去哪個 engine，`body` 是該 engine 的參數。
2. `wait_tags` / `signal_tag` 是 dependency DAG，錯了就會卡住或讀錯資料。
3. `DF_STREAM` descriptor 允許 lookahead bypass，所以 tag reservation 和 priority 都比 normal descriptor 複雜。

你讀後面章節時，請一直把 descriptor 當成主線。看到任何 engine code，都問：

```text
它消費哪一種 body？
它何時 done？
它產生的資料被誰用？
它的 signal tag 被誰 wait？
```

這樣就不會迷路。

> 下一章 → [第 4 章 — Memory Hierarchy：DRAM、UDMA、L1Mesh](04_memory_hierarchy.md)


\newpage

# 第 4 章 — Memory Hierarchy：DRAM、UDMA、L1Mesh

> 上一章：[第 3 章 — Descriptor ISA 與 Dependency Tag](03_descriptor_tag.md)

本章你會學到什麼：

- MDLA7 的 address space 如何區分 L1Mesh 與 DRAM。
- `L1Manager` 在 simulator 裡扮演什麼角色。
- L1Mesh 16 banks、16-byte stripe、256 B/cycle peak 是什麼意思。
- DRAM row hit / row miss / refresh model 如何影響 cycle。
- UDMA 六種 mode 如何搬資料。
- memory hierarchy 如何和 descriptor tag、tiling、cycle accuracy 連在一起。

---

## 4.1 先用一張文字圖理解

MDLA7 的 memory path 可以先看成：

```text
Compute Engines
  CONV / Requant / EWE / POOL
        |
        v
L1Manager
        |
        +--> L1Mesh SRAM  0x00000000 - 0x002FFFFF
        |
        +--> DRAM         0x10000000 - 0xFFFFFFFF

UDMA 也透過 L1Manager 搬資料
```

CONV ACT_R / WGT_R read 各有專線到 CONV Engine，是 direct L1Mesh path；其他 engine / UDMA 透過
[`L1Manager`](../systemc/include/mdla7/memory.h)。這讓大部分 memory latency
入口集中，同時保留 CONV read 的最高服務優先權：

```cpp
l1mgr.read(addr, dst, n);
l1mgr.write(addr, src, n);
```

只要 address 在 L1 range，就走 L1Mesh；address 在 DRAM range，就走 Dram。

---

## 4.2 必讀檔案

本章主要看：

| 檔案 | 重點 |
|---|---|
| [`memory.h`](../systemc/include/mdla7/memory.h) | L1Mesh、Dram、L1Manager |
| [`udma.h`](../systemc/include/mdla7/udma.h) | UDMA descriptor execution |
| [`descriptor.h`](../systemc/include/mdla7/descriptor.h) | address range helper、UdmaBody |
| [`test_model.cpp`](../systemc/src/test_model.cpp) | L1 address allocation、DRAM tensor layout、UDMA descriptor emission |
| [`spec/spec.md`](../spec/spec.md) | memory bandwidth 與 architecture target |

讀法建議：

1. 從 `descriptor.h` 的 `L1MESH_BASE` / `DRAM_BASE` 開始。
2. 讀 `L1Manager::read()` / `write()`。
3. 讀 `L1Mesh::impose_bank_latency()`。
4. 讀 `Dram::impose_latency()`。
5. 讀 `Udma::run()` 和各個 `do_*()`。

這個順序會先建立「位址去哪裡」，再理解「時間怎麼算」。

---

## 4.3 Address space

MDLA7 simulator 的 address space 在 [`descriptor.h`](../systemc/include/mdla7/descriptor.h) 定義：

```cpp
constexpr uint32_t L1MESH_BASE  = 0x0000'0000;
constexpr uint32_t L1MESH_END   = 0x002F'FFFF;
constexpr uint32_t L1MESH_BYTES = L1MESH_END + 1;
constexpr uint32_t DRAM_BASE    = 0x1000'0000;
constexpr uint32_t DRAM_END     = 0xFFFF'FFFF;
```

換成表格：

| Range | 大小 / 意義 |
|---|---|
| `0x00000000` 到 `0x002FFFFF` | L1Mesh，3 MB on-chip SRAM |
| `0x10000000` 到 `0xFFFFFFFF` | DRAM address space |
| 其他 range | illegal，`L1Manager` 會報錯 |

helper：

```cpp
inline bool addr_in_l1mesh(uint32_t a) { return a <= L1MESH_END; }
inline bool addr_in_dram(uint32_t a) { return a >= DRAM_BASE; }
```

初學 debug memory 問題時，第一步就是把 address 分類：

```text
0x000xxxxx -> L1
0x100xxxxx -> DRAM
其他       -> 可疑
```

---

## 4.4 L1Manager：Non-CONV 入口 arbiter + memory router

硬體 spec 上，CONV ACT/WGT Payload R **直接接 L1Mesh，不經過 L1Manager**。
這條 direct path 讓 CONV read 取得最高服務優先權，避免 CONV compute
cluster starvation。

`L1Manager` 是 non-CONV engine / UDMA 進入 L1Mesh 的仲裁點。

L1Manager 入口 arbitration policy **v1 frozen = round-robin**：

| Rule | Meaning |
|---|---|
| participants | Requant / EWE / POOL / TNPS / UDMA non-CONV traffic |
| read / write | read side 與 write side 各自 round-robin |
| no fixed priority | 不用 fixed priority，避免 UDMA 或任一 non-CONV engine 長時間 starvation |
| skip rule | empty / stalled master 直接 skip，grant 給下一個 eligible master |
| CONV bypass | CONV ACT_R / WGT_R 不參與 L1Manager round-robin，直接接 L1Mesh |

目前 SystemC 實作仍是簡化的一階模型，`L1Manager` code path 接近 pass-through router：

```cpp
void read(uint32_t addr, void* dst, uint32_t n) {
    if (addr_in_l1mesh(addr)) mesh_.read(addr, dst, n);
    else if (addr_in_dram(addr)) dram_.read(addr, dst, n);
    else SC_REPORT_ERROR("L1Manager", "addr out of range");
}
```

它負責：

| 功能 | 目前狀態 |
|---|---|
| address decode | implemented |
| route to L1Mesh / DRAM | implemented |
| out-of-range error | implemented |
| Non-CONV entrance arbitration | spec defined; simulator simplified |
| QoS / arbitration | round-robin among non-CONV masters; CONV ACT_R / WGT_R bypass L1Manager via two dedicated direct L1Mesh paths |
| cache coherency | not relevant，這裡是 scratchpad |

換句話說，HW spec 有 CONV ACT_R / WGT_R dedicated direct paths + L1Manager non-CONV round-robin arbitration；
目前 simulator 還不是完整 port-accurate interconnect model。真正的 latency
主要在 L1Mesh / DRAM 裡 imposing wait。

這個設計對 simulator 有好處：

- non-CONV engine code 不需要知道某個 address 在 L1 還是 DRAM。
- UDMA mode 可以用同一套 `read()` / `write()`。
- future 如果要加完整 round-robin contention / port-accurate grant model，可以集中在 `L1Manager`。

---

## 4.5 L1Mesh 的基本 spec

L1Mesh 是 3 MB SRAM scratchpad，採 2 個 parallel 4x4 banked NoC plane。
兩個 mesh plane 共用同一組 16-bank SRAM backend。CONV ACT/WGT read 直接接
L1Mesh，不經過 L1Manager；L1Manager 負責 non-CONV engine / UDMA traffic。

目前 model：

| 參數 | 值 |
|---|---:|
| 容量 | 3 MB |
| bank 數 | 16 |
| NoC topology | 2 x 4x4 mesh planes |
| SRAM macro | 768 x 16B = 12 KB |
| macro 總數 | 256 |
| 每 bank macro | 16 |
| 每 bank 容量 | 192 KB |
| stripe | 16 bytes |
| Payload beat | 16 bytes |
| Payload transaction grouping | tid + last |
| router input FIFO depth | 2 flits provisional |
| per-bank bandwidth | 16 B/cycle |
| sequential peak | 16 banks × 16 B/cycle = 256 B/cycle |
| SRAM clock | 1.3 GHz |
| core clock axis | 1.9 GHz |

NoC edge ports per mesh plane：

目前 simulator 的 logical edge map 是 8 個 perimeter edges：

| Edge | Banks | Payload R lanes | Payload W lanes |
|---|---|---|---|
| W0 | B0, B1 | R0, R1 | W0, W1 |
| E0 | B2, B3 | R2, R3 | W2, W3 |
| W1 | B4, B5 | R4, R5 | W4, W5 |
| E1 | B6, B7 | R6, R7 | W6, W7 |
| W2 | B8, B9 | R8, R9 | W8, W9 |
| E2 | B10, B11 | R10, R11 | W10, W11 |
| W3 | B12, B13 | R12, R13 | W12, W13 |
| E3 | B14, B15 | R14, R15 | W14, W15 |

單一 mesh plane 合計是 16R + 16W edge injection/ejection；2 個 plane 合計是
32R + 32W。這是在降低 NoC 邊界/router/link hot spot，不是把 SRAM backend
bandwidth 乘二；真正的 per-bank service 仍是每 bank 每 SRAM cycle 一個
16B read 或 write beat。

Router input FIFO depth 暫訂為 2 flits。目前 SystemC mesh timing 只記錄
edge/router-output/link finish time，還沒有 input FIFO occupancy / upstream
backpressure，所以這個 depth 先是 architecture knob，不會改變現有 timing。

L1Mesh Payload input probe 可用環境變數打開：

```bash
MDLA7_L1_PAYLOAD_PROBE=batch/output/l1mesh_payload_probe.csv ./systemc/build/test_model ...
```

log 欄位是：

```text
cycle,1st payload (engineid) addr,2nd payload (engineid) addr,...,32nd payload (engineid) addr
```

前 16 欄對應 read Payload input lane 0..15，後 16 欄對應 write Payload input
lane 0..15。每格內容像 `(ewe) 0x00000100`；同一 cycle 同 lane 若有多個
16B beat，會用 `|` 接在同一格。

### 4.5.1 怎麼讀 `L1Mesh-4x4-NoC` 圖

如果你對 NoC 不熟，先不要把圖看成「一大塊 SRAM」。比較好的看法是：

```text
外面的 engine / L1Manager
        |
        v
  mesh 邊界入口 ingress
        |
        v
  2 x 4x4 router grid plane
        |
        v
  shared 目標 bank 的 SRAM macro port
```

圖上的每個名詞可以這樣拆：

| 名詞 | 白話意思 | 在 L1Mesh 圖上代表什麼 |
|---|---|---|
| NoC | Network-on-Chip，晶片內的小網路 | 4x4 router grid，用來把 request 送到正確 SRAM bank |
| mesh | 網格狀連線 | router 只跟上下左右鄰居相連，不是所有點互接 |
| router | 交叉路口 | 決定下一步往左、右、上、下，或送進本地 bank |
| link | 兩個 router 中間的路 | 同一方向同一 cycle 只能過有限資料，會塞車 |
| bank | 可以獨立服務的 SRAM 區塊 | B0..B15，總共 16 個 |
| macro | bank 裡更小的 SRAM 顆粒 | 每顆 `768 x 16B = 12 KB` |
| ingress | 進入 NoC 的入口 | request 從外部 engine 進到 mesh 的邊界 port |
| egress | 離開 NoC 的出口 | read data 從 SRAM bank 回 engine，或 write ack 離開 |

`ingress` 是最容易誤會的詞。它不是一顆 SRAM，也不是一個 bank。
它只是「traffic 從外面進入 mesh 的門口」。

用日常比喻：

```text
SRAM bank = 目的地建築物
router    = 路口
link      = 道路
ingress   = 高速公路交流道入口
```

所以「ACT_R left-edge ingress」的意思是：

```text
CONV 要讀 activation。
這些 read request 不走 L1Manager。
ACT_R 是 CONV Engine 專用線，不和 WGT_R 或 L1Manager_R 共用同一組入口。
它們從 L1Mesh 左邊的入口進入 NoC。
進去後再沿著其中一個 4x4 mesh plane 路由到目標 bank。
```

同理：

```text
WGT_R top-edge ingress
```

意思是：

```text
CONV 要讀 weight。
這些 read request 也不走 L1Manager。
WGT_R 是另一組 CONV Engine 專用線，不和 ACT_R 共用。
它們從 L1Mesh 上方的入口進入 NoC。
```

為什麼要分 left / top / right / bottom？因為不同 traffic 如果全部從同一邊進來，
很容易在同一批 edge port 和前幾個 router link 塞住。四邊分流的目的，是讓
traffic 一開始就分散：

```text
left   edge -> ACT_R dedicated CONV link
top    edge -> WGT_R dedicated CONV link
right  edge -> L1Manager_R
bottom edge -> L1Manager_W
```

這裡的重點是「分散入口」，不是「容量變成四倍」。最後資料還是要進某個 SRAM
bank，而 bank 本身每 cycle 能服務的 16B beat 數仍然有限。

### 4.5.2 一個 16B read beat 怎麼走

假設 CONV 要讀一個 activation byte range，其中某個 16B beat 的 address 算出來是
bank 6：

```text
bank_id = (addr >> 4) & 0xF = 6
```

4x4 bank 位置可以這樣看：

```text
row 0: B0  B1  B2  B3
row 1: B4  B5  B6  B7
row 2: B8  B9  B10 B11
row 3: B12 B13 B14 B15
```

所以 bank 6 在：

```text
x = bank_id % 4 = 2
y = bank_id / 4 = 1
```

如果 request 目標是 bank 6，logical edge map 會讓它從比較近的 E1 edge 進入：

```text
E1 edge -> B7 router -> B6 router -> local SRAM bank
```

用座標表示：

```text
source = (3, 1)
dest   = (2, 1)
route  = (3,1) -> (2,1)
```

目前 mesh mode 不再把每個 16B beat 的 hop latency 串起來加到 critical path；
它會為 edge/router/link/local resource 做 contention reservation，真正完成時間
主要由 bank SRAM service 和 queue wait 決定。

如果目標是 bank 14：

```text
B14 -> x=2, y=3
```

bank 14 走 E3 edge：

```text
source = (3, 3)
dest   = (2, 3)
route  = (3,3) -> (2,3)
```

目前 `L1Mesh-Payload-Edge-Map` 頁面列出 16R / 16W lanes 對 W/E edge 和 bank 的完整 mapping。

### 4.5.3 什麼叫 router/link conflict

假設兩個 request 同一個 cycle 都想走同一條路：

```text
request A: B4 -> B5 link
request B: B4 -> B5 link
```

如果這條 directed link 一個 cycle 只能送一個 flit，那第二個 request 就要等。
這就是 link conflict。

router output conflict 類似。假設同一個 router 同一時間有兩個 request 都想從 east
output 出去：

```text
router B4 east output -> B5
```

那也要仲裁，一個先走，一個等。

在 `--l1-timing=mesh` 裡，一個 16B beat 會依序消耗：

```text
1. long blocking call 先切成 simulator scheduling chunks
2. edge/router/link/local resource 做 contention reservation
3. 目標 bank 的 SRAM lane 服務 Payload beat/chunk
```

所以 mesh mode 比 port-conflict mode 多看了「路上資源是否有 queue buildup」，
但不再把每個 hop 固定 latency 疊到每個 16B stripe 上。

### 4.5.4 edge ports 為什麼不是 SRAM bandwidth

圖上寫：

```text
top edge    : 4R + 4W
right edge  : 4R + 4W
bottom edge : 4R + 4W
left edge   : 4R + 4W
per plane   : 16R + 16W
aggregate   : 32R + 32W across 2 planes
```

其中 left/top edge 分別對應 ACT_R / WGT_R 到 CONV Engine 的專用線。這代表 NoC 邊界有很多入口/出口，讓 traffic 比較容易進出 mesh。dual-plane
model 會把每個 16B flit 放到較不忙的 mesh plane，但最後仍然進同一個
SRAM bank backend。

但 SRAM backend 是另一回事。最後每個 bank 還是：

```text
bank service = 16B / SRAM cycle
```

因此不要把 `16R edge ports` 解讀成：

```text
每個 bank 都變 16 倍快
```

正確解讀是：

```text
很多 request 可以從不同邊進 mesh，降低入口塞車。
但是如果它們最後都打到同一個 bank，同一個 bank 還是會 serialize。
```

這也是為什麼我們分三種 timing mode：

| Mode | 看什麼瓶頸 | 沒看的東西 |
|---|---|---|
| `fast` | 總頻寬大概夠不夠 | bank hotspot、router/link hotspot |
| `conflict` | bank port 會不會撞 | NoC 入口、router、link 會不會撞 |
| `mesh` | edge/router/link/bank 一起估 | 還沒分 ACT/WGT/UDMA 的完整 QoS priority |

### 4.5.5 現在 simulator 的 mesh mode 做了什麼、沒做什麼

目前 code 在 [`memory.h`](../systemc/include/mdla7/memory.h) 裡，核心是
`L1Mesh::impose_mesh_latency()`。

它做了：

```text
Payload lane width = 16B
long blocking API call 先切成 simulator scheduling chunks
Payload 本身只帶 tid + last，不帶 burst metadata
每個 chunk 再分散到 16 banks
bank swizzle 減少 repeated-slice row/column 熱點
edge/router/link/local resource 只當 contention reservation
SRAM bank port service 仍是真正完成時間的主要限制
```

它還沒做完整硬體 QoS：

```text
還沒有把 requester 精確分成 CONV ACT_R / CONV WGT_R / L1Manager_R / L1Manager_W。
目前 L1Manager read/write API 沒帶 requester class。
所以 mesh mode 是 NoC 壅塞近似模型，不是 final RTL-grade NoC simulator。
```

這個限制很重要。它表示 `mesh/fast` 很高時，你應該把它當成：

```text
這個 layer 的 access pattern 可能有 NoC hotspot，值得看。
```

而不是馬上解讀成：

```text
硬體一定會慢這麼多。
```

source 裡的 constants：

```cpp
static constexpr unsigned N_BANKS = 16;
static constexpr unsigned BANK_STRIDE = 16;
static constexpr unsigned BYTES_PER_CYCLE = 16;
static constexpr unsigned PAYLOAD_SCHED_CHUNK_BEATS = 16;
static constexpr double CORE_CLOCK_GHZ = 1.9;
static constexpr double SRAM_CLOCK_GHZ = 1.3;
```

最重要的一句：

```text
Sequential access fans out across all 16 banks.
```

因為 stripe 是 16 bytes：

```text
address 0x0000 - 0x000F -> bank 0
address 0x0010 - 0x001F -> bank 1
address 0x0020 - 0x002F -> bank 2
...
address 0x00F0 - 0x00FF -> bank 15
address 0x0100 - 0x010F -> bank 0
```

連續大塊資料可以分散到 16 banks，理想 peak 就是 256 B/cycle。

Bank 內 macro mapping：

```text
byte_addr[3:0] = byte offset inside 16B beat
bank_id        = (byte_addr >> 4) & 0xF
bank_line      = byte_addr >> 8
macro_id       = bank_line / 768
row_addr       = bank_line % 768
```

每個 bank 的 read priority：

```text
1. CONV ACT_R
2. CONV WGT_R
3. L1Manager_R
```

write slot 由 `L1Manager_W` 使用。

Simulator 提供三種 L1 timing mode：

| Mode | CLI | 用途 |
|---|---|---|
| fast estimate | `--l1-timing=fast` | 預設。用 aggregate bandwidth 估算，不逐 bank 計算 port conflict，適合 regression sweep。 |
| port conflict | `--l1-timing=conflict` | 逐 bank finish-array 模型，read/read、write/write、read/write 同 bank 都會 serialize，適合架構分析。 |
| mesh conflict | `--l1-timing=mesh` | 逐 bank shared 1R/W SRAM service + dual-4x4 transparent NoC contention reservation；長 blocking call 依 Payload scheduling chunk 拆開估 timing；適合找 lane imbalance / queue buildup。 |
| mesh optimistic | `--l1-timing=mesh-opt` | 使用 mesh-style Payload chunk + SRAM bank service，但跳過 NoC resource reservation。 |

明確地說，`mesh` 不是取代 `conflict`，而是：

```text
mesh = SRAM bank/port conflict + NoC edge/router/link conflict
```

因此 `conflict/fast` 看 SRAM bank/port overhead，`mesh/conflict` 看額外 NoC
overhead，`mesh/fast` 則是兩者合併後的總 overhead。

mesh profile 的 HTML 會額外顯示 `L1Mesh Payload lane latency` 表：

| 欄位 | 意義 |
|---|---|
| accesses | 此 lane 被分配到的 Payload scheduling chunks 數 |
| KB | 此 lane 總服務資料量 |
| avg cyc / max cyc | request 到 scheduling chunk 完成的 latency |
| avg wait / max wait | queue wait，包含前序 chunk / resource contention |
| avg service / max service | 單一 scheduling chunk 在 SRAM lane 上的服務時間 |

如果 `max service` 很高，通常代表 scheduling chunk 或 beat 寬度不合理。
若 `max latency` 仍很高，多半是 large FC / weight transfer 的後段 chunks 排隊。

`batch/run_model.py` 可用 `--l1-timing` 單跑其中一種模式。`batch/run_mdla6_pattern.py`、
`batch/run_hotspot.py`、`batch/run_ethz_v5.py`、`batch/run_ethz_v6.py`、
`batch/run_mlperf.py` 則預設一次跑 fast / conflict / mesh 三種模式，方便同一份 HTML
裡直接比較 L1Mesh overhead。

---

## 4.6 L1 bank latency 怎麼算

`L1Mesh::read()` 和 `write()` 先 `memcpy`，再排入 beat-level timing queue：

```cpp
std::memcpy(dst, &mem[offset], n);
if (in_process()) schedule_latency(offset, n, is_read);
```

這表示 functional data 先被複製，然後 simulation time 被推進。對 single-thread deterministic simulator 來說，這是常見寫法。

latency model 的精神：

```text
每個 bank 有自己的 shared R/W finish time。
同一個 bank 的 access 會 serialize。
不同 bank 的 access 可以 parallel。
一個 request 的完成時間是所有 touched banks 的 max finish。
```

pseudo code：

```cpp
for each 16-byte stripe touched:
    bank = (addr / 16) % 16
    start = max(sram_bank_finish[bank], now)
    finish = start + beat_time
    sram_bank_finish[bank] = finish
    max_finish = max(max_finish, finish)

wait(max_finish - now)
```

如果一筆 access 是連續 256 bytes，它剛好 touch 16 banks：

```text
bank 0..15 各拿 16 bytes
理想上約 1 SRAM beat
換成 core cycle 要乘 1.9 / 1.3
```

如果一筆 access 每次都打同一個 bank，例如 stride 剛好讓地址落在同 bank：

```text
bank conflict 增加
parallelism 下降
latency 上升
```

這就是 banked SRAM model 想捕捉的現象。

---

## 4.7 SRAM clock 與 core clock 的縮放

Simulator 的時間軸用 core clock cycle 表示，註解裡提到 core clock 是 1.9 GHz。但 L1Mesh SRAM 是 1.3 GHz。

因此一個 SRAM beat 對 core cycle 的成本是：

```text
CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ = 1.9 / 1.3 ~= 1.46 core cycles
```

source 裡：

```cpp
const sc_core::sc_time access(
    beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
    sc_core::SC_NS);
```

這裡 `SC_NS` 在此 simulator 裡被當成抽象 cycle unit。你會看到很多地方用：

```cpp
wait(cycles, sc_core::SC_NS);
```

所以讀 code 時不要把它理解成真實 nanosecond，而要理解成：

```text
1 SC_NS ~= 1 simulator cycle
```

再由外層用 1.9 GHz 換算成 ms。

---

## 4.8 L1 read / write 共用 SRAM macro port

`L1Mesh` 裡每個 bank 只有一組 SRAM macro finish time：

```cpp
sc_core::sc_time sram_bank_finish_[N_BANKS];
```

這代表 SRAM macro 是 **1R/W**，同一個 bank 同時間只能服務 read 或 write 其中之一，不是 1R+1W。

對 model 的意義：

| 行為 | 模型 |
|---|---|
| read-read 同 bank | 會 serialize |
| write-write 同 bank | 會 serialize |
| read-write 同 bank | 會 serialize |
| different banks | 可 overlap |

這反映 spec 裡 L1Mesh 有讀寫 ingress lane，但最後共享同一組 16-bank SRAM macro backend。

如果未來要更接近 RTL，可以補：

- engine master arbitration。
- outstanding transaction depth。
- requester class（CONV ACT_R / CONV WGT_R / L1Manager_R / L1Manager_W）傳到 `L1Mesh`。

目前的 model 已足夠讓 tiling、bank conflict、DRAM pressure 對 performance 有一階效果。

---

## 4.9 DRAM model 的基本 spec

`Dram` model 是 LPDDR-class abstract model。

目前 constants：

| 參數 | 值 |
|---|---:|
| default capacity | 256 MB |
| row size | 8 KB |
| banks | 16 |
| bandwidth | 48 B/cycle |
| AXI burst length | 16 beats |
| AXI burst bytes | 256 B |
| row miss penalty | 50 cycles |
| refresh period | 7800 cycles |
| refresh stall | 200 cycles |

註解中的 bandwidth 推導：

```text
LPDDR5X-10667 dual x32:
10.667 Gbps/pin * 32 pins * 2 channels / 8 = 85.3 GB/s
85.3 GB/s / 1.9 G cycles/s ~= 44.9 B/cycle
round to 48 B/cycle
```

所以 DRAM sequential access 的理想 bandwidth 是 48 B/cycle，比 L1Mesh sequential peak 256 B/cycle 小很多。

DRAM timing 以 AXI burst window 收費：一個 beat 是 128b = 16B，burst
length 是 16 beats，所以一個 burst 是 256B。未對齊或很小的 transfer
會被 round 到它碰到的 256B burst window；跨兩個 window 就收兩個 burst。

這也是為什麼 NPU performance 很重視：

- L1 reuse
- tiling
- weight / activation locality
- L1-resident handoff
- avoiding redundant store/read

---

## 4.10 DRAM row hit / row miss

DRAM model 追蹤每個 bank 目前 open row：

```cpp
int32_t open_row_[N_BANKS];
```

address 會被拆成：

```cpp
off  = addr - DRAM_BASE;
bank = (off / ROW_BYTES) % N_BANKS;
row  = (off / ROW_BYTES) / N_BANKS;
```

如果同一個 bank 再次讀寫相同 row：

```text
row hit -> no row miss penalty
```

如果 row 不同：

```text
row miss -> +50 cycles
```

access bandwidth 本身：

```cpp
access = ceil(bytes / 48) cycles
```

總 latency：

```text
finish = start + row_miss_penalty_if_any + ceil(bytes / 48)
```

這是一個簡化模型，但可以捕捉兩件重要事情：

| 現象 | 模型反映 |
|---|---|
| 大塊連續讀寫比較有效 | row hit 多，bandwidth 主導 |
| 小碎片、跳躍讀寫比較慢 | row miss penalty 比例變大 |

---

## 4.11 DRAM refresh

DRAM 需要週期性 refresh。model 裡：

```cpp
REFRESH_PERIOD = 7800 cycles
REFRESH_STALL  = 200 cycles
```

當 access start time 跨過新的 refresh period：

```cpp
start += missed * REFRESH_STALL;
```

refresh overhead 比例約：

```text
200 / 7800 ~= 2.6%
```

這個數字不會主導每個 layer，但長模型、DRAM-heavy workload 會累積。

debug performance 時，如果你看到某些 UDMA access 比單純 bytes / 48 更慢，可能包含：

- row miss
- refresh stall
- previous DRAM access serialization

---

## 4.12 DRAM 目前是 single finish time

`Dram` 裡有一個 `last_finish_`：

```cpp
sc_core::sc_time last_finish_{sc_core::SC_ZERO_TIME};
```

每次 access start：

```cpp
start = max(last_finish_, now)
```

這代表 DRAM requests 在 model 裡大致 serialize。雖然它有 16 banks 與 open rows，但沒有完整 bank-level parallel outstanding model。

這是比較保守還是比較樂觀？

| 面向 | 影響 |
|---|---|
| 沒有多 request overlap | 對高併發 DRAM workload 較保守 |
| row hit / row miss 有建模 | 對 access locality 有區分 |
| fixed 48 B/cycle + 256B burst window | 保留 burst length / alignment penalty 的一階效果 |
| refresh 有建模 | 對長時間 bandwidth 有 overhead |

對目前 SystemC simulator，這是合理的一階模型。未來若要更接近 DRAM controller，可加入 per-bank finish time、read/write turnaround、outstanding queue。

---

## 4.13 UDMA 的角色

UDMA 是 DRAM 與 L1Mesh 之間的 data mover，也支援一些 layout transform。

在 descriptor graph 裡，UDMA 常見位置：

```text
DRAM input / weights
        |
        v
UDMA read
        |
        v
L1 tensors
        |
        v
CONV / EWE / POOL
        |
        v
L1 output
        |
        v
UDMA write
        |
        v
DRAM output
```

UDMA descriptor 的 `direction`：

| direction | 語意 | profile lane |
|---:|---|---|
| 0 | DRAM 到 L1，load | `tasks_read` |
| 1 | L1 到 DRAM，store | `tasks_write` |

UDMA implementation 裡會把 read / write busy time 分開記：

```cpp
busy_time_read
busy_time_write
tasks_read
tasks_write
```

這對 profile 很有用，因為 load 和 store 對 pipeline 的意義不同：

- load 餵 compute。
- store drain output，通常可以背景化。

---

## 4.14 UDMA descriptor 的共同欄位

`UdmaBody`：

| 欄位 | 說明 |
|---|---|
| `mode` | UDMA mode |
| `direction` | 0 read / 1 write |
| `src_addr` | source address |
| `dst_addr` | destination address |
| `length` | bytes；依 mode 有不同語意 |
| `src_stride` | source stride bytes |
| `dst_stride` | destination stride bytes |
| `num_chunks` | row / chunk count |
| `idx_table_addr` | gather / concat table |
| `slice_begin[4]` | slice / d2s metadata |
| `slice_end[4]` | slice metadata |

UDMA 的 `run()`：

```cpp
DescriptorBody body = cfg_in.read();
const UdmaBody& u = body.udma;

switch (u.mode) {
case UM_LINEAR_COPY:    do_linear(u); break;
case UM_STRIDED_2D:     do_strided(u); break;
case UM_INDEXED_GATHER: do_gather(u); break;
case UM_SCATTER_CONCAT: do_concat(u); break;
case UM_STRIDED_SLICE:  do_slice(u); break;
case UM_DEPTH_TO_SPACE: do_depth_to_space(u); break;
}
done_tag_out.write(0);
```

UDMA 本身只加 16-cycle decode startup：

```cpp
void wait_bytes(uint64_t) {
    wait(16, sc_core::SC_NS);
}
```

真正 memory bandwidth cost 由 `l1mgr.read()` / `write()` 裡的 L1Mesh / Dram 來 impose。

---

## 4.15 UDMA mode 0：LINEAR_COPY

最單純的 mode：

```text
copy length bytes from src_addr to dst_addr
```

source shape：

```cpp
std::vector<uint8_t> buf(u.length);
l1mgr.read(u.src_addr, buf.data(), u.length);
l1mgr.write(u.dst_addr, buf.data(), u.length);
wait_bytes(u.length);
```

常見用途：

| 用途 | 方向 |
|---|---|
| input tensor load | DRAM -> L1 |
| weight tile load | DRAM -> L1 |
| output tensor store | L1 -> DRAM |
| params blob load | DRAM -> L1 或直接放 L1 |

debug LINEAR_COPY：

| 檢查 | 說明 |
|---|---|
| `length` 是否等於 tensor bytes | dtype 會影響 byte count |
| source / destination 是否重疊 | L1 scratchpad reuse 時要小心 |
| direction 是否符合語意 | 雖非合法性必要，但 profile 會依它分 lane |
| wait tag 是否保護 consumer | compute 不可早於 load |

---

## 4.16 UDMA mode 1：STRIDED_2D

STRIDED_2D 用來複製多列資料：

```text
for r in rows:
    copy length bytes
    src += src_stride
    dst += dst_stride
```

source：

```cpp
for (uint16_t r = 0; r < u.num_chunks; ++r) {
    l1mgr.read(u.src_addr + r * u.src_stride, buf.data(), u.length);
    l1mgr.write(u.dst_addr + r * u.dst_stride, buf.data(), u.length);
}
```

常見用途：

- 只搬 tensor 的一個 rectangular tile。
- 從 full tensor row 裡取部分 columns / channels。
- 把 compact tile 寫回有 stride 的 destination layout。

你可以把欄位理解成：

| 欄位 | 語意 |
|---|---|
| `length` | 每 row 要複製多少 bytes |
| `num_chunks` | row 數 |
| `src_stride` | source 下一 row 距離 |
| `dst_stride` | destination 下一 row 距離 |

常見 bug：

| 現象 | 可能原因 |
|---|---|
| 每 row 開頭正確但下一 row 錯 | stride 設錯 |
| tile 只對第一 row | `num_chunks` 錯 |
| channel tail 錯 | `length` 沒乘 dtype bytes 或 channel count |

---

## 4.17 UDMA mode 2：INDEXED_GATHER

INDEXED_GATHER 先讀 index table：

```cpp
std::vector<uint32_t> idx(u.num_chunks);
l1mgr.read(u.idx_table_addr, idx.data(), idx.size() * sizeof(uint32_t));
```

再依 index 複製：

```text
dst[i] = src[idx[i]]
```

精確一點：

```cpp
s = src_addr + idx[i] * src_stride;
d = dst_addr + i * dst_stride;
copy length bytes
```

常見用途：

- gather 非連續 tensor block。
- 重新排列資料。
- 某些 op lowering 後需要 indirect copy。

debug 時注意：

| 檢查 | 說明 |
|---|---|
| `idx_table_addr` 在哪個 memory space | table 可能在 L1 或 DRAM |
| index 是 element index 還是 row index | source 用 `idx[i] * src_stride` |
| `dst_stride` 是否等於 output element pitch | gather 結果可能有 padding |

---

## 4.18 UDMA mode 3：SCATTER_CONCAT

SCATTER_CONCAT 用 metadata table 描述多個 source：

```cpp
struct ConcatEntry {
    uint32_t src_addr;
    uint32_t length;
};
```

流程：

```text
cursor = dst_addr
for each entry:
    copy entry.length bytes from entry.src_addr to cursor
    cursor += entry.length
```

常見用途：

- TFLite CONCATENATION lowering。
- 多個 branch output 合併成一個 tensor。
- 某些 logical concat 如果不能 L1 view 化，就需要實際 copy。

容易出錯的地方：

| 問題 | 說明 |
|---|---|
| concat axis 不是最後一維 | memory layout 可能不是單純 append |
| source 還沒完成 | concat descriptor wait tags 不完整 |
| source 留在 L1 但被覆蓋 | 需要 store barrier 或 dependency 保護 |
| non-conservative path | 若 compiler 做 view / handoff，要確認 data lifetime |

CONCAT 類 bug 通常不是 UDMA copy loop 本身錯，而是 upstream descriptor DAG 或 layout assumption 錯。

---

## 4.19 UDMA mode 4：STRIDED_SLICE

STRIDED_SLICE 支援 2D slice：

```text
rows = [slice_begin[0], slice_end[0])
col_off = slice_begin[1]
each output row length = length bytes
```

source address：

```cpp
s = src_addr + r * src_stride + col_off;
d = dst_addr + (r - r0) * dst_stride;
```

常見用途：

- TFLite STRIDED_SLICE。
- 取 tensor 的 row / column 子區域。
- 作為 compiler lowering 的簡化 copy primitive。

注意這裡的 `col_off` 是 byte offset，不一定是 element index。如果 dtype 是 int16 / fp16，要記得乘 2。

---

## 4.20 UDMA mode 5：DEPTH_TO_SPACE

DEPTH_TO_SPACE 是 NHWC layout transform。

descriptor encoding：

| 欄位 | 語意 |
|---|---|
| `num_chunks` | input H |
| `slice_begin[0]` | input W |
| `slice_begin[1]` | input Cin |
| `slice_begin[2]` | block size |
| `slice_begin[3]` | output Cout |
| `length` | element bytes |
| `src_stride` | input row bytes |
| `dst_stride` | output row bytes |

合法性檢查：

```text
Cin == Cout * block * block
```

核心 mapping：

```text
input  [ih, iw, ic]
q  = ic / Cout
oc = ic % Cout
bh = q / block
bw = q % block
output [ih * block + bh, iw * block + bw, oc]
```

這類 op 最容易被誤認為「只是 reshape」。但在 NHWC memory layout 下，它通常需要真的搬動 bytes。

---

## 4.21 Memory latency 與 compute latency 的 overlap

這份 simulator 裡，engine 常見 pattern 是：

```cpp
t_begin = sc_time_stamp();

l1mgr.read(...);   // memory latency pushes time
compute functional data

cyc = compute_cycle_formula(...);
elapsed = sc_time_stamp() - t_begin;
if (cyc > elapsed) wait(cyc - elapsed);
```

這代表：

```text
engine wall time = max(memory latency already paid, compute cycle estimate)
```

而不是：

```text
memory latency + compute cycles
```

這是刻意的。真實硬體中，operand streaming 與 compute pipeline 通常 overlap。若直接相加，會太悲觀。

例子：

```text
CONV input/weight read 花 100 cycles
CONV compute formula 250 cycles

engine total ~= 250 cycles
額外 wait = 250 - 100 = 150
```

如果 memory 花 300 cycles：

```text
engine total ~= 300 cycles
compute wait = 0
```

這個設計會在後面 cycle accuracy 章再深入。

---

## 4.22 UDMA 與 engine 的 overlap

UDMA 是獨立 `SC_THREAD`，CONV / EWE / POOL / Requant 也各自是獨立 thread。因此只要 dependency tag 允許，它們可以在 SystemC time 上 overlap。

例如：

```text
tile 0 CONV 正在跑
tile 1 UDMA read input 可以同時跑
tile -1 UDMA write output 也可能背景跑
```

這取決於 descriptor DAG：

| DAG 寫法 | 結果 |
|---|---|
| tile 1 load wait tile 0 store | 太保守，overlap 少 |
| tile 1 load 只 wait L1 buffer safe tag | overlap 較多 |
| compute 不 wait load | functional hazard |
| store 不 wait requant | output wrong |

memory hierarchy 的效能不是單看 L1 / DRAM bandwidth，還要看 Command Engine 是否把工作排得出來。

---

## 4.23 L1 capacity 與 tiling

L1Mesh 只有 3 MB。大模型 layer 不可能把所有 input、weight、output 都同時放進 L1。

所以 compiler / scheduler 需要 tiling：

```text
把 tensor 拆成 tile
每個 tile 搬入 L1
compute
寫出或 handoff
重用 L1 buffer 給下一個 tile
```

一個 tile 至少要考慮：

| 區塊 | 佔用 |
|---|---|
| input tile | activation bytes |
| weight tile | weight bytes |
| output tile | output bytes |
| params blob | scale / bias / LUT |
| scratch / correction map | op-specific temporary |
| double buffer | 若要 overlap load/compute，可能需要兩份 |

junior 常犯錯是只算 input + output，忘記 weight 和 params。對 convolution，weight 有時比 activation tile 更大。

---

## 4.24 L1-resident handoff

如果 producer layer 的 output 可以留在 L1，consumer layer 直接讀 L1，就可以避免：

```text
producer output L1 -> DRAM store
consumer input DRAM -> L1 load
```

這是 performance 上很有價值的 optimization。

但它要求 compiler 正確管理 lifetime：

```text
producer output buffer 不能在 consumer 讀完前被覆蓋
```

handoff 問題常見於：

- multi-tile layer。
- concat / branch。
- depth-to-space / reshape 類 tail op。
- suppressed producer store。
- buffer ping-pong slot reuse。

如果 functional regression 出現：

```text
single tile PASS
multi tile FAIL
```

請優先懷疑 L1 buffer lifetime 或 store barrier。

---

## 4.25 Descriptor wait tag 如何保護 memory

Memory correctness 通常不是由 `L1Manager` 保護，而是由 descriptor dependency 保護。

例如：

```text
UDMA load input
  signal tag 10

CONV
  wait tag 10
```

這保證 CONV 不會在 input load 完成前讀 L1。

另一個例子：

```text
Requant writes L1 output
  signal tag 20

UDMA store output
  wait tag 20
```

這保證 store 不會在 output tensor 完成前讀 L1。

L1 buffer reuse 也需要 tag：

```text
UDMA store old tile from buffer A
  signal tag 30

UDMA load new tile into buffer A
  wait tag 30
```

如果缺了這種 wait，functional data 可能被覆蓋，`L1Mesh` 不會阻止你。scratchpad 的精神就是 compiler / scheduler 自己管理。

---

## 4.26 看 profile 時怎麼判讀 memory bottleneck

Profile 裡通常會有 per-engine busy time 或 Gantt-like timeline。你可以問：

| 問題 | 解讀 |
|---|---|
| UDMA read lane 很長嗎？ | input / weight load 可能是瓶頸 |
| UDMA write lane 很長嗎？ | output store 或 concat / d2s tail 可能重 |
| CONV lane 有空洞嗎？ | load 太晚、dependency 太保守、tiling 不佳 |
| DRAM-heavy layer 是否接近 bytes / 48？ | bandwidth 主導 |
| 很多小 UDMA descriptor 嗎？ | 16-cycle decode startup 和 row miss 可能累積 |
| L1 access 是否跨很多 small stride？ | bank conflict 或 fragmented access |

簡單估算：

```text
DRAM cycles ~= bytes / 48 + row_miss_penalty + refresh overhead
L1 cycles   ~= bytes / 256 * 1.46，若 sequential 且 bank conflict 少
UDMA extra  ~= 16 cycles per descriptor
```

這只是 first-order estimate，但足夠幫你判斷 cycle 是否離譜。

---

## 4.27 Activation Compression 在 memory hierarchy 的位置

ACT compression/decompression path 已正式納入 v1 spec。當 profile 顯示 `UDMA_R` 長期 dominate，代表 DRAM→L1 的 activation load 可能比 compute 更重要，ACTC 就是 memory hierarchy 的標準解法之一。

最保守的設計是：

```text
DRAM compressed activation
  -> UDMA_R + ACT_DECOMP
  -> L1 raw NHWC tile
  -> CONV / EWE / POOL existing engines
```

這個設計有一個關鍵原則：**L1 裡仍然放 raw tensor**。所以 CONV 的 3x3 window、padding、stride、halo address 都不用知道 compression。Compression 只存在 DRAM storage format 與 UDMA_R/UDMA_W path 中。

如果一開始就做：

```text
L1 compressed activation -> CONV on-the-fly decompress
```

會馬上遇到幾個困難：

| 困難 | 原因 |
|---|---|
| random window access | CONV 會重複讀 halo row / overlapping window |
| block boundary | 3x3 window 可能跨 compressed block |
| stride / padding | address mapping 不再是簡單 NHWC offset |
| cycle model | 每個 MAC 讀 operand 前可能有 decompress latency |
| L1 lifetime | compressed block 和 raw window cache 都要管理 |

所以 v1 spec 固定採用 DRAM compressed、L1 decompressed。

### 4.27.1 UDMA ACTC mode

ACTC 使用 UDMA mode：

```text
UM_ACT_DECOMP_COPY:
  src_addr = DRAM compressed stream
  dst_addr = L1 raw tile
  idx_table_addr = block metadata table
  length = raw output bytes

UM_ACT_COMP_COPY:
  src_addr = L1 raw tile
  dst_addr = DRAM compressed stream
  idx_table_addr = block metadata table
  length = raw input bytes
```

v1 格式與 cycle model：

| 項目 | v1 決定 |
|---|---|
| raw block size | 128B |
| metadata | per-block compressed length + raw fallback flag；v1 offset implicit |
| fallback | 壓不下去就 raw block |
| codec throughput | 512 B/cyc，依 raw bytes 計 |
| DRAM charge | compressed bytes + metadata bytes |
| L1 charge | raw bytes |

Cycle model 至少要計：

| 項目 | 是否會變 |
|---|---|
| DRAM read bytes | 降低，讀 compressed bytes |
| metadata read bytes | 增加，讀 block table/header |
| L1 write bytes | 不變，寫 raw tile |
| ACT_DECOMP cycles | 增加，取決於 lanes / bytes per cycle |
| UDMA descriptor startup | 仍存在 |

因此 ACT compression 不是把 `dram_r` 直接除以 2。它是用較少 DRAM bytes 換 ACT_DECOMP / ACT_COMP latency，並且 L1 仍付 raw bytes。

### 4.27.2 Profile 上怎麼看值不值得做

先拆 `dram_r`：

```text
dram_r = activation read + weight read + params read + metadata/layout read
```

ACT compression 只影響 activation read。若 layer 的 DRAM read 主要是 weights，例如大 1x1 convolution 的 weight table，ACTC 幫助就小。若 layer 是高解析度、多 H tile、重複讀 input window，ACTC 才會有明顯收益。

判斷順序：

```text
1. UDMA_R 是否是 peak utilization？
2. top dram_r layer 的 bytes 是否主要來自 activation？
3. weight 是否已經 persistent？
4. fanout input 是否已經共用？
5. halo reload 是否還很多？
6. activation entropy 是否可能壓縮？
```

只有前幾個 scheduling / tiling 問題先排掉後，ACTC 的收益才比較乾淨。

---

## 4.28 Memory debug checklist

遇到 output mismatch，請照這個順序看：

| Step | 檢查 |
|---:|---|
| 1 | fail layer 的 input / output dtype byte width |
| 2 | UDMA `length` 是否等於預期 bytes |
| 3 | `src_addr` / `dst_addr` 是否在正確 address range |
| 4 | `src_stride` / `dst_stride` 是否用 bytes，不是 elements |
| 5 | compute descriptor 是否 wait load tags |
| 6 | store descriptor 是否 wait producer tags |
| 7 | L1 buffer reuse 是否 wait old consumer / store |
| 8 | concat / slice / d2s layout 是否符合 NHWC |
| 9 | INT16 / FP16 output 是否用 2 bytes |
| 10 | stream mode 是否讓 store / load 越序造成 overwrite |

遇到 performance regression，請照這個順序看：

| Step | 檢查 |
|---:|---|
| 1 | DRAM bytes 是否突然變多 |
| 2 | UDMA descriptor count 是否暴增 |
| 3 | L1-resident handoff 是否失效 |
| 4 | tiling 是否變碎，造成 CONV fill 重複付 |
| 5 | `wait_tags` 是否過度 serialize |
| 6 | UDMA write 是否擋住 UDMA read |
| 7 | L1 bank conflict 是否增加 |

---

## 4.29 一個手算例子：linear load

假設有一筆 UDMA read：

```text
src = DRAM
dst = L1
length = 98,304 bytes
```

粗估 DRAM：

```text
bytes / 48 = 2048 cycles
row miss = 50 cycles，至少第一個 row
refresh = 視時間點，可能 0 或多個 200 cycles
```

粗估 L1 write：

```text
bytes / 256 = 384 cycles
乘 SRAM/core ratio 1.46 -> 約 561 core cycles
```

UDMA decode：

```text
16 cycles
```

因為 UDMA `do_linear()` 是：

```text
read src -> write dst -> wait 16
```

所以這筆 descriptor 可能約：

```text
DRAM read 2098 + L1 write 561 + 16 = 2675 cycles
```

這是粗估。實際還會受 row locality、previous DRAM last_finish、L1 bank finish 影響。

---

## 4.30 一個手算例子：compute overlap

假設 CONV descriptor 需要：

```text
input L1 read = 200 cycles
weight L1 read = 500 cycles
compute formula = 1200 cycles
```

因為 model 用 max overlap：

```text
CONV total ~= max(memory elapsed, compute formula)
           ~= max(700, 1200)
           ~= 1200 cycles
```

如果 tiling 改變後：

```text
input L1 read = 200
weight L1 read = 1800
compute formula = 1200
```

那 CONV 變成 memory-dominated：

```text
CONV total ~= 2000 cycles
```

這能幫你理解為什麼有些 layer 增加算力不會變快，因為瓶頸在 memory。

---

## 4.31 常見誤解

| 誤解 | 正確理解 |
|---|---|
| L1 address 和 DRAM address 可以任意混用 | `L1Manager` 用 address range route，錯 range 會錯或報錯 |
| UDMA direction 決定讀寫哪種 memory | 真正讀寫由 `src_addr` / `dst_addr` range 決定；direction 是語意和 profile |
| L1 是無限快 | L1 有 bank latency、clock ratio、bank conflict |
| DRAM 只看 bytes / bandwidth | row miss、refresh、serialization 也會影響 |
| scratchpad 會自動避免 overwrite | 不會，必須靠 descriptor dependency 和 compiler lifetime 管理 |
| `length` 是 element count | UDMA 裡大多是 bytes，dtype 要自己換算 |
| CONV time = memory + compute | 目前 model 對 engine 內部採用 overlap，近似 max(memory, compute) |
| 很多小 UDMA 沒關係 | 每筆有 decode startup，也更容易 row miss |
| ACT compression 會讓 L1 也自動變小 | 若採 DRAM compressed / L1 decompressed，L1 footprint 不變，只省 DRAM bandwidth |

---

## 4.32 本章小結

MDLA7 memory hierarchy 的主線是：

```text
DRAM 大但慢
L1Mesh 小但快
UDMA 負責搬資料
descriptor tags 保護資料生命週期
tiling 決定 L1 是否裝得下與能否 overlap
```

你要特別記住：

1. L1Mesh 是 3 MB、4x4 NoC、16 banks、16-byte stripe，sequential peak 約 256 B/cycle。
2. DRAM 是 48 B/cycle，含 row miss 與 refresh，通常是大模型瓶頸之一。
3. UDMA 的功能正確仰賴 address、length、stride、wait tag 全部正確。
4. Scratchpad 不會自動保護資料，compiler / scheduler 必須管理 lifetime。
5. ACT compression 若採 DRAM compressed / L1 decompressed，可以先省 DRAM activation read，而不打擾 CONV/EWE/POOL 的 raw NHWC path。

下一章會進入 compute engines，看看 CONV、Requant、EWE、POOL 如何消費 descriptor body，並把 memory data 轉成 tensor output。

> 下一章 → [第 5 章 — Compute Engines Overview](05_compute_engines.md)


\newpage

# 第 5 章 — Compute Engines Overview

> 上一章：[第 4 章 — Memory Hierarchy：DRAM、UDMA、L1Mesh](04_memory_hierarchy.md)

本章你會學到什麼：

- MDLA7 compute engines 的共同 SystemC pattern。
- CONV Engine 如何讀 input / weight，並把 partial sum 推到 Requant chain。
- Requant Engine 如何做 INT fixed-point requant、FP clamp、INT16 output。
- EWE Engine 如何處理 ADD / MUL / SUB / softmax / unary activation。
- POOL Engine 如何處理 max / avg / global pooling。
- dtype、memory、descriptor tag、cycle model 如何在 engine 裡合流。
- junior debug compute mismatch 時該怎麼縮小問題。

---

## 5.1 Top view

上一章談 memory，本章談 compute。

MDLA7 目前 simulator 的主要 compute engines：

```text
Command Engine
    |
    +--> CONV Engine
    |       |
    |       v
    |   128-lane chain
    |       |
    |       v
    +--> Requant Engine
    |
    +--> EWE Engine
    |
    +--> POOL Engine
    |
    +--> UDMA
```

其中 UDMA 是 data mover，不算 compute engine，但它和 compute overlap，所以 performance 分析時一定一起看。

一個典型 convolution layer：

```text
UDMA load input / weight
        |
        v
CONV reads L1 input / weight
        |
        v
CONV pushes int32 or FP32 psum to chain[oc % 128]
        |
        v
Requant drains chain, applies bias / scale / clamp
        |
        v
Requant writes output tensor to L1
        |
        v
UDMA store output or next layer reads directly from L1
```

這條 CONV → Requant chain 是本章最重要的 data path。

---

## 5.2 必讀檔案

| 檔案 | 重點 |
|---|---|
| [`conv_engine.h`](../systemc/include/mdla7/conv_engine.h) | convolution functional path 與 bit-mult cycle model |
| [`requant_engine.h`](../systemc/include/mdla7/requant_engine.h) | CONV chain drain、requant params、INT / FP output |
| [`ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) | EWE 與 POOL engine |
| [`replay/softmax_lut.h`](../systemc/include/mdla7/softmax_lut.h) | 如果存在 softmax LUT，可對照 INT softmax |
| [`requant.h`](../systemc/include/mdla7/requant.h) | fixed-point multiplier helper |
| [`fp_utils.h`](../systemc/include/mdla7/fp_utils.h) | FP16 / FP32 conversion |
| [`descriptor.h`](../systemc/include/mdla7/descriptor.h) | body struct 與 dtype enum |
| [`test_model.cpp`](../systemc/src/test_model.cpp) | descriptor generation 與 layer lowering |

註：`softmax_lut.h` 實際 path 是 [`softmax_lut.h`](../systemc/include/mdla7/softmax_lut.h)，上表提醒你從 EWE include 找進去。

建議閱讀順序：

1. `ConvEngine::run()`。
2. `ConvEngine::compute_int()` 與 `compute_fp()`。
3. `RequantEngine::run()`。
4. `EweEngine::run()`。
5. `PoolEngine::run()`。

---

## 5.3 Engine 的共同 SystemC pattern

多數 engine 都長得像：

```cpp
SC_MODULE(SomeEngine) {
    sc_core::sc_fifo_in<DescriptorBody> cfg_in;
    sc_core::sc_fifo_out<uint8_t> done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time;
    std::vector<std::pair<uint64_t, uint64_t>> tasks;

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            t_begin = sc_time_stamp();

            // functional compute
            // memory read / write
            // wait cycle model

            t_end = sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(t_begin, t_end);
            done_tag_out.write(0);
        }
    }
};
```

這個 pattern 帶出幾個重要點：

| 機制 | 意義 |
|---|---|
| `cfg_in.read()` | engine 沒工作時 blocking，不推進時間 |
| `L1Manager& l1mgr` | engine 透過統一 memory path 讀寫 |
| `busy_time` | profiler 統計 engine active 時間 |
| `tasks` | Gantt / timeline 用 |
| `done_tag_out.write(0)` | 告訴 Command Engine 這筆 descriptor 完成 |

payload `0` 不代表 tag 0。真正 signal tag 由 Command Engine 的 pending queue 對應。這點在第 3 章已經講過。

---

## 5.4 dtype latch：header 資訊如何進 engine

每個 engine FIFO 收到的是 `DescriptorBody`，沒有 header。因此 dtype 透過 Command Engine side-channel latch：

```text
CommandEngine sees descriptor.hdr.dtype
        |
        +--> engine.last_dtype = dtype
        |
        +--> engine.cfg_in.write(body)
```

engine 裡常看到：

```cpp
DType dt = static_cast<DType>(last_dtype);
```

目前 dtype 會影響：

| Engine | dtype 影響 |
|---|---|
| CONV | 選 INT8x8、INT16x8、INT16x16、FP compute path |
| Requant | INT requant / FP clamp，output int8 / int16 / fp16 |
| EWE | INT element-wise 或 FP16 storage / FP32 compute |
| POOL | INT8 pool 或 FP16 storage / FP32 compute |
| UDMA | 不直接看 dtype，length 由 compiler 預先換算 bytes |

所以 dtype bug 通常不是局部問題，會一路影響 functional output 和 cycle。

---

## 5.5 CONV Engine 的責任邊界

CONV Engine 做：

- 讀 L1 input tensor。
- 讀 L1 weight tensor。
- 根據 shape / stride / pad / group 做 convolution。
- INT path 產生 int32 partial sum。
- FP path 產生 FP32 partial sum。
- 把 partial sum 寫入 `chain_out[oc % 128]`。
- 根據 bit-mult formula 估算 cycles。

CONV Engine 不做：

- 不直接寫 final quantized output tensor。
- 不做 TFLite fixed-point requant。
- 不做 activation clamp。
- 不做 output store 到 DRAM。

這個責任分界很重要：

```text
CONV = raw accumulation
Requant = bias / scale / zero-point / activation / output write
```

如果 output quantization 錯，不一定是 CONV 錯，常常是 Requant params 或 chain ordering。

---

## 5.6 CONV shape decode

CONV descriptor body 給 input shape、kernel、stride、pad。Engine 會算 output shape：

```cpp
uint32_t s_h = decode_stride(c.stride_dilation & 0x3);
uint32_t s_w = decode_stride((c.stride_dilation >> 2) & 0x3);
uint32_t pad_t = c.pad_tb & 7;
uint32_t pad_b = (c.pad_tb >> 3) & 7;
uint32_t pad_l = c.pad_lr & 7;
uint32_t pad_r = (c.pad_lr >> 3) & 7;

uint32_t out_h = (c.in_h + pad_t + pad_b - c.k_h) / s_h + 1;
uint32_t out_w = (c.in_w + pad_l + pad_r - c.k_w) / s_w + 1;
```

`decode_stride()`：

| encoding | stride |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |

如果 fail layer 的 output shape 不對，先看：

- `in_h` / `in_w`
- `k_h` / `k_w`
- `stride_dilation`
- `pad_tb` / `pad_lr`

這比一開始看 accumulation loop 更省時間。

---

## 5.7 CONV INT path

INT path 支援：

| dtype | activation type | weight type |
|---|---|---|
| `DT_INT8x8` | `int8_t` | `int8_t` |
| `DT_INT16x16` | `int16_t` | `int16_t` |
| `DT_INT16x8` | `int16_t` | `int8_t` |

template：

```cpp
template <typename T_a, typename T_w>
void compute_int(...)
```

資料讀取：

```cpp
std::vector<T_a> in_buf(in_h * in_w * in_c);
std::vector<T_w> wgt_buf(out_c * k_h * k_w * in_per_group);
l1mgr.read(c.in_addr, in_buf.data(), in_buf.size() * sizeof(T_a));
l1mgr.read(c.wgt_addr, wgt_buf.data(), wgt_buf.size() * sizeof(T_w));
```

loop order：

```text
for oh
  for ow
    for oc
      for kh
        for kw
          for icr
            sum += activation * weight
      chain[oc % 128].write(sum)
```

chain order 是 NHWC scan order：

```text
oh major
ow next
oc inner
```

Requant 必須用同樣順序 drain，否則 channel 會錯位。

---

## 5.8 Group convolution / depthwise 的 index

CONV 裡：

```cpp
group = c.group ? c.group : 1;
in_per_group = c.in_c / group;
out_per_group = c.out_c / group;
```

每個 output channel 屬於一個 group：

```cpp
g = oc / out_per_group;
ic_base = g * in_per_group;
```

weight layout：

```cpp
wgt[((oc * k_h + kh) * k_w + kw) * in_per_group + icr]
```

這代表每個 `oc` 只看自己 group 的 input channels。

depthwise convolution 通常可視為：

```text
group = input channels
out_per_group = depth_multiplier
```

如果 depthwise output 錯，常見原因：

| 可能原因 | 說明 |
|---|---|
| `group` 沒設對 | 會讀到錯 input channel group |
| weight layout 錯 | 每個 oc 的 kernel flatten 順序不一致 |
| correction map layout 錯 | asymmetric zp_w correction 常在 depthwise 特別處理 |

---

## 5.9 Padding value 與 quantization

CONV 的 padding value 來自：

```cpp
int16_t in_pad_value;
```

INT path 轉成：

```cpp
const int64_t pad_v = int64_t(c.in_pad_value);
```

在 padding 區域：

```cpp
a = pad_v;
```

量化模型裡，padding value 通常應該是 input zero-point `zp_in`，不是數值 0。這很關鍵：

```text
real value 0 對應 quantized zp_in
```

如果 padding 用 0 而不是 `zp_in`，邊界 pixel 會錯，尤其是：

- SAME padding。
- small feature map。
- depthwise convolution。
- first / last layer。

這就是為什麼 `in_pad_value` 被放進 descriptor。

---

## 5.10 CONV FP path

FP path 支援 FP dtype：

- `DT_FP8`
- `DT_FP16`
- `DT_BFP16`

目前 functional storage 主要是 FP16：

```cpp
std::vector<uint16_t> in_h16;
std::vector<uint16_t> wgt_h16;
```

compute 時轉 FP32：

```cpp
in_buf[i] = fp16_to_fp32(in_h16[i]);
wgt_buf[i] = fp16_to_fp32(wgt_h16[i]);
```

accumulation：

```cpp
float sum = 0.0f;
sum += a * w;
```

寫入 chain 時 bit-cast 成 int32 payload：

```cpp
int32_t bits;
std::memcpy(&bits, &sum, 4);
chain_out[lane]->write(bits);
```

這是 simulator trick：chain FIFO 型別是 `int32_t`，但 FP path 把 FP32 bits 放進去。Requant FP path 再 bit-cast 回 float。

---

## 5.11 CONV cycle model：bit-mult invariant

CONV cycle 公式：

```text
cycles = ceil(MAC_total * activation_bits * weight_bits / 1,048,576) + 64
```

其中：

```text
MAC_total = k_h * k_w * in_per_group * out_count
out_count = out_h * out_w * out_c
```

source：

```cpp
uint64_t bit_mult = mac_total * a * b;
return (bit_mult + 1048575) / 1048576 + 64;
```

解讀：

| 項目 | 意義 |
|---|---|
| `a * b` | dtype bit width 影響 throughput |
| `1,048,576` | abstract bit-multiply throughput |
| `+64` | tile fill / pipeline startup |

重要觀念：

```text
每個 CONV descriptor 都付一次 64-cycle fill。
```

所以 tiling 拆得越碎，fill overhead 越多。這是 cycle accuracy 章會深入的重點。

---

## 5.12 Requant Engine 的責任邊界

Requant Engine 做：

- 從 128-lane chain 讀 CONV partial sum。
- INT path 加 bias / correction。
- INT path 做 fixed-point requant。
- INT path 加 output zero-point。
- activation clamp。
- 寫 int8 或 int16 output tensor 到 L1。
- FP path 加 bias / clamp / FP16 output。

Requant Engine 不做：

- 不重新計算 convolution。
- 不讀 CONV input / weight。
- 不 store DRAM。

可以把它想成：

```text
CONV raw psum -> Requant output tensor
```

如果 CONV log 顯示 psum count 正確，但 output mismatch，下一個要看 Requant params blob。

---

## 5.13 Requant chain drain order

Requant 讀 chain：

```cpp
for oh
  for ow
    for oc
      lane = oc % 128
      psum = chain_in[lane]->read()
```

這必須和 CONV push order 完全一致。

chain lane 選擇：

```text
lane = oc % 128
```

如果 output channel count 不是 16 的倍數，也沒有問題，因為每個 `oc` 都固定對應 lane。

常見錯誤：

| 現象 | 可能原因 |
|---|---|
| channel 週期性錯位 | CONV push / Requant drain order 不一致 |
| simulation 卡在 Requant | CONV push 的 psum 數量不足 |
| Requant 提早 done | total shape `h*w*c` 填小 |
| 下一層 input 錯 | Requant output address 或 byte width 錯 |

---

## 5.14 Requant INT params blob

INT path params layout：

```text
[ int32 zp_out
| int32 act_min
| int32 act_max
| int32 mult[OC_layer]
| int8  shift[OC_layer]
| int32 bias_eff[OC_layer] ]
```

source offset：

```cpp
off_mult     = 12;
off_shift    = off_mult + 4 * OC_layer;
off_bias_eff = off_shift + OC_layer;
```

注意 `OC_layer` 和 this dispatch 的 `OC` 不一定一樣：

```text
OC_layer = r.scale_count ? r.scale_count : OC
OC       = r.c
oc_start = r.oc_start
```

讀取 this dispatch 的參數：

```cpp
l1mgr.read(scale_lut + off_mult + 4 * oc_start, mult.data(), OC * 4);
l1mgr.read(scale_lut + off_shift + oc_start, shift.data(), OC);
l1mgr.read(scale_lut + off_bias_eff + 4 * oc_start, bias_eff.data(), OC * 4);
```

這讓 OC tiling 可以共用同一份 full-layer params blob。

---

## 5.15 bias_eff 與 zero-point folding

量化 convolution 的理想形式：

```text
sum((q_in - zp_in) * (q_w - zp_w)) + bias
```

但 CONV Engine 為了簡化，主要做：

```text
sum(q_in * q_w)
```

其他 correction 被 compiler fold 到 Requant params：

```text
bias_eff = bias - zp_in * sum_w
```

Requant 裡：

```cpp
with_bias = psum + bias_eff[oc];
```

如果是 asymmetric uint8 weight zero-point，還可能有 per-pixel correction：

```cpp
with_bias += corr_val;
```

所以 quantization correctness 的責任分布是：

| 部分 | 責任 |
|---|---|
| CONV | raw `sum(q_in * q_w)` |
| compiler | 產生 `bias_eff`、`mult`、`shift`、correction map |
| Requant | 套用 params、clamp、寫 output |

debug 時不要只看 CONV sum，要連 params blob 一起看。

---

## 5.16 Requant fixed-point path

INT requant 核心：

```cpp
scaled = multiply_by_quantized_multiplier(psum_b, mult[oc], shift[oc]);
v = scaled + zp_out;
v = clamp(v, act_min, act_max);
```

對 int8 output：

```text
range = [-128, 127]
```

對 int16 output：

```text
range = [-32768, 32767]
```

code：

```cpp
const bool int16_out = (last_dtype == DT_INT16x16 || last_dtype == DT_INT16x8);
```

這是 INT16 model 的關鍵。`DT_INT16x8` 是 16-bit activation / 8-bit weight，但 output 仍然是 int16 path。

若 INT16 model 全部錯，請確認：

- input bytes 是否用 2。
- CONV activation type 是否 `int16_t`。
- Requant output buffer 是否 `int16_t`。
- UDMA store length 是否乘 2。

---

## 5.17 Requant FP path

FP params layout：

```text
[ f32 act_min | f32 act_max | f32 bias[OC_layer] ]
```

流程：

```cpp
bits = chain_in[lane]->read();
float psum = bitcast(bits);
v = psum + bias[oc];
v = clamp(v, act_min, act_max);
out_h16[idx] = fp32_to_fp16(v);
```

重點：

| 項目 | 說明 |
|---|---|
| chain payload | FP32 bits 放在 int32 FIFO |
| compute | FP32 |
| storage | FP16 |
| clamp | 使用 params blob 的 act_min / act_max |

FP mismatch 常見原因：

- compiler reference 和 simulator reduction order 不一致。
- FP16 round-trip 時機不同。
- params blob layout 和 INT path 混用。
- dtype 沒 latch，結果走到 INT path。

---

## 5.18 Requant lanes 與 cycle

RequantEngine 裡：

```cpp
static constexpr uint64_t LANES = 512;
```

這代表 CONV / EWE 共用的 quantize-pack / clamp resource。functional 仍然透過 chain FIFO，但 timing 上用 512 elem/cycle。

cycle：

```text
pipe = ceil(total_elements / 512)
```

同 CONV，一樣使用 overlap 概念：

```cpp
elapsed = sc_time_stamp() - t_begin;
if (pipe > elapsed) wait(pipe - elapsed);
```

如果 output tensor 很小，Requant cycle 很少；如果 output 很大但 L1 write latency 大，可能 memory-dominated。

---

## 5.19 EWE Engine 的責任邊界

EWE = Element-Wise Engine。

目前支援：

| subtype | INT | FP |
|---|---|---|
| ADD | yes | yes |
| MUL | yes | yes |
| SUB | yes | yes |
| SOFTMAX | yes | yes |
| HARD_SWISH | no / limited | yes |
| GELU | no / limited | yes |

EWE 的共同欄位：

```text
in_a_addr
in_b_addr
out_addr
n, h, w, c
lut_addr
subtype
```

element count：

```cpp
elems = uint64_t(e.h) * e.w * e.c;
```

目前 `n` 多數 path 沒特別乘進去，代表 compiler 產生 descriptor 時通常把 shape flatten 到 h/w/c 或 n=1。讀新的 model lowering 時要注意這個 assumption。

---

## 5.20 EWE ADD INT path

INT8 ADD params layout 是 12 個 int32：

```text
[ zp_a | zp_b | zp_out
| mult_a | shift_a
| mult_b | shift_b
| mult_out | shift_out
| left_shift | act_min | act_max ]
```

流程：

```cpp
a = (q_a - zp_a) << left_shift;
b = (q_b - zp_b) << left_shift;
sa = MBQM(a, mult_a, shift_a);
sb = MBQM(b, mult_b, shift_b);
s = sa + sb;
v = MBQM(s, mult_out, shift_out) + zp_out;
v = clamp(v, act_min, act_max);
```

這是 TFLite-style quantized ADD。

ADD mismatch 時看：

| 檢查 | 說明 |
|---|---|
| 兩個 input tensor address | branch output 是否已完成 |
| params blob 12 int32 | layout 是否對 |
| activation clamp | fused activation 是否 fold |
| broadcasting | 目前 EWE path 對 broadcast 支援需看 lowering |

---

## 5.21 EWE MUL / SUB INT path

MUL：

```text
raw = (a - zp_a) * (b - zp_b)
out = MBQM(raw, mult_out, shift_out) + zp_out
```

SUB：

```text
sa = MBQM((a - zp_a) << left_shift, mult_a, shift_a)
sb = MBQM((b - zp_b) << left_shift, mult_b, shift_b)
out = MBQM(sa - sb, mult_out, shift_out) + zp_out
```

MUL 和 SUB 共用 ADD-like params blob，只是有些欄位 unused 或語意不同。

debug 時不要因為都是 12 int32 就以為三者完全一樣。要看 subtype。

---

## 5.22 EWE FP binary path

FP ADD / MUL / SUB：

```cpp
uint16_t a16, b16;
float a = fp16_to_fp32(a16);
float b = fp16_to_fp32(b16);

v = a + b;  // or a * b, a - b
v = clamp(v, act_min, act_max);
out16 = fp32_to_fp16(v);
```

params blob 前 8 bytes：

```text
[ f32 act_min | f32 act_max ]
```

storage 是 FP16，compute 是 FP32。這和 CONV / Requant FP path 一致。

---

## 5.23 EWE unary FP：HARD_SWISH / GELU

HARD_SWISH：

```text
y = x * relu6(x + 3) / 6
```

GELU 使用 tanh approximation：

```text
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

source 裡使用 `std::tanh`，並註解要求 reference 要 mirror 相同 loop order / invocation，避免 FP 微小差異。

FP nonlinear op 的 regression 容易受這些因素影響：

- FP16 input rounding。
- FP32 intermediate。
- `std::tanh` vs numpy implementation 差異。
- output FP16 rounding。
- clamp sentinel 值。

---

## 5.24 EWE softmax

INT softmax：

```cpp
softmax_int8(in_buf.data(), out_buf.data(), elems);
```

FP softmax 使用 numerically stable 3-pass：

```text
1. max reduce
2. exp(x - max) and sum
3. exp / sum and cast to FP16
```

cycle model：

```text
per_pass = ceil(elems / lanes)
cycles = 3 * per_pass
```

其中 lanes 依 dtype 決定：INT8=64、INT16=32、FP=32。

softmax 常見 mismatch：

| 原因 | 說明 |
|---|---|
| axis flatten 錯 | softmax 應該沿特定 axis，不一定整個 tensor |
| LUT / reference 不一致 | INT softmax 很依賴 LUT |
| FP reduction order 不一致 | sum 順序會影響最後 bit |
| dtype latch 錯 | INT / FP path 搞混 |

---

## 5.25 EWE cycle model

EWE cycle 基本公式：

| op | cycle |
|---|---|
| ADD / MUL / SUB | `ceil(elems / lanes)` |
| HARD_SWISH / GELU | `ceil(elems / lanes)` |
| SOFTMAX | `3 * ceil(elems / lanes)` |

lanes 依 dtype：

| dtype | lanes |
|---|---:|
| INT8 | 64 |
| INT16 | 32 |
| FP | 32 |

memory read / write latency 由 L1Manager imposing。EWE code 目前多數 path 是：

```text
read input(s)
functional compute
write output
wait compute cycles
```

這裡不像 CONV / Requant 那樣明顯做 `max(memory, compute)` 補差，而是 read/write 後再 wait compute cycles。讀 timing 時要注意每個 engine 的 overlap 模型不完全一樣。

這也是後續 cycle accuracy 可以改善的地方。

---

## 5.26 POOL Engine 的責任邊界

POOL Engine 支援：

| mode | 說明 |
|---|---|
| `PM_MAX` | max pooling |
| `PM_AVG` | average pooling |
| `PM_GLOBAL` | global pooling，實作上走 avg-like path |

POOL descriptor 給：

```text
input shape
output shape
kernel
stride
padding
count_include_pad
```

POOL 讀 input tensor，直接寫 output tensor到 L1。它不走 CONV/Requant chain。

---

## 5.27 POOL INT path

INT path 目前用 int8 buffer：

```cpp
std::vector<int8_t> in_buf;
std::vector<int8_t> out_buf;
```

MAX：

```text
best = max(valid input window)
```

AVG：

```text
s = sum(valid input window)
div = count_include_pad ? k_h * k_w : valid_count
q = rounded_divide(s, div)
clamp to [-128, 127]
```

注意 average pooling 的 rounding：

```cpp
q = (s >= 0)
  ? (s + div / 2) / div
  : -((-s + div / 2) / div);
```

如果 avgpool 差 1，常常是 rounding rule 或 count_include_pad 差異。

---

## 5.28 POOL FP path

FP path：

```text
storage = FP16
compute = FP32
output  = FP16
```

MAX：

```text
best = max(fp32 input values)
```

AVG：

```text
s = running FP32 sum
div = pad policy
out = fp32_to_fp16(s / div)
```

source 註解強調 reduction order 要和 reference 一致。這是 FP regression 的核心原則：

```text
同樣數學公式，不同加總順序，也可能 bit 不同。
```

---

## 5.29 POOL stride 與 global kernel

POOL stride decode：

```cpp
pool_decode_stride(enc)
```

支援：

| encoding | stride |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |

kernel 特例：

```cpp
k_h = (p.k_h == 255) ? p.in_h : p.k_h;
k_w = (p.k_w == 255) ? p.in_w : p.k_w;
```

這讓 `255` 可代表 global / full input dimension。

如果某個 global avgpool output wrong，先檢查：

- `k_h` / `k_w` 是否被 encode 成 255。
- stride 是否正確。
- output shape 是否 1×1 或預期 shape。
- `count_include_pad`。

---

## 5.30 POOL cycle model

POOL cycle：

```text
out_elems = out_h * out_w * out_c
per_lane = ceil(out_elems / lanes)
cycles = per_lane * max(k_h * k_w, 1)
```

lanes 依 dtype 決定：INT8=64、INT16=32、FP=32。每個 output element 要看 `k_h*k_w` 個 input。

這對大 kernel global average pool 很敏感：

```text
global avgpool 8x8
k_h*k_w = 64
cycles = ceil(out_elems/lanes) * 64
```

所以某些模型 tail 的 pooling 可能不是完全免費。

Scheduler 也能把 POOL 放進 microblock fused tail：

- `CONV/Requant -> EWE -> POOL`：用 POOL output row 切 microblock，再反推
  producer rows，支援 real-window / global pool 的保守 handoff。
- `POOL -> ADD/MUL/SUB`：POOL output tile 直接作為 binary EWE input-A。
- `POOL -> GELU/HARD_SWISH`：POOL output tile 直接作為 unary EWE input。
- `POOL -> D2SPACE`：POOL output tile 交給 TNPS D2S。

POOL 仍是獨立 engine；microblock fuse 只是 Command Engine 用 L1 slot 和
dependency tag 串起 producer / consumer，省掉中間 DRAM checkpoint。

---

## 5.31 TNPS / D2SPACE 在目前 notebook 的位置

TNPS engine 現在是 layout transform 的主路徑之一，負責 transpose / slice / concat / space-depth 類 tensor movement。D2SPACE 要分三種情況看：

- `CONV/FC/DWCONV -> final DEPTH_TO_SPACE`：Requant final-store address swizzle，TNPS cycles 為 0。
- `CONV/FC/DWCONV -> DEPTH_TO_SPACE -> consumer`：TNPS tiled streaming path。
- standalone / non-CONV producer D2SPACE：TNPS `TM_DEPTH_TO_SPACE`。
- UDMA `UM_DEPTH_TO_SPACE`：legacy / debug fallback。

另外，若 GraphMeta 確認 `TRANSPOSE/PACK/UNPACK/SPLIT` 只是 intermediate
layout handoff，scheduler 可 suppress 這個 DRAM checkpoint，讓後續 compute
consumer 代表 functional verification anchor。這是 layout handoff model，不等於
任意 transpose tile kernel 已完全實作。

所以本章讀 compute engines 時要記得：D2SPACE 不一定在同一個 module 做；看 producer/consumer pattern 決定。

讀 spec 時要分清楚：

| 類型 | 意義 |
|---|---|
| HW spec target | 架構希望支援的 block |
| simulator implemented path | 目前 code 實際跑 regression 的 path |

這個 distinction 對新同事很重要。不要看到 spec 有一個 block，就以為 SystemC 已經有完整 functional model。

---

## 5.32 Engine 與 descriptor body 的對照

| Engine | FIFO body | 主要 source |
|---|---|---|
| CONV | `body.conv` | `ConvEngine::run()` |
| Requant | `body.requant` | `RequantEngine::run()` |
| EWE | `body.ewe` | `EweEngine::run()` |
| POOL | `body.pool` | `PoolEngine::run()` |
| UDMA | `body.udma` | `Udma::run()` |

Command Engine 只 dispatch：

```text
op_class -> corresponding cfg FIFO
```

engine 自己相信 body 欄位是對的。如果 compiler 把 `OC_EWE` descriptor 填成 `ConvBody` layout，engine 不會知道，它只會把 bytes 用 `EweBody` 解讀，結果一定怪。

---

## 5.33 Engine 與 memory 的對照

| Engine | 讀 memory | 寫 memory | 特殊 channel |
|---|---|---|---|
| CONV | input、weight | 不直接寫 final output | writes psum chain |
| Requant | params、correction map | output tensor | reads psum chain |
| EWE | input A / B、params | output tensor | none |
| POOL | input tensor | output tensor | none |
| UDMA | source address | destination address | none |

這張表可以幫你 debug：

```text
如果 output tensor in L1 錯，是誰最後寫它？
```

答案不一定是 CONV。對 conv layer，最後寫 output tensor 的是 Requant。

---

## 5.34 Engine done 與 tag completion

每個 engine 做完 descriptor 後：

```cpp
done_tag_out.write(0);
```

Command Engine 收到後：

```text
pop pending_tags[engine_class]
set tag_done[signal_tag] = true
notify tag_changed
```

因此 engine functional code 如果提早 `done_tag_out.write(0)`，後面 descriptor 就可能太早開始。

檢查 engine code 時，確認：

| 檢查 | 說明 |
|---|---|
| memory write 是否在 done 前 | output 必須寫完才 signal |
| wait cycle 是否在 done 前 | timing completion 應該包含 compute |
| exceptional path 是否也 done | unknown mode 不可卡死 |
| continue path 是否記 busy_time | profiler 不要漏 |

---

## 5.35 Functional correctness 的縮小法

遇到 layer mismatch，不要一次看全部。先分類：

| 問題類型 | 第一個懷疑 |
|---|---|
| CONV layer output wrong | CONV raw sum、Requant params、chain order |
| only edge pixels wrong | padding value / pad shape |
| only some channels wrong | OC tiling、oc_start、params offset |
| depthwise wrong | group、weight layout、correction map |
| INT16 wrong | dtype、byte width、Requant int16 output |
| FP wrong | FP16 conversion、reduction order、dtype latch |
| ADD / MUL / SUB wrong | EWE params blob、input branch dependency |
| avgpool off by 1 | rounding / count_include_pad |
| softmax wrong | axis / LUT / reduction order |
| concat tail wrong | UDMA concat layout / source lifetime |

然後問四個固定問題：

1. 這個 output tensor 是哪個 engine 寫的？
2. 這個 engine 讀了哪些 input / params？
3. descriptor wait tags 是否保證 input 已完成？
4. dtype 和 byte count 是否一致？

---

## 5.36 Cycle debug 的縮小法

遇到 cycle regression，也先分類：

| 現象 | 可能原因 |
|---|---|
| CONV time 變長 | dtype bits、MAC count、tile count、L1 read time |
| Requant time 變長 | output elements、L1 write、params/corr read |
| EWE time 變長 | element count、softmax 3-pass、memory read/write |
| POOL time 變長 | kernel size、output elements |
| UDMA time 變長 | bytes、row miss、more descriptors、stride fragmentation |
| wall time 變長但 busy time 沒變多 | dependency 太保守，engine idle |
| busy time 高但 wall time 不高 | overlap 好，這反而可能是正常 |

對 CONV 先手算：

```text
MAC_total = k_h * k_w * in_per_group * out_h * out_w * out_c
cycles = ceil(MAC_total * a_bits * b_bits / 1048576) + 64
```

對 UDMA 先手算：

```text
DRAM ~= bytes / 48 + row effects
L1 ~= bytes / 256 * 1.46
extra ~= 16 per descriptor
```

如果手算和 profile 差很多，再看 overlap、wait tags、bank conflict。

---

## 5.37 Code reading：從 CONV log 追到 output

CONV log 會印：

```text
[CONV] in=HxWxC k=... s=... pad=... out=OHxOWxOC dtype=...
[CONV] pushed N psums to chain
[CONV] estimated X cycles
```

Requant log：

```text
[Requant] OHxOWxOC oc_start=... scale_lut=...
```

你可以這樣對：

| 檢查 | 公式 |
|---|---|
| psum count | `OH * OW * OC` |
| Requant total | `r.h * r.w * r.c` |
| chain lanes | `oc % 128` |
| output bytes | `total * 1` for int8，`total * 2` for int16/fp16 |

如果 CONV pushed count 和 Requant total 不同，通常就是 descriptor generation bug。

---

## 5.38 Exercises for junior

### 練習 1：手算一個 CONV cycle

挑一個 CONV log：

```text
in=28x28x64
k=3x3
out=28x28x128
dtype=INT8x8
```

手算：

```text
MAC_total = 3 * 3 * 64 * 28 * 28 * 128
bit_mult = MAC_total * 8 * 8
cycles = ceil(bit_mult / 1048576) + 64
```

再和 log 的 estimated cycles 對。

### 練習 2：追一個 output tensor 的 writer

找 fail layer 的 output address，搜尋：

```bash
rg "out_addr|dst_addr" systemc/src/test_model.cpp
```

問：

- 是 Requant 寫的？
- 是 EWE 寫的？
- 是 POOL 寫的？
- 是 UDMA transform 寫的？

先找 writer，再找 reader。

### 練習 3：畫 chain order

假設 `OC = 20`，列出 oc 對 lane：

```text
oc 0  -> lane 0
oc 1  -> lane 1
...
oc 15 -> lane 15
oc 16 -> lane 0
oc 17 -> lane 1
```

然後想像 Requant 用同樣順序 drain。這能幫你理解為什麼 chain 不需要 20 條 lane。

---

## 5.39 常見誤解

| 誤解 | 正確理解 |
|---|---|
| CONV 直接寫 output tensor | CONV 寫 psum chain，Requant 寫 final tensor |
| Requant 只是簡單 cast | Requant 包含 bias、scale、zero-point、activation、correction |
| INT16x8 output 是 int8 | 目前 `DT_INT16x8` 走 int16 output path |
| FP path 全程 FP16 | storage FP16，但 compute / accumulation 多為 FP32 |
| EWE 所有 op 都只是逐 element 一 pass | softmax 是 3-pass，GELU / hardswish 有 nonlinear compute |
| POOL avg 差 1 一定是 bug in sum | 也可能是 rounding 或 count_include_pad |
| engine busy time 越低越好 | 如果 wall time 也高，可能是 dependency idle，不是效率好 |
| cycle model 全部都用同一種 overlap | 各 engine 實作略不同，需要讀 source |

---

## 5.40 本章小結

本章把主要 compute engines 串起來：

```text
CONV       raw MAC accumulation -> chain
Requant    chain -> quantized / FP output tensor
EWE        element-wise / softmax / activation
POOL       max / avg / global pooling
UDMA       data movement and layout transform
```

最重要的 mental model：

1. `DescriptorBody` 是 engine 的參數。
2. dtype 由 Command Engine latch，會決定 functional path 和 byte width。
3. CONV 和 Requant 是一條 chain，不能分開看。
4. Memory latency 和 compute cycle 都會影響 engine time。
5. Functional correctness 先追 writer / reader / params / dependency，再追 cycle。

下一個 Part 會開始進入 compiler：TFLite model 如何被拆成 layers、tensor、descriptor、reference output，最後餵給這些 SystemC engines。

> 下一章 → [第 6 章 — TFLite Flatbuffer 與 Op Extraction](06_tflite_flatbuffer.md)


\newpage

# 第 6 章 — TFLite Flatbuffer 與 Op Extraction

> 上一章：[第 5 章 — Compute Engines Overview](05_compute_engines.md)

本章你會學到什麼：

- `.tflite` 在 MDLA7 flow 裡如何被讀進來。
- 為什麼 compiler 使用 FlatBuffer parser，而不是只靠 TFLite Interpreter。
- `SUPPORTED_OPS` 如何決定哪些 op 會進入 MDLA7 program。
- shape、padding、stride、producer / consumer sidecar 如何被抽出。
- unsupported op / unsupported shape 為什麼會被 skip。
- junior 要如何從 compile log 追到 layer metadata。

---

## 6.1 Compiler 在整體 flow 的位置

MDLA7 end-to-end flow：

```text
.tflite
  |
  v
systemc/scripts/compile_model.py
  |
  v
program.bin
  |
  v
systemc/build/test_model
  |
  v
Mdla7System + verification + profile
```

本章聚焦 [`compile_model.py`](../systemc/scripts/compile_model.py) 前半段：怎麼從 `.tflite` 取得 op、tensor、shape、options、quantization metadata。

它不是完整 production compiler，而是 SystemC simulator 的 model compiler。它的主要責任是：

| 責任 | 說明 |
|---|---|
| parse TFLite graph | 讀 subgraph、operator、tensor、buffer |
| select supported ops | 只收 MDLA7 目前能模擬的 op |
| synthesize inputs | 產生 deterministic random input 或 chained reference |
| create references | 用 numpy reference 算每層 expected output |
| pack program.bin | 輸出給 C++ test_model 消費 |

---

## 6.2 為什麼用 FlatBuffer parser

TFLite Interpreter 可以執行模型，但它不是最好用的 compiler input。MDLA7 compiler 需要讀：

- op builtin options
- stride / padding / dilation
- fused activation
- tensor buffer bytes
- quantization scale / zero-point
- tensor producer / consumer relation

這些資訊用 FlatBuffer 直接讀比較穩定。

source：

```python
def _load_flatbuffer(path: str):
    import tflite as fb
    with open(path, "rb") as f:
        buf = f.read()
    model = fb.Model.GetRootAsModel(buf, 0)
    return fb, model, model.Subgraphs(0)
```

得到三個物件：

| 物件 | 用途 |
|---|---|
| `fb` | TFLite enum / options class namespace |
| `model` | whole flatbuffer model |
| `sg` | subgraph 0，主要 inference graph |

目前 compiler 只讀 subgraph 0。這對大多數 inference `.tflite` 足夠。

---

## 6.3 Operator name decode

TFLite op 在 flatbuffer 裡不是直接存字串，而是 builtin code。

compiler 用 `_opcode_name()` 轉成人類可讀名稱：

```python
def _opcode_name(fb, model, op):
    code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    return next((k for k, v in fb.BuiltinOperator.__dict__.items() if v == code),
                f"opcode_{code}")
```

例如：

| builtin op | compiler name |
|---|---|
| convolution | `CONV_2D` |
| depthwise convolution | `DEPTHWISE_CONV_2D` |
| fully-connected | `FULLY_CONNECTED` |
| pooling | `AVERAGE_POOL_2D` / `MAX_POOL_2D` |
| activation / elementwise | `ADD` / `MUL` / `SUB` / `HARD_SWISH` / `GELU` |
| data movement | `RESHAPE` / `CONCATENATION` / `GATHER` / `DEPTH_TO_SPACE` |

---

## 6.4 Supported ops

`SUPPORTED_OPS` 是 compiler 的第一道門：

```python
SUPPORTED_OPS = (
    "CONV_2D", "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "AVERAGE_POOL_2D", "MAX_POOL_2D",
    "SOFTMAX", "RESHAPE",
    "ADD", "CONCATENATION", "GATHER",
    "MUL", "SUB", "HARD_SWISH", "GELU", "MEAN",
    "DEPTH_TO_SPACE",
)
```

在 `main()` 裡：

```python
ops = []
for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    name = _opcode_name(fb, model, op)
    if name not in SUPPORTED_OPS:
        continue
    ops.append((i, name, op))
```

這代表 unsupported op 會被 compiler 略過。這不是一般 compiler 的最終行為，但對這個 simulator 有明確目的：

```text
讓已支援的 op 可以繼續被 regression，
同時清楚紀錄哪些 op 還不在 SystemC model 範圍內。
```

---

## 6.5 OP_KIND：Python 和 C++ 必須對齊

Python compiler 定義 `OP_CONV`、`OP_DWCONV` 等 enum。C++ [`test_model.cpp`](../systemc/src/test_model.cpp) 也有相同 enum。

Python：

```python
OP_CONV       = 0
OP_DWCONV     = 1
OP_AVG_POOL   = 2
OP_MAX_POOL   = 3
OP_SOFTMAX    = 4
OP_RESHAPE    = 5
OP_FC         = 6
OP_ADD        = 7
OP_CONCAT     = 8
OP_GATHER     = 9
OP_MUL        = 10
OP_SUB        = 11
OP_HARD_SWISH = 12
OP_GELU       = 13
OP_D2SPACE    = 14
OP_MATERIALIZE = 15
```

C++：

```cpp
enum OpKindEnum : uint16_t {
    OK_CONV = 0,
    OK_DWCONV = 1,
    ...
};
```

這是 binary interface。若 Python 新增 enum 但 C++ 沒同步，`program.bin` 會被錯誤解讀。

新增 op 時 checklist：

1. Python `OP_*` enum。
2. Python `OP_NAME`。
3. C++ `OpKindEnum`。
4. C++ `op_name()`。
5. compiler lowering。
6. test_model descriptor emission。
7. engine functional implementation。
8. reference / verification。

---

## 6.6 Tensor shape：從 TFLite rank 轉成 HWC

MDLA7 internal notebook 多用 HWC / NHWC。TFLite tensor shape 可能是：

| TFLite shape | 語意 |
|---|---|
| `[1, H, W, C]` | image tensor |
| `[N, features]` | FC / transformer tensor |
| `[C]` | scalar-like / vector |
| `[]` | scalar |
| higher rank | attention / sequence tensors |

compiler 用 `to_hwc()` canonicalize：

```python
def to_hwc(shape):
    if not shape:
        return (1, 1, 1)
    shape = shape if shape[0] == 1 else [1] + shape
    shape = shape[1:]
    while len(shape) < 3:
        shape = [1] + shape
    if len(shape) > 3:
        shape = shape[-3:]
    return tuple(shape)
```

這是 simulator-friendly simplification。它讓很多 transformer / vector op 可以用 `(H, W, C)` 的 metadata 走同一套 descriptor schema。

風險：

```text
高 rank tensor 被壓成最後三維時，axis semantics 可能被簡化。
```

所以 softmax、reshape、mean、gather 這類 axis-sensitive op 要特別小心。

---

## 6.7 Shape propagation

有些 FP / quantized model 的 static TFLite shape 會像 placeholder：

```text
[1, 1, 1, C]
```

但實際在 graph 裡應該有更大的 spatial shape。compiler 有 `_infer_shapes()` 做簡化 shape propagation，並用 `_pick()` 選擇 static 或 propagated shape：

```python
def _pick(static_hwc, prop_hwc):
    sH, sW, sC = static_hwc
    if prop_hwc is None:
        return static_hwc
    pH, pW, pC = prop_hwc
    if sH <= 1 and sW <= 1 and (pH > 1 or pW > 1):
        return prop_hwc
    return static_hwc
```

原則：

| 情況 | 選擇 |
|---|---|
| static shape 看起來正常 | 信任 TFLite static shape |
| static shape 像 placeholder | 使用 propagated shape |
| propagation 沒資料 | 使用 static shape |

這是保守策略，避免 shape propagation 對正常 INT8 model 造成 regression。

---

## 6.8 Padding 解析

TFLite convolution / pooling 常用 SAME / VALID padding。compiler 轉成 explicit pad：

```python
def _padding_to_pad(fb, padding_enum, K, in_dim, out_dim, stride):
    if padding_enum == fb.Padding.VALID:
        return 0, 0
    total = max(0, (out_dim - 1) * stride + K - in_dim)
    return total // 2, total - total // 2
```

對 SAME：

```text
total_pad = max(0, (out - 1) * stride + kernel - input)
top       = total_pad // 2
bottom    = total_pad - top
```

對 2D convolution，要分別算 H 和 W：

```text
pT, pB = padding_to_pad(H dimension)
pL, pR = padding_to_pad(W dimension)
```

這些值最後進入 `LayerMeta`，再由 C++ 填到 `ConvBody.pad_tb` / `pad_lr`。

---

## 6.9 Stride 限制

MDLA7 descriptor 的 stride encoding 是 2-bit log2：

| stride | encoding |
|---:|---:|
| 1 | 0 |
| 2 | 1 |
| 4 | 2 |
| 8 | 3 |

compiler 會 skip 不能表示的 stride：

```python
if s_h not in (1, 2, 4, 8) or s_w not in (1, 2, 4, 8):
    print("skipped ... stride not in {1,2,4,8}")
    last_output_arr = None
    continue
```

這比硬塞一個錯的 encoding 好。錯的 stride 會直接造成 output shape / data mismatch，而且很難 debug。

---

## 6.10 Tensor buffer bytes

constant tensor 的 raw bytes 來自 TFLite buffer：

```python
def _tensor_buffer_bytes(model, t):
    buf = model.Buffers(t.Buffer())
    if buf.DataLength() == 0:
        return None
    arr = buf.DataAsNumpy()
    return bytes(arr)
```

再由 `_tensor_array()` 依 tensor type 解成 numpy array：

```python
DTYPE_MAP = {
    INT8: np.int8,
    UINT8: np.uint8,
    INT16: np.int16,
    INT32: np.int32,
    FLOAT16: np.float16,
    FLOAT32: np.float32,
}
```

若 weight tensor 沒有 buffer，compiler 會嘗試追 DEQUANTIZE producer：

```text
weight tensor
  produced by DEQUANTIZE
    input has actual constant buffer
```

若仍拿不到，該 layer skip。

---

## 6.11 Producer / consumer graph

compiler 建立兩張 side table：

```python
producer_op_by_tensor[tensor_id] = op_index
consumers_by_tensor[tensor_id].append(op_index)
```

用途：

| 用途 | 說明 |
|---|---|
| chain input | layer N+1 可使用 layer N reference output |
| GraphMeta | 寫入 producer / consumer layer |
| L1 lifetime | C++ test_model 可用 last consumer 決定 store / handoff |
| unsupported op passthrough | producer / consumer 解析可穿過 compiler-elided op |

`GraphMeta` 是第 8 章會詳講的 program sidecar。先記住它不是 functional tensor data，而是 graph provenance。

---

## 6.12 Layer lowering 的總流程

每個 supported op 會進入一個大 loop：

```python
for li, (orig_op_index, opname, op) in enumerate(ops):
    read input tensor / output tensor
    choose H,W,C and OH,OW,OC
    choose dtype
    synthesize or chain input
    lower op-specific params
    compute numpy reference
    append layer dict
```

一個 layer dict 最後包含：

| 欄位 | 來源 |
|---|---|
| input / output shape | TFLite tensor shape + propagation |
| kernel / stride / pad | builtin options |
| `dram_in` / `dram_wgt` / `dram_out` | compiler DRAM region allocator |
| `in_size` / `wgt_size` / `ref_size` | serialized payload bytes |
| `op_kind` | Python enum |
| `dtype` | TFLite dtype lowered to MDLA7 dtype |
| `zp_in_eff` | CONV padding zero-point |
| graph tensor IDs | TFLite input / output tensor index |

---

## 6.13 Chain mode：reference output 當下一層 input

compiler 有一個 `last_output_arr`：

```python
last_output_arr = None
```

如果下一層 input shape / dtype 和上一層 output match，就直接使用上一層 reference output 當 input：

```python
if last_output_arr is not None
   and last_output_arr.shape == (H, W, Cin)
   and last_output_arr.dtype == expected_dtype:
    in_arr = last_output_arr
else:
    in_arr = random_synthetic_input
```

目的：

| 目的 | 說明 |
|---|---|
| reference 更接近真實 graph | 下一層不是完全 random input |
| 啟用 L1-resident fusion | C++ 可看出 producer-consumer relation |
| 降低 false mismatch | chained ops 對同一份資料做 reference |

如果某層 skip，chain 會 break：

```python
last_output_arr = None
```

這很重要，否則 skip op 後的下一層會吃到錯誤 upstream reference。

---

## 6.14 Compile log 怎麼讀

compiler canonical line：

```text
layer 12     conv  in=56x56x32  k=3x3  s=1x1  g=1  out=56x56x64  (200704 INT8)  ready
```

欄位：

| 欄位 | 意義 |
|---|---|
| `layer 12` | compiler layer index |
| op name | lowered op kind |
| `in=H×W×C` | input HWC |
| `k=Kh×Kw` | kernel / op shape hint |
| `s=Sh×Sw` | stride |
| `g` | group count |
| `out=OH×OW×OC` | output HWC |
| `(N TYPE)` | output elements and dtype |
| `ready` | layer was compiled |

Skip line 通常會帶原因：

```text
skipped (stride=3x3 not in {1,2,4,8})
skipped (weights tensor has no buffer)
skipped (shape exceeds descriptor's ushort dim limit)
```

junior debug compile-fail 時，先找第一個 skipped / SystemExit，而不是先看 C++。

現在 Hotspot path 也有 `matrlz` fallback：

```text
layer  4  matrlz  in=1x1x77  k=1x1  s=1x1  g=1  out=1x1x77  (77 INT8)  ready
```

`matrlz` 是 ready layer，不是 skipped。它代表 compiler 把該 op 的
reference tensor 預先 materialize，simulator 再用分塊 UDMA
`DRAM -> L1 -> DRAM` copy 來保留 profile/verification coverage。常見來源：

- non-spatial `MEAN` axes。
- runtime-matmul `FULLY_CONNECTED`。
- INT `GELU` / `HARD_SWISH` fallback。
- attention reshape shape-prop mismatch。
- descriptor `uint16_t` dim overflow。

---

## 6.15 常見誤解

| 誤解 | 正確理解 |
|---|---|
| TFLite Interpreter 是 compiler 唯一入口 | MDLA7 compiler 主要用 FlatBuffer 讀 graph metadata |
| unsupported op 會變成 no-op descriptor | 目前多數 unsupported op 直接不進 compiled layer list |
| HWC 一定等於原始 TFLite rank | HWC 是 simulator canonical form，高 rank 會簡化 |
| SAME padding 是固定 pad 1 | padding 依 input/output/kernel/stride 計算 |
| stride 任意值都支援 | descriptor 目前只支援 1/2/4/8 |
| chain mode 等於真實 TFLite full inference | 它是 self-consistent reference chain，不是完整 runtime fidelity |
| skip op 後還能安全沿用前一層 output | skip 會 break chain，避免 downstream 假資料 |

---

## 6.16 本章小結

`compile_model.py` 的前半段把 TFLite graph 變成一串可 lower 的 MDLA7 layer：

```text
FlatBuffer model
  -> supported op list
  -> HWC shape
  -> options / quant metadata
  -> producer-consumer relation
  -> layer dict
```

你要記住：

1. `SUPPORTED_OPS` 決定哪些 op 進入 simulator。
2. shape / padding / stride 是 descriptor correctness 的第一層。
3. producer / consumer sidecar 是後面 L1 lifetime 和 fusion 的基礎。
4. compile log 是 debug 的第一個入口。

> 下一章 → [第 7 章 — Quantization / FP / INT16 Compile Path](07_quantization_fp_int16.md)


\newpage

# 第 7 章 — Quantization / FP / INT16 Compile Path

> 上一章：[第 6 章 — TFLite Flatbuffer 與 Op Extraction](06_tflite_flatbuffer.md)

本章你會學到什麼：

- INT8 quantized CONV 如何從 TFLite scale / zero-point 變成 Requant params。
- `bias_eff`、`zp_in_eff`、`zp_w` correction map 的意義。
- `multiply_by_quantized_multiplier` 為什麼要對齊 TFLite。
- INT16x16 與 INT16x8 hybrid path 如何 lower。
- FP16 / FP32-source model 如何被轉成 FP16 storage + FP32 compute。
- ADD / MUL / SUB / pool / softmax 的 reference 產生方式。

---

## 7.1 這章為什麼重要

NPU simulator 的 PASS / FAIL 很大一部分不是 MAC 算錯，而是 quantization lowering 差一點：

```text
zero-point 少減一次
bias 沒 fold
multiplier rounding 不同
activation clamp range 錯
uint8 byte representation 沒 shift
INT16 output byte width 錯
FP16 rounding 時機不同
```

本章的核心觀念：

```text
compiler 產生 reference 和 params；
SystemC engine 消費 params；
兩邊的數學與 byte layout 必須一致。
```

---

## 7.2 dtype lowering

TFLite tensor type 會被 mapping 成 MDLA7 descriptor dtype：

| TFLite type | MDLA7 dtype | 說明 |
|---|---|---|
| INT8 | `DT_INT8x8` | signed int8 path |
| UINT8 | `DT_INT8x8` | byte representation shift 到 signed int8 |
| INT16 | `DT_INT16x16` 或 `DT_INT16x8` | 看 weight dtype |
| FLOAT16 | `DT_FP16` | FP16 storage |
| FLOAT32 | `DT_FP16` | deployment policy：FP32 source lowered to FP16 storage |

INT16x8 特例：

```python
if input is INT16 and weight is INT8:
    layer_dtype = DT_INT16x8
```

這對 `unet_int16`、`esrgan_int16` 這類 16x8 quantization model 很重要。

---

## 7.3 INT8 CONV 的理想數學

TFLite quantized convolution 大致是：

```text
acc = sum((q_in - zp_in) * (q_w - zp_w)) + bias
out = clamp(MBQM(acc, mult, shift) + zp_out)
```

展開：

```text
sum(q_in * q_w)
- zp_w * sum(q_in)
- zp_in * sum(q_w)
+ zp_in * zp_w * window_size
+ bias
```

SystemC CONV 為了保持 datapath 簡單，主要做：

```text
psum = sum(q_in * q_w)
```

其他項目由 compiler fold 到 Requant params 或 correction map。

---

## 7.4 zp_in_eff 與 padding

TFLite 的 real zero 在 quantized tensor 裡是 `zp_in`。所以 SAME padding 的 OOB input 應該填：

```text
q_in = zp_in
```

compiler 把這個值放進 `LayerMeta.zp_in_eff`：

```python
zp_in_eff = zp_in if input type is INT8 else 0
```

C++ test_model 會把它填入：

```cpp
ConvBody.in_pad_value
```

CONV Engine OOB 時使用：

```cpp
a = pad_v;
```

如果這個值錯，最典型的症狀是：

| 症狀 | 原因 |
|---|---|
| 邊界 pixel 錯 | padding quantized value 錯 |
| center pixel 對 | kernel 完全在 input 內，不受 padding 影響 |
| depthwise 特別明顯 | 每個 channel 獨立，邊界差異難被平均掉 |

---

## 7.5 bias_eff

compiler fold：

```python
bias_eff = bias - zp_in_eff * sum_w + zp_in_eff * zp_w * window_size
```

然後寫到 params blob：

```text
[zp_out, act_min, act_max, mult[], shift[], bias_eff[]]
```

Requant Engine：

```cpp
with_bias = psum + bias_eff[oc];
```

這樣 CONV Engine 不需要在 inner loop 每次做 `(q_in - zp_in)`，硬體 datapath 也比較接近 raw MAC accumulator。

---

## 7.6 weight zero-point correction map

當 weight zero-point `zp_w` 不為 0 時，展開式裡有一項：

```text
- zp_w * sum_window(q_in)
```

這個值依 output pixel 位置而變，不能只放 per-channel bias。

compiler 會產生 correction map：

| conv type | correction map shape |
|---|---|
| normal conv | `[OH, OW]` |
| depthwise conv | `[OH, OW, OC]` |

RequantBody 欄位：

| 欄位 | 說明 |
|---|---|
| `corr_addr` | correction map address |
| `corr_per_oc` | 是否每個 OC 都有 correction |
| `out_w_layer` | full layer width |
| `oh_start` | height tile offset |
| `oc_start` | channel tile offset |

這套機制是 asymmetric uint8 conv 能 PASS 的關鍵。

---

## 7.7 Quantized multiplier

TFLite requantization 使用 fixed-point multiplier。compiler 裡：

```python
def quantize_multiplier(scale):
    m, e = math.frexp(scale)
    q = floor(m * 2^31 + 0.5)
    return q, e
```

重點是 rounding：

```text
TFLite C++ std::round 是 half-away-from-zero；
Python round 是 banker's rounding。
```

所以 code 用：

```python
floor(x + 0.5)
```

避免 1-LSB mismatch。

SystemC Requant 使用 [`requant.h`](../systemc/include/mdla7/requant.h) 的 helper，必須和 numpy reference 一致。

---

## 7.8 Activation clamp

TFLite fused activation 會被轉成 quantized clamp：

| fused activation | clamp |
|---|---|
| NONE | dtype min / max |
| RELU | `[zp_out, max]` |
| RELU6 | `[zp_out, zp_out + 6 / scale_out]` |
| RELU_N1_TO_1 | `[zp_out - 1/scale, zp_out + 1/scale]` |

對 uint8 output，simulator 使用 signed int8 byte representation：

```text
sim_byte = tflite_uint8 - 128
```

因此 `zp_out`、`act_min`、`act_max` 都會 shift by 128。

---

## 7.9 INT16x16 與 INT16x8

INT16 path 分兩種：

| dtype | activation | weight | output |
|---|---|---|---|
| `DT_INT16x16` | int16 | int16 | int16 |
| `DT_INT16x8` | int16 | int8 | int16 |

compiler 對 INT16x8 的重點：

- input synth 用 int16。
- weight storage 用 1 byte。
- Requant output 仍是 int16。
- UDMA length / ref_size 要用 2 bytes per output element。

SystemC 對應：

```cpp
compute_int<int16_t, int8_t>()   // INT16x8
compute_int<int16_t, int16_t>()  // INT16x16
```

常見錯誤是只改 CONV template，忘了 output store length。那會造成 verification 讀到錯 byte count。

---

## 7.10 FP path：FP16 storage + FP32 compute

FP model policy：

```text
storage: FP16 in DRAM/L1
compute: FP32 accumulator / arithmetic
output: FP16 storage
```

即使 TFLite source 是 FP32，compiler 也 lower 到 FP16 deployment storage：

```python
TYPE_TO_DTYPE[FLOAT32] = DT_FP16
```

CONV FP reference：

```text
input FP16 -> FP32
weight FP16 -> FP32
sum in FP32
bias in FP32
clamp
cast output to FP16
```

params blob：

```text
[f32 act_min | f32 act_max | f32 bias[OC]]
```

SystemC CONV/Requant 使用相同 layout。

---

## 7.11 FP tolerance

INT regression 要求 bit-exact。FP regression 通常使用 tolerance，例如 1e-3。

原因：

- FP32 accumulation order 可能不同。
- `exp` / `tanh` library implementation 可能微差。
- FP16 cast 時機很敏感。
- numpy vectorized path 和 C++ nested loop order不同。

但 simulator 仍盡量讓 reference 和 SystemC reduction order match。例如 FP softmax 註解要求 sequential running sum，以減少 ULP 差異。

---

## 7.12 ADD / MUL / SUB params

INT ADD params 是 12 個 int32：

```text
zp_a, zp_b, zp_out,
mult_a, shift_a,
mult_b, shift_b,
mult_out, shift_out,
left_shift,
act_min, act_max
```

MUL / SUB 重用相近 layout。

FP binary params 則簡化為：

```text
[f32 act_min | f32 act_max]
```

這意味著 EWE Engine 看到同一個 `EweBody`，會依 subtype 和 dtype 解不同 math。

---

## 7.13 Pool / Softmax reference

POOL reference：

- INT avgpool 使用指定 rounding rule。
- FP avgpool 使用 FP32 running sum。
- maxpool 依 dtype 選 int8 或 FP16/FP32 path。

SOFTMAX reference：

- INT 使用 LUT / fixed approximation。
- FP 使用 stable 3-pass：max、exp/sum、divide。

axis-sensitive op 的 reference 要特別小心。若 compiler 把高 rank tensor flatten 成 HWC，softmax axis 是否仍正確是重要限制。

---

## 7.14 常見誤解

| 誤解 | 正確理解 |
|---|---|
| CONV Engine 做完整 TFLite quant math | CONV 做 raw sum，quant correction 在 compiler/Requant |
| padding 用 0 就好 | quantized zero 通常是 `zp_in` |
| uint8 可以直接 view 成 int8 | 目前使用 centered shift，還要修正 zero-point |
| INT16x8 weight 也是 2 bytes | INT16x8 weight 是 int8，只有 activation/output 是 int16 |
| FP32 model 就用 FP32 storage | simulator policy 是 FP16 storage + FP32 compute |
| FP 不需要 reference tolerance | FP path 需要合理 tolerance，尤其 nonlinear op |
| activation clamp 在 engine 裡推導 | clamp range 在 compiler params 裡已經算好 |

---

## 7.15 本章小結

Quantization lowering 是 compiler 和 engine 的合約：

```text
compiler: 讀 TFLite quant metadata，產生 params blob + reference
engine:   消費 params blob，做同樣的 fixed-point / FP math
verify:   比對 SystemC output 與 embedded reference
```

Debug quant mismatch 時，請先確認：

1. dtype 和 byte width。
2. zero-point representation。
3. params blob layout。
4. activation clamp。
5. correction map。
6. reference 和 SystemC reduction / rounding 是否一致。

> 下一章 → [第 8 章 — program.bin 格式與 Reference Generation](08_program_bin_reference.md)


\newpage

# 第 8 章 — program.bin 格式與 Reference Generation

> 上一章：[第 7 章 — Quantization / FP / INT16 Compile Path](07_quantization_fp_int16.md)

本章你會學到什麼：

- `program.bin` 的 binary layout。
- `ProgHeader`、`LayerMeta`、`GraphMeta` 各自負責什麼。
- input / weight / reference payload 如何排在 data section。
- compiler 如何安排 DRAM region。
- C++ `test_model` 如何讀 program、populate DRAM、verification。
- 為什麼 reference 是 simulator regression 的核心。

---

## 8.1 program.bin 是什麼

`program.bin` 是 Python compiler 和 C++ SystemC runner 的 binary contract。

它包含：

```text
Header
LayerMeta table
GraphMeta table
Data section
  inputs
  weights / params
  references
```

它不是真實 silicon firmware 格式，而是 simulator-friendly 的整合檔：

| 內容 | 用途 |
|---|---|
| layer metadata | 讓 C++ 產生 descriptor |
| input blobs | preload 到 simulated DRAM |
| weight / params blobs | preload 到 simulated DRAM |
| reference output | SystemC 跑完後驗證 |
| graph sidecar | lifetime / fusion / debug |

---

## 8.2 Header

Python：

```python
HEADER_FMT = "<IIII"
MAGIC = 0x374C444D
VERSION = 3
```

C++：

```cpp
struct ProgHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_layers;
    uint32_t data_offset;
};
```

欄位：

| 欄位 | 意義 |
|---|---|
| `magic` | `'MDL7'` |
| `version` | 目前支援 v2 / v3 |
| `num_layers` | compiled layer count |
| `data_offset` | data section 起始 file offset |

C++ 檢查：

```cpp
if (magic != 0x374C444D || version not in {2,3})
    bad magic/version
```

---

## 8.3 LayerMeta

`LayerMeta` 固定 64 bytes。Python `LAYER_FMT` 和 C++ struct 必須一致。

主要欄位：

| 欄位 | 說明 |
|---|---|
| `in_h,in_w,in_c` | input HWC |
| `out_h,out_w,out_c` | output HWC |
| `k_h,k_w` | kernel / op hint |
| `s_h,s_w` | stride |
| `p_t,p_b,p_l,p_r` | padding |
| `dram_in,dram_wgt,dram_out` | simulated DRAM absolute address |
| `in_size,wgt_size,ref_size` | payload bytes |
| `in_off,wgt_off,ref_off` | file offset |
| `group` | group / depthwise |
| `op_kind` | compiler op enum |
| `dtype` | MDLA7 dtype enum |
| `zp_in_eff` | CONV padding zero-point |

這個 struct 是 C++ descriptor generation 的主資料來源。

---

## 8.4 GraphMeta

`GraphMeta` 固定 32 bytes：

```cpp
struct GraphMeta {
    int32_t input0_tensor, input1_tensor, output_tensor;
    int32_t producer0_layer, producer1_layer;
    int32_t first_consumer_layer, last_consumer_layer;
    int32_t consumer_count;
};
```

用途：

| 欄位 | 用途 |
|---|---|
| input / output tensor ID | 對回 TFLite graph |
| producer layer | L1-resident handoff 判斷 |
| consumer range | last-use / store suppression |
| consumer count | branch / concat / multi-consumer 風險 |

GraphMeta 不直接影響 engine math，但對 scheduler 很重要。

---

## 8.5 Data section layout

compiler 最後組：

```python
inputs_section  = b"".join(in_blobs)
weights_section = b"".join(wgt_blobs)
refs_section    = b"".join(ref_blobs)
```

file layout：

```text
data_offset
  + inputs_section
  + weights_section
  + refs_section
```

LayerMeta 裡的 `in_off`、`wgt_off`、`ref_off` 是 absolute file offset：

```python
data_offset + L["in_off"]
data_offset + base_w + L["wgt_off"]
data_offset + base_r + L["ref_off"]
```

C++ 可直接：

```cpp
file.data() + L.in_off
```

---

## 8.6 DRAM region allocator

compiler 把 weights、inputs、outputs 放在三個 DRAM region：

```text
DRAM_WGT
DRAM_IN
DRAM_OUT
```

早期曾用固定 +64MB offset，但大 model 會 region overlap。現在做法是：

1. 先用 placeholder address 累計各 region size。
2. layer loop 結束後，知道 `cur_w` / `cur_i` / `cur_o`。
3. 以 64KB alignment 重新配置 region base。
4. patch 每層 `dram_*` address。

目的：

```text
不同 region 不重疊，且支援大 segmentation / transformer 模型。
```

---

## 8.7 Program budget guard

LayerMeta 使用 32-bit file offset 和 32-bit address，因此 compiler 需要 guard：

| guard | 原因 |
|---|---|
| program file > 4GB | `uint32_t` offset 不夠 |
| DRAM end > 0xFFFFFFFF | descriptor address 不夠 |
| dims > 65535 | LayerMeta shape 用 `uint16_t` |
| pool kernel > 255 | `k_h/k_w` 是 byte，global 用 sentinel 255 |

遇到超界時，compiler 會 skip 或 stop compile，避免 C++ 端 struct pack / address overflow。

---

## 8.8 Reference generation

每個 layer 都會產生 reference bytes：

| op | reference |
|---|---|
| CONV / DWCONV | numpy conv + requant or FP conv |
| FC | 1x1 conv style reference |
| ADD / MUL / SUB | numpy element-wise |
| POOL | numpy pool |
| SOFTMAX | numpy softmax |
| RESHAPE / CONCAT / GATHER / D2S | numpy layout transform |

reference 被寫進 `refs_section`。SystemC 跑完後，C++ 讀 simulated DRAM output 和 reference 比對。

這是 MDLA7 regression 的核心：

```text
compiler reference == expected
SystemC output == actual
PASS if expected and actual match
```

---

## 8.9 C++ 如何載入 program

[`test_model.cpp`](../systemc/src/test_model.cpp) 做：

```cpp
read whole file into vector<uint8_t>
reinterpret header
reinterpret LayerMeta table
reinterpret GraphMeta table
```

再把 input / weight preload 到 simulated DRAM：

```cpp
sys.dram.write(L.dram_in,  file.data() + L.in_off,  L.in_size);
sys.dram.write(L.dram_wgt, file.data() + L.wgt_off, L.wgt_size);
```

注意 reference 不寫入 DRAM。reference 留在 file buffer，用於最後 compare。

---

## 8.10 DRAM sizing in test_model

C++ 會掃所有 layer 的 address / size，算需要多大的 DRAM model：

```cpp
max_addr = max(L.dram_in + L.in_size,
               L.dram_wgt + L.wgt_size,
               L.dram_out + L.ref_size)
```

再 round up 到 64MB。

原因：

```text
default 256MB 對大 model 不夠，
直接 sys.dram.write out-of-bounds 會 segfault。
```

這是 simulator robustness 的重要修正。

---

## 8.11 Verification

SystemC 跑完後，C++ 對每層：

```text
read L.dram_out from simulated DRAM
compare with file.data() + L.ref_off
```

INT path 要 bit-exact。FP path 通常用 tolerance。

console canonical line：

```text
layer NN  op  in=... out=... PASS
layer NN  op  in=... out=... FAIL X/Y
```

summary：

```text
summary: P/N layers PASS, F FAIL
sim time: C cycles @ 1.9 GHz (= M ms)
```

Regression scripts 會 parse 這些 line。

---

## 8.12 Profile output

`test_model.cpp` 也輸出：

| file | 用途 |
|---|---|
| `.profile.json` | summary、layers、engine timelines |
| `.profile.csv` | table-friendly layer profile |
| `.profile.png` | matplotlib Gantt |
| `.html` | interactive report，由 batch/run_model.py 生成 |

profile 裡會包含：

- layer cycles
- cumulative cycles
- pass / fail
- tiles
- DRAM / SRAM bytes
- engine busy time
- task timeline

這使 `program.bin` 不只是 functional test input，也是一個 performance experiment input。

---

## 8.13 常見誤解

| 誤解 | 正確理解 |
|---|---|
| program.bin 是硬體正式 ISA | 它是 simulator compiler-to-test_model contract |
| reference 存在 DRAM | reference 存在 file buffer，DRAM 存 actual output |
| LayerMeta 和 Descriptor 一樣 | LayerMeta 是高階 layer schema，C++ 再展開成 descriptors |
| GraphMeta 是 optional debug only | 它也支援 L1 lifetime / fusion decision |
| DRAM address 可用固定 offset | 大 model 需要 region sizing，不能固定 64MB |
| `ref_size` 一定等於 elements | 它是 bytes，dtype 會影響 |

---

## 8.14 本章小結

`program.bin` 把 Python compiler 和 SystemC simulator 接起來：

```text
LayerMeta tells C++ what to run
GraphMeta tells C++ graph lifetime
Data section carries inputs, weights, references
```

你要記住：

1. LayerMeta 是 descriptor generation 的來源。
2. GraphMeta 是 fusion / handoff 的來源。
3. Reference bytes 是 PASS / FAIL 的依據。
4. 所有 offset / address / size 都是 binary interface，Python 和 C++ 必須同步。

> 下一章 → [第 9 章 — Mdla7System Top Wiring](09_system_top_wiring.md)


\newpage

# 第 9 章 — Mdla7System Top Wiring

> 上一章：[第 8 章 — program.bin 格式與 Reference Generation](08_program_bin_reference.md)

本章你會學到什麼：

- `Mdla7System` 如何把 Host、Command Engine、engines、memory 接起來。
- SystemC `sc_fifo` 在這個 simulator 裡扮演什麼角色。
- CONV → Requant chain 如何 wiring。
- done FIFO 和 dtype latch 如何回到 Command Engine。
- top module 和硬體 block diagram 如何對應。

---

## 9.1 Top module 入口

Top module 在 [`system.h`](../systemc/include/mdla7/system.h)：

```cpp
class Mdla7System : public sc_core::sc_module {
    ...
};
```

它的責任不是做 compute，而是 instantiate 和 bind：

```text
Host
CommandEngine
Udma
ConvEngine
RequantEngine
EweEngine
PoolEngine
L1Mesh / Dram / L1Manager
FIFOs
```

這對應 HW spec 的 top-level wiring。

---

## 9.2 Descriptor stream

Host 到 Command Engine：

```cpp
sc_core::sc_fifo<Descriptor> desc_stream{"desc_stream", 256};

host.desc_out(desc_stream);
cmd.desc_in(desc_stream);
```

`desc_stream` 可以看成 command ring buffer 的 simulator 版本。

| FIFO | depth | 說明 |
|---|---:|---|
| `desc_stream` | 256 | Host 上傳 descriptor 給 Command Engine |

Host 每寫一筆 descriptor，Command Engine 就有機會 decode / schedule。

---

## 9.3 Per-engine config FIFO

Command Engine 到 engines：

```cpp
sc_fifo<DescriptorBody> conv_cfg;
sc_fifo<DescriptorBody> requant_cfg;
sc_fifo<DescriptorBody> ewe_cfg;
sc_fifo<DescriptorBody> pool_cfg;
sc_fifo<DescriptorBody> udma_cfg;
```

binding：

```cpp
cmd.conv_cfg_out(conv_cfg);       conv.cfg_in(conv_cfg);
cmd.requant_cfg_out(requant_cfg); requant.cfg_in(requant_cfg);
cmd.ewe_cfg_out(ewe_cfg);         ewe.cfg_in(ewe_cfg);
cmd.pool_cfg_out(pool_cfg);       pool.cfg_in(pool_cfg);
cmd.udma_cfg_out(udma_cfg);       udma.cfg_in(udma_cfg);
```

每個 engine 只看到自己的 `DescriptorBody`。header 的 `dtype`、`layer_id` 等不直接進 FIFO，dtype 透過 latch 處理。

---

## 9.4 Done FIFO

engines 到 Command Engine：

```cpp
sc_fifo<uint8_t> conv_done;
sc_fifo<uint8_t> requant_done;
sc_fifo<uint8_t> ewe_done;
sc_fifo<uint8_t> pool_done;
sc_fifo<uint8_t> udma_done;
```

binding：

```cpp
conv.done_tag_out(conv_done);       cmd.conv_done(conv_done);
requant.done_tag_out(requant_done); cmd.requant_done(requant_done);
...
```

payload 目前固定 0。Command Engine 自己根據 per-engine pending queue 找出對應 signal tag。

這是簡化設計。若未來 engine 內部 out-of-order completion，done payload 必須帶 task ID。

---

## 9.5 CONV → Requant chain

CONV 和 Requant 之間不是透過 L1 output，而是 128 條 int32 FIFO，也就是 4096 bit/cyc：

```cpp
std::array<std::unique_ptr<sc_fifo<int32_t>>, 128> chain;
```

constructor：

```cpp
for (int i = 0; i < 128; ++i) {
    chain[i] = make_unique<sc_fifo<int32_t>>(..., 2);
    conv.chain_out[i] = chain[i].get();
    requant.chain_in[i] = chain[i].get();
}
```

這代表：

```text
CONV pushes psum to chain[oc % 128]
Requant drains chain[oc % 128]
```

chain depth 是 2，足夠讓 functional dataflow block / unblock，並保留 backpressure 味道。

---

## 9.6 Memory subsystem wiring

Top module instantiate：

```cpp
L1Mesh    l1mesh;
Dram      dram;
L1Manager l1mgr;
```

constructor：

```cpp
l1mesh("l1mesh"),
dram("dram", dram_bytes),
l1mgr("l1mgr", l1mesh, dram),
udma("udma", l1mgr),
conv("conv", l1mgr),
requant("requant", l1mgr),
ewe("ewe", l1mgr),
pool("pool", l1mgr)
```

所有 engine 共用同一個 `L1Manager&`：

```text
engine -> L1Manager -> L1Mesh or Dram
```

這讓 latency model 集中在 memory module 裡。

---

## 9.7 dtype latch wiring

Command Engine 有 pointer：

```cpp
uint8_t* conv_dtype_latch;
uint8_t* req_dtype_latch;
uint8_t* ewe_dtype_latch;
uint8_t* pool_dtype_latch;
```

Top module bind：

```cpp
cmd.conv_dtype_latch = &conv.last_dtype;
cmd.req_dtype_latch  = &requant.last_dtype;
cmd.ewe_dtype_latch  = &ewe.last_dtype;
cmd.pool_dtype_latch = &pool.last_dtype;
```

因此 Command Engine issue descriptor 時會先寫 engine 的 `last_dtype`，再寫 body FIFO。

這是一個重要的 simulator shortcut。若未來要更接近真實 command packet，可以把 header 也傳進 engine FIFO。

---

## 9.8 dram_bytes parameter

`Mdla7System` constructor：

```cpp
Mdla7System(sc_module_name nm,
            std::size_t dram_bytes = 256 * 1024 * 1024)
```

`test_model.cpp` 會根據 program 需要的最大 address 動態 sizing DRAM，避免大模型 out-of-bounds。

這代表 top module 可用在兩種情境：

| 情境 | dram size |
|---|---|
| small unit test | default 256 MB |
| full model test | test_model 傳入 computed size |

---

## 9.9 SystemC thread map

每個 module 通常有自己的 `SC_THREAD`：

| Module | thread |
|---|---|
| Host | `run()` |
| CommandEngine | `dispatch()`、`collect()` |
| UDMA | `run()` |
| CONV | `run()` |
| Requant | `run()` |
| EWE | `run()` |
| POOL | `run()` |

這使得只要 dependency tag 允許，engines 可以在 simulation time 上 overlap。

---

## 9.10 常見誤解

| 誤解 | 正確理解 |
|---|---|
| Top module 有 scheduling logic | scheduling 在 Command Engine / test_model descriptor generation |
| CONV output 先寫 L1 再給 Requant | CONV 到 Requant 是 chain FIFO |
| done FIFO payload 就是 tag | payload 目前固定 0，tag 在 Command Engine pending queue |
| cfg FIFO 傳完整 descriptor | 目前傳 body，dtype 另用 latch |
| L1Manager 是 engine | 它是 memory router / latency entry |
| SystemC FIFO depth 不重要 | depth 會影響 blocking / backpressure 行為 |

---

## 9.11 本章小結

`Mdla7System` 是架構圖的具體 wiring：

```text
Host -> desc_stream -> CommandEngine
CommandEngine -> cfg FIFOs -> engines
engines -> done FIFOs -> CommandEngine
CONV -> chain -> Requant
engines -> L1Manager -> L1Mesh / DRAM
```

你讀任何 engine 或 scheduler 問題時，都可以回到本章的 wiring 圖確認資料和 event 走哪條路。

> 下一章 → [第 10 章 — Host 與 Command Engine Module Design](10_host_command_engine.md)


\newpage

# 第 10 章 — Host 與 Command Engine Module Design

> 上一章：[第 9 章 — Mdla7System Top Wiring](09_system_top_wiring.md)

本章你會學到什麼：

- Host module 在 simulator 裡如何上傳 descriptor。
- Command Engine 的 dispatch / collect 兩條 thread。
- pending lookahead queue 如何工作。
- normal descriptor 與 stream descriptor 的 scheduling 差異。
- dependency tag table、pending_tags、tag_changed event 如何互動。
- junior 如何 debug Command Engine 卡住或越序問題。

---

## 10.1 Host module

Host 在 [`host.h`](../systemc/include/mdla7/host.h)：

```cpp
SC_MODULE(Host) {
    sc_fifo_out<Descriptor> desc_out;
    std::vector<Descriptor> program;
    void run();
};
```

`test_model.cpp` 在 `sc_start()` 前填：

```cpp
sys.host.program = std::move(program);
```

Host thread：

```cpp
for (auto& d : program) {
    desc_out.write(d);
    wait(1, SC_NS);
}
```

因此 Host 是簡化的 RISC-V runtime stub：

| 真實系統 | simulator |
|---|---|
| firmware writes command ring in DRAM | Host writes `desc_stream` FIFO |
| AXI-Lite register map + MMIO doorbell | FIFO data_written_event |
| command buffer in DRAM | `std::vector<Descriptor> program` |
| Command/Host interface DRAM AXI master fetches descriptors | descriptors are preloaded into simulator memory vector |

硬體 spec v1 將 Command / Host interface 定義成兩個實體介面：

| Interface | Role |
|---|---|
| AXI-Lite register map | Host CPU 設定 command ring base/size、doorbell、interrupt/status、profile/debug register |
| DRAM AXI master | Command Engine / Host interface 直接從 DRAM fetch command buffer / descriptor，並回寫 completion/status buffer |

控制流程是：Host 在 DRAM 建 command ring，透過 AXI-Lite 寫 register 並敲 doorbell；Command Engine 用 DRAM AXI master 拉 descriptor，dispatch 給 engines，完成後回寫 status 並更新 interrupt/status。

---

## 10.2 Command Engine 兩條 thread

Command Engine 有兩條 `SC_THREAD`：

| thread | 責任 |
|---|---|
| `dispatch()` | 讀 descriptor、檢查 wait_tags、issue 到 engine cfg FIFO |
| `collect()` | 等 engine done、set signal tag、notify scheduler |

這個 split 很自然：

```text
dispatch side = forward path
collect side  = completion path
```

Completion path 會喚醒 dispatch path，讓等待 dependency 的 descriptor 變 ready。

---

## 10.3 tag_done table

Command Engine 內部：

```cpp
bool tag_done[256];
```

初始全部 true：

```cpp
for (int i = 0; i < 256; ++i) tag_done[i] = true;
```

descriptor issue 或進入 stream pending 時：

```text
signal_tag reserved -> tag_done[tag] = false
```

engine done 時：

```text
tag_done[tag] = true
tag_changed.notify()
```

`waits_ready()` 是 AND-check：

```cpp
for wait tag:
    if not done: return false
return true
```

---

## 10.4 pending_tags per engine

Engine done payload 沒帶 tag，所以 Command Engine 需要 queue：

```cpp
std::queue<uint8_t> pending_tags[OC_NUM];
```

issue 時：

```cpp
pending_tags[op_class].push(signal_tag);
```

collect 時：

```cpp
t = pending_tags[cls].front();
pending_tags[cls].pop();
tag_done[t] = true;
```

這假設同一 engine 完成順序等於 issue 順序。現在每個 engine 都是 single-thread in-order，所以成立。

---

## 10.5 dispatch pending queue

`dispatch()` 裡有：

```cpp
std::deque<Descriptor> pending;
constexpr size_t LOOKAHEAD_LIMIT = 64;
```

流程：

```text
fill pending from desc_in until limit
find best ready descriptor
issue it
if none ready, wait new descriptor or tag_changed
```

LOOKAHEAD_LIMIT 的目的：

- 允許 stream descriptor bypass。
- 不讓 8-bit tag wrap hazard 變大。
- 控制 scheduler search cost。

---

## 10.6 normal descriptor scheduling

Normal descriptor 沒有 `DF_STREAM`。

規則：

```text
只看 pending front。
front ready -> issue。
front not ready -> stop，不看後面 normal work。
```

原因是 normal schedule 可能重用固定 L1 region。保守 in-order 可以避免 data overwrite。

這是 correctness-first design。

---

## 10.7 stream descriptor scheduling

Stream descriptor 有 `DF_STREAM`，可以 bypass。

stream metadata：

| 欄位 | 用途 |
|---|---|
| `layer_id` | profile / debug |
| `stream_slot` | ping-pong slot |
| `microblock_id` | wavefront order |
| `stream_meta_flags` | load / compute / store / final tile |

priority：

| 類型 | priority |
|---|---:|
| tail | 0 |
| EWE | 10 |
| UDMA read | 20 |
| CONV | 30 |
| Requant / POOL | 40 |
| UDMA write | 60 |

同類再用 `microblock_id` tie-break。

---

## 10.8 tail waiting

stream tail 還沒 ready 時，Command Engine 允許部分 younger work 先走：

| allowed | 原因 |
|---|---|
| UDMA read | 預取下一個 tile |
| CONV / Requant compute | 保持淺層 compute front |

但不允許任意 store 越過 tail，避免 L1 lifetime 被破壞。

---

## 10.9 last_activity 與 tag_fire_time

Command Engine 記錄：

```cpp
sc_time last_activity;
sc_time tag_fire_time[256];
```

用途：

| 欄位 | 用途 |
|---|---|
| `last_activity` | test harness 報告真正完成時間 |
| `tag_fire_time[tag]` | per-layer cycle reporting |

`sc_start()` 可能跑到預算上限，但真正最後一個 tag 早就完成。`last_activity` 可避免 sim time report 被 budget 污染。

---

## 10.10 Debug：Command Engine 卡住

卡住通常是某個 wait tag 永遠不 done。

檢查順序：

1. 找最後一個 dispatch log。
2. 找 pending descriptor 的 wait tags。
3. 查該 tag 是否有 upstream signal descriptor。
4. 查 upstream descriptor 是否真的 issue 到 engine。
5. 查 engine 是否完成並寫 done FIFO。
6. 查 `pending_tags[op_class]` 是否 pairing 正確。

常見原因：

| 原因 | 現象 |
|---|---|
| wait tag 填錯 | descriptor 永遠不 ready |
| signal_tag = 0 | 完成但不 notify dependency |
| engine path 沒寫 done | collect 收不到 |
| tag wrap live range 太近 | old/new tag state 混淆 |
| stream reservation 錯 | younger descriptor 提早或永遠等 |

---

## 10.11 Debug：越序造成 wrong output

越序 bug 常見於 stream scheduling：

```text
load new tile overwrites buffer
old store / compute 還沒讀完
```

要看：

| 檢查 | 說明 |
|---|---|
| descriptor 是否 `DF_STREAM` | 只有 stream 可 bypass |
| `stream_slot` | ping-pong buffer 是否對 |
| `SMF_STORE` | store 是否被排太後面 |
| tail barrier | 是否缺少 final ordering |
| `allowed_during_tail_wait` | younger work 是否被允許 |
| L1 address overlap | 是否覆蓋同一區 |

這類 bug 不一定會卡住，通常表現為多 tile model mismatch。

---

## 10.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| Host 會執行真正 firmware | 目前 Host 是 descriptor uploader |
| Command Engine 做 compute | 它只 dispatch / dependency tracking |
| tag 由 engine 決定 | tag 由 Command Engine pending queue 對應 |
| normal descriptor 也可 bypass | normal descriptor 保持 in-order |
| lookahead 越大越好 | 太大會增加 tag wrap / lifetime risk |
| tail barrier 只是 performance hint | tail 也保護 ordering |

---

## 10.13 本章小結

Host 和 Command Engine 是 MDLA7 的 control plane：

```text
Host uploads descriptors
CommandEngine checks dependency tags
CommandEngine dispatches body to engines
Engines finish
CommandEngine sets signal tags
```

記住一句話：

```text
Command Engine 不知道 tensor math，但它決定 tensor math 何時能安全開始。
```

> 下一章 → [第 11 章 — CONV / Requant Data Path](11_conv_requant_datapath.md)


\newpage

# 第 11 章 — CONV / Requant Data Path

> 上一章：[第 10 章 — Host 與 Command Engine Module Design](10_host_command_engine.md)

本章你會學到什麼：

- CONV descriptor 如何變成 L1 read、MAC、chain write。
- Requant descriptor 如何把 chain partial sum 轉成 final tensor。
- INT8、INT16x8、INT16x16、FP16 path 的差異。
- OC tiling、OH tiling、correction map 如何接在 Requant。
- chain ordering 為什麼必須保持 NHWC。
- debug conv mismatch 時如何分層定位。

---

## 11.1 Data path 總覽

CONV layer 在 SystemC 裡不是單一 engine 完成全部事情，而是一條 pipeline：

```text
UDMA load input / weight / params
        |
        v
CONV Engine
  read input + weight from L1
  accumulate raw psum
  push psum into chain[oc % 128]
        |
        v
Requant Engine
  drain chain
  read params / correction
  apply bias / scale / clamp
  write output tensor to L1
        |
        v
UDMA store or next layer L1 handoff
```

這章比第 5 章更深入，專看 CONV/Requant 這條主 datapath。

---

## 11.2 Descriptor emission 對 CONV/Requant 的影響

C++ [`test_model.cpp`](../systemc/src/test_model.cpp) 會依 LayerMeta 產生：

```text
UDMA params load
UDMA input tile load
UDMA weight tile load
CONV descriptor
Requant descriptor
UDMA store output tile
```

descriptor 的 wait tags 大致是：

```text
CONV waits input_tag and weight_tag
Requant waits conv_tag
Store waits requant_tag
```

如果 Requant 沒 wait CONV，會讀不到足夠 psum 或讀到前一層殘留 chain data。這種錯通常很嚴重。

---

## 11.3 CONV body 填入欄位

CONV descriptor 的重要欄位：

| 欄位 | 來源 |
|---|---|
| `in_addr` | L1 input tile address |
| `wgt_addr` | L1 weight tile address |
| `in_h,in_w,in_c` | tile input shape |
| `out_c` | tile OC count |
| `k_h,k_w` | LayerMeta kernel |
| `stride_dilation` | encoded stride |
| `pad_tb,pad_lr` | tile-aware padding |
| `group` | normal conv = 1，depthwise = Cin |
| `in_pad_value` | `zp_in_eff` |
| `scale_lut_addr` | params address hint |
| `scale_count` | full layer OC count |

OH tiling 時，tile 的 `in_h` 和 padding 可能不是原 layer shape。這是 halo handling 的重點。

---

## 11.4 Input tile 與 halo

對 convolution，output tile 需要 input halo：

```text
output rows [oh0, oh1)
需要 input rows covering kernel window
```

因此 `tile_in_h` 可能大於 output tile rows。compiler / test_model 必須把正確 input slice 搬到 L1。

常見錯誤：

| 現象 | 原因 |
|---|---|
| tile 邊界 row 錯 | halo 少搬或 pad 計算錯 |
| 第一個 tile 正確，後續錯 | dram input offset 錯 |
| output height seam mismatch | oh_start / tile pad 沒處理好 |

Debug 時先畫：

```text
global output oh
global input ih range
local tile input row
local pad_t / pad_b
```

---

## 11.5 Weight tiling 與 OC tiling

OC tiling 把 output channels 分批處理：

```text
OC layer = 1024
tile OC = 256
tiles_oc = 4
```

每個 OC tile 只 load 該 slice 的 weight：

```text
weight offset = oc_start * Kh * Kw * in_per_group * weight_elem_size
```

Requant 也要用同樣 `oc_start` 讀 params：

```text
mult[oc_start : oc_start + tile_oc]
shift[...]
bias_eff[...]
```

如果 OC tiling 錯，常見現象是某些 channel range 錯，其他 channel 正確。

---

## 11.6 CONV inner loop

INT path核心：

```text
for oh
  for ow
    for oc
      sum = 0
      for kh
        for kw
          for icr
            sum += input * weight
      chain[oc % 128].write(saturate_i32(sum))
```

這裡 accumulator 是 int64，最後 saturate 到 int32 chain payload。

對 INT16x16，int64 很重要。若直接 int32 accumulate，大 kernel / high channel layer 容易 overflow。

---

## 11.7 Padding path

padding 判斷：

```cpp
if ih/iw inside input:
    a = input[index]
else:
    a = in_pad_value
```

對 quantized model：

```text
in_pad_value = zp_in_eff
```

對 FP model：

```text
in_pad_value holds FP16 bit pattern for +0.0 or pad value
```

padding mismatch 是 conv debug 的第一個高收益檢查點。

---

## 11.8 Chain payload

CONV push：

```cpp
lane = oc % 128
chain_out[lane]->write(psum)
```

FP path 也用 int32 FIFO，只是 payload 是 FP32 bit pattern：

```cpp
float sum
bitcast sum -> int32 bits
chain.write(bits)
```

所以 Requant 必須知道 dtype，才能把 int32 payload 解成 int32 psum 或 FP32 bits。

---

## 11.9 Requant body 填入欄位

Requant descriptor 的重要欄位：

| 欄位 | 說明 |
|---|---|
| `out_addr` | L1 output tile address |
| `h,w,c` | this tile output shape |
| `scale_lut_addr` | params in L1 |
| `scale_count` | full layer OC |
| `oc_start` | OC tile offset |
| `out_w_layer` | full layer OW |
| `oh_start` | height tile global offset |
| `corr_addr` | correction map address |
| `corr_per_oc` | correction map layout |
| `reserved/store_mode` | `RQ_STORE_LINEAR` 或 `RQ_STORE_D2SPACE` final-store swizzle |

OH/OC tiling 同時存在時，Requant 需要知道 global offsets 才能讀正確 correction map slice。

---

## 11.10 Requant INT formula

Requant INT path：

```text
psum = chain read
with_bias = psum + bias_eff[oc]
if correction:
    with_bias += corr[oh,ow] or corr[oh,ow,oc]
scaled = MBQM(with_bias, mult[oc], shift[oc])
v = scaled + zp_out
v = clamp(v, act_min, act_max)
write int8 or int16
```

params layout 來自第 7 章。

---

## 11.11 Correction map tiling

非 depthwise：

```text
corr shape = [OH_layer, OW_layer]
corr index = (oh_global * OW_layer + ow)
```

depthwise：

```text
corr shape = [OH_layer, OW_layer, OC_layer]
corr index = (oh_global * OW_layer + ow) * OC_layer + oc_global
```

RequantBody 的 `oh_start`、`out_w_layer`、`oc_start` 就是為此存在。

若 correction map index 錯，通常只在 asymmetric uint8 weight model 出現 mismatch。

---

## 11.12 FP Requant

FP Requant：

```text
bits -> float psum
v = psum + bias[oc]
v = clamp(v, act_min, act_max)
out = fp32_to_fp16(v)
```

它不做 MBQM，不使用 zero-point，也不使用 int activation clamp。

params layout：

```text
[f32 act_min | f32 act_max | f32 bias[OC_layer]]
```

FP output bytes 是 2 bytes per element。

---

## 11.13 Requant output write

Requant 最後寫 L1 output：

| dtype | output vector | bytes |
|---|---|---:|
| int8 | `std::vector<int8_t>` | `total` |
| int16 | `std::vector<int16_t>` | `total * 2` |
| fp16 | `std::vector<uint16_t>` | `total * 2` |

後續 UDMA store 的 length 必須等於這個 byte count。

例外是 `CONV/FC/DWCONV -> final DEPTH_TO_SPACE`。這時 Requant descriptor 會使用 `RQ_STORE_D2SPACE`：

```text
CONV chain order: [oh, ow, ic]
q       = ic / Cout
oc      = ic % Cout
bh      = q / block
bw      = q % block
out_h   = oh * block + bh
out_w   = ow * block + bw
DRAM idx = (out_h * (OW * block) + out_w) * Cout + oc
```

這條 path 的意思是 Requant drain CONV chain 後，直接把 quantized value scatter 到 final D2SPACE layout。它不先寫線性 L1 tile，也不啟動 TNPS，所以 profile 會看到：

```text
D2SPACE layer cycles = 0
TNPS busy            = 0
D2SPACE DRAM W       = final output bytes
```

只有 final-output D2SPACE 走這條。若 D2SPACE 後面還有 ADD / EWE / CONV consumer，仍由 TNPS tiled streaming path 產生 consumer 需要的 tile。

---

## 11.14 Cycle overlap

CONV：

```text
engine time ~= max(L1 read elapsed, conv_cycles)
```

Requant：

```text
engine time ~= max(L1 write elapsed, ceil(elements / 512))
```

這是對 CONV / EWE 共用 quantize-pack / clamp resource 的 pipeline overlap 近似。

若 profile 顯示 CONV 很慢，先比較：

```text
estimated compute cycles
input + weight L1 read time
```

---

## 11.15 Debug checklist

| 問題 | 檢查 |
|---|---|
| output shape 錯 | ConvBody shape / stride / pad |
| edge wrong | `in_pad_value` |
| channel range wrong | OC tile offset / params offset |
| depthwise wrong | `group` / weight transpose |
| asymmetric uint8 wrong | correction map |
| simulation 卡在 Requant | CONV psum count 不足 |
| INT16 wrong | dtype + byte width |
| FP wrong | bitcast chain + FP16 conversion |

---

## 11.16 本章小結

CONV/Requant data path 要一起讀：

```text
CONV raw accumulation
  -> chain[oc % 128]
  -> Requant params / correction
  -> final tensor
```

Functional mismatch 時，先確認：

1. CONV shape。
2. CONV psum count。
3. Requant total elements。
4. params offset。
5. output byte width。

> 下一章 → [第 12 章 — UDMA、DRAM Model、L1Manager Module Design](12_udma_dram_l1manager.md)


\newpage

# 第 12 章 — UDMA、DRAM Model、L1Manager Module Design

> 上一章：[第 11 章 — CONV / Requant Data Path](11_conv_requant_datapath.md)

本章你會學到什麼：

- 第 4 章 memory spec 如何落到 SystemC module design。
- CONV ACT/WGT read 為什麼 direct 接 L1Mesh。
- `L1Manager` 為什麼是 non-CONV engine / UDMA memory access 的入口。
- UDMA descriptor 如何變成 memory copy / transform。
- DRAM / L1 timing 在 module 裡如何 impose。
- profile 裡的 UDMA read/write lane 從哪裡來。

---

## 12.1 Module map

memory-related modules：

| Module | Source | 責任 |
|---|---|---|
| `L1Mesh` | [`memory.h`](../systemc/include/mdla7/memory.h) | SRAM storage + bank latency |
| `Dram` | [`memory.h`](../systemc/include/mdla7/memory.h) | DRAM storage + row/refresh latency |
| `L1Manager` | [`memory.h`](../systemc/include/mdla7/memory.h) | address decode + route |
| `Udma` | [`udma.h`](../systemc/include/mdla7/udma.h) | descriptor-driven data movement |

Engine 不直接知道 address 在哪裡：

```cpp
l1mgr.read(addr, dst, bytes);
l1mgr.write(addr, src, bytes);
```

---

## 12.2 L1Manager design

硬體 spec 上，CONV ACT_R / WGT_R Payload R 各有專線到 CONV Engine，
直接接 L1Mesh，不經過 L1Manager。
這條 direct path 讓 CONV read 取得最高服務優先權，目標是讓 CONV cluster
不被 EWE / POOL / UDMA background traffic 餓住。

`L1Manager` 是 non-CONV engine / UDMA 到 L1Mesh 的入口 arbiter。

L1Manager 入口 arbitration policy **v1 frozen = round-robin**：

| Rule | Meaning |
|---|---|
| participants | Requant / EWE / POOL / TNPS / UDMA non-CONV traffic |
| read / write | read side 與 write side 各自 round-robin |
| skip rule | empty / stalled master 直接 skip，grant 給下一個 eligible master |
| CONV bypass | CONV ACT_R / WGT_R 不參與 L1Manager round-robin，直接接 L1Mesh |

目前 SystemC implementation 還是簡化 router：

```cpp
if addr_in_l1mesh(addr):
    mesh.read/write
elif addr_in_dram(addr):
    dram.read/write
else:
    SC_REPORT_ERROR
```

也就是說，HW spec 已定義 CONV direct read path 與 L1Manager
non-CONV 入口 round-robin arbitration；SystemC 尚未做完整 port-accurate arbitration。
這是刻意簡化，因為現階段重點是：

- functional correctness
- first-order bandwidth / latency
- descriptor scheduling
- tiling effects

未來可在 L1Manager / L1Mesh service model 擴充 CONV direct path 與
non-CONV traffic 的 contention。

---

## 12.3 L1Mesh storage

L1Mesh 是 3 MB dual-4x4 banked SRAM NoC。兩個 mesh plane 共用同一組
16-bank SRAM backend：

| Item | Value |
|---|---:|
| banks | 16 |
| mesh planes | 2 x 4x4 |
| router input FIFO depth | 2 flits provisional |
| macro | 768 x 16B = 12 KB |
| macros per bank | 16 |
| bank capacity | 192 KB |
| total macros | 256 |
| total capacity | 3 MB |

NoC edge ports 採四邊分流，每個 mesh plane 都有：

| Edge | Ports | Primary traffic |
|---|---:|---|
| left | 4R + 4W | CONV ACT_R dedicated direct ingress |
| top | 4R + 4W | CONV WGT_R dedicated direct ingress |
| right | 4R + 4W | L1Manager_R |
| bottom | 4R + 4W | L1Manager_W |

單一 plane 是 16R + 16W；兩個 plane 合計是 32R + 32W NoC
injection/ejection capacity。每個 bank 的 SRAM backend service 仍是每
SRAM cycle 一個 16B read beat / 一個 16B write beat。

目前 SystemC functional storage 仍用：

```cpp
std::vector<uint8_t> mem;
```

read / write：

```cpp
memcpy(...)
impose_bank_latency(...)
```

對 simulator 來說，data copy 和 time wait 分開：

```text
functional state 更新
simulation time 推進
```

這讓 engine code 寫起來像普通 memory access，但仍有 timing。

---

## 12.4 L1 bank finish arrays

L1 有：

```cpp
sram_bank_finish_[16]
```

這代表每個 SRAM macro/bank port 是 **1R/W**。read/read、write/write、read/write
只要打到同一 bank 都會 serialize；不同 bank 可 overlap。

硬體 spec 的 per-bank read priority 是：

```text
1. CONV ACT_R
2. CONV WGT_R
3. L1Manager_R
```

write slot 由 L1Manager_W 使用。

---

## 12.5 Dram storage

Dram 也用 vector：

```cpp
std::vector<uint8_t> mem;
```

address 是 DRAM absolute address，所以存取時 subtract base：

```cpp
mem[addr - DRAM_BASE]
```

若 test_model 沒 sizing 夠大，這裡可能 out-of-bounds。v8.22 之後由 program scan 動態 sizing。

---

## 12.6 Dram timing

DRAM timing 包含：

| 項目 | 值 |
|---|---:|
| bandwidth | 48 B/cycle |
| row miss | 50 cycles |
| refresh period | 7800 cycles |
| refresh stall | 200 cycles |

每次 access：

```text
start = max(last_finish, now)
penalty = row miss ? 50 : 0
access = ceil(bytes / 48)
refresh maybe added
finish = start + penalty + access
```

目前 `last_finish_` 是 single queue，所以 DRAM access 大致 serialize。

---

## 12.7 UDMA run loop

UDMA：

```cpp
DescriptorBody body = cfg_in.read();
switch (body.udma.mode):
    do_linear
    do_strided
    do_gather
    do_concat
    do_slice
    do_depth_to_space     // legacy/debug fallback; D2SPACE main path is TNPS or Requant final-store
done_tag_out.write(0)
```

每個 UDMA descriptor 是 atomic task：做完才 signal done。

---

## 12.8 UDMA read / write profiling

UDMA 根據 `direction` 分 lane：

```cpp
if direction == 1:
    busy_time_write += ...
    tasks_write.push(...)
else:
    busy_time_read += ...
    tasks_read.push(...)
```

所以 profile 裡可以分辨：

```text
UDMA_R: DRAM -> L1 loads
UDMA_W: L1 -> DRAM stores
```

這對看 overlap 很重要。好的 scheduler 會讓 UDMA_R 提早餵 compute，UDMA_W 背景 drain。

---

## 12.9 UDMA startup cost

UDMA 每筆 descriptor 有：

```cpp
wait(16, SC_NS)
```

代表 decode / startup cost。

大量小 UDMA descriptor 會累積：

```text
1000 descriptors * 16 cycles = 16000 cycles
```

所以 tiling 太碎不只增加 CONV fill，也增加 UDMA startup。

---

## 12.10 Data transform correctness

UDMA 不只 copy，也保留 legacy layout transform mode。主路徑上，tensor layout 類 op 已移到 TNPS；`CONV -> final D2SPACE` 會併入 Requant final-store。

| Mode | 風險 |
|---|---|
| STRIDED_2D | stride 是 bytes，不是 elements |
| INDEXED_GATHER | index table dtype / location |
| SCATTER_CONCAT | concat axis layout assumption |
| STRIDED_SLICE | column offset 是 byte offset |
| DEPTH_TO_SPACE | legacy fallback；主路徑請先看 Requant final-store / TNPS 分工 |

layout op fail 時，先判斷它實際走哪個 engine：

| Pattern | 先看哪裡 |
|---|---|
| `CONV -> final D2SPACE` | Requant descriptor 的 store mode / final DRAM address |
| intermediate / standalone D2SPACE | TNPS `TM_DEPTH_TO_SPACE` descriptor |
| `TRANSPOSE/PACK/UNPACK/SPLIT -> compute` intermediate handoff | GraphMeta no-store classification / downstream synthetic input |
| legacy fallback | UDMA `UM_DEPTH_TO_SPACE` descriptor |

確認 engine 後，再看 compiler reference 是否同一 mapping。GraphMeta no-store
只代表中間 DRAM checkpoint 被 suppress，不代表 UDMA 或 TNPS 已經執行完整
layout transform tile kernel；functional correctness 由後續 consumer layer verify。

---

## 12.11 Memory and dependency

Memory module 不知道「這塊 L1 是否還有人要讀」。資料生命週期靠 descriptor tags：

```text
producer write done tag
consumer wait tag
reuse wait last consumer / store tag
```

所以 memory bug 常常要回到 scheduler 找，而不是改 `L1Mesh`。

---

## 12.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| L1Manager 會防止資料 hazard | 它只 route address，不管理 lifetime |
| UDMA direction 決定合法 address | 合法性由 address range 決定 |
| DRAM 有完整 multi-bank parallel model | 目前有 row/bank identity，但 request queue 是簡化 single finish |
| L1 memcpy 代表零時間 | memcpy 後仍 impose bank latency |
| UDMA transform 不需 reference | layout transform 最需要 reference 對齊 |

---

## 12.13 本章小結

memory module design 的主線：

```text
L1Manager routes
L1Mesh / Dram impose timing
UDMA consumes descriptors and performs data movement
Dependency tags protect memory lifetime
```

Debug memory 問題時，把它拆成：

1. address range
2. byte count
3. stride / layout
4. dependency
5. timing

> 下一章 → [第 13 章 — EWE / POOL / SOFTMAX / D2SPACE](13_ewe_pool_softmax_d2space.md)


\newpage

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


\newpage

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


\newpage

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

## 15.8.2 CONV-CONV microblock streaming

`try_stream_conv_chain()` 現在把 Conv chain 分成兩種安全等級：

- `CONV(1x1) -> CONV(1x1) -> ...` linear pointwise chain 可以用 generic
  microblock streaming，只要每個 intermediate 是 direct single-consumer
  boundary。
- `CONV(3x3 SAME) -> ... -> DEPTH_TO_SPACE -> EWE` 保留既有的 deep stream
  path，會為下游 spatial layer 擴張 halo rows。

Plain `CONV(3x3 SAME) -> CONV(3x3 SAME)` chain 仍先走保守 per-layer tiler。
U-Net 類 tiled pair 會立刻暴露 seam correctness 風險；要開這條路徑，需要
更明確的 line-buffer / halo ownership，而不是只靠 row-range expansion。

## 15.8.3 Binary EWE microblock chain

`try_stream_binary_ewe_chain()` 針對線性的 `ADD/MUL/SUB -> ADD/MUL/SUB`
chain 做真正的 L1 handoff。條件刻意保守：

- INT8 binary EWE。
- 每層 shape / dtype 相同。
- `GraphMeta` 確認 producer output 是下一層的 `input0`，避免 ADD/SUB/MUL
  量化參數因 input0/input1 對調而失真。
- 中間 producer 允許 `no-store`。
- chain 尾端若直接接 `SOFTMAX`，會退回原本的 per-layer EWE wavefront；
  這條 path 能把 attention matrix 以 contiguous L1 tensor 留給 softmax，
  避免 softmax 重新從 DRAM 讀整張 matrix。

每個 microblock 會先載入第一層 input-A tile，接著對同一個 tile 依序跑
多個 EWE stage。中間結果只在兩個 L1 data buffer ping-pong，不寫回 DRAM；
每層自己的 input-B tile 和 48B params 仍從 DRAM 載入。Profile 的 `flow`
欄位會把這些 stage 標成同一個 flow，例如 sd decoder slice 的
`L9 -> L10 -> L11`。

同時，Hotspot 裡 `RESHAPE/MATERIALIZE` 若只是中間 graph boundary 且大小不變，
會被當成 metadata/reference checkpoint 跳過 DRAM copy；下游 layer 的 synthetic
input 已由 compiler materialize，所以 functional correctness 仍由後續可驗證 layer
錨定。

同樣的 rule 也套到中間 `CONV/DWCONV/FC` 與 unary EWE (`GELU/HARD_SWISH`)
producer：GraphMeta 顯示後面還有 consumer 時，不再把這個 transient activation
寫回 DRAM；若現有 fused/tiled path 能保留 L1 layout，就走真 on-chip handoff，
否則 consumer 仍使用 compiler 已 materialize 的 synthetic input。

## 15.8.4 Path 7：CONV/Requant -> EWE -> POOL/TNPS consumer

`try_stream_conv_ewe()` 現在不只支援 binary/unary/D2S tail，也支援 POOL tail。
POOL 不再只限 `1x1 / stride1 / no-pad` safe subset；scheduler 會用 consumer
POOL 的 output row 切 microblock，再反推這個 POOL window 需要的 producer rows：

```text
producer rows = output rows * pool_stride + pool_kernel - 1
conv input rows = producer rows * conv_stride + conv_kernel - 1
```

因此一個 microblock 內可以是：

```text
UDMA load CONV input rows
CONV + Requant -> L1 data0
UDMA load EWE input-B rows
EWE ADD/MUL/SUB -> L1 data1
POOL real/global window -> L1 data0
optional store or forward
```

這條 path 的限制仍然保守：

- dtype / channel 必須一致。
- GraphMeta 若存在，producer 必須是 direct single consumer。
- POOL padding 必須能放進 descriptor 的 3-bit pad field。
- 任意 overlapping multi-consumer POOL live range 還不走這條。

## 15.8.5 Path 8：POOL -> EWE/TNPS consumer

新增 `try_stream_pool_consumer()`，讓大型 POOL producer 可以直接接後續 consumer，
不必先完整寫回 DRAM。支援的 tail：

| Tail | Microblock 行為 |
|---|---|
| `POOL -> ADD/MUL/SUB` | POOL output tile 當 EWE input-A，載入 input-B tile 後 compute |
| `POOL -> GELU/HARD_SWISH` | POOL output tile 直接進 unary EWE |
| `POOL -> D2SPACE` | POOL output tile 交給 TNPS D2S |

這條 path 仍只開 INT8 safe subset，並要求 GraphMeta direct producer-consumer。
如果 POOL output 本身可以完整留在 L1，舊的 layer-level fuse 仍可用；這條 path
主要服務 output / input 太大、需要 row microblock 的情況。

## 15.8.6 Path 9：TNPS/layout -> compute consumer

Layout op 現在分兩層看：

- 真 kernel / materialized layout：`TRANSPOSE`、`SLICE`、`STRIDED_SLICE`、
  `D2SPACE/S2D` 等仍可走 TNPS descriptor。
- GraphMeta-confirmed intermediate handoff：`TRANSPOSE/PACK/UNPACK/SPLIT`
  若只是 producer 到後續 compute 的 transient boundary，scheduler 會 suppress
  DRAM checkpoint，讓 profile/cycle 反映 layout producer -> compute consumer 的
  on-chip handoff intent。

這不是宣稱任意 transpose 都已經 tile-kernel 化。它的安全假設是 compiler 已經為
consumer materialize synthetic input bytes；functional correctness 由後續 layer
verify 錨定。真正的任意 permute tile kernel 仍是後續工作。

Standalone `CONCAT -> CONV/DWCONV` slice 目前常見另一種情況：`CONCAT`
在 compiler 端已 materialize 成 consumer input blob，不再保留多個真 source
branch descriptor。這種 case 不應假裝成 true concat L1 pack；現在做法是讓後續
FP H-tiled `CONV/DWCONV` consumer 走一般 ping-pong microblock streaming，
用真 `DF_STREAM` descriptor overlap：

```text
mb0: UDMA_R activation tile -> CONV/DWCONV -> Requant -> UDMA_W store
mb1: UDMA_R activation tile -> CONV/DWCONV -> Requant -> UDMA_W store
...
```

這讓 layout bridge slice 在 Gantt 第二張圖能看到 `load / conv / requant / store`
lane，也會讓 Command Engine 真的做 lookahead overlap。它改善的是 consumer
compute side 的 tile pipeline；任意 multi-source concat 的 on-chip pack /
scatter live-range model 仍屬於 layout bridge 後續工作。

## 15.8.7 Path 10：producer fanout

`try_stream_conv_fanout()` 從原本的 `CONV` special-case 放寬到 safe
`CONV/DWCONV/FC` subset。多個 consecutive branch 若共用同一 logical input，
且後面接 concat-like boundary，可以共用 input microblock：

```text
UDMA load shared input tile
branch0 CONV/DW/FC + Requant
branch1 CONV/DW/FC + Requant
...
logical CONCAT / fanout boundary
```

目前仍是保守 fanout framework：

- branch 的 spatial shape / dtype / kernel / stride / pad 要匹配。
- DW 只接受 depthwise-safe shape。
- FC 只接受 `1x1` output subset。
- 通用 producer -> multiple arbitrary consumers 還需要更完整的 live-range /
  slot ownership model。

## 15.8.8 目前 10 path 狀態

| Path | Pipeline | 狀態 |
|---|---|---|
| 1 | `CONV/DW/FC -> Requant -> store/forward` | 可用；INT8/FP H-tiled ping-pong stream 已開，INT16 仍保守 |
| 2 | `CONV -> Requant -> EWE ADD/MUL/SUB -> store/forward` | 可用 |
| 3 | `CONV -> Requant -> D2SPACE` | 可用 |
| 4 | `ADD/MUL/SUB -> ADD/MUL/SUB -> store/forward` | 可用 |
| 5 | `ADD/MUL/SUB chain -> D2SPACE` | 可用 |
| 6 | `CONV/Requant -> EWE -> unary EWE` | 可用 |
| 7 | `CONV/Requant -> EWE -> POOL/TNPS consumer` | 可用，real-window POOL row microblock |
| 8 | `POOL -> unary/binary EWE/TNPS consumer` | 可用，INT8 safe subset |
| 9 | `TNPS/layout -> compute consumer` | `CONCAT -> FP CONV/DWCONV` consumer 可走 MB；GraphMeta handoff no-store 可用；true arbitrary layout tile kernel 待補 |
| 10 | `producer tile -> multiple consumers / concat fanout` | 可用於 safe CONV/DW/FC fanout subset |

---

## 15.8.9 FP H-tiled CONV/DWCONV streaming

一般 H-tiled CONV path 原本只讓非 FP / 非 INT16 的 tile 進
`DF_STREAM` ping-pong scheduling。現在 FP storage path 也打開，原因是 FP
activation tile 在 L1 裡仍是固定 byte stream，hazard model 和 INT8 一樣靠
兩個 slot 的 `slot_free_tag` 保護：

```text
slot0: UDMA_R input/weight -> CONV/DWCONV -> Requant -> store/tail
slot1: UDMA_R input/weight -> CONV/DWCONV -> Requant -> store/tail
```

`stream_slot`、`microblock_id`、`SMF_LOAD_A`、`SMF_LOAD_B`、`SMF_COMPUTE`
和 `SMF_STORE` 都會被寫進 descriptor header，所以這不是 metadata-only
profile；Command Engine 會依 stream priority 發射 ready descriptors。

目前仍保守擋住：

- `DT_INT16x8` / `DT_INT16x16`：producer/consumer ABI 較寬，暫不放進這條
  ping-pong stream。
- large INT8 upsample conv：先避免過度 aggressive 的 activation streaming。

代表驗證：

| Slice | Before | After |
|---|---:|---:|
| `layout_bridge/deeplab_v3_plus_float_L68_L69` | `0.093 ms / mb=0` | `0.070 ms / mb=5` |
| `layout_bridge/deeplab_v3_plus_float_L73_L74` | `1.412 ms / mb=0` | `0.902 ms / mb=64` |
| `producer_compute/deeplab_v3_plus_float_L2_L2` | `0.377 ms / mb=0` | `0.285 ms / mb=16` |
| `streaming_preload/deeplab_v3_plus_float_L16_L16` | `0.197 ms / mb=0` | `0.131 ms / mb=10` |

---

## 15.8.10 FC OC-slice microblock

`1x1xK -> 1x1xOC` 的 safe FC subset 現在可以用 output-channel slicing
產生 microblock。這條 path 不是 pseudo metadata；它真的把 weight matrix
按 OC rows 切成多段：

```text
load FC params once
load input vector once
mb0: UDMA_R weight OC[0:256]   -> FC full-K -> Requant OC[0:256]
mb1: UDMA_R weight OC[256:512] -> FC full-K -> Requant OC[256:512]
mb2: UDMA_R weight OC[512:768] -> FC full-K -> Requant OC[512:768]
```

每個 microblock 的 `Requant.oc_start` 指向原 layer 的 OC offset，output
slice 寫回完整 output tensor 的對應位置。若 producer store 被 suppress，
完整 output 仍留在 L1；否則每個 OC slice 以 UDMA_W store drain。

目前邊界：

- 只吃 `OK_FC`、`in/out H=W=1`、`group=1`、非 FP dtype。
- 預設 `tile_oc=256`，不足時按 16-channel alignment 往下縮。
- 這是 `OC slice x full-K`；真正 partial-K accumulation 需要 CONV engine
  增加 psum buffer / accumulate descriptor 協定。

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

## 15.10 Layer 到 Microblock 的分工

從高階 layer graph 到 microblock wavefront，可以這樣看：

```text
Layer graph
  ↓  Compiler / graph compiler
Layer fuse / graph-level fuse
  ↓  Compiler / tiler
Tile split
  ↓  Compiler + Command Engine ABI
Tile-level fuse / L1 handoff
  ↓  Command Engine
Microblock pipeline / wavefront
```

分工重點：

| 階段 | 主要負責者 | 做什麼 |
|---|---|---|
| `Layer graph` | Compiler | parse TFLite graph，建立 tensor producer / consumer relation |
| `Layer fuse / graph-level fuse` | Compiler | 決定哪些 layer boundary 合法 fuse，哪些 output 是 intermediate |
| `Tile split` | Compiler / tiler | 根據 L1 budget、H/OC/K 方向決定 tile shape |
| `Tile-level fuse / L1 handoff` | Compiler + Command Engine ABI | 用 metadata / descriptor contract 表示 producer tile 可以留在 L1，不必寫回 DRAM |
| `Microblock pipeline / wavefront` | Command Engine | 展開 UDMA_R、CONV、Requant、EWE、POOL、TNPS、UDMA_W descriptors，分配 L1 slot 和 dependency tags |

一句話：

```text
Compiler decides what can be fused.
Command Engine decides how fused tiles become engine microblocks.
```

目前 prototype 還不是完全產品化切法。實作上：

| 模組 | 目前做的事 |
|---|---|
| `compile_model.py` | TFLite -> `LayerMeta` / `GraphMeta` / weights / reference output |
| `test_model.cpp` scheduler | layer fuse decision、tile split、L1 handoff、microblock descriptor expansion |
| Command Engine model | dependency tag scheduling、engine dispatch arbitration |

未來 cleaner architecture 會把更多 tile / microblock expansion 收斂到 Command Engine：

```text
Compiler:
  Layer graph -> fused groups -> tile plan -> descriptor template / microblock hints

Command Engine:
  descriptor template -> actual microblock descriptors -> runtime dependency scheduling
```

---

## 15.11 Cleaner architecture 的方向

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

## 15.12 Debug checklist

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


\newpage

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

Requant 有兩個吞吐數字：

```cpp
CONV_REQUANT_CHAIN_LANES = 128   // 4096 bit/cyc
PACK_LANES = 512                 // MBQM / clamp / pack
```

chain drain cycle：

```text
ceil(output_elements / 128)
```

pack cycle：

```text
ceil(output_elements / 512)
```

`CONV_REQUANT_CHAIN_LANES` 代表 CONV → Requant 的 INT32 psum chain 寬度；`PACK_LANES` 代表 CONV / EWE 共用的 quantize-pack / clamp resource。

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
CONV 的 `MAC util` 用 ideal CONV cycle 除以實際 CONV engine task
duration；不要拿 ideal cycle 除以 per-layer wall window，因為 stream/fuse
會讓下一層在前一層 tail 尚未全部 retired 前就開始，layer window 可能被壓短。

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


\newpage

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


\newpage

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
id, flow, op, in_h, in_w, in_c, out_h, out_w, out_c,
tiles_h, tiles_oc, pass, cycles_layer, cycles_cum,
conv_util_pct, dram_r, dram_w, sram_r, sram_w
```

`flow` 是 L1 handoff group 的起點 layer id。沒有 fuse 的 layer 會滿足
`flow == id`；例如 L9/L10/L11 若真的形成同一個 on-chip flow，三列都會
顯示 `flow = 9`。

適合丟到 pandas / spreadsheet，找 top cycle layers。

---

## 18.6 profile PNG / HTML

`plot_profile.py` 產生 static Gantt PNG。

`run_model.py` 的 HTML report 更完整：

- summary chips
- interactive Gantt
- microblock stage Gantt
- layer table, including flow id and CONV MAC util
- engine utilization
- compile log
- sim log
- notes / candidates

HTML 對 debug overlap 和 labeling 很有用。CONV MAC util 是用 ideal
CONV cycle / 實際 CONV task duration，不用 layer wall window，避免 fused
或 overlapped layer window 被壓短時出現大於 100% 的假象。

第二張 microblock stage Gantt 會把 task meta 依 stage 重分：

```text
load | conv | requant | ewe/pool/tnps | store
```

判定條件是 descriptor 同時有 `DF_STREAM` 與非零 `stream_meta_flags`。因此
第二張圖空白只代表沒有真 stream descriptor，不必然代表 layer fuse 沒發生。
例如單 tile L1 handoff 可能只在 layer flow 上看得到；materialized layout input
也可能已 fuse graph boundary，但 compute consumer 仍要從 DRAM reload。

近期 FP H-tiled `CONV/DWCONV` 已打開 ping-pong microblock streaming，所以像
`deeplab_v3_plus_float_L73_L74_layout_bridge` 會從 `mb=0` 變成
`mb=64:load+conv+requant+store`。

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


\newpage

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


\newpage

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
./batch/run_model.py --list
./batch/run_model.py inception_v3_quant --keep-intermediate
./batch/run_model.py unet_int16 --keep-intermediate
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
- chain lane `oc % 128` 如何 drain？
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

這本 notebook 到第 20 章已經具備完整 onboarding 主線。下一章會把 performance debug 獨立整理成實戰方法。

你可以把目前內容拿來做：

- 新人 onboarding。
- debug checklist。
- architecture review。
- coverage planning。
- cycle tuning baseline。

> 下一章 → [第 21 章 — Performance Bug：如何看 Profile 與 Fix DRAM Write](21_performance_bug_playbook.md)


\newpage

# 第 21 章 — Performance Bug：如何看 Profile 與 Fix DRAM Write

> 上一章：[第 20 章 — Junior Exercises 與 Roadmap](20_junior_exercises_roadmap.md)

本章你會學到什麼：

- 如何從 `.profile.csv` 找 performance bug。
- 怎麼判斷問題是 tile fuse、microblock fuse、還是 DRAM write 沒省掉。
- 如何看 `dram_r / dram_w / sram_r / sram_w / tiles_h / tiles_oc`。
- 如何用小 model reproducer 驗證修法。
- 如何把「中間層 writeback」改成 on-chip handoff 或 metadata-only boundary。

---

## 21.1 Performance bug 不是只有 cycles 變大

Junior 一開始看 performance，常常只看：

```text
sim time: 5.088 ms
```

但在 MDLA7 裡，cycles 只是結果。真正要看的原因通常在 per-layer profile：

```text
id,op,in_h,in_w,in_c,out_h,out_w,out_c,...,tiles_h,tiles_oc,...,dram_r,dram_w,sram_r,sram_w
```

Performance bug 常見不是「某個 engine 算太慢」，而是：

| 現象 | 可能 root cause |
|---|---|
| `dram_w` 很大 | 中間 tensor 被寫回 DRAM |
| `dram_r` 很大 | 下一層又從 DRAM 讀回中間 tensor |
| `tiles_h` 很大 | 單層 input/output 放不進 L1，必須 H-tiling |
| `tiles_oc` 很大 | OC slice 太大，weight/output 要分 channel tile |
| `conv_util_pct` 很低 | UDMA / store / dependency 等待吃掉時間 |
| 很多 1-byte `dram_w` | writeback 已被 suppress，只剩 scheduling barrier |

所以看 performance 的第一步不是猜，而是把 profile 做成表。

---

## 21.2 快速找出 DRAM write 熱點

先跑單一 model，保留 profile：

```bash
./batch/run_ethz_v6.py --filter imdn_quant
```

然後看最大 `dram_w` layer：

```bash
python3 - <<'PY'
import csv
from pathlib import Path

csv_path = Path("batch/output/imdn_quant.profile.csv")
rows = list(csv.DictReader(csv_path.open()))

for r in sorted(rows, key=lambda x: int(x["dram_w"]), reverse=True)[:20]:
    print(
        f"L{int(r['id']):03d} {r['op'].strip():8s} "
        f"{r['in_h']}x{r['in_w']}x{r['in_c']} -> "
        f"{r['out_h']}x{r['out_w']}x{r['out_c']} "
        f"tiles={r['tiles_h']}x{r['tiles_oc']} "
        f"dram_w={int(r['dram_w'])/1024/1024:.2f} MB"
    )
PY
```

如果你看到很多中間層有 8 MB、16 MB、24 MB write，先不要急著改 engine。要先問：

```text
這個 output 真的需要落 DRAM 嗎？
還是只是 simulator 為了 per-layer verification 寫出去？
```

---

## 21.3 判斷 final output 與 intermediate output

不是所有 `dram_w` 都該省。

| 類型 | 是否可省 | 原因 |
|---|---|---|
| 最後輸出 | 通常不可省 | host / checker 要看到 final tensor |
| multi-tile conv tile output | 通常不可完全省 | full tensor 分散在 DRAM，下一層不能直接整顆從 L1 讀 |
| single-tile producer output | 常可省 | output 可留在 L1 給下一層 |
| concat 前 branch output | 常可省 | concat 可能已是 logical / metadata-only |
| reshape / gather copy | 視情況 | 若 downstream input 已由 compiler pre-materialize，可省 |
| softmax / gelu / h_swish | 視情況 | 若下一層可接 L1 output，可做 source-fusion |

在這份 simulator 裡，有一個很重要的設計：

```text
compile_model.py 會替每個 compiled layer 產生 synthetic input。
所以很多中間層的 DRAM writeback 只是 verification boundary，
不是下一層真正必須讀取的資料來源。
```

這表示有些 `dram_w` 可以安全省掉，但要留下 dependency barrier，避免排程 race。

---

## 21.4 Tile fuse：省掉 producer store 與 consumer load

Tile fuse 的核心想法：

```text
Producer output stays in L1
Consumer reads from producer's L1_OUT
Producer skips UDMA_W
Consumer skips UDMA_R
```

典型條件：

| 條件 | 說明 |
|---|---|
| shape match | producer `out_h/out_w/out_c` 等於 consumer `in_h/in_w/in_c` |
| dtype match | INT8 / INT16 / FP16 storage width 要一致 |
| single tile | producer full output 必須能完整留在 L1 |
| L1 layout 不 overlap | input、weight、params、output 區域不能互相覆蓋 |
| dependency tag 正確 | consumer 必須 wait producer compute done，不是 wait skipped store |

在 [test_model.cpp](/Volumes/4T_OFFICE/_Codex/MDLA7_Codex/systemc/src/test_model.cpp) 裡，這類狀態通常包含：

```cpp
fuse_prev_l1_out_addr
fuse_prev_l1_out_size
fuse_prev_done_tag
fuse_prev_out_h
fuse_prev_out_w
fuse_prev_out_c
fuse_prev_dtype
fuse_prev_single_tile
```

Tile fuse 常見 bug：

| Bug | Profile 現象 |
|---|---|
| producer store 沒 defer | producer `dram_w = full output size` |
| consumer 沒 fuse | consumer `dram_r = full input size` |
| single tile 判斷太保守 | 明明可留 L1，卻落 DRAM |
| pending store 沒 drop | 下一層已 fused，但前一層仍 writeback |
| done tag 用錯 | cycles 不穩、FAIL、或 Gantt dependency 看起來怪 |

---

## 21.5 Pending store：先不要急著寫 DRAM

一個好用的技巧是 pending store：

```text
producer 完成後，先把 UDMA_W descriptor 放進 pending
看到下一層後再決定：
  - 下一層能 fuse：drop pending store
  - 下一層不能 fuse：flush pending store
```

概念流程：

```text
Layer i compute done
  -> create pending udma_w

Layer i+1 starts
  -> if fused with layer i:
       skip pending store
     else:
       emit pending store before layer i+1 body
```

這可以避免 producer 一完成就寫 DRAM，給下一層 fusion 一個機會。

看 profile 時：

| Profile | 意義 |
|---|---|
| `dram_w = output bytes` | store 沒省 |
| `dram_w = 1` | full store 被省，只留 1-byte barrier |
| `streamed = true` in JSON | 此層 writeback 被視為 streamed / skipped |
| layer log 顯示 `FUSED` | output 留在 L1，該層不做 DRAM readback verify |

---

## 21.6 Microblock fuse：不是只省 bytes，也省等待

Tile fuse 是 layer-level。Microblock fuse 是更細的 tile / block-level overlap。

常見在 binary EWE：

```text
ADD / MUL / SUB
```

也常見在 fused consumer tail：

```text
CONV/DWCONV -> Requant -> store/forward
CONV/Requant -> EWE -> POOL
POOL -> EWE / TNPS
layout/TNPS -> compute consumer
fanout producer -> multiple branch consumers
```

一顆大 tensor 如果放不進 2 MB L1，就要切 microblock：

```text
load A tile 0
load B tile 0
compute tile 0
store tile 0

同時 load tile 1 / compute tile 0 / store tile -1
```

這就是 wavefront：

```text
UDMA_R(tile+1) overlaps EWE(tile) overlaps UDMA_W(tile-1)
```

對 H-tiled `CONV/DWCONV`，同樣可以是：

```text
UDMA_R(tile+1) overlaps CONV/Requant(tile) overlaps UDMA_W(tile-1)
```

目前 INT8 與 FP storage path 都可以走 ping-pong `DF_STREAM` descriptor；
INT16 路徑因 producer/consumer ABI 較寬，仍先保守不打開。

在 profile 裡要看：

| 欄位 | 怎麼判斷 |
|---|---|
| `dram_r` | 是否每個 tile 都重讀 A/B |
| `dram_w` | intermediate tile output 是否被 suppress |
| `sram_r/sram_w` | L1 traffic 是否符合 2-input + 1-output |
| Gantt | UDMA_R、EWE、UDMA_W 是否形成 staggered wavefront |
| `task_meta` | 第二張 microblock Gantt 是否能看到 load / compute / store stage |

Microblock fuse 的 bug 常見是：

| Bug | 現象 |
|---|---|
| tile slot reuse 太早 | intermittent FAIL |
| barrier tag 不夠 | consumer race |
| final tile 標記錯 | layer done time 太早 |
| suppress store 後沒有 barrier | 下一層可能提早開始 |
| tile size 太大 | L1 overflow 或 fallback |
| POOL window row 算錯 | real-window/global pool output mismatch |
| layout no-store 用錯 | transpose/pack/unpack 後 consumer shape 對不上 |

目前 10 條 path 的 debug 方向：

| Path 類型 | 優先看 |
|---|---|
| CONV/EWE/POOL tail | producer rows、pool kernel/stride/pad、slot ping-pong |
| POOL producer tail | pool output bytes、consumer input shape、GraphMeta direct consumer |
| layout -> compute | GraphMeta consumer、是否只是 intermediate checkpoint |
| fanout | branch 是否共用 logical input、concat boundary 是否真的是 downstream consumer |

---

## 21.7 省 DRAM write：GraphMeta 是重要線索

`compile_model.py` 會在 `program.bin` 裡寫入 GraphMeta：

```text
input0_tensor
input1_tensor
output_tensor
producer0_layer
producer1_layer
first_consumer_layer
last_consumer_layer
consumer_count
```

這讓 simulator 可以知道：

```text
這個 compiled layer 的真實 TFLite output 後面還有沒有 consumer？
```

如果有 consumer，而且目前 simulator 會替 consumer pre-load synthetic input，那 producer 的 DRAM write 很可能只是中間 verification boundary。

修法方向：

```cpp
if (suppressible && G.consumer_count > 0 && G.last_consumer_layer > int32_t(k)) {
    producer_no_store[k] = true;
}
```

`suppressible` 不能亂加。每個 op path 必須真的支援 suppress store。

目前比較安全的類別：

| Op | 為什麼相對安全 |
|---|---|
| CONV / DWCONV / FC | path 已有 `suppress_producer_store` |
| ADD / MUL / SUB | EWE path 已有 barrier / streamed handling |
| AVG_POOL / MAX_POOL | pool path 已支援 deferred store |
| D2SPACE | final `CONV -> D2SPACE` 可併入 Requant final-store；intermediate D2SPACE 走 TNPS streaming；UDMA 只作 fallback |

比較需要小心的類別：

| Op | 風險 |
|---|---|
| RESHAPE | 現在多是 DRAM copy，若省掉要確定 layout / consumer |
| GATHER | index semantics，不一定只是 layout boundary |
| SOFTMAX | row-wise tiled path，要補 L1-resident handoff |
| GELU / HARD_SWISH | unary path 要補 pending / source-fusion |
| CONCAT | logical concat 可省，但 branch producer 判斷要精準 |

---

## 21.8 Case Study：ETHZ_v6 `imdn_quant`

修之前，`imdn_quant` profile 會看到：

```text
L001 conv  512x512x32 -> 512x512x32  dram_w = 8 MB
L002 conv  512x512x24 -> 512x512x32  dram_w = 8 MB
L003 conv  512x512x24 -> 512x512x32  dram_w = 8 MB
L004 conv  512x512x24 -> 512x512x8   dram_w = 2 MB
L005 concat                              dram_w = 1 byte
```

`L005 concat` 已經是 metadata-only，但 concat 前的 branch producer 還是把 output 寫回 DRAM。這表示：

```text
concat 自己省了，但是 concat input producer 沒省。
```

用 GraphMeta 檢查會看到這些 branch conv 的 output 後面仍有 consumer：

```text
consumer_count > 0
last_consumer_layer > producer_layer
```

修完後：

```text
imdn_quant:
  middle dram_w: ~104 MB -> ~0 MB
  total dram_w : ~107 MB -> ~3 MB
  sim time     : 6.656 ms -> 5.088 ms
```

剩下的 3 MB 是最後 `D2SPACE` output，這是 final output，不應該省。若 pattern 是 `CONV -> final D2SPACE`，可優化的是把 D2SPACE address swizzle 併入 Requant final-store，省掉 TNPS tail；final DRAM W bytes 仍然存在。

---

## 21.9 Case Study：ETHZ_v6 `imdn_float`

同一個修法也適用 FP16 storage path。

修完後：

```text
imdn_float:
  middle dram_w: ~208 MB -> ~0 MB
  total dram_w : only final output ~6 MB
```

這說明省 DRAM write 不是 quant-only optimization，而是 memory scheduling / graph boundary optimization。

---

## 21.10 什麼時候不要省 DRAM write

Performance optimization 最危險的地方是：省掉看似中間的 store，但其實後面真的需要。

不要省的情況：

| 情況 | 原因 |
|---|---|
| final output | host / verification 要讀 |
| graph output tensor | external visible boundary |
| multi-tile producer full tensor 不在 L1 | 下一層無法整顆 source-fuse |
| consumer 需要不同 layout | L1 bytes 不等於 consumer input bytes |
| op path 沒有 barrier | 省 store 可能讓 done time 太早 |
| graph metadata 不可靠 | producer/consumer 可能被 unsupported op 隱藏 |

省 store 前要回答三個問題：

```text
1. 下一層的 input bytes 從哪裡來？
2. 如果不寫 DRAM，誰提供 dependency done tag？
3. 這層還需要 per-layer verification 嗎？
```

答不出來就先不要省。

---

## 21.11 Fix performance bug 的標準流程

建議 junior 照這個流程做：

```text
1. 找 reproducer
2. 讀 profile CSV
3. 找 top dram_w / dram_r layer
4. 分類：final output / intermediate / multi-tile / logical boundary
5. 找 source code path
6. 加最小修法
7. rebuild
8. rerun target model
9. rerun neighboring model
10. 比較 before/after profile
```

對應 command：

```bash
make -C systemc -s
./batch/run_ethz_v6.py --filter imdn_quant
./batch/run_ethz_v6.py --filter imdn_float
./batch/run_ethz_v6.py --filter resnet_quant
```

比較 profile：

```bash
python3 - <<'PY'
import csv

for m in ["imdn_quant", "imdn_float", "resnet_quant"]:
    rows = list(csv.DictReader(open(f"batch/output/{m}.profile.csv")))
    mid = sum(int(r["dram_w"]) for r in rows[:-1])
    total = sum(int(r["dram_w"]) for r in rows)
    print(f"{m:14s} mid_w={mid/1024/1024:.6f} MB total_w={total/1024/1024:.6f} MB")
PY
```

---

## 21.12 看 Gantt：確認不是假省

省掉 `dram_w` 後還要看 Gantt。

要確認：

| Gantt 現象 | 正確期待 |
|---|---|
| producer 沒有大段 UDMA_W | 中間 write 已省 |
| 仍有小 barrier | dependency 還在 |
| consumer 沒有不合理提前 | tag dependency 正確 |
| UDMA_R/EWE/UDMA_W 有 overlap | microblock wavefront 正常 |
| layer done time 沒變 0 | profile accounting 正確 |

如果 profile 變好但 Gantt 有 race 味道，不能收工。

---

## 21.13 如果 UDMA_R 還是 dominate：Activation Compression

前面的 tile fuse、microblock fuse、DRAM write suppress，主要是在省中間 boundary 的 read/write。可是有些模型修完後，profile 仍然會像這樣：

```text
per-engine busy:
   udma_r:  1974579 cyc  (82.1 %)
     conv:   665035 cyc  (27.7 %)
  requant:   995813 cyc  (41.4 %)
```

這代表主要瓶頸還在「從 DRAM 把 activation tile 搬進 L1」。這時候只增加 CONV MAC 或 EWE lanes，幫助會有限，因為 compute engine 還是在等資料。

v1 spec 已正式加入 **ACTC（Activation Compression / Decompression）**：

```text
DRAM compressed ACT
    -> UDMA_R + ACT_DECOMP
    -> L1 normal NHWC tile
    -> CONV / EWE / POOL / REQUANT existing path
```

v1 固定採用：

```text
DRAM compressed, L1 decompressed
```

不要一開始就讓 CONV 直接讀 compressed L1。原因是 CONV 需要 3x3 window、halo、stride、padding，input address 必須像一般 NHWC tile 一樣連續。先在 UDMA_R path decompress 到 L1，可以讓既有 CONV/EWE/POOL 完全不用改，風險最低。

### 21.13.1 ACTC 放在哪裡

推薦資料路徑：

```text
UDMA_R normal:
  DRAM raw bytes -> L1 raw bytes

UDMA_R + ACT_DECOMP:
  DRAM compressed block stream -> ACT_DECOMP -> L1 raw NHWC tile

UDMA_W + ACT_COMP:
  L1 raw NHWC tile -> ACT_COMP -> DRAM compressed block stream
```

也就是 ACTC 是 memory path resource，不是 CONV engine 的一部分。

硬體 block 可以想成：

| Block | 功能 |
|---|---|
| `ACT_DECOMP` | DRAM compressed activation 解回 L1 raw tile |
| `ACT_COMP` | L1 raw activation 壓成 DRAM compressed format |
| block metadata reader | 讀每個 compressed block 的 offset / size / raw fallback flag |
| raw fallback path | 壓不下去時直接搬 raw bytes |

### 21.13.2 Compression format 要保守

v1 不追求最強壓縮率，要追求硬體簡單、latency 可估、worst case 安全。

建議 block granularity：

| 欄位 | 建議 |
|---|---|
| block size | 128B raw block |
| granularity | row-major NHWC，盡量不要跨太多 row |
| dtype | INT8 / INT16 / FP16 都以 storage byte stream 處理 |
| metadata | per-block compressed length + raw flag；offset v1 用 sequential stream implicit |
| fallback | compressed size >= raw size 時存 raw |

可先支援的 lossless scheme：

| Scheme | 適合資料 | 硬體成本 |
|---|---|---|
| zero-run / repeated-value RLE | activation sparse 或大量相同值 | 低 |
| base-delta | 鄰近 activation 數值變化小 | 中 |
| small dictionary | 小 block 內常見 byte pattern | 中 |
| raw block | 壓縮無效時 fallback | 低 |

最重要的是 raw fallback。沒有 raw fallback，worst case 可能變大，compiler 和 DRAM allocator 會很難保證空間。

### 21.13.3 Performance 怎麼估

先把 DRAM read 拆成：

```text
total_udma_r = act_read + weight_read + params_read + layout_read
```

ACT compression 只會改善 `act_read`，不會改善 weight/params。

粗估：

```text
effective_act_read = act_read / compression_ratio
effective_udma_r   = effective_act_read + weight_read + params_read + metadata_read
```

例如一個 model：

```text
DRAM total read = 74 MB
其中 activation read = 50 MB
weight + params = 24 MB
ACT compression ratio = 2.0
metadata overhead = 1 MB
```

那新的 DRAM read 近似：

```text
50 / 2 + 24 + 1 = 50 MB
```

這不代表 latency 會直接從 74/50 等比例下降，因為 bottleneck 可能轉移到 Requant、CONV、EWE、或 ACT_DECOMP 自己。但如果原本 `udma_r` 是 80% 以上，通常會有感。

### 21.13.4 Cycle model 要加哪些東西

Simulator 不應該只把 bytes 乘上一個 compression ratio。比較正確的建模要包含：

| 成本 | 說明 |
|---|---|
| compressed DRAM read bytes | 真的少從 DRAM 讀 |
| metadata read bytes | offset table / block header 也要讀 |
| decompress cycles | ACT_DECOMP lanes / bytes per cycle |
| L1 write raw bytes | 解壓後寫入 L1 的 bytes 不變 |
| descriptor startup | 新 UDMA mode 或 ACTC descriptor 仍有 decode cost |
| fallback ratio | 部分 block 可能 raw，不會壓縮 |

v1 使用兩個 UDMA mode：

```text
UM_ACT_DECOMP_COPY
UM_ACT_COMP_COPY
```

descriptor body 可以沿用 `src_addr / dst_addr / length`，再把 `idx_table_addr` 指向 compressed block table。

### 21.13.5 什麼時候 ACT compression 幫助小

ACT compression 不是萬靈丹。

| 情況 | 為什麼幫助小 |
|---|---|
| weight read dominate | ACT 不是主要 DRAM traffic |
| Requant dominate | UDMA_R 降低後瓶頸轉到 Requant |
| activation entropy 高 | 壓縮率接近 1 |
| metadata 太碎 | block table overhead 吃掉收益 |
| L1 write dominate | L1 raw write bytes 沒變 |
| latency-critical 小 tensor | descriptor / metadata overhead 可能比省 bytes 大 |

所以 ACTC patch 必須用 profile 驗證：

```text
before:
  udma_r busy
  dram_r bytes
  top act-read layers

after:
  effective compressed bytes
  ACT_DECOMP busy
  total sim time
  correctness PASS
```

### 21.13.6 對 junior 的判斷口訣

看到 `UDMA_R dominate` 時，先照順序問：

```text
1. 是不是重複讀 weight？先做 persistent weight。
2. 是不是 branch 共用 input？先做 fanout input tile reuse。
3. 是不是 H tile halo 重複讀？需要 rolling halo / L1 rotate。
4. ACT read 還是很大？才考慮 ACT compression。
```

ACT compression 是硬體資源，不是 scheduling 小修。它已進 v1 spec；simulator 用 conservative ratio / raw fallback 先估趨勢，RTL 需實作 lossless 128B block codec。

---

## 21.14 常見誤解

| 誤解 | 正確理解 |
|---|---|
| `dram_w` 越少一定越正確 | 可能只是錯誤 skip final output |
| `dram_w=1` 是 bug | 可能是 deliberate 1-byte barrier |
| concat 已經 metadata-only 就沒事 | concat 的 branch producer 也可能還在寫 DRAM |
| multi-tile conv 一定能 fuse | full tensor 不在 L1，通常不能直接 layer fuse |
| `ok` 代表 performance model 正確 | `ok` 只代表 byte check pass，不代表 memory schedule 最佳 |
| 只看 total ms 就能 debug | 要看 per-layer `dram_r/w` 和 Gantt |
| 加 ACT compression 一定會變快 | 只有 ACT DRAM read 是瓶頸且壓縮率夠好時才會明顯 |

---

## 21.15 本章小結

Performance debug 的核心不是「看到慢就調參」，而是建立一條證據鏈：

```text
profile hotspot
  -> layer shape / op / dtype
  -> memory traffic classification
  -> source code scheduling path
  -> minimal fix
  -> before/after profile
  -> neighboring regression
```

本章最重要的三句話：

```text
Tile fuse 省 layer boundary。
Microblock fuse 省 tile boundary 等待。
DRAM write suppress 省中間 verification boundary。
ACT compression 省 DRAM activation read bandwidth。
```

真正好的 performance patch，應該同時做到：

- correctness 還是 `ok`。
- `dram_w` 明確下降。
- Gantt dependency 合理。
- 對鄰近 model 沒 regression。
- 你能說清楚哪些 write 省了，哪些 write 不該省。

> 下一步：把本章流程用在 `pynet_v2_*`、`sam_float`、`sd_*`，逐一分類剩下的 `reshape / softmax / gelu / h_swish` 中間 write。


\newpage
