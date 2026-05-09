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

這裡有一個很重要的設計：CONV 不直接把最終 output tensor 寫回 L1。CONV 把 partial sums 推到 16 條 chain，Requant Engine 再 drain chain、做 fixed-point requantization、寫出 output。若後面是 final `DEPTH_TO_SPACE`，Requant 也可以直接做 final-store address swizzle。這會在第 11 章詳細走讀。

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
