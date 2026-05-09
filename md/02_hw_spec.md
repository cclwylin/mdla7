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
| Host | RISC-V host / runtime，負責下 descriptor |
| Command Engine | 中央 controller，依 dependency tag dispatch 工作 |
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
Host -> Command Engine -> engine config FIFO -> engine done tag
```

在 SystemC 裡對應：

| HW 概念 | SystemC 對應 |
|---|---|
| Host 下 descriptor | [`Host`](../systemc/include/mdla7/host.h) |
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

這表示 CONV Engine 和 Requant Engine 中間有 16 條 int32 FIFO。這不是一般 descriptor config FIFO，而是 data path。

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
| CONV -> Requant 16-lane chain | implemented conceptually |
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
