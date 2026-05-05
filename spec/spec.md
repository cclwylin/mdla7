# MDLA7 Spec（v0 — 從 block 圖推導）

> 本檔是讀完 [`mdla7.drawio`](mdla7.drawio) 後的初步推導，作為 SystemC modelling 的起點。
> **凡標 `[TBD]` 的條目都需要使用者補正確 spec**；其餘項目是直接從圖上抄的，原則上不會錯。

---

## 1. 系統概觀

MDLA7 是一個類 Edge-TPU 的 NN 加速器，採用「RISC-V host + 中央 controller + 多運算引擎 + 雙層 memory hierarchy」結構：

```
   Host (RISC-V) ──descr──▶ ┌─────────────── Command Engine ───────────────┐
                            │  W↓     W↓      W↓      W↓        W↓         │
                            ▼        ▼        ▼        ▼         ▼
                        CONV Eng  Requant   EWE Eng  POOL Eng   TNPS Eng
                            │        │        │        │         │
                            └──────────────  L1_Manager  ────────┘
                                          │           │
                                 L1Mesh 2M SRAM     UDMA
                                                      │
                                                   4G DRAM
```

- **Host 控制流**：RISC-V 把 NN graph 編成 descriptor stream（含 dependency tag），送進 Command Engine。
- **NPU 內控制流**：Command Engine 含 **dependency tracker**，依 tag 狀態 dispatch task 給各 Engine（透過 `W` 1T 介面）。Engine 完成後上報 done tag。
- **資料流**：DRAM (LPDDR5X-10667) ⇄ UDMA ⇄ L1_Manager ⇄ {L1Mesh SRAM, 5 Engines}。
- 5 條 compute pipeline 共用一顆 L1_Manager 作為 on-chip 記憶體仲裁中心。

---

## 2. Block 階層

| 層 | Block | 容量 / 寬度 | 角色 |
|---|---|---|---|
| Host | **Host (RISC-V)** | SoC 主 CPU | NN compiler、runtime、descriptor 產生器、ISR；下發 descriptor 給 Command Engine `[TBD]`：core 數、ISA extension、SoC 上是內嵌 NPU 還是外掛 |
| Ctrl | **Command Engine** | top controller | 解 host 下來的 descriptor，把 layer 拆成 micro-op 派給各 Engine |
| Compute | **CONV Engine** | **65,536 4×4 base cell**（16 cluster，4 個帶 FP）| Convolution / MAC 主運算；bit-decomposable，hybrid INT+FP；見 §3A.4 |
| Compute | **Requant Engine** | — | INT8/INT32 重新量化（scale + shift + zero-point）`[TBD]` |
| Compute | **EWE Engine** | — | Element-Wise（add / mul / activation, ReLU 系），**64 elem/cyc** |
| Compute | **POOL Engine** | — | Max / Avg pooling `[TBD]` |
| Compute | **TNPS Engine** | — | Tensor 軸重排 / transpose（permute、dim swap、NHWC↔NCHW、attention Q/K/V 重排）；data-movement only，不做 MAC `[TBD]` |
| Mem ctrl | **L1_Manager** | 中央仲裁 | 5 Engine + UDMA + L1Mesh SRAM 之間的 traffic arbiter |
| On-chip mem | **L1Mesh SRAM** | **2 MB @ 1.3 GHz** | 工作 buffer（activation / weight tile） |
| DMA | **UDMA** | 5 op modes | DRAM ↔ on-chip 搬運；含 LINEAR / STRIDED_2D / INDEXED_GATHER / SCATTER_CONCAT / STRIDED_SLICE（見 §3A.9） |
| Off-chip | **DRAM (LPDDR5X-10667)** | **4 GB** | host shared / model weight / activation tensor；JEDEC LPDDR5X，10.667 Gbps/pin，dual-channel x32 → 85.3 GB/s peak |

---

## 3. 介面（從圖上的標記讀出）

### 3.0 圖上線條的約定

| 線條樣式 | 語意 | 寬度 |
|---|---|---|
| **藍色線** | **AXI bus** | 每條 **128 bits = 16 bytes**，含完整 AXI handshake |
| **白色線** | **單拍 (1T) payload**，無 AXI 協議 | 視 payload 大小，假設一個 cycle 寫完 |
| 線旁邊的數字 | **該方向 AXI bus 的條數**（並列度） | **暫定值，後續會評估修改** |

→ 同一條 link 的「總頻寬 = 條數 × 128 bits/cycle」。下表的數字都是「條數」。

### 3.1 各 Engine ↔ L1_Manager（藍線 = AXI 128b × N）

| Engine | AXI_R (ACT) | AXI_R (WGT) | AXI_R | AXI_W | 該 Engine 對 L1_Manager 的 peak BW @ 1.9 GHz |
|---|---|---|---|---|---|
| CONV | **32** | **32** | — | — | R: 64 × 16B = **1024 B/cyc** = 1.024 TB/s；**無 AXI_W** |
| Requant | — | — | 16 | 8 | R: **256 B/cyc**；W: **128 B/cyc** |
| EWE | — | — | 16 | 8 | R: **256 B/cyc**；W: **128 B/cyc** |
| POOL | — | — | 16 | 8 | R: **256 B/cyc**；W: **128 B/cyc** |
| **TNPS** | — | — | **8** | **8** | R: **128 B/cyc**；W: **128 B/cyc**（讀寫對稱：每個 element 一進一出） |

- CONV 對 L1_Manager **只有讀**（32 組 ACT + 32 組 WGT，全部都是 AXI_R）。
- CONV 的輸出（INT32 partial sum）**不直接寫回 L1**；推測有專用 chain 路徑串到 Requant Engine（路徑形式 `[TBD]`，見 §6.2）。
- Requant / EWE / POOL / TNPS：保留 AXI_R + AXI_W，partial sum / output / 重排後 tensor 透過 L1_Manager 仲裁進出 L1Mesh。
- TNPS 純做 data movement（讀 → 重新 index → 寫），不參與 MAC；R/W 條數設成對稱（8+8）反映 1:1 流量；BW 設成 EWE/POOL 的一半，因為 transpose 多半發生在 attention 相關的 small-tensor reshape，不是 throughput-critical 路徑。

### 3.2 L1_Manager 邊界（藍線）

| 對端 | 介面 | 條數 × 128b |
|---|---|---|
| L1Mesh SRAM | `AXI_R` 雙向各一組 | 16 + 16 @ **1.3 GHz SRAM clock**（peak 等效 core-cycle BW 乘 1.3/1.9） |
| UDMA | `AXI_R` + `AXI_W` | 16 + 16 |

### 3.3 UDMA ↔ DRAM（白線 = 1T 模型假設）

- 真實 HW 上仍是 AXI / DDR controller，但 v0 SystemC 模型假設 **DRAM 為零延遲、單拍完成 R / W**。
- 何時升級成有 latency 的 AXI 模型 `[TBD]`（可能在 §5.3 v3）。

### 3.4 Host (RISC-V) ↔ Command Engine（白線 = 1T 模型假設）

- v0 模型：RISC-V 以 1T 把整包 descriptor 寫進 Command Engine（fire-and-forget）。
- 真實 HW 預期是 **AXI-Lite / AXI MMIO**（host 對 NPU 的 CSR 寫；descriptor 本體可能放在 DRAM，host 只寫 doorbell）`[TBD]`。
- 反向（NPU → Host）：通常用 interrupt 通知 layer/全圖完成；v0 暫不模 interrupt path。
- Host 是否也直接走 AXI 存取 4G DRAM（與 UDMA 共享 DRAM）`[TBD]`。

### 3.5 Command Engine ↔ 各 Engine（白線）

- `W` 標記 = 單拍 (1T) 寫入 payload，**不走 AXI**。
- 假設：每下一個 micro-op／descriptor，Command Engine 在一個 cycle 內把整包 config 推給 target Engine（target 內部 latch）。
- Payload 結構（descriptor 欄位）`[TBD]`。
- 是否要 back-pressure / ack `[TBD]`（單純 1T fire-and-forget 的話，需在 Command Engine 端保證 target 已就緒）。

### 3.6 其他白線

- L1_Manager ↔ Engine 的單拍訊號（如 done / ready）——若有的話，目前圖上看起來主要 data path 都是藍線，白線集中在 Host / Command Engine 派工。`[TBD]`
- Engine → Command Engine 的 done flag：v0 假設 1T 上行 ack，cycle 級時序 `[TBD]`。

---

## 3A. 系統參數 / 算力 / BW 平衡

### 3A.1 已固定

| 參數 | 值 |
|---|---|
| 系統時脈 | **1.9 GHz** |
| CONV base cell | **4 × 4 INT multiplier**（BitFusion-style 可融合） |
| CONV base cell 總數 | **65,536 個**（= 1,048,576 bit-mult ÷ 16 bit-mult/cell） |
| Cluster 組織 | **16 cluster**，每 cluster 4,096 base cell |
| FP datapath | **4 個 cluster** 內嵌 FP unit（dedicated FP slice，其餘 12 個 INT-only） |
| FP / INT 並行性 | **Layer-serial only**（同一時間整個 array 只跑 INT 或只跑 FP，不混跑） |
| Accumulator 寬度 | **INT48**（INT16×16 累加可承受 ~2^16 次乘加；INT8×8 上限遠大於任何 conv layer 需求） |
| MAC array 總算力 | **~1.05 M bit-mult / cycle**（bit-level 守恆，見 §3A.2） |
| 支援精度 | **Hybrid INT** + **Hybrid FP**（見 §3A.2 表） |
| Requant 路徑 | **16 條 per-cluster Requant lane**，CONV 直接 chain 到 Requant（不經 L1_Manager） |

### 3A.2 Hybrid-precision MAC 表（per cycle, @ 1.9 GHz）

CONV array 是 **bit-decomposable**：固定矽面積、可依 operand 精度重組 MAC 顆數。表記法 `INTa × b` = `a-bit × b-bit` 混精乘法。

#### INT 系（`INTa × b`，相對 INT8×8 baseline）

| Data type | MAC / cycle | Ratio | Peak ops @ 1.9 GHz | bit-mult / cycle |
|---|---|---|---|---|
| INT8 × 4 | 32,768 | 2.00 | **124.5 TOPS** | 1,048,576 |
| **INT8 × 8 (baseline)** | **16,384** | **1.00** | **62.3 TOPS** | **1,048,576** |
| INT16 × 4 | 16,384 | 1.00 | 62.3 TOPS | 1,048,576 |
| INT16 × 8 | 8,192 | 0.50 | 31.1 TOPS | 1,048,576 |
| INT16 × 16 | 4,096 | 0.25 | 15.6 TOPS | 1,048,576 |

→ **invariant**：`MAC × a × b = 1,048,576 bit-mult/cycle`。
→ 可推論：4 條 64×64 ring 每條有 64×64 = 4096 個 baseline INT8 MAC slot；每個 slot 內含 64 bit-mult cell（8×8），可拆 / 拼成更小或更大的 operand。

#### FP 系（4 種 16-bit 級格式 + FP8）

| Data type | Exp / Mant | MAC / cycle | Ratio | Peak ops @ 1.9 GHz |
|---|---|---|---|---|
| FP8 (E4M3) | 4 / 3 | 4,096 | 0.25 | **15.6 TFLOPS** |
| FP16 (E5M10) | 5 / 10（IEEE 754） | 4,096 | 0.25 | 15.6 TFLOPS |
| BFP16 (E8M7) | 8 / 7（Brain Float） | 4,096 | 0.25 | 15.6 TFLOPS |

→ FP 全走 4096 MAC 的子集；可能是 array 中有 1/4 slot 內含 FP-capable cell，或共用 INT mantissa 路徑 + 額外 exponent 邏輯 `[TBD]`。

#### Spec 中的「peak throughput」

- 報帳口徑（marketing 等）：**INT8×4 = 124.5 TOPS**（最高 INT 數字）
- 工程口徑（程式設計師預期）：**INT8 = 62.3 TOPS**（baseline）
- FP：**15.6 TFLOPS**（任一 FP 格式）

> 對照：Edge TPU ≈ 4 TOPS、Apple ANE ≈ 17 TOPS、Tesla FSD ≈ 36 TOPS、TPU v4 MXU ≈ 275 TOPS / 137 TFLOPS BF16。
> MDLA7 @ 1.9 GHz **INT8×4 124 TOPS / FP 15.6 TFLOPS**，進入 mid-range datacenter 量級。

#### FP32 deployment policy

> **MDLA7 不直接支援 FP32 native compute。**
> FP32 source model（如 TFLite FP32、ONNX FP32）必須由 compiler / runtime 轉成 **FP16 / BFP16 / INT8** 之一才能部署。
> 建議：
> - **CNN（weight-heavy）** → INT8 PTQ（post-training quantization）
> - **Transformer（attention 對精度敏感）** → BFP16 (E8M7)
> - **混合（vision + transformer）** → FP16 (E5M10)
> 這是業界 deployment workflow 標準做法（TFLite Converter / ONNX Runtime / TVM / CoreML 都支援），不算 HW 限制。

### 3A.3 BW vs Compute 平衡（CONV）

CONV 對 L1_Manager 的 R 介面（**只有讀，沒有寫**）：

| 來源 | 條數 | bytes/cycle @ 128b | bytes/sec @ 1.9 GHz |
|---|---|---|---|
| AXI_R (ACT) | 32 | 512 | **973 GB/s** |
| AXI_R (WGT) | 32 | 512 | **973 GB/s** |
| 合計 R | 64 | 1024 | **1.946 TB/s** |

**單位 cycle 的 operand 需求（理論上限，未計 reuse）：**

- 16384 MAC × (1B ACT + 1B WGT) = 16,384 + 16,384 B = 32,768 B/cycle ⇨ **32 TB/s**
- 對比可用 BW 1.024 TB/s ⇒ **頻寬只夠 1/32 的 naïve 需求**
- ⇒ 必須仰賴 **systolic / weight-stationary 等 dataflow 的 operand reuse** 才能餵滿 array

**Output-stationary 假設下穩態 BW（`[TBD]`，等使用者確認）：**

16 cluster 並列、每 cluster 在 OS 模式穩態每 cycle：
- 注入新 ACT：32 byte
- 注入新 WGT：32 byte
- 16 cluster 合計 → 512 B ACT + 512 B WGT per cycle
- ACT_R 可用 512 B/cyc、WGT_R 可用 512 B/cyc → **剛好飽和**（無 BW 餘裕；要 double buffer 需在 cluster 內部開 buffer）

→ 32 + 32 條 AXI 設計剛好對應 16 cluster 的穩態餵料。
→ 若 dataflow 是 weight-stationary，WGT_R 只在 tile load 時忙碌，可拿來預取下一 tile 的權重。

> 真正的 dataflow（OS / WS / IS / row-stationary）`[TBD]`。

---

## 3A.4 MAC Array 組織（推薦組合，已決定）

3 層階層：

```
L0 base cell    : 4×4 INT multiplier (16 bit-mult/cell)
L1 fusion unit  : 4 base cell = 1 INT8×8 MAC（可拆/拼）
L2 cluster      : 4096 base cell = 1024 INT8 MAC = 256 INT16×16 MAC
L3 array        : 16 cluster
```

### 16 cluster 構造

| 屬性 | 值 |
|---|---|
| 形狀 | 4 × 4 cluster grid（邏輯） |
| 每 cluster 內部 | 64×64 of 4-bit base cell（systolic 4-bit 視角）<br>= 32×32 of INT8 MAC（8-bit 視角）<br>= 16×16 of INT16 MAC（16-bit 視角） |
| INT-only cluster | **12 個**（cluster 0–11） |
| FP-capable cluster | **4 個**（cluster 12–15）— dedicated FP datapath，內含 1024 base cell × 1/4 FP-overhead = 256 FP MAC / cluster |

### 各 dtype 的 cluster 分布

| Data type | MAC per cluster (INT-only / FP-capable) | cluster 數 | 總 MAC |
|---|---|---|---|
| INT8×4 | 2048 / 2048 | 16 | **32,768** |
| INT8×8 (baseline) | 1024 / 1024 | 16 | **16,384** |
| INT16×16 | 256 / 256 | 16 | **4,096** |
| FP8 / FP16 / BFP16 | 0 / 1024 | **4**（僅 FP cluster） | **4,096** |

→ FP 任務 dispatch 時只啟動 cluster 12–15，其餘 cluster idle（節能）。
→ INT 任務 16 cluster 全開，FP-capable cluster 的 FP datapath 在 INT 模式下不耗能。

### Cluster 內部 dataflow 預設

- 每 cluster 採 **output-stationary**：accumulator 留在 cluster 內部，ACT / WGT 從邊緣注入流經 array。
- Partial sum 不在 cluster 內部繞 ring；reduce 發生在 array level（16 cluster → reduce tree → Requant chain）。
- Tile fill latency = cluster 邊長 ≈ 64 cycle（base cell 視角；實際與 dtype 相關）`[TBD]`。

---

## 3A.5 CONV → Requant Chain（推薦組合，已決定）

CONV 沒有 AXI_W，partial sum 不寫回 L1Mesh。改走 **per-cluster Requant lane chain**：

```
                        per-cluster INT32 chain
CONV cluster 0  ──INT32──▶ Requant lane 0  ──┐
CONV cluster 1  ──INT32──▶ Requant lane 1  ──┤
...                                          ├── merge
CONV cluster 15 ──INT32──▶ Requant lane 15 ──┘
                                              ▼
                                        8 條 AXI_W ──▶ L1_Manager ──▶ L1Mesh
```

### Chain 介面

| 屬性 | 值 |
|---|---|
| 條數 | **16 條 INT32 lane**（一對一對應 CONV cluster ↔ Requant lane） |
| 寬度 | 每 lane 32 bit（INT32 partial sum） |
| 拍數 | 1T per psum（cluster 完成一次 reduction 即推一筆 psum 上 chain） |
| 同步 | cluster 完成 reduction → 自動 push to chain；Requant lane 隨後同 cycle latch |
| Reverse path | 無；Requant 不回 CONV |

### Requant lane 內部

每條 lane 處理 1 個 cluster 的輸出，邏輯：

```
psum (INT32) ─▶ × scale ─▶ >> shift ─▶ + zero_point ─▶ saturate to INT8 ─▶ FIFO out
```

- **(scale, shift, zp) per output channel**：每條 lane 帶自己的 LUT，layer 開始時由 Command Engine config 灌入。
- → **per-output-channel quantization 天然支援**（modern NN 標準需求）。

### Merge → AXI_W

16 條 INT8 輸出在 Requant Engine 末端 merge，走 **8 條 AXI_W**（Requant ↔ L1_Manager）寫回 L1Mesh：

- 16 lane × 1 byte/cycle = **16 byte/cycle 穩態** = 1 AXI 128b lane 即可
- 8 條 AXI_W 大量 over-provision → 用於 burst 寫、多 layer fused 模式 `[TBD]`。

### SystemC 介面

- **CONV cluster ↔ Requant lane**：`sc_fifo<int32_t> chain[16]`，depth=2（cycle latch + 1 cushion）
- **Requant lane 內 LUT**：`sc_signal<requant_param_t>`，由 CommandEngine 在 layer 啟動時 set
- **merge → AXI_W**：Requant Engine 內部 collector → `simple_initiator_socket`
- **Accumulator**：`sc_signal<sc_int<48>>` 或 `int64_t` 簡化版（v0 用 int64 即可，v1 改成嚴格 sc_int<48> 模 saturation）

---

## 3A.6 CONV 支援的 op 範圍（**proposal — 待確認**）

依「Edge-TPU 級 mobile-to-edge NN HW」常見支援度提案。實際決定影響 ACT line buffer / WGT staging 大小、address generator 複雜度、descriptor 欄位寬度。

### Kernel size

| 軸 | 範圍 |
|---|---|
| K_h, K_w 各自 | **1, 2, 3, 5, 7**（cover 95% modern CNN：pointwise / 標準 / spatial expand） |
| 是否支援 K_h ≠ K_w | **是**（asymmetric kernel；e.g. 1×7 + 7×1 separable conv） |
| 上限 | **11×11**（Alexnet-era 第一層，少用但留 capability） |
| descriptor 編碼 | 4 bit / 軸（值域 1..15） |

### Stride

| 軸 | 範圍 |
|---|---|
| S_h, S_w 各自 | **1, 2, 4** |
| Stride > kernel 是否合法 | 否（無意義） |
| descriptor 編碼 | 2 bit / 軸（值域 1, 2, 4） |

### Dilation

| 軸 | 範圍 |
|---|---|
| D_h, D_w 各自 | **1, 2, 4** |
| descriptor 編碼 | 2 bit / 軸（值域 1, 2, 4） |

### Padding

| 屬性 | 範圍 |
|---|---|
| P_top, P_bottom, P_left, P_right 各自 | **0..7** |
| 模式 | explicit value（不走 SAME / VALID 抽象，由 host compiler 算好填欄位）|
| 邊界值 | zero-pad（最常見）；reflect / replicate 視 use case 加 `[TBD]` |

### Group conv

| 模式 | 支援 |
|---|---|
| 標準 conv | group = 1 |
| Grouped conv | group ∈ {2, 4, 8, 16, 32} |
| **Depthwise** | group = C_in（每 channel 獨立 kernel；MobileNet / EfficientNet 必備） |
| Group 切分要求 | C_in 必須是 group 倍數 |

### Datatype × group conv 限制

- 所有 hybrid INT / FP dtype 對所有 kernel / stride / dilation / group 組合都支援。
- Depthwise + INT16×16：可能會在 16 cluster 切分上產生奇數 channel、需額外 padding to multiple-of-16 `[TBD]`。

### Layer-level fused 行為

- CONV → Requant 是硬接 chain，**所有 conv layer 都必跑 Requant**（不可 bypass）。若 layer 不需要 quant，Requant lane 走 identity（scale=1, shift=0, zp=0）。
- CONV → Requant → EWE / POOL 不在 chain 上 fused，需要 round-trip L1Mesh。

---

## 3A.7 EWE / POOL 支援的 op

### EWE Engine — 全收，分四類

| 類別 | Op | 形式 | 實作策略 |
|---|---|---|---|
| **Arithmetic** | add, sub, mul, div, mac (a×b+c) | 2-tensor 或 tensor + scalar | 直 HW ALU |
| **Activation (PWL)** | ReLU, ReLU6, LeakyReLU, PReLU | piecewise linear | 直 HW（compare + select + mul） |
| **Activation (LUT)** | sigmoid, tanh, GELU, SiLU/Swish, hardsigmoid, hardswish, ELU, exp, log, rsqrt, sqrt | smooth nonlinear | **64-entry LUT + 線性內插**（共用一顆 LUT，每 layer config swap） |
| **Compare** | max, min, equal, less, greater, less_eq, greater_eq | 2-tensor → mask 或 select | 直 HW comparator |
| **Bit / Logic** | and, or, xor, not, shift_l, shift_r | int 操作 | 直 HW |
| **Reduce-style** | sum, mean, max, min（reduce 軸由 descriptor 指定） | tensor → tensor (rank-reduce) | HW iterator + accumulator |

**Datatype 支援：**
- 所有 INT dtype（INT8 / INT16 / INT32）
- 所有 FP dtype（FP8 E4M3 / FP16 / BFP16）
- mixed-precision 操作（e.g. INT32 + INT8 → INT32）由 Requant chain 處理；EWE 不主動做精度轉換 `[TBD]`

**Broadcasting：**
- 支援 NumPy-style broadcasting（scalar / per-channel / per-tile）
- broadcast 軸由 descriptor 顯式指定（不自動推導）

**Fused 模式：**
- 標準 fused：`mul + add`（線性變換）、`add + ReLU`（residual + activation）
- transformer 必備：`softmax`（exp + reduce-sum + div）、`layernorm`（reduce-mean + sub + reduce-var + rsqrt + mul + add）→ 是 multi-pass，由 Cmd Engine 串成 EWE op 序列，不 fused 在硬體一拍內 `[TBD]`

### POOL Engine — proposal

| 模式 | 範圍 |
|---|---|
| **Max pool** | 標準 |
| **Avg pool** | 除數行為由 descriptor flag `count_include_pad` 決定：<br>• `0`：除以 valid count（pad 區不算）<br>• `1`：除以 K_h×K_w（PyTorch 預設） |
| **Global pool** | 整個 H×W reduce 成 1×1（max / avg 兩種） |

| 軸 | 範圍 |
|---|---|
| Kernel K_h, K_w 各自 | **2, 3, 5, 7**（cover 95% NN：2×2 是 ResNet 主流、3×3 maxpool 早期 CNN、7×7 是 ResNet50 head 前的 avgpool） |
| 上限 | **K_h, K_w ≤ 11**（與 CONV 對齊，省 descriptor 編碼 bit） |
| 是否支援 K_h ≠ K_w | **是** |
| Stride S_h, S_w 各自 | **1, 2**（pool 通常 stride=K，少有大 stride） |
| Dilation | **不支援**（已確認） |
| Padding 各邊 | **0..3**（pool 不需要 conv 那麼大的 padding） |

**Datatype：**
- INT8 / INT16 / FP8 / FP16 / BFP16
- max pool：output dtype = input dtype
- avg pool：累加用 INT32（INT 模式）或 FP32（FP 模式），最後除以 `count_include_pad ? K_h×K_w : valid_count`，再 truncate / round 回 input dtype

**Global pool 實作：**
- 內部：HW iterator over (H, W)，accumulate 到 per-channel reg → 出 1×1×C
- 不走 K_h × K_w 的 kernel size encoding，而是獨立 op type

---

## 3A.7b TNPS Engine — proposal

### 用途

軸重排 / transpose / permute。常見場景：
- **Attention path**：`[B, S, H, D]` → `[B, H, S, D]`（split heads），以及 Q·Kᵀ 之前的 `[B, H, S, D]` → `[B, H, D, S]`。
- **Layout 轉換**：NHWC ↔ NCHW（外部 framework 偶爾要求 NCHW 輸出）。
- **Conv → FC 銜接**：`[B, H, W, C]` flatten 前的維度重排（多數情況 RESHAPE 即可，但若有 channel-last 排序需求則走 TNPS）。

設計 rationale：把 transpose 從 UDMA 的 STRIDED_SLICE / SCATTER_CONCAT path 拆出來成獨立 engine，避免 UDMA 在 transformer block 內被 attention reshape 拖到 DRAM round-trip。TNPS 走 L1↔L1，純 on-chip。

### 操作模型

| 屬性 | 規格 |
|---|---|
| 輸入 / 輸出 | **L1Mesh ↔ L1Mesh**（不直接觸 DRAM；如需要，由 UDMA 接力） |
| 軸數 | 最多 **6D**（cover NCHW + batch + head + spatial 細分） |
| Permute 表達 | descriptor 帶 6-byte `axis_perm[6]`（input axis index → output position） |
| Stride | 任意 stride（不限 power-of-2）；HW 內部 6 個 nested counter |
| Dtype 支援 | INT8 / INT16 / FP8 / FP16 / BFP16（純 byte-wise 搬運，沒有 numerical op） |
| In-place | 不支援（src ≠ dst region；L1 budget 端調度） |
| Broadcast | 不支援（單純重排，不複製） |

### 算力 / BW

| 參數 | 值 |
|---|---|
| AXI_R 條數 | **8** × 128 b = **128 B/cyc** |
| AXI_W 條數 | **8** × 128 b = **128 B/cyc** |
| Throughput | 一 cycle 搬 128 B（讀 → permute index 計算 → 寫，1:1 流量） |
| 最不利 stride pattern | dst-side scatter 跨多個 L1 bank → 受 L1Mesh write port 仲裁限制（見 §3.2） |

設計取捨：**不做 8+8 以上**因為：
- attention 內 transpose 多半是 `S × D` 量級的小 tensor（1k–16k element），128 B/cyc 已能在幾百 cycle 內完成；
- 給太多 port 會跟 EWE/POOL 搶 L1Mesh 仲裁，拉低主路徑 (CONV/Requant) 的可用 bandwidth。

### Cycle 模型

```
cycles = ceil(total_bytes / 128)
       + perm_setup_overhead   (估 16 cyc，descriptor decode + counter init)
       + drain_overhead        (估 8 cyc)
```

無 row-miss penalty（純 L1 操作）。

### 跟其他 engine 的搭配

- **Attention block** 典型流程：
  ```
  Q·Kᵀ:  CONV/FC → Requant → TNPS (transpose K) → BMM (CONV-as-matmul) → Requant
  attn·V: softmax (EWE) → BMM → TNPS (combine heads) → CONV/FC
  ```
- **Conv → channel-shuffle (ShuffleNet)**：CONV → Requant → TNPS（reshape + permute）→ 下一 CONV。
- 跟 RESHAPE 的差異：RESHAPE 是純 metadata 改動（DRAM bytes 不動），TNPS 是真實的 byte permute（必須 read-modify-write）。

### `[TBD]`

- 是否需要 fuse 連續兩個 transpose（merge perm tables）`[TBD]`
- 是否支援 quantize-on-the-fly（搬運途中 INT8↔INT16）`[TBD]` — 目前傾向**否**，留給 Requant Engine
- 6D 上限是否夠（GPT-style head split 偶爾 7D `[B, layers, S, H, D, ...]`）`[TBD]`

---

## 3A.8 Dependency Tag 同步機制

Engine 之間的同步**不靠 barrier**（會浪費資源），而採 **dependency tag**：每筆 task 攜帶「等什麼 / 完成後通知誰」資訊，由 Command Engine 內部的 **dependency tracker** 控制 dispatch。

### Tag 模型

| 概念 | 說明 |
|---|---|
| Tag ID | unsigned int，**`[TBD]`** 寬度（提案 8 bit = 256 個並存 tag）|
| Tag state | 1 bit：`pending` / `done` |
| 初始化 | 整片 tag table 在系統 reset 時清為 `done`（語意：未產生過） |

### Task descriptor 上的 tag 欄位

每筆下發給 Engine 的 descriptor 至少帶這三個欄位：

```
struct EngineTask {
    op_t          op;           // CONV / Requant / EWE / POOL / UDMA op 細節
    tag_mask_t    wait_mask;    // bit-mask of tag IDs this task waits on (AND-of-all)
    tag_id_t      signal_tag;   // tag ID this task will set to 'done' on completion (0 = no signal)
    // ... 其它 op-specific 參數
};
```

- `wait_mask`：必須等到 mask 中所有 bit 對應的 tag 都是 `done` 才能 dispatch。
- `signal_tag`：task 完成時把該 tag set 為 `done`；同時把 wait 在它身上的 task 從 wait queue 喚起。

### Command Engine 內部 dependency tracker

```
┌─ Tag Table  ─────────┐    ┌─ Dispatch FIFO ────┐
│  [tag 0..255: 1 bit] │    │  per-Engine queue  │
└─────────┬────────────┘    └──────┬─────────────┘
          │                        │
          ▼                        ▼
   ┌─── Dependency Resolver ────┐
   │  walk pending tasks        │
   │  for each: AND-check       │  ───▶ ready ───▶ dispatch via W (1T)
   │  wait_mask vs Tag Table    │
   └────────────────────────────┘
```

**演算法（每 cycle）**：
1. Engine 完成 task → 透過 done flag (white line, 1T) 通知 Cmd Engine 該 task 的 `signal_tag`
2. Cmd Engine 把 `Tag Table[signal_tag] = done`，喚起 wait queue 中等該 tag 的 task
3. 對每個 ready-to-dispatch task：用 W (1T) 介面推到對應 Engine

### Dependency graph

Host 在 compile time 構出整張 NN graph 的 task 依賴關係（CONV →requires DMA preload→ Requant→ ...），編譯成 task descriptor stream 含 wait/signal tag → 灌進 Cmd Engine。

- **graph 節點數量上限** = tag 數量（提案 256） `[TBD]`
- **同時 in-flight task 數** = per-Engine dispatch queue depth × Engine 數量（每 Engine 提案 4 deep） `[TBD]`

### SystemC 介面

- **Tag table**：`bool tag_done[256];` 在 CommandEngine 模組內部 array
- **Wait check**：`sc_event_and_list waitlist;` 動態組合（v0 直接用 polling 的 SC_THREAD 即可）
- **Signal**：Engine 完成 → 透過 `sc_signal<tag_id_t> done_tag;` 上報 Cmd Engine
- **Dispatch**：CmdEngine SC_METHOD on tag_done change，掃 dispatch FIFO，能跑就 push 到 Engine config fifo

→ 簡化做法：v0 用 `std::queue<EngineTask> per_engine[5]` + 每 cycle scan，速度足夠 functional model。

---

## 3A.9 UDMA Op Modes

UDMA 不只是線性 copy；為了支援 transformer / detector / inception 系列模型，必須提供下列 5 種 op mode：

| Mode | 行為 | 用途 / 對應模型 op |
|---|---|---|
| `LINEAR_COPY` | `dst[i] = src[i]`，一段連續 byte | 一般 weight / activation 預載 |
| `STRIDED_2D` | 2D tile 含 src/dst stride | CONV tile preload、transpose |
| `INDEXED_GATHER` | `dst[i] = src[idx_table[i] × elem_size]` | **token embedding lookup**（BERT / DistilBERT / MobileBERT / Albert） |
| `SCATTER_CONCAT` | 多個 source 寫到 dst 連續區段 | **concat**（Inception / DenseNet / YOLO / EfficientDet BiFPN） |
| `STRIDED_SLICE` | 從大 tensor 取子區段（含 begin/end/stride 各軸） | **patch embedding**（ViT / MobileViT）、tensor slice |

### UDMA descriptor body 修訂

```
mode (4)            -- enum 上述 5 個
direction (1)       -- 0: DRAM→L1Mesh, 1: L1Mesh→DRAM
src_addr (32)
dst_addr (32)
length (32)         -- bytes
src_stride (32)     -- 用於 STRIDED_2D, INDEXED_GATHER 的 elem stride
dst_stride (32)     -- 用於 STRIDED_2D, SCATTER_CONCAT
num_chunks (16)     -- STRIDED_2D rows / SCATTER_CONCAT 來源段數
idx_table_addr (32) -- INDEXED_GATHER 的 index 表 addr
slice_begin[4] (16×4) -- STRIDED_SLICE 各軸起點（最多 4D tensor）
slice_end[4] (16×4)
```

→ 仍 fit 在 48 byte body 內（合計約 360 bit，剩餘給 reserved）。

### v0 / v1 切分

| Mode | v0 必做 | v1 補 |
|---|---|---|
| `LINEAR_COPY` | ✓ | |
| `STRIDED_2D` | ✓ | |
| `INDEXED_GATHER` | | ✓（解 NLP transformer） |
| `SCATTER_CONCAT` | | ✓（解 Inception / DenseNet / YOLO） |
| `STRIDED_SLICE` | | ✓（解 ViT patch embed） |

---

## 3A.10 Descriptor 格式

固定 **64 byte / 512 bit** descriptor，分 16 byte common header + 48 byte engine-specific body。
delivery 走 **DRAM ring buffer + MMIO doorbell**（host 不直接 MMIO 寫整個 desc）。

### Common Header（16 byte / 128 bit）

| Bits | 欄位 | 說明 |
|---|---|---|
| [4] | `op_class` | enum：CONV / Requant / EWE / POOL / **TNPS** / UDMA + 2 reserved |
| [4] | `op_subtype` | engine 內 op id（EWE 用最多，~30 ops；其他 engine ≈ 0） |
| [8] | `flags` | bit0 preempt / bit1 chain-source / bit2 chain-sink / bit3 debug-trace / 4-7 reserved |
| [8] | `dtype` | INT8×4 / INT8×8 / INT16×4 / INT16×8 / INT16×16 / FP8 / FP16 / BFP16 + reserved |
| [8] | `signal_tag` | 完成後 set 的 tag id（**0..255；0 = no signal**） |
| [8] | `wait_count` | 0..4，這個 task 等幾個 tag |
| [32] | `wait_tags[4]` | 4 × 8 bit，**AND-of-all** 條件 |
| [16] | `layer_id` | host 編 layer 序號（debug / trace 用，HW 不解碼） |
| [40] | reserved | 未來擴展（QoS、power gating hint、etc） |

### Body union（48 byte / 384 bit）

依 `op_class` 解出六種 body：CONV / Requant / EWE / POOL / **TNPS** / UDMA。
詳細欄位定義：

#### CONV body（最複雜）

```
in_addr (32) | wgt_addr (32) | out_addr (32) |
in_h (16) | in_w (16) | in_c (16) | out_c (16) |
k_h (4) | k_w (4) | s_h (2) | s_w (2) | d_h (2) | d_w (2) |
pad_t (3) | pad_b (3) | pad_l (3) | pad_r (3) |
group (16) | cluster_mask (16) |
bias_addr (32; 0 = no bias) |
scale_lut_addr (32) | scale_count (16) | reserved (16)
```

#### Requant body（獨立 op；CONV chain 不走這條）

```
in_addr (32) | out_addr (32) | n,h,w,c (16×4) |
scale_lut_addr (32) | scale_count (16) | per_channel_flag (1) |
shift_global (8) | zp_global (8) | reserved
```

#### EWE body

```
in_a_addr (32) | in_b_addr (32; 0 = unary op) | out_addr (32) |
n,h,w,c (16×4) | broadcast_axes (8) |
lut_addr (32; 用於 sigmoid/tanh/...) |
scalar_imm (16) | reduce_axes (4) | reserved
```

#### POOL body

```
in_addr (32) | out_addr (32) | in/out shape (16×4 + 16×4) |
mode (2: max/avg/global) | k_h (4) | k_w (4) |
s_h (2) | s_w (2) | pad (3×4) |
count_include_pad (1) | reserved
```

#### TNPS body

```
in_addr (32) | out_addr (32) |
in_shape  (16×6)  — d0..d5（先後存 input dim 大小，未用維度填 1） |
out_shape (16×6)  — derived from axis_perm but stored explicitly for HW |
axis_perm (4×6)   — output axis k 取自 input axis axis_perm[k]（0..5） |
elem_size (4)     — 1 / 2 / 4 byte per element（dtype-derived） |
reserved
```

#### UDMA body

見 §3A.9 修訂版。

### Address 空間

**Unified 32-bit address space**，不分 mem_type 欄位；由 addr 高位區分：

| Range | 對應 |
|---|---|
| `0x0000_0000 .. 0x001F_FFFF` | L1Mesh 2 MB |
| `0x0020_0000 .. 0x0FFF_FFFF` | reserved（CSR / future） |
| `0x1000_0000 .. 0xFFFF_FFFF` | DRAM 4 GB（high half） |

### Delivery 機制

```
DRAM
┌──── descriptor ring buffer (16 KB = 256 × 64 byte) ──────┐
│  [desc 0][desc 1][desc 2]...                              │
└──┬────────────────────────────────────────────────────────┘
   │ Host 直接 memcpy 寫入
   ▼
Host 寫 MMIO doorbell:
   CmdEng_TAIL_PTR = next slot
   ↓
CmdEng (內部小 UDMA) 從 ring 拉 descriptor → 解碼 → dispatch
   ↓
完成後 CmdEng 寫 IRQ status MMIO，觸發 Host RISC-V interrupt
```

| 介面 | 細節 |
|---|---|
| Ring buffer 容量 | **256 entry × 64 byte = 16 KB** |
| Head pointer | NPU 寫，Host 讀（NPU 處理進度） |
| Tail pointer | Host 寫 MMIO doorbell，NPU latch |
| MMIO 區段 | **4 KB AXI-Lite slave port** for CSR：tail_ptr / head_ptr / status / IRQ_clear / config |
| Endian | **Little-endian**（RISC-V 標準） |

### SystemC 映射

```cpp
struct DescriptorHeader {
    uint8_t op_class:4, op_subtype:4;
    uint8_t flags;
    uint8_t dtype;
    uint8_t signal_tag;
    uint8_t wait_count;
    uint8_t wait_tags[4];
    uint16_t layer_id;
    uint8_t _pad[5];
};  // 16 byte

union DescriptorBody {
    ConvBody conv; RequantBody requant; EweBody ewe;
    PoolBody pool; TnpsBody tnps; UdmaBody udma;
    uint8_t raw[48];
};  // 48 byte

struct Descriptor { DescriptorHeader hdr; DescriptorBody body; };
static_assert(sizeof(Descriptor) == 64);

sc_fifo<Descriptor> descriptor_stream;          // depth=256
sc_fifo<DescriptorBody> conv_cfg, requant_cfg,  // depth=4
                        ewe_cfg, pool_cfg, tnps_cfg, udma_cfg;
```

---

## 3A.11 Host-side Post-processing Scope

下列 op **不下發 NPU**，由 RISC-V host 軟體完成：

| Op | 為何給 host | 出現於 |
|---|---|---|
| **NMS (Non-Max Suppression)** | control-flow heavy、small data、需要 sort | SSD / EfficientDet / YOLO |
| **TopK / ArgMax** | small data，per-batch 1 個結果 | classifier final layer |
| **Box decode** | 序列邏輯，運算量微小 | 所有 detector |
| **Tokenizer / Detokenizer** | 純字元串處理 | NLP / speech 模型 |
| **Beam search** | tree search + sort | whisper decode |
| **Anchor generation** | static table lookup | detector |

設計理由：這些 op 運算量極小（≪ 1 GFLOPs），但 control-flow / sort / branch 多，NPU 強項是 dense linear algebra，由 host 做更省功耗也省 spec 複雜度。

---

## 3A.12 Model Coverage（61 個目標模型）

| 類別 | 模型 | 數量 | Coverage |
|---|---|---|---|
| **CNN — vision** | ResNet, VGG, MobileNet v1/v2/v3, EfficientNet/Det, Inception v3/v4, Xception, SqueezeNet, NASNet, DenseNet, AlexNet, FaceMesh, BlazeFace, MoveNet, PoseNet, StyleTransfer, MobileViT | ~30 | ✓ 全部 |
| **CNN — segmentation** | DeepLab v3, DeepLab MNV2 | 2 | ✓（dilated conv D∈{1,2,4} 涵蓋） |
| **Detector** | SSD MobileNet, EfficientDet Lite0–3, YOLOv8n | 7 | ✓（NMS 由 host 做） |
| **Vision Transformer** | ViT tiny / base, MobileViT | 3 | ✓（patch embed = STRIDED_SLICE，attention = 1×1 CONV） |
| **NLP Transformer** | BERT base, BERT LLM, MobileBERT(QA), DistilBERT, Albert | 6 | ✓（embedding = INDEXED_GATHER，softmax/LN = EWE multi-pass） |
| **Audio Transformer** | Whisper-tiny | 1 | ✓（1D conv → 2D conv H=1） |
| **Audio CNN** | YAMNet | 1 | ✓ |
| **Attention block** | bmm_softmax_bmm (FP16/FP32) | 2 | ✓（標準 transformer attention pattern） |
| **Pose / mesh** | MoveNet, PoseNet, FaceMesh | 3 | ✓ |
| **量化測試** | INT8/INT16/UINT8 quant 模型 | 9 | ✓（Hybrid INT 涵蓋 INT8×8 / INT16×16） |

**精度部署對應：**

| Source | 推薦 deploy precision | 對應 hybrid 模式 |
|---|---|---|
| FP32 source（44 個） | INT8 PTQ（CNN）/ BF16（transformer）/ FP16（混合） | INT8×8 / BFP16 / FP16 |
| FP16 source（7 個） | 原生 FP16 部署 | FP16 (E5M10) |
| INT8 / UINT8（9 個） | 原生 INT8 部署 | INT8×8 |
| INT16（1 個） | 原生 INT16 部署 | INT16×16 |

**結論**：全 61 個模型可在 MDLA7 上部署。需要的全部 op 都已在 §3A.6 (CONV) / §3A.7 (EWE/POOL) / §3A.9 (UDMA) 涵蓋；不適合 NPU 的 op（NMS / TopK / 等）由 RISC-V host 做（§3A.11）。

---

## 4. Dataflow（典型 layer 執行流程，推測）

1. **Command Engine** 收到 NN graph layer，發 config 給：
   - UDMA：把這層的 weight / activation tile 從 DRAM 拉到 L1Mesh SRAM。
   - CONV / Requant / EWE / POOL：依 layer 類別啟動。
2. **CONV Engine** 從 L1_Manager 讀 ACT + WGT，做 MAC，把 partial sum 寫回 L1Mesh。
3. **Requant Engine** 讀 INT32 partial sum，做 scale/shift，寫 INT8 結果回 L1Mesh。
4. **EWE Engine**：殘差連接 / activation function。
5. **POOL Engine**：spatial downsample。
6. **TNPS Engine**：軸重排 / transpose（attention Q·Kᵀ 之間、layout 轉換）。
7. **UDMA** 在背景把上一個 tile 的結果寫回 DRAM、預取下一個 tile（double-buffer `[TBD]`）。

> 5 個 compute engine 之間的次序、是否能 chain 為 fused layer、partial sum 是否常駐 L1Mesh — 全部 `[TBD]`。

---

## 5. SystemC Modelling 規劃（草稿）

預設用 **TLM-2.0 LT (loosely-timed)** 起手，必要時針對 CONV / L1_Manager 升級到 AT (approximately-timed)。

### 5.1 模組對應

| HW Block | SystemC 模組 | TLM 角色 |
|---|---|---|
| Host (RISC-V) | `sc_module Host` | v0：descriptor stream 產生器 stub（讀測試 input → 推 descriptor）；v1+：可換成 RISC-V ISS（如 Spike / etiss / RVV-capable model） |
| Command Engine | `sc_module CommandEngine` | target（對 Host）+ initiator（對 Engine config socket） |
| CONV / Requant / EWE / POOL | `sc_module XxxEngine` | initiator（對 L1_Manager）+ target（對 Command Engine config） |
| L1_Manager | `sc_module L1Manager` | multi-port target / arbiter |
| L1Mesh SRAM | `sc_module L1Mesh` | target（plain memory） |
| UDMA | `sc_module UDMA` | initiator 雙向；內部含 5 個 op-mode handler（v0 只實作 LINEAR + STRIDED_2D；GATHER/CONCAT/SLICE 留 v1） |
| DRAM | `sc_module DRAM` | target（含 latency model） |

### 5.2 介面選擇

- **Host → Command Engine（白線, 1T descriptor）**：v0 用 `sc_fifo<descriptor_t>`，host 把 descriptor `write()` 進去，CmdEngine 用 `read()` 取出。v1 改成 `simple_initiator_socket`（AXI-Lite stub）後可掛 RISC-V ISS。
- **Command Engine → Engine config（白線, 1T payload）**：用 `sc_signal<descriptor_t>` 或自定 `sc_fifo<>`(depth=1)，1 cycle latch 即可，**不需要 TLM socket**。
- **Engine ↔ L1_Manager（藍線, AXI 128b × N）**：`tlm_utils::simple_initiator_socket` / `simple_target_socket`，AXI-like payload；N 條並列在 v0 可先合併成「N×16B / cycle」單條 socket，等 contention 模型上線再展開。
- **CONV → Requant chain**：`std::array<sc_fifo<int32_t>, 16> chain;`，每 lane depth=2；CONV 內部 `Cluster` 子模組各自 push、Requant 內部 `RequantLane` 子模組各自 read，**繞過 L1_Manager**。
- **L1_Manager 內部 arbiter**：用 `sc_event_queue` 或 round-robin 排程 `[TBD]`。
- **DRAM latency**：用 `wait()` 模 fixed latency，第一版不模 DDR controller。

### 5.3 Cycle accuracy

- **v0**：functional + transaction count（只算 op 對不對，不準 cycle）。
- **v1**：CONV Engine 上 cycle model（**bit-mult invariant 版**）
  - 物理 throughput：**~1,048,576 bit-mult / cycle**（bit-level 守恆）
  - INT layer cycle 數 = ⌈ MAC_total × bit_a × bit_b / 1,048,576 ⌉ + tile fill
  - **FP layer cycle 數**：用 cluster 12–15 的 4096 MAC ⇒ ⌈ MAC_total / 4096 ⌉ + tile fill
  - 等價寫法：`cycle = ⌈ MAC_total / MAC_per_cycle(dtype) ⌉ + tile fill`，其中 `MAC_per_cycle` 查 §3A.2 表
  - INT8×8 → 16384、INT8×4 → 32768、INT16×16 → 4096、FP* → 4096
  - tile fill = 64 cycle（systolic 從注入到第一筆輸出）`[TBD]`
  - 1.9 GHz 換算：wall time = cycle 數 / 1.9 GHz ≈ cycle 數 × 0.526 ns
  - **SystemC 介面**：CONV Engine config descriptor 必須帶 `dtype` + `output_channel_group_id` (per cluster) 欄位，模型查表決定 cycle 數
- **v2**：L1_Manager arbitration + bandwidth contention
  - 80 條 R + 8 條 W 對 L1Mesh 的 16 條 R 收斂時的 stall 模型
  - 4 個 Engine 同時想存取 L1Mesh 的衝突
- **v3**：DRAM controller（page hit/miss、refresh）`[TBD]`，視 UDMA 是否成 bottleneck 再決定要不要做。

---

## 6. 待補 Spec 清單（請使用者填）

下列項目決定後，spec.md 會升級到 v1。

### 6.1 系統參數
- [x] ~~介面寬度 `32 / 16 / 8` 是 byte width / channel / 其他？~~ → **是 AXI 128b bus 的條數（暫定值，後續會評估）**
- [x] ~~系統時脈頻率？~~ → **1.9 GHz**（v8.25 起；原 1 GHz baseline 已升至 1.9 GHz 對齊 LPDDR5X-10667 BW）
- [x] ~~CONV throughput target？~~ → **62.3 TOPS @ INT8**（由 16384 MAC × 2 × 1.9 GHz 推得）
- [ ] 介面條數的最終值何時凍結？哪些是會調整的熱點？

### 6.2 CONV Engine
- [x] ~~PE array 形狀~~ → **65,536 base 4×4 cell，組成 16 cluster**（見 §3A.4）
- [x] ~~CONV ↔ L1_Manager 介面~~ → **只有 32 AXI_R (ACT) + 32 AXI_R (WGT)，無 AXI_W**
- [x] ~~支援的 datatype~~ → **Hybrid INT + Hybrid FP**（見 §3A.2）
- [x] ~~CONV → Requant 的 chain 路徑~~ → **16 條 INT32 per-cluster lane，1T 同步**（見 §3A.5）
- [x] ~~"ring" 拓樸 / 4 ring 工作切分軸~~ → **改成 16 cluster + reduce tree，無 ring**；output channel 切分跨 cluster（每 cluster 對應 16 個 output channel group）
- [x] ~~Bit-decomposable MAC cell 的微結構~~ → **BitFusion-style 4×4 base cell，可融合**
- [x] ~~FP 子集 4096 MAC 的位置~~ → **dedicated FP slice：cluster 12–15 共 4 個，每 cluster 1024 FP MAC**
- [x] ~~Accumulator 寬度~~ → **INT48**（v0 SystemC 用 `int64_t` 簡化，v1 切 `sc_int<48>` 嚴格 saturation）
- [x] ~~FP / INT 並行性~~ → **Layer-serial only**（不混跑；scheduler 簡化）
- [x] ~~支援 kernel / stride / dilation / padding / group~~ → **見 §3A.6**（K∈{1,2,3,5,7}, K 上限 11, S∈{1,2,4}, D∈{1,2,4}, P∈[0,7], group 含 depthwise）
- [ ] **Dataflow** 細節（OS 已預設，但 ACT 沿哪邊注入？WGT 怎麼預載？）
- [ ] **Tile fill latency** = 64 cycle 是否正確？對應 4-bit 視角還是 8-bit 視角？
- [x] ~~Kernel 上限~~ → **11×11 保留**
- [x] ~~Dilation 上限~~ → **降到 4**（{1, 2, 4}）
- [ ] Group conv 上限是否 ≥ 32（含 depthwise = C_in）？
- [ ] Padding 是否只支援 zero-pad？需 reflect / replicate 嗎？
- [ ] Depthwise + C_in 不是 16 倍數時，host compiler pad-up 還是 array 內部處理？

### 6.3 其他 Engine
- [ ] Requant：scale 是 per-tensor / per-channel？INT32 → INT8 路徑？（CONV chain 那條已是 per-channel；獨立 Requant op 待確認）
- [x] ~~EWE op list~~ → **全收**（arithmetic / activation / math / compare / bit；見 §3A.7）
- [x] ~~POOL 模式~~ → **max / avg / global**；avg 帶 `count_include_pad` flag；無 dilation；見 §3A.7

### 6.4 Memory
- [ ] L1Mesh 2M SRAM 的 bank 切法、port 數
- [x] ~~DRAM 規格~~ → **LPDDR5X-10667**（JEDEC LPDDR5X，10.667 Gbps/pin）
- [x] ~~channel 數 / bus width~~ → **dual-channel x32**，peak BW **85.3 GB/s**（@ 1.9 GHz core ≈ 44.9 B/cyc → 模型用 48 B/cyc）
- [ ] L1_Manager 的 arbitration policy

### 6.5 Command / Programming model
- [x] ~~Command 格式~~ → **64 byte fixed descriptor，16+48 layout**（見 §3A.10）
- [x] ~~host ↔ Command Engine 介面~~ → **DRAM ring buffer + MMIO doorbell**（見 §3A.10）
- [x] ~~Memory map~~ → **Unified 32-bit addr space**：L1Mesh 0x0000_0000–0x001F_FFFF；DRAM 0x1000_0000–0xFFFF_FFFF（見 §3A.10）
- [ ] CSR 4 KB AXI-Lite slave 的詳細 register map（tail_ptr / head_ptr / status / IRQ_clear 各 register offset）
- [ ] Ring buffer head/tail wrap-around / overflow policy

### 6.6 同步
- [x] ~~Engine 之間同步機制~~ → **Dependency tag**（每筆 task descriptor 帶 wait-list + signal-tag；見 §3A.8）
- [x] ~~Command Engine 支援 dependency graph~~ → **是**（CmdEngine 內含 dependency tracker，依 tag 狀態 dispatch）

### 6.7 Host (RISC-V)
- [x] ~~Host CPU 種類？~~ → **RISC-V**
- [ ] RISC-V 規格：core 數、ISA extension（RV64GC？M/A/F/D/C？V (RVV) for SIMD？）
- [ ] SoC 拓樸：NPU 是 RISC-V 的 close-coupled accelerator（同 die、共用 cache）還是 loose-coupled（PCIe / 系統 bus）？
- [ ] Host ↔ Command Engine 介面：AXI-Lite MMIO / mailbox / shared queue in DRAM？
- [ ] Host 是否直接擁有 DRAM AXI master（與 UDMA 共享 4G DRAM）？
- [ ] Interrupt 路徑（NPU done → RISC-V）：哪條 IRQ line？是否需 PLIC 模型？
- [ ] SystemC 端要 ISS 還是 stub？
  - v0：stub（從 JSON / 測試檔讀 descriptor 推進 fifo）
  - v1+：可考慮接 Spike / etiss / SystemC-RISC-V model

---

## 附錄 A — 與 Edge TPU / TPU v4 對照（推測）

| 項目 | MDLA7（推測） | Edge TPU | 備註 |
|---|---|---|---|
| 控制 | Command Engine | Scalar Core / VLIW | TPU 用 VLIW，MDLA7 似為 descriptor |
| 主算 | CONV Engine | MXU (128×128 systolic) | MDLA7 PE 規模 `[TBD]` |
| 後處理 | Requant + EWE + POOL | Vector Unit (VPU) | TPU 把這些合在 VPU，MDLA7 拆成 3 個 |
| L1 | 2 MB L1Mesh | VMEM / CMEM | 容量同等級 |
| DMA | UDMA | HBM controller | MDLA7 用 DRAM 而非 HBM |

---

*v0 — 待使用者補 detail spec 後改寫為 v1*
