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
    |   16-lane chain
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
CONV pushes int32 or FP32 psum to chain[oc % 16]
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
- 把 partial sum 寫入 `chain_out[oc & 0xF]`。
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
      chain[oc % 16].write(sum)
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

- 從 16-lane chain 讀 CONV partial sum。
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
      lane = oc & 0xF
      psum = chain_in[lane]->read()
```

這必須和 CONV push order 完全一致。

chain lane 選擇：

```text
lane = oc % 16
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
static constexpr uint64_t LANES = 256;
```

註解說明這是把 requant 視為融合到 CONV cluster output stage 的效果。functional 仍然透過 chain FIFO，但 timing 上用 256 elem/cycle。

cycle：

```text
pipe = ceil(total_elements / 256)
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
per_pass = ceil(elems / 64)
cycles = 3 * per_pass
```

其中 `EWE_LANES = 64`。

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
| ADD / MUL / SUB | `ceil(elems / 64)` |
| HARD_SWISH / GELU | `ceil(elems / 64)` |
| SOFTMAX | `3 * ceil(elems / 64)` |

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
per_lane = ceil(out_elems / 16)
cycles = per_lane * max(k_h * k_w, 1)
```

也就是 16 lanes，每個 output element 要看 `k_h*k_w` 個 input。

這對大 kernel global average pool 很敏感：

```text
global avgpool 8x8
k_h*k_w = 64
cycles = ceil(out_elems/16) * 64
```

所以某些模型 tail 的 pooling 可能不是完全免費。

---

## 5.31 TNPS 在目前 notebook 的位置

Spec 裡提到 TNPS engine，目標是處理 transpose / reshape 類 tensor processing。就目前 source 導讀而言，主要可執行 path 集中在：

- UDMA 的 `DEPTH_TO_SPACE`
- UDMA 的 `STRIDED_SLICE`
- UDMA 的 `SCATTER_CONCAT`
- EWE / POOL 的 tensor op

所以本章先把 TNPS 視為 architecture roadmap，不把它列為已完整實作的 compute engine。

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
| chain lanes | `oc & 0xF` |
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
