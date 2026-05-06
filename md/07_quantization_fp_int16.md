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
