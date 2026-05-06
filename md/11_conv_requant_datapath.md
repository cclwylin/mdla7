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
  push psum into chain[oc % 16]
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
      chain[oc & 0xF].write(saturate_i32(sum))
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
lane = oc & 0xF
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

---

## 11.14 Cycle overlap

CONV：

```text
engine time ~= max(L1 read elapsed, conv_cycles)
```

Requant：

```text
engine time ~= max(L1 write elapsed, ceil(elements / 256))
```

這是對 pipeline overlap 的近似。

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
  -> chain[oc % 16]
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
