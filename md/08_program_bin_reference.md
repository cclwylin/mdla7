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
