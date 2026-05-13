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
all_ops = []
unsupported_ops = []
for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    name = _opcode_name(fb, model, op)
    if name in SUPPORTED_OPS:
        all_ops.append((i, name, op))
    else:
        unsupported_ops.append((i, name, op))
```

unsupported op 不能再安靜消失。compiler 會把 unsupported rows 印進 compile
log，regression wrapper 會把 `compile-skipped:N` 視為 failure。若要讓 corpus
clean，該 op 必須補成 native lowering，或補成明確的 `matrlz`
supported-but-not-native fallback。

```text
unsupported=0 代表沒有原始 TFLite op 被跳過。
matrlz>0 代表有 supported-but-not-native coverage boundary。
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

Hotspot、BMM、ETHZ path 都可能有 `matrlz` fallback：

```text
layer  4  matrlz  in=1x1x77  k=1x1  s=1x1  g=1  out=1x1x77  (77 INT8)  ready
```

`matrlz` 是 ready layer，不是 skipped。它代表 compiler 把該 op 的
reference tensor 預先 materialize，simulator 再用分塊 UDMA
`DRAM -> L1 -> DRAM` copy 來保留 profile/verification coverage。常見來源：

- non-spatial `MEAN` axes。
- `BATCH_MATMUL` / attention score fallback。
- runtime-matmul `FULLY_CONNECTED`。
- INT `GELU` / `HARD_SWISH` / `LOGISTIC` fallback。
- attention reshape shape-prop mismatch。
- descriptor `uint16_t` dim overflow。

---

## 6.15 常見誤解

| 誤解 | 正確理解 |
|---|---|
| TFLite Interpreter 是 compiler 唯一入口 | MDLA7 compiler 主要用 FlatBuffer 讀 graph metadata |
| unsupported op 會變成 no-op descriptor | unsupported row 會出現在 compile log；ETHZ/BMM regression 會 fail |
| `matrlz` 等於 native engine | `matrlz` 是 supported-but-not-native correctness boundary |
| HWC 一定等於原始 TFLite rank | HWC 是 simulator canonical form，高 rank 會簡化 |
| SAME padding 是固定 pad 1 | padding 依 input/output/kernel/stride 計算 |
| stride 任意值都支援 | descriptor 目前只支援 1/2/4/8 |
| chain mode 等於真實 TFLite full inference | 它是 self-consistent reference chain，不是完整 runtime fidelity |
| skip op 後還能安全沿用前一層 output | skip 會 break chain；clean corpus 不應有 skipped original op |

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
