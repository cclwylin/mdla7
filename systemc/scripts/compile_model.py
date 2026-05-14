#!/usr/bin/env python3
"""Compile a .tflite into an MDLA7 descriptor-stream "program" (program.bin).

The program describes one full Mdla7System invocation:
  Host → CommandEngine → (UDMA load wgt, UDMA load act, CONV, UDMA writeback) × N
Each layer is independent (synthetic int8 input, no inter-layer chaining yet),
but all layers run in a single sc_start so we exercise the real dispatch +
dependency-tag flow and get one global cycle count.

Blob layout (little-endian):

  Header (16 byte):
    uint32 magic = 'MDL7' (0x374C444D)
    uint32 version = 3 or 4
    uint32 num_layers
    uint32 data_offset           -- byte offset to start of data section

  LayerMeta[num_layers] (64 byte each for v3, 76 byte each for v4):
    uint16 in_h, in_w, in_c
    uint16 out_h, out_w, out_c
    uint8  k_h, k_w, s_h, s_w
    uint8  p_t, p_b, p_l, p_r
    uint32 dram_in,  dram_wgt,  dram_out   -- DRAM placement
    uint32 in_size,  wgt_size,  ref_size   -- bytes
    uint32/uint64 in_off, wgt_off, ref_off -- absolute file offsets
    uint32 _reserved[2]

  GraphMeta[num_layers] (32 byte each, v3):
    int32 input0_tensor, input1_tensor, output_tensor
    int32 producer0_layer, producer1_layer
    int32 first_consumer_layer, last_consumer_layer
    int32 consumer_count

  Data:
    inputs (concatenated)
    weights (concatenated)
    refs    (concatenated, INT32 each)
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


MAGIC, VERSION_V3, VERSION_V4 = 0x374C444D, 3, 4
HEADER_FMT     = "<IIII"                # 16 byte
HEADER_SIZE    = struct.calcsize(HEADER_FMT)
LAYER_FMT_V3   = "<HHHHHHBBBBBBBBIIIIIIIIIHHHh"   # 64 byte (last short = zp_in_eff)
LAYER_FMT_V4   = "<HHHHHHBBBBBBBBIIIIIIQQQHHHh"   # 76 byte, 64-bit file offsets
LAYER_SIZE_V3  = struct.calcsize(LAYER_FMT_V3)
LAYER_SIZE_V4  = struct.calcsize(LAYER_FMT_V4)
assert LAYER_SIZE_V3 == 64, f"LayerMeta v3 size mismatch: {LAYER_SIZE_V3}"
assert LAYER_SIZE_V4 == 76, f"LayerMeta v4 size mismatch: {LAYER_SIZE_V4}"
LAYER_SIZE     = LAYER_SIZE_V3
GRAPH_META_FMT  = "<iiiiiiii"           # 32 byte tensor-level producer/consumer sidecar
GRAPH_META_SIZE = struct.calcsize(GRAPH_META_FMT)
assert GRAPH_META_SIZE == 32, f"GraphMeta size mismatch: {GRAPH_META_SIZE}"
UINT32_MAX     = (1 << 32) - 1
FP_BINARY_BCAST_MAGIC = 0x46424357  # "WCBF", FP binary compact-broadcast marker
FP_BINARY_BCAST_SCALAR = 1
FP_BINARY_BCAST_MIN_BYTES = 1 << 20

def _pack_tnps_meta(rank, elem_size, in_shape, out_shape, a_vals, b_vals=None):
    def pad_u(vals):
        vals = [int(v) & 0xFFFFFFFF for v in list(vals)[:6]]
        return vals + [1] * (6 - len(vals))
    def pad_i(vals):
        vals = [int(v) & 0xFFFFFFFF for v in list(vals or [])[:6]]
        return vals + [0] * (6 - len(vals))
    words = [int(rank), int(elem_size)]
    words += pad_u(in_shape)
    words += pad_u(out_shape)
    words += pad_i(a_vals)
    words += pad_i(b_vals)
    return struct.pack("<26I", *words)

def _fp_binary_params(act_min: np.float32, act_max: np.float32,
                      compact_mode: int = 0, compact_count: int = 0) -> bytes:
    meta = b"\x00" * 40
    if compact_mode:
        meta = (
            struct.pack("<III", FP_BINARY_BCAST_MAGIC, int(compact_mode), int(compact_count))
            + b"\x00" * 28
        )
    return struct.pack("<ff", float(act_min), float(act_max)) + meta

def _fp_binary_b_payload(in_b_arr: np.ndarray, params_b: bytes) -> bytes:
    b = np.asarray(in_b_arr, dtype=np.float16)
    full_b = b.tobytes(order="C")
    if len(full_b) >= FP_BINARY_BCAST_MIN_BYTES and b.size > 0:
        first = b.reshape(-1)[0]
        if np.all(b == first):
            params_b = _fp_binary_params(
                np.frombuffer(params_b[:4], dtype="<f4")[0],
                np.frombuffer(params_b[4:8], dtype="<f4")[0],
                FP_BINARY_BCAST_SCALAR,
                1,
            )
            return np.asarray([first], dtype=np.float16).tobytes(order="C") + params_b
    return full_b + params_b

# op_kind enum — must mirror C++ in mdla7/program_image.h
OP_CONV       = 0
OP_DWCONV     = 1
OP_AVG_POOL   = 2
OP_MAX_POOL   = 3
OP_SOFTMAX    = 4
OP_RESHAPE    = 5
OP_FC         = 6         # FC mapped to 1x1 conv but displayed separately
OP_ADD        = 7         # element-wise binary add (residual / SE)
OP_CONCAT     = 8         # channel concat (Inception/DenseNet branch merge)
OP_GATHER     = 9         # indexed lookup (BERT embeddings, audio mel-bins)
OP_MUL        = 10        # v8.30: element-wise binary mul (mobilenet_v3 SE gate)
OP_SUB        = 11        # v8.30: element-wise binary sub (transformer attention)
OP_HARD_SWISH = 12        # v8.30: x * relu6(x+3) / 6 (mobilenet_v3, 21 in fp16)
OP_GELU       = 13        # v8.30: tanh-approx GELU (transformer activations)
OP_D2SPACE    = 14        # v8.32: DEPTH_TO_SPACE / pixel shuffle, UDMA layout op
OP_MATERIALIZE = 15       # compiler fallback: pre-materialized reference bytes
OP_TRANSPOSE  = 16
OP_S2SPACE    = 17
OP_SQUEEZE    = 18
OP_EXPAND_DIMS = 19
OP_SLICE      = 20
OP_STRIDED_SLICE = 21
OP_PAD        = 22
OP_PACK       = 23
OP_UNPACK     = 24
OP_TILE       = 25
OP_SPLIT      = 26
OP_LOGISTIC   = 27       # v10: sigmoid unary EWE (EfficientNet swish gate)
OP_RSQRT      = 28       # 1/sqrt(x) INT8 LUT EWE (LayerNorm/RMSNorm normaliser)
OP_TANH       = 29       # tanh(x)   INT8 LUT EWE (RNN/transformer activation)
OP_FC_BMM     = 30       # BATCH_MATMUL lowered to 1x1 CONV; same engine as OP_FC
OP_SHAPE      = 31       # TFLite SHAPE: compile-time constant shape vector
OP_REVERSE    = 32       # TFLite REVERSE_V2: compile-time pre-flipped bytes (UDMA load)

# dtype enum — must mirror DType in include/mdla7/descriptor.h
DT_INT8x4   = 0
DT_INT8x8   = 1
DT_INT16x4  = 2
DT_INT16x8  = 3
DT_INT16x16 = 4
DT_FP8      = 8
DT_FP16     = 9
DT_BFP16    = 10

OP_NAME = {OP_CONV:"   conv", OP_DWCONV:" dwconv",
           OP_AVG_POOL:"avgpool", OP_MAX_POOL:"maxpool",
           OP_SOFTMAX:"softmax", OP_RESHAPE:"reshape",
           OP_FC:        "     fc",
           OP_ADD:       "    add",
           OP_CONCAT:    " concat",
           OP_GATHER:    " gather",
           OP_MUL:       "    mul",
           OP_SUB:       "    sub",
           OP_HARD_SWISH:"h_swsh",
           OP_GELU:      "   gelu",
           OP_D2SPACE:   "d2spac",
           OP_MATERIALIZE:"matrlz",
           OP_TRANSPOSE: " trnps",
           OP_S2SPACE:   "s2spac",
           OP_SQUEEZE:   "squeez",
           OP_EXPAND_DIMS:"expand",
           OP_SLICE:     " slice",
           OP_STRIDED_SLICE:"sslice",
           OP_PAD:       "   pad",
           OP_PACK:      "  pack",
           OP_UNPACK:    "unpack",
           OP_TILE:      "  tile",
           OP_SPLIT:     " split",
           OP_LOGISTIC:  "logist",
           OP_RSQRT:     " rsqrt",
           OP_TANH:      "  tanh",
           OP_FC_BMM:    "fc(bmm)",
           OP_SHAPE:     "  shape",
           OP_REVERSE:   "reverse"}


def _load_interpreter(path: str):
    """Legacy path used by single-layer extract_conv.py — still uses TFLite
    Interpreter because that codepath only needs tensors, not op options."""
    try:
        import tflite_runtime.interpreter as tflr
        return tflr.Interpreter(model_path=path)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)
    except ImportError:
        import platform, sys as _sys
        py = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
        cmd = (f"{py} -m pip install --user tensorflow"
               if platform.machine() == "arm64"
               else f"{py} -m pip install --user tflite-runtime")
        raise SystemExit(f"compile_model: TFLite missing.\n  install with:  {cmd}")


# ---- v1.1: real flatbuffer parsing (gets us actual op options) -----------

def _load_flatbuffer(path: str):
    """Return (model, subgraph) from raw .tflite flatbuffer."""
    try:
        import tflite as fb
    except ImportError:
        raise SystemExit("compile_model: missing 'tflite' package.\n"
                         "  install with:  pip install tflite")
    with open(path, "rb") as f:
        buf = f.read()
    model = fb.Model.GetRootAsModel(buf, 0)
    return fb, model, model.Subgraphs(0)


def _opcode_name(fb, model, op):
    code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    return next((k for k, v in fb.BuiltinOperator.__dict__.items() if v == code),
                f"opcode_{code}")


def _tensor_shape(t):
    return tuple(int(t.Shape(i)) for i in range(t.ShapeLength()))


def _tensor_buffer_bytes(model, t):
    buf = model.Buffers(t.Buffer())
    if buf.DataLength() == 0:
        return None
    arr = buf.DataAsNumpy()
    return bytes(arr)


def _padding_to_pad(fb, padding_enum, K, in_dim, out_dim, stride):
    """Resolve TF Lite's SAME/VALID enum into explicit (top, bottom) values."""
    if padding_enum == fb.Padding.VALID:
        return 0, 0
    # SAME: total pad = max(0, (out - 1) * stride + K - in)
    total = max(0, (out_dim - 1) * stride + K - in_dim)
    return total // 2, total - total // 2


SUPPORTED_OPS = ("CONV_2D", "DEPTHWISE_CONV_2D",
                 "FULLY_CONNECTED",
                 "AVERAGE_POOL_2D", "MAX_POOL_2D",
                 "SOFTMAX", "RESHAPE",
                 "ADD", "CONCATENATION", "GATHER",
                 # v8.30
                 "MUL", "SUB", "HARD_SWISH", "GELU", "MEAN",
                 "LOGISTIC",
                 "QUANTIZE", "CAST", "RELU", "MINIMUM", "GREATER",
                 "LEAKY_RELU", "PRELU", "TANH", "RSQRT",
                 "SQUARED_DIFFERENCE", "SUM",
                 "BATCH_MATMUL",
                 "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
                 "TRANSPOSE_CONV",
                 "DEPTH_TO_SPACE",
                 "SPACE_TO_DEPTH", "TRANSPOSE",
                 "SQUEEZE", "EXPAND_DIMS",
                 "SLICE", "STRIDED_SLICE",
                 "SPLIT", "SPLIT_V",
                 "PAD", "PADV2", "PACK", "UNPACK", "TILE",
                 # v11: graph-helper ops needed by qwen35_attention; emitted as
                 # materialize fallbacks so compile no longer skips them.
                 "SHAPE", "REVERSE_V2", "RANDOM_STANDARD_NORMAL")

# Supported-by-materialization means the compiler emits an explicit reference
# byte boundary (`matrlz`) rather than a native MDLA7 arithmetic/data-movement
# engine layer. Keep this set separate from SUPPORTED_OPS so corpus audits can
# distinguish "no skipped unsupported op" from "native datapath implemented".
MATERIALIZED_FALLBACK_OPS = frozenset((
    "QUANTIZE", "CAST", "RELU", "MINIMUM", "GREATER",
    "LEAKY_RELU", "PRELU",
    "SQUARED_DIFFERENCE", "SUM", "BATCH_MATMUL",
    "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
    "TRANSPOSE_CONV",
    # v11: qwen35 graph helpers. REVERSE_V2 flips along given axes;
    # RANDOM_STANDARD_NORMAL is materialized deterministically from a fixed
    # seed so reference bytes are reproducible across compile runs.
    # NOTE: SHAPE is no longer a fallback — it is lowered to OK_SHAPE (L1
    # constant-load) and handled before this block.
    "REVERSE_V2", "RANDOM_STANDARD_NORMAL",
))

# FP variants of these ops have native EWE/unary support below. Quantized/int
# variants route through a 256-byte INT8 LUT in the EWE engine; the empty set
# below is intentional now that HARD_SWISH/GELU INT8 are LUT-lowered too.
DTYPE_MATERIALIZED_FALLBACK_OPS = frozenset()

# Native FP CONV coverage exceptions found by full ETHZ correctness runs.
# These become explicit materialized boundaries instead of pretending the
# current simulator datapath is bit-exact for the layer.
FORCED_MATERIALIZED_NATIVE = {
    ("inception_v3_float", 2): "fp conv stem mismatch in fast/cx",
    ("inception_v3_float", 5): "fp conv stem mismatch in fast/cx",
    ("unet_quant", 8): "int conv mismatch in fast/cx",
    ("unet_quant", 14): "int conv mismatch in fast/cx",
    ("unet_float", 1): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 2): "fp encoder maxpool mismatch in fast/cx",
    ("unet_float", 3): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 4): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 5): "fp encoder maxpool mismatch in fast/cx",
    ("unet_float", 6): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 7): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 8): "fp encoder maxpool mismatch in fast/cx",
    ("unet_float", 9): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 10): "fp encoder conv mismatch in fast/cx",
    ("unet_float", 11): "fp encoder maxpool mismatch in fast/cx",
    ("unet_float", 12): "fp bottleneck conv mismatch in fast/cx",
    ("unet_float", 13): "fp bottleneck conv mismatch in fast/cx",
    ("imdn_float", 1): "fp space-to-depth mismatch in fast/cx",
    ("dped_float", 1): "fp space-to-depth mismatch in fast/cx",
    ("efficientnet_b4_float", 473): "fp avgpool mismatch in fast/cx",
    ("efficientnet_b4_float", 474): "fp fc mismatch in fast/cx",
    ("yolo_v8_quant", 207): "int conv branch input mismatch in fast",
}


def list_supported_ops(interp):
    fn = interp._get_ops_details if hasattr(interp, "_get_ops_details") \
         else interp.get_ops_details
    return [op for op in fn() if op["op_name"] in SUPPORTED_OPS]


# Compatibility — older callers (extract_conv.py) used list_conv_ops.
def list_conv_ops(interp):
    fn = interp._get_ops_details if hasattr(interp, "_get_ops_details") \
         else interp.get_ops_details
    return [op for op in fn() if op["op_name"] == "CONV_2D"]


def quantize_multiplier(scale: float):
    """TFLite QuantizeMultiplier: scale → (mult_q31, shift). int32 mult, int shift.
    shift > 0 means effective_scale > 1 (left shift first); shift <= 0 right shift.

    v7: TFLite C++ uses std::round (round-half-away-from-zero); Python's
    built-in round() is banker's (round-half-to-even). The two disagree
    only on exact halves but in practice scale-quantization values land
    at-or-very-near halves often enough that this caused 1-LSB output
    differences. m is in [0.5, 1) so always positive; floor(m*2^31 + 0.5)
    gives the desired rounding."""
    if scale <= 0:
        return 0, 0
    import math
    m, e = math.frexp(scale)
    q = int(math.floor(m * (1 << 31) + 0.5))
    if q == (1 << 31):
        q //= 2; e += 1
    return q, e


def saturating_doubling_high_mul_np(a, b):
    """vectorized version of TFLite's SaturatingDoublingHighMul."""
    x = a.astype(np.int64) * np.int64(b)
    r = (x + (1 << 30)) >> 31
    return np.clip(r, -(1 << 31), (1 << 31) - 1).astype(np.int32)


def rounding_divide_by_pot_np(x, exponent):
    # v8.30: upcast to int64 internally — for SUB on tensors with very
    # asymmetric scale ratios (e.g. sam_quant's int8 SUB whose mq_b shift
    # produces right_shift=33), the int32 mask overflows. int64 covers all
    # plausible shifts (QuantizeMultiplier emits exponents bounded by float
    # range). Clamp at 62 to stay safely below int64 width.
    if exponent <= 0:
        return x
    e = min(int(exponent), 62)
    x64 = x.astype(np.int64)
    mask = (1 << e) - 1
    remainder = x64 & mask
    threshold = (mask >> 1) + np.where(x64 < 0, 1, 0).astype(np.int64)
    out64 = (x64 >> e) + np.where(remainder > threshold, 1, 0).astype(np.int64)
    return out64.astype(np.int32)


def multiply_by_quantized_multiplier_np(x, mult, shift):
    """Per-element shift array allowed (mult/shift can be scalars or arrays)."""
    left  = np.where(shift > 0,  shift, 0).astype(np.int32)
    right = np.where(shift > 0,  0,    -shift).astype(np.int32)
    shifted = x << left
    high = saturating_doubling_high_mul_np(shifted, mult)
    # rounding_divide_by_pot with possibly per-element exponent
    out = high.copy()
    for r in np.unique(right):
        if r == 0: continue
        sel = right == r
        out = np.where(sel, rounding_divide_by_pot_np(high, int(r)), out)
    return out


def conv_int8_ref(act_i8, wgt_i8, s_h, s_w, pad, group=1, pad_value=0):
    """Group-aware reference. wgt shape: [OC, Kh, Kw, in_per_group].
    int64 accumulator (spec §3A.1: INT48); narrowed to int32 with saturation
    so the requant input matches what the sim's chain carries.

    v7: pad_value (= zp_in_eff for asymmetric int8 input) replaces the
    previous "skip OOB" behaviour, so the bias_eff fold matches TFLite at
    boundaries even when zp_in != 0.

    v8.25: vectorised over (oh, ow) — old Python `for oh: for ow:` triple
    loop dispatched OH*OW*Kh*Kw small einsums (~58M for xlsr_quant's
    360×640 conv layers, ~10 min compile time). Now we loop only (kh, kw)
    and emit one big einsum per kernel position over the full OH×OW grid.
    np.pad with constant_values=pad_value handles boundaries (no manual
    OOB skip / pad_w_per_kk dance). Order-independent (INT add is
    associative) so this stays bit-identical to the old reference."""
    H, W, Cin = act_i8.shape
    OC, K_h, K_w, in_per_group = wgt_i8.shape
    out_per_group = OC // group
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - K_h) // s_h + 1
    OW = (W + pL + pR - K_w) // s_w + 1
    pad_const = int(pad_value)
    a_pad = np.pad(act_i8.astype(np.int64),
                   ((pT, pB), (pL, pR), (0, 0)),
                   mode="constant", constant_values=pad_const)
    w = wgt_i8.astype(np.int64).reshape(group, out_per_group, K_h, K_w, in_per_group)
    out = np.zeros((OH, OW, group, out_per_group), dtype=np.int64)
    for kh in range(K_h):
        for kw in range(K_w):
            # Strided view of (OH, OW, Cin) for kernel position (kh, kw).
            in_slice = a_pad[kh:kh + OH * s_h:s_h, kw:kw + OW * s_w:s_w, :]
            in_slice = in_slice.reshape(OH, OW, group, in_per_group)
            # (OH, OW, G, Ig) × (G, Opg, Ig) → (OH, OW, G, Opg).
            out += np.einsum("hwgi,gci->hwgc", in_slice, w[:, :, kh, kw, :])
    out = out.reshape(OH, OW, OC)
    return np.clip(out, -(1 << 31), (1 << 31) - 1).astype(np.int32)


def _to_hwc(shape):
    sh = list(shape)
    if sh and sh[0] == 1 and len(sh) >= 2:
        sh = sh[1:]
    while len(sh) < 3:
        sh = [1] + sh
    if len(sh) > 3:
        sh = sh[-3:]
    return tuple(sh)


def _infer_shapes(fb, model, sg):
    """v8: walk the graph in op order, computing each tensor's (H, W, C) shape
    from input shape + op semantics. TFLite often stores placeholder static
    shapes ([1,1,1,C] etc.) for FP16-quantized models; propagating from the
    real model input recovers the correct dimensions."""
    shapes = {}
    for i in range(sg.InputsLength()):
        idx = sg.Inputs(i)
        shapes[idx] = _to_hwc(_tensor_shape(sg.Tensors(idx)))

    def _conv_out_dims(H, W, Kh, Kw, s_h, s_w, pad_enum):
        if pad_enum == fb.Padding.SAME:
            return (H + s_h - 1) // s_h, (W + s_w - 1) // s_w
        return max(0, (H - Kh)) // s_h + 1, max(0, (W - Kw)) // s_w + 1

    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        op_name = _opcode_name(fb, model, op)
        in_idx = op.Inputs(0) if op.InputsLength() > 0 else -1
        in_shape = shapes.get(in_idx)
        if in_shape is None:
            t = sg.Tensors(in_idx) if in_idx >= 0 else None
            in_shape = _to_hwc(_tensor_shape(t)) if t is not None else (1, 1, 1)
        H, W, C = in_shape
        out_shape = (H, W, C)        # default: shape-preserving
        opt_table = op.BuiltinOptions()

        try:
            if op_name == "CONV_2D":
                co = fb.Conv2DOptions(); co.Init(opt_table.Bytes, opt_table.Pos)
                wgt_sh = list(_tensor_shape(sg.Tensors(op.Inputs(1))))   # [OC, Kh, Kw, Cin]
                if len(wgt_sh) == 4:
                    OC, Kh, Kw, _ = wgt_sh
                    OH, OW = _conv_out_dims(H, W, Kh, Kw,
                                            co.StrideH(), co.StrideW(), co.Padding())
                    out_shape = (OH, OW, OC)
            elif op_name == "DEPTHWISE_CONV_2D":
                co = fb.DepthwiseConv2DOptions(); co.Init(opt_table.Bytes, opt_table.Pos)
                wgt_sh = list(_tensor_shape(sg.Tensors(op.Inputs(1))))   # [1, Kh, Kw, OC]
                if len(wgt_sh) == 4:
                    _, Kh, Kw, dw_OC = wgt_sh
                    mult = int(co.DepthMultiplier()) or max(1, dw_OC // max(1, C))
                    OC = C * mult
                    OH, OW = _conv_out_dims(H, W, Kh, Kw,
                                            co.StrideH(), co.StrideW(), co.Padding())
                    out_shape = (OH, OW, OC)
            elif op_name in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
                po = fb.Pool2DOptions(); po.Init(opt_table.Bytes, opt_table.Pos)
                OH, OW = _conv_out_dims(H, W, po.FilterHeight(), po.FilterWidth(),
                                        po.StrideH(), po.StrideW(), po.Padding())
                out_shape = (OH, OW, C)
            elif op_name == "MEAN":
                out_shape = (1, 1, C)         # global mean (mobilenet_v3 uses this)
            elif op_name == "FULLY_CONNECTED":
                wgt_sh = list(_tensor_shape(sg.Tensors(op.Inputs(1))))   # [OC, in]
                if len(wgt_sh) >= 1:
                    out_shape = (1, 1, wgt_sh[0])
            elif op_name == "RESHAPE":
                out_shape = _to_hwc(_tensor_shape(sg.Tensors(op.Outputs(0))))
            elif op_name == "DEPTH_TO_SPACE":
                try:
                    d2s = fb.DepthToSpaceOptions(); d2s.Init(opt_table.Bytes, opt_table.Pos)
                    b = int(d2s.BlockSize())
                except Exception:
                    osh = _to_hwc(_tensor_shape(sg.Tensors(op.Outputs(0))))
                    b = max(1, osh[0] // max(H, 1))
                out_shape = (H * b, W * b, C // (b * b))
            elif op_name == "CONCATENATION":
                # CONCAT may happen on non-NHWC-channel axes in sequence models
                # (e.g. LSTM rank-2 [1,500]+[1,512] -> [1,1012]). The op is a
                # byte-preserving copy in our reduced program, so trust the
                # TFLite output tensor's canonical HWC shape instead of trying
                # to infer the axis in the collapsed HWC domain.
                out_shape = _to_hwc(_tensor_shape(sg.Tensors(op.Outputs(0))))
            # else (ADD/MUL/SUB/HARD_SWISH/etc.): preserve input shape.
        except Exception:
            pass

        for k in range(op.OutputsLength()):
            shapes[op.Outputs(k)] = out_shape
    return shapes


def _find_op_producer(fb, model, sg, tensor_idx):
    """Return (op, op_name) of whatever op writes `tensor_idx`, or (None, None)."""
    for j in range(sg.OperatorsLength()):
        op = sg.Operators(j)
        for k in range(op.OutputsLength()):
            if op.Outputs(k) == tensor_idx:
                return op, _opcode_name(fb, model, op)
    return None, None


def conv_fp_ref(act_f, wgt_f, s_h, s_w, pad, group, bias, act_min, act_max):
    """FP32 conv reference matching the sim's compute_fp reduction order
    bit-for-bit. Each output element accumulates in (kh, kw, icr) nested order
    via Kh*Kw*in_per_group element-wise IEEE 754 FP32 multiply-then-add passes
    — no BLAS/einsum pairwise reduction. Combined with the sim's same loop
    structure, this lets sim and ref produce identical FP32 values when fed
    identical FP16 inputs (necessary for fused FP chains to stay byte-exact).
    Bias add and activation clamp also mirror sim: clamp uses the same
    +/-3.4e38 sentinels packed into the params blob."""
    H, W, Cin = act_f.shape
    OC, Kh, Kw, in_per_group = wgt_f.shape
    out_per_group = OC // group
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - Kh) // s_h + 1
    OW = (W + pL + pR - Kw) // s_w + 1
    a_pad = np.pad(act_f, ((pT, pB), (pL, pR), (0, 0)), mode="constant").astype(np.float32)
    w  = wgt_f.astype(np.float32).reshape(group, out_per_group, Kh, Kw, in_per_group)
    out = np.zeros((OH, OW, group, out_per_group), dtype=np.float32)
    for kh in range(Kh):
        for kw in range(Kw):
            in_slice = a_pad[kh:kh + OH * s_h:s_h, kw:kw + OW * s_w:s_w, :]
            in_slice = in_slice.reshape(OH, OW, group, in_per_group)
            for icr in range(in_per_group):
                in_one = in_slice[:, :, :, icr]              # (OH, OW, G)
                w_one  = w[:, :, kh, kw, icr]                # (G, Opg)
                # Element-wise FP32 mul + add into running sum — same per-element
                # operation order the sim does inside compute_fp's inner loop.
                out += in_one[:, :, :, None] * w_one[None, None, :, :]
    out = out.reshape(OH, OW, OC)
    out += bias.reshape(1, 1, -1).astype(np.float32)
    # Match sim: it always clips with the sentinel from the params blob, even
    # when the original act_min/act_max were +/-inf.  So apply the same finite
    # sentinels here unconditionally rather than gating on isfinite.
    amin_sent = np.float32(-3.4e38 if not np.isfinite(act_min) else float(act_min))
    amax_sent = np.float32( 3.4e38 if not np.isfinite(act_max) else float(act_max))
    out = np.maximum(out, amin_sent)
    out = np.minimum(out, amax_sent)
    return out.astype(np.float32)


def _conv_window_sum(in_arr, Kh, Kw, s_h, s_w, pad, OH, OW, in_per_group,
                     pad_value=0):
    """v7: per-pixel input sum for a non-DW conv (pad-aware).
    Returns int64 array of shape [OH, OW]. OOB kernel positions contribute
    `pad_value * in_per_group` so the sum mirrors what the CONV engine
    produces with the corresponding `in_pad_value` set."""
    H, W, Cin = in_arr.shape
    pT, pB, pL, pR = pad
    a = in_arr.astype(np.int64)
    out = np.zeros((OH, OW), dtype=np.int64)
    pv = int(pad_value)
    pv_per_kk = pv * in_per_group   # OOB kernel-position contribution
    for kh in range(Kh):
        for kw in range(Kw):
            for oh in range(OH):
                ih = oh * s_h + kh - pT
                ih_ok = (0 <= ih < H)
                for ow in range(OW):
                    iw = ow * s_w + kw - pL
                    if ih_ok and (0 <= iw < W):
                        out[oh, ow] += a[ih, iw, :in_per_group].sum()
                    elif pv != 0:
                        out[oh, ow] += pv_per_kk
    return out


def _conv_window_sum_dw(in_arr, Kh, Kw, s_h, s_w, pad, OH, OW, pad_value=0):
    """v7: per-(oh, ow, oc) input sum for a DW conv (group == Cin), pad-aware."""
    H, W, Cin = in_arr.shape
    pT, pB, pL, pR = pad
    a = in_arr.astype(np.int64)
    out = np.zeros((OH, OW, Cin), dtype=np.int64)
    pv = int(pad_value)
    for kh in range(Kh):
        for kw in range(Kw):
            for oh in range(OH):
                ih = oh * s_h + kh - pT
                ih_ok = (0 <= ih < H)
                for ow in range(OW):
                    iw = ow * s_w + kw - pL
                    if ih_ok and (0 <= iw < W):
                        out[oh, ow, :] += a[ih, iw, :]
                    elif pv != 0:
                        out[oh, ow, :] += pv
    return out


def _decode_padding(opts, Kh, Kw):
    pad = opts.get("padding", "SAME")
    if pad in ("SAME", b"SAME", 0):
        pT = (Kh - 1) // 2; pB = (Kh - 1) - pT
        pL = (Kw - 1) // 2; pR = (Kw - 1) - pL
        return pT, pB, pL, pR
    return 0, 0, 0, 0


# LUT for v1 SOFTMAX — must stay byte-identical with C++ copy in
# include/mdla7/softmax_lut.h.  Generated as round(exp(-i/32) * 16384).
SOFTMAX_LUT = np.array([
    16384, 15880, 15391, 14918, 14459, 14014, 13583, 13165,
    12760, 12367, 11987, 11618, 11261, 10914, 10578, 10253,
     9937,  9632,  9335,  9048,  8770,  8500,  8238,  7985,
     7739,  7501,  7270,  7047,  6830,  6620,  6416,  6219,
     6027,  5842,  5662,  5488,  5319,  5155,  4997,  4843,
     4694,  4550,  4410,  4274,  4143,  4015,  3892,  3772,
     3656,  3543,  3434,  3329,  3226,  3127,  3031,  2937,
     2847,  2760,  2675,  2592,  2513,  2435,  2360,  2288,
     2217,  2149,  2083,  2019,  1957,  1897,  1838,  1782,
     1727,  1674,  1622,  1572,  1524,  1477,  1432,  1388,
     1345,  1304,  1263,  1225,  1187,  1150,  1115,  1081,
     1047,  1015,   984,   954,   924,   896,   868,   842,
      816,   791,   766,   743,   720,   698,   676,   655,
      635,   616,   597,   578,   561,   543,   527,   510,
      495,   480,   465,   450,   437,   423,   410,   398,
      385,   373,   362,   351,   340,   330,   319,   310,
      300,   291,   282,   273,   265,   257,   249,   241,
      234,   227,   220,   213,   206,   200,   194,   188,
      182,   176,   171,   166,   161,   156,   151,   146,
      142,   137,   133,   129,   125,   121,   118,   114,
      110,   107,   104,   101,    97,    94,    92,    89,
       86,    83,    81,    78,    76,    74,    71,    69,
       67,    65,    63,    61,    59,    57,    56,    54,
       52,    51,    49,    47,    46,    45,    43,    42,
       41,    39,    38,    37,    36,    35,    34,    33,
       32,    31,    30,    29,    28,    27,    26,    25,
       25,    24,    23,    22,    22,    21,    20,    20,
       19,    19,    18,    17,    17,    16,    16,    15,
       15,    14,    14,    14,    13,    13,    12,    12,
       12,    11,    11,    11,    10,    10,    10,     9,
        9,     9,     9,     8,     8,     8,     8,     7,
        7,     7,     7,     6,     6,     6,     6,     6,
], dtype=np.int32)


def softmax_int8_ref(logits_i8):
    """Deterministic INT8 softmax over the last axis.

    Must match mdla7_model_runner.cpp's row-wise EWE dispatch: each H×W position is
    sent to softmax_int8() as one contiguous C-vector.
    """
    x = logits_i8.astype(np.int32)
    max_v = np.max(x, axis=-1, keepdims=True)
    diff = np.clip(max_v - x, 0, 255)
    exp_q = SOFTMAX_LUT[diff]
    sum_q = np.sum(exp_q, axis=-1, keepdims=True).astype(np.int64)
    sum_q = np.maximum(sum_q, 1)
    out = (exp_q.astype(np.int64) * 127) // sum_q
    return np.clip(out, 0, 127).astype(np.int8)


def softmax_int8_decomp_ref(logits_i8, zp_in, zp_out=-128):
    """v13: INT8 softmax via the descriptor-level decomposition chain
    (ES_DEQUANT_INT8 → POOL_MAX → EWE_SUB → EWE_EXP → POOL_SUM → EWE_DIV →
    ES_QUANT_FP_INT8). Must match EweEngine + PoolEngine FP paths when
    MDLA7_DECOMPOSE_SOFTMAX=1. FP16 storage between every sub-op; FP32
    internal compute. EXP uses libm `expf` (via ctypes) for bit-parity with
    C++ `std::exp(float)` — same pattern as softmax_fp_ref.

    NOT bit-equal to softmax_int8_ref (the LUT path); selected only when the
    runner emits the decomp chain. TFLite softmax output convention: zp=-128,
    scale=1/256.
    """
    expf = _libm_expf()
    flat = logits_i8.reshape(-1, logits_i8.shape[-1])
    rows, K = flat.shape
    # Phase 1 dequant — INT8 -> FP16 (vectorised: int32 subtract -> fp32 -> fp16).
    dq16 = (flat.astype(np.int32) - int(zp_in)).astype(np.float32).astype(np.float16)
    out_i8 = np.empty_like(flat)
    for r in range(rows):
        row32 = dq16[r].astype(np.float32)
        # Phase 2 POOL_MAX (FP16 storage, FP32 compute).
        mx16 = np.float16(row32.max())
        mx32 = np.float32(mx16)
        # Phase 3 EWE_SUB (broadcast row max, vector op).
        ctr16 = (row32 - mx32).astype(np.float16)
        ctr32 = ctr16.astype(np.float32)
        # Phase 4 EWE_EXP — libm expf if loadable for bit-parity with std::exp.
        if expf:
            exp32 = np.empty(K, dtype=np.float32)
            for k in range(K):
                exp32[k] = expf(float(ctr32[k]))
        else:
            exp32 = np.exp(ctr32).astype(np.float32)
        exp16 = exp32.astype(np.float16)
        # Phase 5 POOL_SUM (sequential FP32 running add — engine adds kw=0..K-1
        # in order so reduction order matters for bit-parity).
        s32 = np.float32(0.0)
        for v in exp16:
            s32 = np.float32(s32 + np.float32(v))
        if s32 == np.float32(0.0):
            s32 = np.float32(1.0)
        sum32 = np.float32(np.float16(s32))
        # Phase 6 EWE_DIV (broadcast row sum, vector op).
        div16 = (exp16.astype(np.float32) / sum32).astype(np.float16)
        # Phase 7 ES_QUANT_FP_INT8 (round to nearest even via np.rint).
        q = np.rint(div16.astype(np.float32) * np.float32(256.0)).astype(np.int32) + int(zp_out)
        out_i8[r] = np.clip(q, -128, 127).astype(np.int8)
    return out_i8.reshape(logits_i8.shape)


def _libm_expf():
    """Cache a ctypes handle for libm's float expf(float) — used by
    softmax_fp_ref to match C++ std::exp(float) bit-for-bit. numpy.exp on
    float32 differs by 1 ULP for certain inputs (e.g. exp(3.14), exp(-2.7))
    likely due to SIMD intrinsic vs libm scalar implementation differences."""
    if getattr(_libm_expf, "_cached", None) is None:
        import ctypes, ctypes.util
        libm_path = ctypes.util.find_library("m")
        libm = ctypes.CDLL(libm_path) if libm_path else None
        if libm is not None and hasattr(libm, "expf"):
            libm.expf.argtypes = [ctypes.c_float]
            libm.expf.restype  = ctypes.c_float
            _libm_expf._cached = libm.expf
        else:
            _libm_expf._cached = False
    return _libm_expf._cached


def _libm_tanhf():
    """Cache libm tanhf(float) for byte-identical FP GELU reference."""
    if getattr(_libm_tanhf, "_cached", None) is None:
        import ctypes, ctypes.util
        libm_path = ctypes.util.find_library("m")
        libm = ctypes.CDLL(libm_path) if libm_path else None
        if libm is not None and hasattr(libm, "tanhf"):
            libm.tanhf.argtypes = [ctypes.c_float]
            libm.tanhf.restype  = ctypes.c_float
            _libm_tanhf._cached = libm.tanhf
        else:
            _libm_tanhf._cached = False
    return _libm_tanhf._cached


def softmax_fp_ref(logits_f16):
    """v8.28 / v8.30: FP softmax reference. Must match EweEngine::run_softmax_fp
    byte-for-byte. Runs over the last axis, one contiguous C-vector at a time.
    Standard 3-pass numerically-stable form: subtract max, expf, sequential
    running-add sum, divide, cast to FP16.

    v8.30: route through libm's `expf` via ctypes instead of `np.exp` so
    every element-wise exp() result is bit-identical to C++ `std::exp(float)`.
    numpy.exp(float32) disagrees with libm expf by 1 ULP on certain inputs
    (vector intrinsic vs scalar libm), which used to cause swin_float
    softmax layers to FAIL by 1-3 bytes out of 14k-115k.
    """
    expf  = _libm_expf()
    rows = logits_f16.reshape(-1, logits_f16.shape[-1]).astype(np.float32)
    out = np.empty(rows.shape, dtype=np.float16)
    for row_idx, row in enumerate(rows):
        max_v = np.float32(row.max())
        diff = (row - max_v).astype(np.float32)
        if expf:
            exp_v = np.empty(diff.size, dtype=np.float32)
            for i in range(diff.size):
                exp_v[i] = expf(float(diff[i]))
        else:
            # Fallback when libm isn't loadable (rare; bytes may diverge by 1
            # ULP on a small fraction of large softmax tensors).
            exp_v = np.exp(diff).astype(np.float32)
        # Sequential sum (matches sim's running-add accumulator order).
        s = np.float32(0.0)
        for v in exp_v:
            s = np.float32(s + v)
        if s == 0.0:
            s = np.float32(1.0)
        out[row_idx] = (exp_v / s).astype(np.float16)
    return out.reshape(logits_f16.shape)


def pool_int8_ref(in_i8, k_h, k_w, s_h, s_w, pad, mode, count_include_pad):
    """Reference avg/max pool — must match POOL engine implementation byte-for-byte."""
    H, W, C = in_i8.shape
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - k_h) // s_h + 1
    OW = (W + pL + pR - k_w) // s_w + 1
    out_dtype = np.int16 if in_i8.dtype == np.int16 else np.int8
    min_v = -32768 if out_dtype == np.int16 else -128
    max_v =  32767 if out_dtype == np.int16 else  127
    out = np.zeros((OH, OW, C), dtype=out_dtype)
    for oh in range(OH):
        for ow in range(OW):
            for c in range(C):
                if mode == OP_MAX_POOL:
                    best = min_v
                    for kh in range(k_h):
                        ih = oh * s_h + kh - pT
                        if not (0 <= ih < H): continue
                        for kw in range(k_w):
                            iw = ow * s_w + kw - pL
                            if not (0 <= iw < W): continue
                            v = int(in_i8[ih, iw, c])
                            if v > best: best = v
                    out[oh, ow, c] = best
                else:  # AVG_POOL
                    s, n = 0, 0
                    for kh in range(k_h):
                        ih = oh * s_h + kh - pT
                        if not (0 <= ih < H): continue
                        for kw in range(k_w):
                            iw = ow * s_w + kw - pL
                            if not (0 <= iw < W): continue
                            s += int(in_i8[ih, iw, c]); n += 1
                    div = (k_h * k_w) if count_include_pad else max(n, 1)
                    # round-to-nearest, half away from zero
                    if s >= 0:  q = (s + div // 2) // div
                    else:       q = -((-s + div // 2) // div)
                    if q > max_v: q = max_v
                    if q < min_v: q = min_v
                    out[oh, ow, c] = q
    return out


def pool_fp_ref(in_f16, k_h, k_w, s_h, s_w, pad, mode, count_include_pad):
    """v8.17: FP avg/max pool reference matching PoolEngine::run_pool_fp
    bit-for-bit. AVG sums in (kh outer, kw inner) order via element-wise FP32
    add into a running scalar — same loop structure the sim uses, so values
    come out identical when ref and sim see the same FP16 input."""
    H, W, C = in_f16.shape
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - k_h) // s_h + 1
    OW = (W + pL + pR - k_w) // s_w + 1
    in_f32 = in_f16.astype(np.float32)
    if mode == OP_MAX_POOL:
        out = np.full((OH, OW, C), np.float32(-3.4e38), dtype=np.float32)
        for kh in range(k_h):
            for kw in range(k_w):
                # Tile views with bounds clipping. We iterate (oh, ow) only over
                # positions whose kernel cell is in-bounds — preserves "skip OOB"
                # semantics without padding the input.
                for oh in range(OH):
                    ih = oh * s_h + kh - pT
                    if not (0 <= ih < H): continue
                    for ow in range(OW):
                        iw = ow * s_w + kw - pL
                        if not (0 <= iw < W): continue
                        v = in_f32[ih, iw, :]
                        out[oh, ow, :] = np.maximum(out[oh, ow, :], v)
        return out.astype(np.float16)
    # AVG path
    sums   = np.zeros((OH, OW, C), dtype=np.float32)
    counts = np.zeros((OH, OW),    dtype=np.int32)
    for kh in range(k_h):
        for kw in range(k_w):
            for oh in range(OH):
                ih = oh * s_h + kh - pT
                if not (0 <= ih < H): continue
                for ow in range(OW):
                    iw = ow * s_w + kw - pL
                    if not (0 <= iw < W): continue
                    sums[oh, ow, :] += in_f32[ih, iw, :]
                    counts[oh, ow] += 1
    if count_include_pad:
        divs = np.full((OH, OW), k_h * k_w, dtype=np.int32)
    else:
        divs = np.where(counts > 0, counts, 1)
    out = sums / divs[..., None].astype(np.float32)
    return out.astype(np.float16)


def main():
    here = Path(__file__).resolve().parent.parent
    repo = here.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?",
                    default=str(repo / "model/INT8/efficientnet_lite0_int8.tflite"))
    ap.add_argument("output", nargs="?",
                    default=str(here / "build/program.bin"))
    ap.add_argument("--max-layers", type=int, default=0,
                    help="cap number of CONV_2D layers (0 = all)")
    args = ap.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    fb, model, sg = _load_flatbuffer(args.model)
    model_stem = Path(args.model).stem

    # v8: propagate tensor shapes through the graph so FP-quant models with
    # placeholder [1,1,1,C] static shapes get real (H, W, C) at every op.
    shape_dict = _infer_shapes(fb, model, sg)

    producer_op_by_tensor = {}
    consumers_by_tensor = {}
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        for k in range(op.OutputsLength()):
            tidx = int(op.Outputs(k))
            if tidx >= 0:
                producer_op_by_tensor[tidx] = i
        for k in range(op.InputsLength()):
            tidx = int(op.Inputs(k))
            if tidx >= 0:
                consumers_by_tensor.setdefault(tidx, []).append(i)

    # Collect ops in execution order. Keep unsupported ops visible in the
    # compile log instead of silently dropping them; run_systemc.py surfaces
    # these rows in the per-model HTML report.
    all_ops = []
    unsupported_ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        name = _opcode_name(fb, model, op)
        if name in SUPPORTED_OPS:
            all_ops.append((i, name, op))
        else:
            unsupported_ops.append((i, name, op))
    ops = all_ops
    if args.max_layers:
        ops = ops[: args.max_layers]
    if not ops:
        sys.exit("no supported op found in model")

    print(f"compile_model: {args.model}")
    op_counts = {}
    for _, name, _ in ops: op_counts[name] = op_counts.get(name, 0) + 1
    print(f"  {len(ops)} ops: " + ", ".join(f"{k}={v}" for k, v in op_counts.items()))
    if unsupported_ops:
        unsupported_counts = {}
        for _, name, _ in unsupported_ops:
            unsupported_counts[name] = unsupported_counts.get(name, 0) + 1
        print(
            "compile_model: unsupported ops skipped: " +
            ", ".join(f"{k}={v}" for k, v in unsupported_counts.items())
        )
        for orig_op_index, opname, op in unsupported_ops:
            in_idx = int(op.Inputs(0)) if op.InputsLength() > 0 else -1
            in_shape = shape_dict.get(in_idx)
            if in_shape is None and in_idx >= 0:
                in_shape = _to_hwc(_tensor_shape(sg.Tensors(in_idx)))
            if in_shape is None:
                in_shape = (1, 1, 1)
            H, W, C = (int(v) for v in in_shape)
            print(
                f"  layer {orig_op_index:>2d} {opname:<8s} "
                f"in={H}x{W}x{C} skipped (unsupported op)"
            )

    # DRAM bump allocator: weights → inputs → outputs in disjoint regions.
    # v8.23: region bases are placeholders here; the real DRAM_IN / DRAM_OUT
    # offsets are recomputed AFTER the layer loop so each region is sized to
    # the actual cumulative payload (not a hardcoded +64MB stride). Hardcoded
    # offsets used to silently overlap on big-tensor models — e.g. deeplab's
    # input tensors total >64MB, so layer 8's `dram_in` ran into the start of
    # the DRAM_OUT region and was overwritten by layer 0's `udma_w`. The
    # placeholder values below let the per-layer dict carry an
    # in-region offset (cur_w / cur_i / cur_o); patch_dram_addrs below
    # rewrites them to non-overlapping absolutes once totals are known.
    # Keep DRAM above the L1/L1Mesh address space while leaving as much of the
    # 32-bit descriptor address range as possible for huge FP image models.
    DRAM_BASE = 0x00300000
    DRAM_WGT  = DRAM_BASE + 0x00000000
    DRAM_IN   = DRAM_BASE + 0x04000000     # placeholder (resized post-loop)
    DRAM_OUT  = DRAM_BASE + 0x08000000     # placeholder (resized post-loop)
    cur_w = cur_i = cur_o = 0
    in_off = wgt_off = ref_off = 0

    layers = []
    op_to_compiled = {}
    in_blobs, wgt_blobs, ref_blobs = [], [], []

    # DRAM addresses are still descriptor uint32_t values. File offsets stay
    # v3/uint32 for normal programs, but very large complete programs can be
    # emitted as v4 with uint64 offsets instead of silently dropping tail layers.
    REGION_ALIGN = 64 * 1024
    def _round_up(x, a):
        return (x + a - 1) & ~(a - 1)

    def _program_budget_error(next_in_b, next_wgt_b, next_ref_b):
        cand_i = cur_i + next_in_b
        cand_w = cur_w + next_wgt_b
        cand_o = cur_o + next_ref_b
        cand_layers = len(layers) + 1
        dram_end = DRAM_BASE + _round_up(cand_w, REGION_ALIGN) \
                 + _round_up(cand_i, REGION_ALIGN) + cand_o
        if dram_end - 1 > UINT32_MAX:
            return (f"DRAM end 0x{dram_end - 1:08x} exceeds uint32 address "
                    f"limit")
        return None

    # v8.12: chain mode — when layer N+1's expected input shape & dtype match
    # layer N's reference output, reuse N's output as N+1's input (instead of a
    # fresh rng draw).  This makes the per-layer rng-synth align with what the
    # actual model would feed forward, AND lets mdla7_model_runner.cpp detect fusable
    # adjacent pairs and skip layer N+1's udma_r (the input is already in L1
    # at layer N's L1_OUT slot).  Models where the chain breaks (skipped FP
    # ADD / HARD_SWISH / etc.) just get a fresh rng draw at the break.
    rng = np.random.default_rng(0)
    last_output_arr = None
    last_output_tensor = None
    tensor_values = {}
    tensor_input_cache = {}
    DTYPE_MAP = {
        fb.TensorType.INT8:    np.int8,
        fb.TensorType.UINT8:   np.uint8,
        fb.TensorType.INT16:   np.int16,
        fb.TensorType.INT32:   np.int32,
        fb.TensorType.FLOAT16: np.float16,
        fb.TensorType.FLOAT32: np.float32,
    }
    # TFLite tensor type → MDLA7 dtype enum (int8 / int16 / fp paths).
    TYPE_TO_DTYPE = {
        fb.TensorType.INT8:    DT_INT8x8,
        fb.TensorType.UINT8:   DT_INT8x8,
        fb.TensorType.INT16:   DT_INT16x16,
        fb.TensorType.FLOAT16: DT_FP16,
        fb.TensorType.FLOAT32: DT_FP16,    # spec policy: FP32 source lowered to FP16 deployment
    }
    FP_TFLITE_TYPES = {fb.TensorType.FLOAT16, fb.TensorType.FLOAT32}

    def _elem_size_for_layer_dtype(dtype):
        return 2 if dtype in (DT_INT16x16, DT_FP16, DT_BFP16, DT_FP8) else 1

    def _storage_dtype_for_tensor(t):
        if t.Type() in FP_TFLITE_TYPES:
            return np.float16
        if t.Type() == fb.TensorType.INT16:
            return np.int16
        if t.Type() == fb.TensorType.INT32:
            return np.int32
        if t.Type() == fb.TensorType.BOOL:
            return np.int8
        return np.int8

    def _pack_hwc_for_elems(elems):
        """Pack an element count into descriptor-safe HWC dimensions."""
        elems = int(max(1, elems))
        dim_max = 65535
        if elems <= dim_max:
            return (1, 1, elems)
        for h in range(min(dim_max, int(np.sqrt(elems)) + 1), 0, -1):
            if elems % h:
                continue
            rem = elems // h
            if rem <= dim_max:
                return (1, h, rem)
            for w in range(min(dim_max, int(np.sqrt(rem)) + 1), 0, -1):
                if rem % w == 0 and rem // w <= dim_max:
                    return (h, w, rem // w)
        # Fallback that should cover only prime-ish huge tensors; keeping the
        # product exact matters more than preserving logical axes.
        h = min(dim_max, elems)
        rem = (elems + h - 1) // h
        w = min(dim_max, rem)
        c = (elems + h * w - 1) // (h * w)
        return (h, w, min(dim_max, c))

    def _synth_output_array(out_tensor, out_h, out_w, out_c):
        dt = _storage_dtype_for_tensor(out_tensor)
        shape = (max(1, int(out_h)), max(1, int(out_w)), max(1, int(out_c)))
        if dt == np.float16:
            return (rng.standard_normal(shape) * 0.5).astype(np.float16)
        if dt == np.int16:
            return rng.integers(-128, 128, size=shape, dtype=np.int16)
        return rng.integers(-8, 8, size=shape, dtype=np.int8)

    def _input_array_for_tensor(tensor_idx, shape, dtype):
        """Return stable synthetic bytes for a graph tensor input."""
        shape = tuple(int(x) for x in shape)
        dtype = np.dtype(dtype)
        if tensor_idx is not None and tensor_idx >= 0:
            t = sg.Tensors(int(tensor_idx))
            const = _tensor_array(t)
            if const is not None:
                if t.Type() == fb.TensorType.UINT8:
                    const = (const.astype(np.int16) - 128).astype(np.int8)
                elif t.Type() in FP_TFLITE_TYPES:
                    const = const.astype(np.float16)
                elif t.Type() == fb.TensorType.BOOL:
                    const = const.astype(np.int8)
                const = np.asarray(const, dtype=dtype)
                if const.shape == shape:
                    return const
                if const.size == 1:
                    return np.broadcast_to(const.reshape(()), shape)
                if const.size == int(np.prod(shape)):
                    return const.reshape(shape)
                try:
                    return np.broadcast_to(const, shape)
                except ValueError:
                    pass
            arr = tensor_values.get(int(tensor_idx))
            if arr is not None and arr.size == int(np.prod(shape)) and arr.dtype == dtype:
                return arr.reshape(shape)
            key = (int(tensor_idx), shape, dtype.str)
            cached = tensor_input_cache.get(key)
            if cached is not None:
                return cached
        if dtype == np.float16:
            arr = (rng.standard_normal(shape) * 0.5).astype(np.float16)
        elif dtype == np.int16:
            arr = rng.integers(-128, 128, size=shape, dtype=np.int16)
        elif dtype == np.int32:
            arr = rng.integers(-8, 8, size=shape, dtype=np.int32)
        elif dtype == np.bool_:
            arr = rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.bool_)
        else:
            arr = rng.integers(-8, 8, size=shape, dtype=np.int8)
        if tensor_idx is not None and tensor_idx >= 0:
            tensor_input_cache[(int(tensor_idx), shape, dtype.str)] = arr
        return arr

    def _last_output_matches(tensor_idx, shape, dtype):
        if tensor_idx is None or tensor_idx < 0 or last_output_tensor != int(tensor_idx):
            return False
        if last_output_arr is None:
            return False
        return (last_output_arr.shape == tuple(int(x) for x in shape)
                and last_output_arr.dtype == np.dtype(dtype))

    def _input_or_last(tensor_idx, shape, dtype):
        if _last_output_matches(tensor_idx, shape, dtype):
            return last_output_arr
        return _input_array_for_tensor(tensor_idx, shape, dtype)

    def _tensor_array(t):
        b = _tensor_buffer_bytes(model, t)
        if b is None:
            return None
        npt = DTYPE_MAP.get(t.Type())
        if npt is None:
            return None
        arr = np.frombuffer(b, dtype=npt).copy()
        sh  = _tensor_shape(t)
        if sh: arr = arr.reshape(sh)
        return arr

    def _qscale(q, default=1.0):
        if not q or not q.ScaleLength():
            return float(default)
        v = float(q.Scale(0))
        return v if np.isfinite(v) and v > 0.0 else float(default)

    def _qzero(t):
        q = t.Quantization()
        return int(q.ZeroPoint(0)) if q and q.ZeroPointLength() else 0

    def _internal_to_numeric(arr, t):
        arr = np.asarray(arr)
        if t.Type() == fb.TensorType.UINT8:
            return arr.astype(np.int32) + 128
        if t.Type() == fb.TensorType.BOOL:
            return arr.astype(np.bool_)
        return arr

    def _internal_to_real(arr, t):
        arr = np.asarray(arr)
        if t.Type() in FP_TFLITE_TYPES:
            return arr.astype(np.float32)
        if t.Type() in (fb.TensorType.INT8, fb.TensorType.UINT8,
                        fb.TensorType.INT16):
            return (_internal_to_numeric(arr, t).astype(np.float32)
                    - np.float32(_qzero(t))) * np.float32(_qscale(t.Quantization()))
        return _internal_to_numeric(arr, t).astype(np.float32)

    def _quantize_real_to_tensor(real, t):
        real = np.asarray(real, dtype=np.float32)
        if t.Type() in FP_TFLITE_TYPES:
            return real.astype(np.float16)
        if t.Type() == fb.TensorType.BOOL:
            return real.astype(bool).astype(np.int8)
        scale = np.float32(_qscale(t.Quantization()))
        zp = np.float32(_qzero(t))
        q = np.floor(real / scale + zp + np.float32(0.5)).astype(np.int64)
        if t.Type() == fb.TensorType.UINT8:
            return (np.clip(q, 0, 255) - 128).astype(np.int8)
        if t.Type() == fb.TensorType.INT16:
            return np.clip(q, -32768, 32767).astype(np.int16)
        if t.Type() == fb.TensorType.INT32:
            return np.clip(np.rint(real), -(1 << 31), (1 << 31) - 1).astype(np.int32)
        return np.clip(q, -128, 127).astype(np.int8)

    def _cast_numeric_to_tensor(values, t):
        values = np.asarray(values)
        if t.Type() in FP_TFLITE_TYPES:
            return values.astype(np.float16)
        if t.Type() == fb.TensorType.BOOL:
            return values.astype(bool).astype(np.int8)
        if t.Type() == fb.TensorType.UINT8:
            return (np.clip(values.astype(np.int64), 0, 255) - 128).astype(np.int8)
        if t.Type() == fb.TensorType.INT16:
            return np.clip(values.astype(np.int64), -32768, 32767).astype(np.int16)
        if t.Type() == fb.TensorType.INT32:
            return np.clip(values.astype(np.int64), -(1 << 31), (1 << 31) - 1).astype(np.int32)
        return np.clip(values.astype(np.int64), -128, 127).astype(np.int8)

    def _tensor_operand_internal(tensor_idx, fallback_hwc, dtype=None):
        t = sg.Tensors(int(tensor_idx))
        arr = _tensor_array(t)
        if arr is None:
            dt = dtype if dtype is not None else _storage_dtype_for_tensor(t)
            arr = _input_array_for_tensor(int(tensor_idx), fallback_hwc, dt)
        else:
            if t.Type() == fb.TensorType.UINT8:
                arr = (arr.astype(np.int16) - 128).astype(np.int8)
            elif t.Type() in FP_TFLITE_TYPES:
                arr = arr.astype(np.float16)
            elif t.Type() == fb.TensorType.BOOL:
                arr = arr.astype(np.int8)
        return np.asarray(arr)

    def _reshape_or_broadcast(arr, hwc):
        arr = np.asarray(arr)
        if arr.shape == tuple(hwc):
            return arr
        if arr.size == 1:
            return np.broadcast_to(arr.reshape(()), tuple(hwc))
        if arr.size == int(np.prod(hwc)):
            return arr.reshape(tuple(hwc))
        return np.broadcast_to(arr, tuple(hwc))

    def _resize_nearest_ref(src, out_h, out_w, align_corners=False,
                            half_pixel_centers=False):
        src = np.asarray(src)
        if src.ndim == 3:
            src = src.reshape((1,) + src.shape)
        n, in_h, in_w, c = src.shape
        oy = np.arange(out_h, dtype=np.float32)
        ox = np.arange(out_w, dtype=np.float32)
        if align_corners and out_h > 1:
            y_idx = np.rint(oy * (in_h - 1) / (out_h - 1)).astype(np.int64)
        elif half_pixel_centers:
            y_idx = np.floor((oy + 0.5) * in_h / out_h).astype(np.int64)
        else:
            y_idx = np.floor(oy * in_h / out_h).astype(np.int64)
        if align_corners and out_w > 1:
            x_idx = np.rint(ox * (in_w - 1) / (out_w - 1)).astype(np.int64)
        elif half_pixel_centers:
            x_idx = np.floor((ox + 0.5) * in_w / out_w).astype(np.int64)
        else:
            x_idx = np.floor(ox * in_w / out_w).astype(np.int64)
        y_idx = np.clip(y_idx, 0, in_h - 1)
        x_idx = np.clip(x_idx, 0, in_w - 1)
        return src[:, y_idx, :, :][:, :, x_idx, :]

    def _resize_bilinear_ref(src, out_h, out_w, align_corners=False,
                             half_pixel_centers=False):
        src = np.asarray(src, dtype=np.float32)
        if src.ndim == 3:
            src = src.reshape((1,) + src.shape)
        n, in_h, in_w, c = src.shape
        def map_coord(o, in_size, out_size):
            if align_corners and out_size > 1:
                return o * (in_size - 1) / (out_size - 1)
            if half_pixel_centers:
                return (o + 0.5) * in_size / out_size - 0.5
            return o * in_size / out_size
        fy = map_coord(np.arange(out_h, dtype=np.float32), in_h, out_h)
        fx = map_coord(np.arange(out_w, dtype=np.float32), in_w, out_w)
        y0 = np.floor(np.maximum(fy, 0.0)).astype(np.int64)
        x0 = np.floor(np.maximum(fx, 0.0)).astype(np.int64)
        y0 = np.clip(y0, 0, in_h - 1)
        x0 = np.clip(x0, 0, in_w - 1)
        y1 = np.clip(y0 + 1, 0, in_h - 1)
        x1 = np.clip(x0 + 1, 0, in_w - 1)
        wy = np.where(fy >= 0.0, fy - y0.astype(np.float32), 0.0).astype(np.float32)
        wx = np.where(fx >= 0.0, fx - x0.astype(np.float32), 0.0).astype(np.float32)
        top = (src[:, y0, :, :][:, :, x0, :] * (1.0 - wx)[None, None, :, None] +
               src[:, y0, :, :][:, :, x1, :] * wx[None, None, :, None])
        bot = (src[:, y1, :, :][:, :, x0, :] * (1.0 - wx)[None, None, :, None] +
               src[:, y1, :, :][:, :, x1, :] * wx[None, None, :, None])
        return top * (1.0 - wy)[None, :, None, None] + bot * wy[None, :, None, None]

    for li, (orig_op_index, opname, op) in enumerate(ops):
        # TFLite SPLIT is encoded as SPLIT(axis, value); value is the data tensor.
        # Keep compiler chaining and GraphMeta centered on the real data input,
        # not the scalar axis input.
        primary_input_slot = 1 if opname == "SPLIT" and op.InputsLength() > 1 else 0
        if opname == "TRANSPOSE_CONV" and op.InputsLength() > 2:
            primary_input_slot = 2
        in_t   = sg.Tensors(op.Inputs(primary_input_slot))
        out_t  = sg.Tensors(op.Outputs(0))

        # ---- shape extraction ----
        ish = list(_tensor_shape(in_t))
        osh = list(_tensor_shape(out_t))
        # Canonicalise to NHWC: pad with 1s on the left if rank<4, else take last 3 spatial dims.
        # v8.30: rank-0 (scalar) and rank-1 tensors common in transformer
        # graphs (mobilebert MUL by scalar, embeddings) — synthesise the
        # missing axes with 1s rather than indexing past the end.
        def to_hwc(shape):
            if not shape:
                return (1, 1, 1)
            shape = shape if shape[0] == 1 else [1] + shape  # ensure batch dim
            shape = shape[1:]                                # drop batch
            while len(shape) < 3: shape = [1] + shape
            if len(shape) > 3: shape = shape[-3:]            # collapse extra dims to last 3
            return tuple(shape)
        # v8: prefer the propagated shape only when the .tflite static shape
        # looks like a placeholder ([1,1,*,C]). For normal INT8 models the
        # static shape is correct and trusted. This avoids regressions when
        # propagation gets confused by ops we don't model precisely.
        def _pick(static_hwc, prop_hwc):
            sH, sW, sC = static_hwc
            if prop_hwc is None: return static_hwc
            pH, pW, pC = prop_hwc
            if sH <= 1 and sW <= 1 and (pH > 1 or pW > 1):
                return prop_hwc
            return static_hwc
        def _tensor_original_array(tensor_idx):
            t = sg.Tensors(int(tensor_idx))
            sh = _tensor_shape(t)
            arr = tensor_values.get(int(tensor_idx))
            if arr is not None and int(np.asarray(arr).size) == int(np.prod(sh)):
                return np.asarray(arr).reshape(sh)
            fallback_hwc = to_hwc(list(sh))
            arr = _tensor_operand_internal(
                int(tensor_idx), fallback_hwc, _storage_dtype_for_tensor(t))
            if int(np.asarray(arr).size) == int(np.prod(sh)):
                return np.asarray(arr).reshape(sh)
            return np.asarray(arr)
        H, W, Cin   = _pick(to_hwc(ish), shape_dict.get(op.Inputs(0)))
        OH, OW, OC  = _pick(to_hwc(osh), shape_dict.get(op.Outputs(0)))

        # Defaults (overwritten per-op below)
        Kh = Kw = 1; s_h = s_w = 1
        pT = pB = pL = pR = 0
        group = 1
        op_kind = OP_CONV
        wgt = np.zeros((0,), dtype=np.int8)            # weights buffer (may be empty)
        layout_wgt_payload = None
        fc_label = False                                # only set true for FC
        is_fp_layer = False                             # v8: set true inside CONV/DWCONV FP path
        zp_in_eff = 0                                   # int default; FP path keeps it at 0

        # v4.1: pick element width from the input tensor's TFLite dtype.
        layer_dtype = TYPE_TO_DTYPE.get(in_t.Type(), DT_INT8x8)
        is_int16    = (layer_dtype == DT_INT16x16)
        is_fp_input = layer_dtype in (DT_FP16, DT_BFP16, DT_FP8)
        elem_size   = 2 if (is_int16 or is_fp_input) else 1
        np_in_dt    = _storage_dtype_for_tensor(in_t)

        # v9.3: int16 path is implemented for conv-class, pool, binary EWE,
        # reshape/concat byte movement, and spatial MEAN via AVG_POOL.
        int16_supported_ops = {
            "CONV_2D", "DEPTHWISE_CONV_2D",
            "ADD", "MUL", "SUB",
            "AVERAGE_POOL_2D", "MAX_POOL_2D", "MEAN",
            "RESHAPE", "CONCATENATION",
            "QUANTIZE", "CAST", "RELU", "MINIMUM", "GREATER",
            "LEAKY_RELU", "PRELU", "TANH", "RSQRT",
            "SQUARED_DIFFERENCE", "SUM", "BATCH_MATMUL",
            "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
            "TRANSPOSE_CONV",
            "HARD_SWISH", "GELU", "LOGISTIC",
        }
        if is_int16 and opname not in int16_supported_ops:
            print(f"  layer {li:>2d}  {opname.lower():>7s}  in={H}x{W}x{Cin} "
                  f"skipped (int16 {opname} not yet supported in v4.1)")
            continue
        # v8.17: FP path handles ADD (FP32 sum + clamp) and AVG/MAX POOL
        # (FP32 reduction in (kh,kw) order). v8.24: FP FULLY_CONNECTED routes
        # through the FP CONV path. v8.28: FP SOFTMAX added (numerically
        # stable 3-pass exp/sum/divide in FP32, FP16 storage). Remaining
        # FP-only-skip set is empty; RESHAPE/CONCAT/GATHER stay byte-passthrough.

        # v8.12: prefer the previous layer's reference output when its shape +
        # dtype match the current layer's expected input — enables L1-resident
        # fusion in mdla7_model_runner.cpp.  Falls back to fresh rng on first layer or
        # whenever the chain breaks (skipped op, shape mismatch).
        expected_dtype = np_in_dt   # FP conv/FC paths may override this later
        input0_idx = int(op.Inputs(primary_input_slot)) if op.InputsLength() > primary_input_slot else -1
        if (opname in ("SPACE_TO_DEPTH", "TRANSPOSE",
                       "SQUEEZE", "EXPAND_DIMS",
                       "SLICE", "STRIDED_SLICE", "SPLIT", "SPLIT_V",
                       "PAD", "PADV2", "PACK", "UNPACK", "TILE")
            and _last_output_matches(input0_idx, (H, W, Cin), expected_dtype)):
            H, W, Cin = last_output_arr.shape
        in_arr = _input_or_last(input0_idx, (H, W, Cin), expected_dtype)
        in_i8 = in_arr        # legacy name kept for downstream readability

        opt_table = op.BuiltinOptions()  # may be None for some ops

        force_materialize_native_reason = FORCED_MATERIALIZED_NATIVE.get((model_stem, li))
        materialize_op = (
            opname in MATERIALIZED_FALLBACK_OPS or
            (opname in DTYPE_MATERIALIZED_FALLBACK_OPS
             and in_t.Type() not in FP_TFLITE_TYPES)
        )
        # ---- v12 BATCH_MATMUL → FC-style CONV lowering -------------------------
        # BATCH_MATMUL(A[...,M,K], B[...,K,N]) == 1×1 CONV with H=M, W=1,
        # Cin=K, OC=N. Lower when ZP conditions allow exact int8 accumulation
        # (or always for FP16). Each head/batch slice becomes one OP_FC layer.
        if opname == "BATCH_MATMUL":
            _a_idx_bmm = int(op.Inputs(0)); _b_idx_bmm = int(op.Inputs(1))
            _a_t_bmm   = sg.Tensors(_a_idx_bmm); _b_t_bmm = sg.Tensors(_b_idx_bmm)
            _is_fp_bmm = _a_t_bmm.Type() in FP_TFLITE_TYPES

            def _bmm_can_lower(a_t, b_t):
                # v12 Phase 7: always lower INT8 BATCH_MATMUL to CONV; non-zero
                # ZPs are handled at runtime via per-sample activation-sum correction.
                if a_t.Type() in FP_TFLITE_TYPES:
                    return True
                return True  # INT8: removed zp_a/zp_b==0 guard

            _b_arr_raw = _tensor_original_array(_b_idx_bmm)
            if _bmm_can_lower(_a_t_bmm, _b_t_bmm) and _b_arr_raw is not None:
                # Read adjoint flags
                _adj_x = _adj_y = False
                if opt_table:
                    try:
                        _bo = fb.BatchMatMulOptions()
                        _bo.Init(opt_table.Bytes, opt_table.Pos)
                        _adj_x = bool(_bo.AdjX()); _adj_y = bool(_bo.AdjY())
                    except Exception:
                        pass
                # Get A from chain or raw tensor
                _a_arr_raw = _tensor_original_array(_a_idx_bmm)
                _a_arr_bmm = np.asarray(
                    in_arr if in_arr is not None else _a_arr_raw,
                    dtype=np.float16 if _is_fp_bmm else np.int8,
                )
                _b_arr_bmm = np.asarray(_b_arr_raw,
                                         dtype=np.float16 if _is_fp_bmm else np.int8)
                if _adj_x: _a_arr_bmm = np.swapaxes(_a_arr_bmm, -1, -2)
                if _adj_y: _b_arr_bmm = np.swapaxes(_b_arr_bmm, -1, -2)
                # Shape: a[...,M,K], b[...,K,N] — both may have batch leading dims
                # Broadcast contracting dim (K), mirrors the materialised path.
                _a_k = _a_arr_bmm.shape[-1]; _b_k = _b_arr_bmm.shape[-2]
                if _a_k != _b_k:
                    if _b_k == 1:
                        _tgt = list(_b_arr_bmm.shape); _tgt[-2] = _a_k
                        _b_arr_bmm = np.broadcast_to(_b_arr_bmm, _tgt).copy()
                    elif _a_k == 1:
                        _tgt = list(_a_arr_bmm.shape); _tgt[-1] = _b_k
                        _a_arr_bmm = np.broadcast_to(_a_arr_bmm, _tgt).copy()
                _a_flat_shape = _a_arr_bmm.shape
                _b_flat_shape = _b_arr_bmm.shape
                if _a_flat_shape[-1] != _b_flat_shape[-2]:
                    pass  # K mismatch — fall through to materialise
                else:
                    _M_bmm  = int(_a_flat_shape[-2])
                    _K_bmm  = int(_a_flat_shape[-1])
                    _N_bmm  = int(_b_flat_shape[-1])
                    # Compute total batch (head) count from remaining leading dims
                    _batch_a = _a_flat_shape[:-2]
                    _batch_b = _b_flat_shape[:-2]
                    # Broadcast batch dims following numpy matmul semantics
                    try:
                        _batch_out = np.broadcast_shapes(_batch_a, _batch_b)
                    except Exception:
                        _batch_out = _batch_a if _batch_a else _batch_b
                    _B_total = int(np.prod(_batch_out)) if _batch_out else 1
                    _a_3d = np.broadcast_to(
                        _a_arr_bmm.reshape(_batch_a + (_M_bmm, _K_bmm)),
                        _batch_out + (_M_bmm, _K_bmm),
                    ).reshape(_B_total, _M_bmm, _K_bmm)
                    _b_3d = np.broadcast_to(
                        _b_arr_bmm.reshape(_batch_b + (_K_bmm, _N_bmm)),
                        _batch_out + (_K_bmm, _N_bmm),
                    ).reshape(_B_total, _K_bmm, _N_bmm)
                    # Quantisation params for INT8 path
                    if not _is_fp_bmm:
                        _in_q_bmm  = in_t.Quantization()
                        _out_q_bmm = out_t.Quantization()
                        _b_t_q_bmm = _b_t_bmm.Quantization()
                        _sc_in  = _qscale(_in_q_bmm)
                        _sc_out = _qscale(_out_q_bmm)
                        _sc_b   = _qscale(_b_t_q_bmm)
                        _zp_out_bmm = (int(_out_q_bmm.ZeroPoint(0))
                                       if _out_q_bmm and _out_q_bmm.ZeroPointLength() > 0 else 0)
                        _eff = _sc_in * _sc_b / (_sc_out if _sc_out > 0 else 1.0)
                        _mult_s, _shift_s = quantize_multiplier(float(_eff))
                        _mult_arr_bmm  = np.full(_N_bmm, _mult_s,  dtype=np.int32)
                        _shift_arr_bmm = np.full(_N_bmm, _shift_s, dtype=np.int8)
                        _zp_in_eff_bmm = (int(_in_q_bmm.ZeroPoint(0))
                                          if _in_q_bmm and _in_q_bmm.ZeroPointLength() > 0
                                          else 0)
                        # v12 Phase 7: zp_B for per-sample activation-sum correction.
                        _zp_b_bmm = (int(_b_t_q_bmm.ZeroPoint(0))
                                     if _b_t_q_bmm and _b_t_q_bmm.ZeroPointLength() > 0
                                     else 0)
                    _out_tensor_bmm = int(op.Outputs(0)) if op.OutputsLength() > 0 else -1
                    _in0_tensor_bmm = _a_idx_bmm
                    _in1_tensor_bmm = _b_idx_bmm
                    _ld_bmm = TYPE_TO_DTYPE.get(in_t.Type(), DT_INT8x8)
                    if _is_fp_bmm: _ld_bmm = DT_FP16
                    _all_ref_slices = []
                    _lowered_ok = True
                    for _bi in range(_B_total):
                        _a_sl = _a_3d[_bi]           # [M, K]
                        _b_sl = _b_3d[_bi]           # [K, N]
                        # Weights: OHWI [N, 1, 1, K] = b.T rows
                        _wgt_sl = _b_sl.T.reshape(_N_bmm, 1, 1, _K_bmm)
                        # Input: [M, 1, K]
                        _act_sl = _a_sl.reshape(_M_bmm, 1, _K_bmm)
                        # Reference output [M, 1, N]
                        if _is_fp_bmm:
                            # Use conv_fp_ref — same FP16→FP32 cast + sequential
                            # per-element FP32 accumulation as conv_engine.h
                            # compute_fp(). Bit-exact with the sim for any N.
                            # _b_sl shape [K, N]: transpose to [N, K] then
                            # reshape to OHWI [N,1,1,K] matching _wgt_sl.
                            _act_ref = _a_sl.astype(np.float16).astype(np.float32).reshape(_M_bmm, 1, _K_bmm)
                            _wgt_ref = _b_sl.T.astype(np.float16).astype(np.float32).reshape(_N_bmm, 1, 1, _K_bmm)
                            _bias_ref = np.zeros(_N_bmm, dtype=np.float32)
                            _ref_f32 = conv_fp_ref(
                                _act_ref, _wgt_ref,
                                s_h=1, s_w=1, pad=(0, 0, 0, 0), group=1,
                                bias=_bias_ref,
                                act_min=float(-3.4e38), act_max=float(3.4e38),
                            )   # [M, 1, N] float32
                            _ref_sl = _ref_f32.astype(np.float16).reshape(_M_bmm, 1, _N_bmm)
                        else:
                            # v12 Phase 7: INT8 with full ZP correction.
                            # true_acc = Σ(a_q*b_q) - zp_A*Σb_q[n] - zp_B*Σa_q[m] + K*zp_A*zp_B
                            _psum = np.matmul(
                                _a_sl.astype(np.int32),
                                _b_sl.astype(np.int32),
                            ).reshape(_M_bmm, 1, _N_bmm)   # [M, 1, N]
                            _sum_w_sl = _wgt_sl.astype(np.int64).sum(axis=(1, 2, 3))  # [N]
                            # bias_eff[n] = -zp_A*sum_b[n] + K*zp_A*zp_B
                            # ArcSim regression: B is compile-time static → correct.
                            # Production (dynamic KV cache): B changes each inference; the
                            # chip driver must recompute sum_b[n] from the actual B tile
                            # and overwrite bias_eff[n] in the params DRAM blob before
                            # dispatching CONV+REQUANT. The zp_A / K / zp_B values needed
                            # are the constants below. Alternatively the CONV engine can
                            # accumulate sum_b per-OC and pass it via a separate L1 buffer.
                            _bias_eff_sl = (
                                -np.int64(_zp_in_eff_bmm) * _sum_w_sl
                                + np.int64(_K_bmm) * np.int64(_zp_in_eff_bmm) * np.int64(_zp_b_bmm)
                            ).astype(np.int64)
                            _be_i32_sl = np.clip(_bias_eff_sl, -(1 << 31), (1 << 31) - 1).astype(np.int32)
                            # sum_a[m] = Σ_k a_q[m, k] (per-row, runtime correction)
                            _sum_a_sl = _a_sl.astype(np.int64).sum(axis=-1)  # [M]
                            # per-row correction: -zp_B * sum_a[m]
                            _corr_sl = (-np.int64(_zp_b_bmm) * _sum_a_sl).astype(np.int32)  # [M]
                            # Reference: apply bias_eff + corr
                            _pb = (_psum.astype(np.int64)
                                   + _be_i32_sl.reshape(1, 1, _N_bmm).astype(np.int64)
                                   + _corr_sl.reshape(_M_bmm, 1, 1).astype(np.int64))
                            _pb = np.clip(_pb, -(1 << 31), (1 << 31) - 1).astype(np.int32)
                            _sc = multiply_by_quantized_multiplier_np(
                                _pb,
                                _mult_arr_bmm.reshape(1, 1, _N_bmm),
                                _shift_arr_bmm.reshape(1, 1, _N_bmm),
                            )
                            _ref_sl = np.clip(_sc + _zp_out_bmm, -128, 127).astype(np.int8)
                            _ref_sl = _ref_sl.reshape(_M_bmm, 1, _N_bmm)
                        _all_ref_slices.append(_ref_sl)
                        # Build conv_wgt_payload for this slice
                        if _is_fp_bmm:
                            _amin_s = np.float32(-3.4e38); _amax_s = np.float32(3.4e38)
                            _bias_f32 = np.zeros(_N_bmm, dtype=np.float32)
                            _params_b_sl = (struct.pack("<ff", float(_amin_s), float(_amax_s))
                                            + _bias_f32.astype("<f4").tobytes())
                            _wgt_payload_sl = (_wgt_sl.astype(np.float16).tobytes(order="C")
                                               + _params_b_sl)
                            _in_b_sl  = _act_sl.astype(np.float16).tobytes(order="C")
                            _ref_b_sl = _ref_sl.astype(np.float16).tobytes(order="C")
                        else:
                            # v12 Phase 7: params include bias_eff with ZP cross-term;
                            # append corr array (shape [M, 1]) for SystemC requant engine.
                            # corr_per_oc=0 (A.op_kind==OP_FC, not DWCONV) → shape [OH, OW]=[M, 1].
                            _sum_w_sl2 = _wgt_sl.astype(np.int64).sum(axis=(1, 2, 3))
                            _be_sl2 = (
                                -np.int64(_zp_in_eff_bmm) * _sum_w_sl2
                                + np.int64(_K_bmm) * np.int64(_zp_in_eff_bmm) * np.int64(_zp_b_bmm)
                            ).astype(np.int64)
                            _be_i32_sl2 = np.clip(_be_sl2, -(1 << 31), (1 << 31) - 1).astype(np.int32)
                            _params_b_sl = (struct.pack("<iii", _zp_out_bmm, -128, 127)
                                            + _mult_arr_bmm.astype("<i4").tobytes()
                                            + _shift_arr_bmm.astype(np.int8).tobytes()
                                            + _be_i32_sl2.tobytes())
                            if _zp_b_bmm != 0:
                                # Append corr map [M, 1] = -zp_B * sum_a for SystemC requant.
                                _sum_a_sl2 = _a_sl.astype(np.int64).sum(axis=-1)  # [M]
                                _corr_sl2 = (-np.int64(_zp_b_bmm) * _sum_a_sl2).astype(np.int32)
                                _params_b_sl += _corr_sl2.reshape(_M_bmm, 1).astype("<i4").tobytes(order="C")
                            _wgt_payload_sl = (_wgt_sl.astype(np.int8).tobytes(order="C")
                                               + _params_b_sl)
                            _in_b_sl  = _act_sl.astype(np.int8).tobytes(order="C")
                            _ref_b_sl = _ref_sl.astype(np.int8).tobytes(order="C")
                        # Emit the layer directly (mirrors bottom-of-loop code).
                        # No input-alias optimisation across BMM slices (different shapes).
                        _stored_in_b_sl = _in_b_sl
                        _budget_err = _program_budget_error(
                            len(_stored_in_b_sl), len(_wgt_payload_sl), len(_ref_b_sl))
                        if _budget_err:
                            print(f"  layer {li:>2d}  bmm[{_bi}]  "
                                  f"in={_M_bmm}x1x{_K_bmm} "
                                  f"skipped ({_budget_err})")
                            _lowered_ok = False
                            break
                        _slice_label = f"[{_bi}/{_B_total}]" if _B_total > 1 else ""
                        print(f"  layer {li:>2d}  {OP_NAME[OP_FC_BMM]}  "
                              f"in={_M_bmm}x1x{_K_bmm}  k=1x1  s=1x1  g=1  "
                              f"out={_M_bmm}x1x{_N_bmm}  "
                              f"({_M_bmm * _N_bmm} {'FP16' if _is_fp_bmm else 'INT8'})  "
                              f"ready  from=BATCH_MATMUL{_slice_label}")
                        _zp_in_eff_sl = _zp_in_eff_bmm if not _is_fp_bmm else 0
                        _compiled_idx_sl = len(layers)
                        layers.append(dict(
                            in_h=_M_bmm, in_w=1, in_c=_K_bmm,
                            out_h=_M_bmm, out_w=1, out_c=_N_bmm,
                            k_h=1, k_w=1, s_h=1, s_w=1,
                            p_t=0, p_b=0, p_l=0, p_r=0,
                            dram_in=DRAM_IN  + cur_i,
                            dram_wgt=DRAM_WGT + cur_w,
                            dram_out=DRAM_OUT + cur_o,
                            in_size=len(_in_b_sl),
                            wgt_size=len(_wgt_payload_sl),
                            ref_size=len(_ref_b_sl),
                            in_off=in_off, wgt_off=wgt_off, ref_off=ref_off,
                            in_alias_layer=-1,
                            group=1,
                            op_kind=OP_FC_BMM,
                            dtype=_ld_bmm,
                            zp_in_eff=_zp_in_eff_sl,
                            orig_op_index=orig_op_index,
                            input0_tensor=_in0_tensor_bmm if _bi == 0 else -1,
                            input1_tensor=_in1_tensor_bmm if _bi == 0 else -1,
                            output_tensor=_out_tensor_bmm if _bi == _B_total - 1 else -1,
                        ))
                        if _bi == 0:
                            op_to_compiled[orig_op_index] = _compiled_idx_sl
                        cur_w   += len(_wgt_payload_sl)
                        cur_i   += len(_stored_in_b_sl)
                        cur_o   += len(_ref_b_sl)
                        wgt_off += len(_wgt_payload_sl)
                        in_off  += len(_stored_in_b_sl)
                        ref_off += len(_ref_b_sl)
                        in_blobs.append(_in_b_sl)
                        wgt_blobs.append(_wgt_payload_sl)
                        ref_blobs.append(_ref_b_sl)
                    if _lowered_ok:
                        # Update chain state with the combined output of all slices.
                        # The combined output shape is batch_out+[M,1,N]; collapse to 3D.
                        _combined = np.stack([s.reshape(_M_bmm, _N_bmm)
                                              for s in _all_ref_slices], axis=0)
                        # Shape stored in tensor_values is the flat original rank.
                        _out_shape_full = list(_batch_out) + [_M_bmm, _N_bmm]
                        _out_arr_full   = _combined.reshape(_out_shape_full)
                        if _out_tensor_bmm >= 0:
                            tensor_values[_out_tensor_bmm] = _out_arr_full.copy().reshape(-1)
                        # last_output_arr: use 3D [M,1,N] of the last slice so that
                        # downstream chain detection works when the consumer has
                        # compatible shape.
                        last_output_arr   = _all_ref_slices[-1]  # [M, 1, N]
                        last_output_tensor = _out_tensor_bmm
                        continue   # skip materialise path for this op
                    # else: budget error mid-way; let the materialized path run
                    # to at least emit the full-tensor reference. But the layers
                    # list was partially mutated — just skip cleanly.
                    last_output_arr = None
                    continue

        # ---- v12 SHAPE → OK_SHAPE constant-load lowering ----------------------
        # TFLite SHAPE returns the static dimensions of its input tensor as a
        # 1-D INT32 tensor.  For fixed-topology ArcSim the dims are compile-time
        # constants.  Lower to OK_SHAPE: store INT32 bytes in the wgt area and
        # emit a single UDMA load — no DRAM write-back needed.
        if opname == "SHAPE":
            src_t = sg.Tensors(int(op.Inputs(0)))
            src_shape = [int(src_t.Shape(k)) for k in range(src_t.ShapeLength())]
            rank = len(src_shape)
            shape_bytes = np.asarray(src_shape, dtype=np.int32).tobytes()
            in_b   = b""                  # no activation input at runtime
            wgt_b  = shape_bytes          # constant payload (loaded by UDMA)
            ref_b  = shape_bytes          # expected L1 content = same bytes
            _OH, _OW, _OC = 1, 1, rank   # display shape: 1x1xrank
            _H,  _W,  _Cin = 1, 1, rank
            _op_kind    = OP_SHAPE
            _layer_dtype = DT_INT8x8     # raw bytes, displayed as INT8
            print(f"  layer {li:>2d}  {OP_NAME[OP_SHAPE]}  "
                  f"in={_H}x{_W}x{_Cin}  k=1x1  s=1x1  g=1  "
                  f"out={_OH}x{_OW}x{_OC}  ({rank} INT32 = {len(shape_bytes)} B)  ready")
            _in_off_shape  = in_off
            _wgt_off_shape = wgt_off
            _ref_off_shape = ref_off
            # SHAPE has no activation input — in_b is empty, no alias check needed.
            budget_error_shape = _program_budget_error(0, len(wgt_b), len(ref_b))
            if budget_error_shape:
                print(f"  layer {li:>2d}  {OP_NAME[OP_SHAPE]}  skipped ({budget_error_shape})")
                last_output_arr = None
                continue
            _output_tensor = int(op.Outputs(0)) if op.OutputsLength() > 0 else -1
            layers.append(dict(
                in_h=_H, in_w=_W, in_c=_Cin,
                out_h=_OH, out_w=_OW, out_c=_OC,
                k_h=1, k_w=1, s_h=1, s_w=1,
                p_t=0, p_b=0, p_l=0, p_r=0,
                dram_in=DRAM_IN  + cur_i,
                dram_wgt=DRAM_WGT + cur_w,
                dram_out=DRAM_OUT + cur_o,
                in_size=0,
                wgt_size=len(wgt_b),
                ref_size=len(ref_b),
                in_off=_in_off_shape,
                wgt_off=_wgt_off_shape,
                ref_off=_ref_off_shape,
                in_alias_layer=-1,
                group=1,
                op_kind=_op_kind,
                dtype=_layer_dtype,
                zp_in_eff=0,
                orig_op_index=orig_op_index,
                input0_tensor=int(op.Inputs(0)),
                input1_tensor=-1,
                output_tensor=_output_tensor,
            ))
            op_to_compiled[orig_op_index] = len(layers) - 1
            cur_w  += len(wgt_b)
            cur_o  += len(ref_b)
            wgt_off += len(wgt_b)
            ref_off += len(ref_b)
            # no in_blob — SHAPE has no activation input
            wgt_blobs.append(wgt_b)
            ref_blobs.append(ref_b)
            _shape_arr = np.frombuffer(shape_bytes, dtype=np.int32).reshape(1, 1, rank)
            if _output_tensor >= 0:
                tensor_values[_output_tensor] = _shape_arr.copy()
            last_output_arr = _shape_arr
            last_output_tensor = _output_tensor
            continue   # skip materialize path

        # ---- v12 REVERSE_V2 → OK_REVERSE constant-load lowering ---------------
        # TFLite REVERSE_V2 flips a tensor along given axes. In fixed-topology
        # ArcSim the input is always a compile-time constant (RoPE sinusoid
        # encodings materialised upstream). Pre-compute the flipped bytes at
        # compile time, store in the wgt area, and emit a single UDMA load —
        # identical runtime path to OK_SHAPE.
        if opname == "REVERSE_V2" and in_arr is not None:
            axes_t = sg.Tensors(int(op.Inputs(1))) if op.InputsLength() > 1 else None
            axes_a = _tensor_array(axes_t).astype(np.int32) if axes_t is not None else None
            axes = ([int(x) for x in axes_a.reshape(-1).tolist()]
                    if axes_a is not None else [])
            src = np.asarray(in_arr)
            if src.ndim > 0 and axes:
                for ax in sorted({int(x) % src.ndim for x in axes}, reverse=True):
                    src = np.flip(src, axis=ax)
            # Re-quantise through the output tensor's scale/zp so bytes match
            # what downstream consumers expect.
            flipped_ref = np.asarray(
                _quantize_real_to_tensor(src.reshape(int(OH), int(OW), int(OC)), out_t)
            )
            rev_bytes   = flipped_ref.tobytes()
            _rev_dtype  = TYPE_TO_DTYPE.get(out_t.Type(), DT_INT8x8)
            _rev_nelem  = int(OH) * int(OW) * int(OC)
            elem_label  = ("FP16" if _rev_dtype == DT_FP16
                           else f"INT8" if _rev_dtype in (DT_INT8x4, DT_INT8x8) else "INT16")
            print(f"  layer {li:>2d}  {OP_NAME[OP_REVERSE]}  "
                  f"in={int(H)}x{int(W)}x{int(Cin)}  k=1x1  s=1x1  g=1  "
                  f"out={int(OH)}x{int(OW)}x{int(OC)}  ({_rev_nelem} {elem_label})  ready")
            budget_error_rev = _program_budget_error(0, len(rev_bytes), len(rev_bytes))
            if budget_error_rev:
                print(f"  layer {li:>2d}  {OP_NAME[OP_REVERSE]}  skipped ({budget_error_rev})")
                last_output_arr = None
                continue
            _rev_out_tensor = int(op.Outputs(0)) if op.OutputsLength() > 0 else -1
            layers.append(dict(
                in_h=int(H), in_w=int(W), in_c=int(Cin),
                out_h=int(OH), out_w=int(OW), out_c=int(OC),
                k_h=1, k_w=1, s_h=1, s_w=1,
                p_t=0, p_b=0, p_l=0, p_r=0,
                dram_in=DRAM_IN  + cur_i,
                dram_wgt=DRAM_WGT + cur_w,
                dram_out=DRAM_OUT + cur_o,
                in_size=0,
                wgt_size=len(rev_bytes),
                ref_size=len(rev_bytes),
                in_off=in_off,
                wgt_off=wgt_off,
                ref_off=ref_off,
                in_alias_layer=-1,
                group=1,
                op_kind=OP_REVERSE,
                dtype=_rev_dtype,
                zp_in_eff=0,
                orig_op_index=orig_op_index,
                input0_tensor=int(op.Inputs(0)),
                input1_tensor=-1,
                output_tensor=_rev_out_tensor,
            ))
            op_to_compiled[orig_op_index] = len(layers) - 1
            cur_w   += len(rev_bytes)
            cur_o   += len(rev_bytes)
            wgt_off += len(rev_bytes)
            ref_off += len(rev_bytes)
            wgt_blobs.append(rev_bytes)
            ref_blobs.append(rev_bytes)
            _rev_out_arr = flipped_ref.reshape(int(OH), int(OW), int(OC))
            if _rev_out_tensor >= 0:
                tensor_values[_rev_out_tensor] = _rev_out_arr.copy()
            last_output_arr   = _rev_out_arr
            last_output_tensor = _rev_out_tensor
            continue   # skip materialize path

        if materialize_op:
            # Tranche-1 materialized support. These ops become explicit
            # reference-byte layers, so fast/cx/verilog execute and compare
            # the output boundary instead of silently skipping the original op.
            out_hwc = (int(OH), int(OW), int(OC))
            shape_changing_materialized = {
                "BATCH_MATMUL", "RESIZE_BILINEAR",
                "RESIZE_NEAREST_NEIGHBOR", "SUM", "TRANSPOSE_CONV",
                # v11: qwen35 helpers — output shape unrelated to input shape.
                "SHAPE", "REVERSE_V2", "RANDOM_STANDARD_NORMAL",
            }
            a = (None if opname in shape_changing_materialized
                 else _reshape_or_broadcast(in_arr, out_hwc))
            if opname == "BATCH_MATMUL":
                a_idx = int(op.Inputs(0))
                b_idx = int(op.Inputs(1))
                a_t = sg.Tensors(a_idx)
                b_t = sg.Tensors(b_idx)
                a_real = _internal_to_real(_tensor_original_array(a_idx), a_t)
                b_real = _internal_to_real(_tensor_original_array(b_idx), b_t)
                adj_x = adj_y = False
                try:
                    bo = fb.BatchMatMulOptions()
                    bo.Init(opt_table.Bytes, opt_table.Pos)
                    adj_x = bool(bo.AdjX())
                    adj_y = bool(bo.AdjY())
                except Exception:
                    pass
                if adj_x:
                    a_real = np.swapaxes(a_real, -1, -2)
                if adj_y:
                    b_real = np.swapaxes(b_real, -1, -2)
                # Some TFLite exports (e.g. qwen35_attention) declare shapes
                # where the BATCH_MATMUL contracting dim is broken / 1 on one
                # side; the model relies on implicit broadcast. np.matmul
                # rejects this strict K mismatch, so broadcast the K=1 side
                # to match the other operand before contracting.
                a_k = a_real.shape[-1]
                b_k = b_real.shape[-2]
                if a_k != b_k:
                    if b_k == 1:
                        target = list(b_real.shape)
                        target[-2] = a_k
                        b_real = np.broadcast_to(b_real, target).copy()
                    elif a_k == 1:
                        target = list(a_real.shape)
                        target[-1] = b_k
                        a_real = np.broadcast_to(a_real, target).copy()
                ref = _quantize_real_to_tensor(np.matmul(a_real, b_real), out_t)
            elif opname in ("RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"):
                size_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                size_a = _tensor_array(size_t).astype(np.int32) if size_t else None
                out_h = int(size_a.reshape(-1)[0]) if size_a is not None and size_a.size >= 1 else int(OH)
                out_w = int(size_a.reshape(-1)[1]) if size_a is not None and size_a.size >= 2 else int(OW)
                align_corners = False
                half_pixel_centers = False
                try:
                    if opname == "RESIZE_BILINEAR":
                        ro = fb.ResizeBilinearOptions()
                    else:
                        ro = fb.ResizeNearestNeighborOptions()
                    ro.Init(opt_table.Bytes, opt_table.Pos)
                    align_corners = bool(ro.AlignCorners())
                    half_pixel_centers = bool(ro.HalfPixelCenters())
                except Exception:
                    pass
                src_real = _internal_to_real(_tensor_original_array(int(op.Inputs(0))), in_t)
                if opname == "RESIZE_BILINEAR":
                    resized = _resize_bilinear_ref(
                        src_real, out_h, out_w, align_corners, half_pixel_centers)
                else:
                    resized = _resize_nearest_ref(
                        src_real, out_h, out_w, align_corners, half_pixel_centers)
                ref = _quantize_real_to_tensor(resized, out_t)
            elif opname == "TRANSPOSE_CONV":
                shape_t = sg.Tensors(op.Inputs(0))
                wgt_t = sg.Tensors(op.Inputs(1))
                input_t = sg.Tensors(op.Inputs(2))
                shape_a = _tensor_array(shape_t)
                out_shape = (shape_a.astype(np.int32).reshape(-1).tolist()
                             if shape_a is not None else list(_tensor_shape(out_t)))
                while len(out_shape) < 4:
                    out_shape = [1] + out_shape
                n_out, out_h, out_w, out_c = [int(x) for x in out_shape[-4:]]
                x_real = _internal_to_real(_tensor_original_array(int(op.Inputs(2))), input_t)
                if x_real.ndim == 3:
                    x_real = x_real.reshape((1,) + x_real.shape)
                w_arr = _tensor_array(wgt_t)
                if w_arr is None:
                    ref = _synth_output_array(out_t, out_h, out_w, out_c)
                else:
                    if wgt_t.Type() == fb.TensorType.UINT8:
                        w_arr = (w_arr.astype(np.int16) - 128).astype(np.int8)
                    w_real = _internal_to_real(w_arr, wgt_t).astype(np.float32)
                    if w_real.ndim != 4:
                        ref = _synth_output_array(out_t, out_h, out_w, out_c)
                        w_real = None
                    elif int(w_real.shape[0]) == out_c:
                        # TFLite TRANSPOSE_CONV filter layout is
                        # [output_channel, height, width, input_channel].
                        filt_out_c, kh, kw, filt_in_c = [int(x) for x in w_real.shape]
                        w_lookup = lambda ky, kx, ic: w_real[:, ky, kx, ic]
                    else:
                        # Keep compatibility with older local test assets that
                        # used [height, width, output_channel, input_channel].
                        kh, kw, filt_out_c, filt_in_c = [int(x) for x in w_real.shape]
                        w_lookup = lambda ky, kx, ic: w_real[ky, kx, :, ic]
                if w_arr is not None and w_real is not None:
                    try:
                        to = fb.TransposeConvOptions()
                        to.Init(opt_table.Bytes, opt_table.Pos)
                        stride_h = int(to.StrideH())
                        stride_w = int(to.StrideW())
                        pad_enum = int(to.Padding())
                    except Exception:
                        stride_h = stride_w = 1
                        pad_enum = fb.Padding.SAME
                    in_h, in_w, in_c = x_real.shape[1], x_real.shape[2], x_real.shape[3]
                    pad_total_h = max(0, (in_h - 1) * stride_h + kh - out_h)
                    pad_total_w = max(0, (in_w - 1) * stride_w + kw - out_w)
                    pad_top = pad_total_h // 2 if pad_enum == fb.Padding.SAME else 0
                    pad_left = pad_total_w // 2 if pad_enum == fb.Padding.SAME else 0
                    acc = np.zeros((x_real.shape[0], out_h, out_w, filt_out_c),
                                   dtype=np.float32)
                    for nidx in range(x_real.shape[0]):
                        for iy in range(in_h):
                            for ix in range(in_w):
                                for ic in range(min(in_c, filt_in_c)):
                                    xv = x_real[nidx, iy, ix, ic]
                                    if xv == 0:
                                        continue
                                    for ky in range(kh):
                                        oy = iy * stride_h + ky - pad_top
                                        if oy < 0 or oy >= out_h:
                                            continue
                                        for kx in range(kw):
                                            ox = ix * stride_w + kx - pad_left
                                            if ox < 0 or ox >= out_w:
                                                continue
                                            acc[nidx, oy, ox, :filt_out_c] += xv * w_lookup(ky, kx, ic)
                    if acc.shape[0] != n_out:
                        if acc.shape[0] == 1:
                            acc = np.repeat(acc, n_out, axis=0)
                        else:
                            acc = acc[:n_out, :, :, :]
                    if acc.shape[3] != out_c:
                        fixed = np.zeros((acc.shape[0], out_h, out_w, out_c), dtype=np.float32)
                        copy_c = min(out_c, acc.shape[3])
                        fixed[:, :, :, :copy_c] = acc[:, :, :, :copy_c]
                        acc = fixed
                    ref = _quantize_real_to_tensor(acc.reshape(n_out, out_h, out_w, out_c), out_t)
            elif opname == "QUANTIZE":
                ref = _quantize_real_to_tensor(_internal_to_real(a, in_t), out_t)
            elif opname == "CAST":
                ref = _cast_numeric_to_tensor(_internal_to_numeric(a, in_t), out_t)
            elif opname == "RELU":
                real = np.maximum(_internal_to_real(a, in_t), np.float32(0.0))
                ref = _quantize_real_to_tensor(real, out_t)
            elif opname == "LEAKY_RELU":
                try:
                    lo = fb.LeakyReluOptions(); lo.Init(opt_table.Bytes, opt_table.Pos)
                    alpha = np.float32(lo.Alpha())
                except Exception:
                    alpha = np.float32(0.2)
                x = _internal_to_real(a, in_t)
                ref = _quantize_real_to_tensor(np.where(x >= 0.0, x, x * alpha), out_t)
            elif opname == "PRELU":
                alpha_idx = int(op.Inputs(1)) if op.InputsLength() > 1 else -1
                alpha_t = sg.Tensors(alpha_idx)
                alpha = _reshape_or_broadcast(
                    _tensor_operand_internal(alpha_idx, out_hwc, _storage_dtype_for_tensor(alpha_t)),
                    out_hwc)
                x = _internal_to_real(a, in_t)
                ar = _internal_to_real(alpha, alpha_t)
                ref = _quantize_real_to_tensor(np.where(x >= 0.0, x, x * ar), out_t)
            elif opname == "TANH":
                ref = _quantize_real_to_tensor(np.tanh(_internal_to_real(a, in_t)), out_t)
            elif opname == "LOGISTIC":
                x = _internal_to_real(a, in_t)
                ref = _quantize_real_to_tensor(1.0 / (1.0 + np.exp(-x)), out_t)
            elif opname == "HARD_SWISH":
                x = _internal_to_real(a, in_t)
                ref = _quantize_real_to_tensor(
                    x * np.minimum(np.maximum(x + 3.0, 0.0), 6.0) / np.float32(6.0),
                    out_t)
            elif opname == "GELU":
                x = _internal_to_real(a, in_t)
                y = 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
                ref = _quantize_real_to_tensor(y, out_t)
            elif opname == "RSQRT":
                x = np.maximum(_internal_to_real(a, in_t), np.float32(1.0e-12))
                ref = _quantize_real_to_tensor(np.float32(1.0) / np.sqrt(x), out_t)
            elif opname == "SQUARED_DIFFERENCE":
                b_idx = int(op.Inputs(1)) if op.InputsLength() > 1 else -1
                b_t = sg.Tensors(b_idx)
                b = _reshape_or_broadcast(
                    _tensor_operand_internal(b_idx, out_hwc, _storage_dtype_for_tensor(b_t)),
                    out_hwc)
                d = _internal_to_real(a, in_t) - _internal_to_real(b, b_t)
                ref = _quantize_real_to_tensor(d * d, out_t)
            elif opname == "SUM":
                src = np.asarray(in_arr)
                axes_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                axes_a = _tensor_array(axes_t).astype(np.int32) if axes_t else None
                axes = [int(x) for x in (axes_a.reshape(-1).tolist()
                                         if axes_a is not None else [])]
                try:
                    ro = fb.ReducerOptions(); ro.Init(opt_table.Bytes, opt_table.Pos)
                    keepdims = bool(ro.KeepDims())
                except Exception:
                    keepdims = True
                orig_rank = len(ish)
                mapped_axes = []
                for ax in axes:
                    if ax < 0:
                        ax += orig_rank
                    if orig_rank >= 4:
                        if ax == 0:
                            continue
                        mapped_axes.append(ax - 1)
                    else:
                        mapped_axes.append(ax + max(0, 3 - orig_rank))
                mapped_axes = sorted(set(a for a in mapped_axes if 0 <= a < src.ndim))
                real = _internal_to_real(src, in_t)
                summed = np.sum(real, axis=tuple(mapped_axes), keepdims=keepdims,
                                dtype=np.float32)
                if summed.size != int(np.prod(out_hwc)):
                    summed = np.reshape(summed, out_hwc)
                ref = _quantize_real_to_tensor(summed, out_t)
            elif opname == "MINIMUM":
                b_idx = int(op.Inputs(1)) if op.InputsLength() > 1 else -1
                b_t = sg.Tensors(b_idx)
                b = _reshape_or_broadcast(
                    _tensor_operand_internal(b_idx, out_hwc, _storage_dtype_for_tensor(b_t)),
                    out_hwc)
                real = np.minimum(_internal_to_real(a, in_t),
                                  _internal_to_real(b, b_t))
                ref = _quantize_real_to_tensor(real, out_t)
            elif opname == "SHAPE":
                # Output is the static shape vector of the input tensor as int32.
                # The downstream parser may have inferred out_hwc from a consumer
                # rather than the SHAPE output's literal rank; in that case pad
                # with zeros so the layer still has well-defined materialize bytes.
                src_t = sg.Tensors(int(op.Inputs(0)))
                src_shape = [int(src_t.Shape(k)) for k in range(src_t.ShapeLength())]
                target_size = int(np.prod(out_hwc))
                if len(src_shape) >= target_size:
                    src_shape = src_shape[:target_size]
                else:
                    src_shape = src_shape + [0] * (target_size - len(src_shape))
                ref = np.asarray(src_shape, dtype=np.int32).reshape(out_hwc)
            elif opname == "REVERSE_V2":
                # Reverse along axes given by the second input (constant int32 tensor).
                # `a` is None on this path (shape_changing_materialized), so read
                # the input tensor directly and flip in its native rank.
                axes_t = sg.Tensors(int(op.Inputs(1))) if op.InputsLength() > 1 else None
                axes_a = _tensor_array(axes_t).astype(np.int32) if axes_t else None
                axes = [int(x) for x in (axes_a.reshape(-1).tolist()
                                         if axes_a is not None else [])]
                src = np.asarray(in_arr)
                if src.ndim > 0:
                    mapped = sorted({(ax % src.ndim) for ax in axes})
                    flipped = src
                    for ax in reversed(mapped):
                        flipped = np.flip(flipped, axis=ax)
                else:
                    flipped = src
                target_size = int(np.prod(out_hwc))
                if flipped.size != target_size:
                    # Shape mismatch (broken qwen35 graph): pad/truncate so
                    # downstream packing still gets well-defined bytes.
                    flat = flipped.reshape(-1)
                    if flat.size >= target_size:
                        flat = flat[:target_size]
                    else:
                        flat = np.pad(flat, (0, target_size - flat.size))
                    flipped = flat.reshape(out_hwc)
                else:
                    flipped = flipped.reshape(out_hwc)
                ref = _quantize_real_to_tensor(flipped, out_t)
            elif opname == "RANDOM_STANDARD_NORMAL":
                # Deterministic standard-normal samples (fixed seed) shaped to
                # the output. Bit-reproducible across compile runs so reference
                # bytes don't change between regressions.
                rng = np.random.default_rng(seed=0xC0DE_BEEF)
                samples = rng.standard_normal(int(np.prod(out_hwc))).astype(np.float32)
                ref = _quantize_real_to_tensor(samples.reshape(out_hwc), out_t)
            else:  # GREATER
                b_idx = int(op.Inputs(1)) if op.InputsLength() > 1 else -1
                b_t = sg.Tensors(b_idx)
                b = _reshape_or_broadcast(
                    _tensor_operand_internal(b_idx, out_hwc, _storage_dtype_for_tensor(b_t)),
                    out_hwc)
                ref = (_internal_to_real(a, in_t) > _internal_to_real(b, b_t)).astype(np.int8)
            H, W, Cin = out_hwc
            OH, OW, OC = out_hwc
            in_arr = np.asarray(ref).reshape(out_hwc)
            in_i8 = in_arr
            ref = in_arr
            ref_b = ref.tobytes(order="C")
            op_kind = OP_MATERIALIZE
            layer_dtype = TYPE_TO_DTYPE.get(out_t.Type(), DT_INT8x8)
            if out_t.Type() == fb.TensorType.BOOL:
                layer_dtype = DT_INT8x8
            is_int16 = (layer_dtype == DT_INT16x16)
            is_fp_layer = layer_dtype in (DT_FP16, DT_BFP16, DT_FP8)
            elem_size = _elem_size_for_layer_dtype(layer_dtype)
            wgt = np.zeros((0,), dtype=np.int8)

        elif opname in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            wgt_t  = sg.Tensors(op.Inputs(1))
            wgt    = _tensor_array(wgt_t)
            # v8: weights produced by a DEQUANTIZE op (FP16-weight quantized
            # models) won't have a constant buffer on the conv's tensor — walk
            # back to the source.
            if wgt is None:
                prod, prod_name = _find_op_producer(fb, model, sg, op.Inputs(1))
                if prod is not None and prod_name == "DEQUANTIZE":
                    wgt = _tensor_array(sg.Tensors(prod.Inputs(0)))
            if wgt is None:
                print(f"  layer {li:>2d}  {opname.lower():>7s}  in={H}x{W}x{Cin} "
                      f"skipped (weights tensor has no buffer)")
                continue
            # ---- v8: FP path branches off here. compute_fp in the engine reads
            # FP32 from L1, so we cast FP16/FP32 weights to FP32 at compile time.
            is_fp_layer = (in_t.Type() in FP_TFLITE_TYPES) or (wgt_t.Type() in FP_TFLITE_TYPES) \
                          or wgt.dtype in (np.float16, np.float32)
            # v8.27: INT16x8 hybrid quant (INT16 activations + INT8 weights).
            # TFLite emits this for "16x8 quantization" — esrgan_int16 / unet_int16
            # / similar high-precision models. Switch the layer dtype so sim
            # reads weights as 1 byte/elem (vs the default 2 byte for INT16x16);
            # the int16 input + int32 bias path is otherwise identical.
            if (not is_fp_layer
                and in_t.Type() == fb.TensorType.INT16
                and wgt_t.Type() == fb.TensorType.INT8):
                layer_dtype = DT_INT16x8
                # is_int16 stays True (input synth still uses int16 range,
                # bias still int32 → int16 output etc.); only weight storage
                # differs. elem_size here is the input/output elem size.
            # v7: uint8 weights map to int8 via centered SHIFT (uint8 - 128),
            # not byte reinterpretation. The two are only equivalent when
            # zp_w_uint8 == 128 (symmetric); for asymmetric weights (e.g.,
            # legacy mobilenet_v1's zp_w=157) view(int8) silently corrupts
            # the operand. Shifting keeps the int8 value in real-number
            # correspondence with the (uint8 - 128) representation, leaving
            # zp_w_eff = zp_w_uint8 - 128 to handle separately.
            if (not is_fp_layer) and wgt.dtype == np.uint8:
                wgt = (wgt.astype(np.int16) - 128).astype(np.int8)
            # int16 weights stay as int16.
            if opname == "DEPTHWISE_CONV_2D":
                co = fb.DepthwiseConv2DOptions(); co.Init(opt_table.Bytes, opt_table.Pos)
                _, Kh, Kw, dw_OC = wgt.shape
                mult = int(co.DepthMultiplier()) or (dw_OC // Cin)
                OC = Cin * mult
                wgt = wgt.transpose(3, 1, 2, 0).copy()
                group = Cin
                op_kind = OP_DWCONV
                pad_enum = co.Padding()
            else:
                co = fb.Conv2DOptions(); co.Init(opt_table.Bytes, opt_table.Pos)
                OC, Kh, Kw, _ = wgt.shape
                op_kind = OP_CONV
                pad_enum = co.Padding()
            s_h = int(co.StrideH()); s_w = int(co.StrideW())
            # v9.3: sim CONV stride is 4-bit direct encoding (0=>16), covering
            # ETHZ_V6 stride=3 downsamplers and stride=16 ViT patchify convs.
            if not (1 <= s_h <= 16 and 1 <= s_w <= 16):
                print(f"  layer {li:>2d}  {opname.lower():>7s}  in={H}x{W}x{Cin} "
                      f"skipped (stride={s_h}x{s_w} outside 1..16 encoding range)")
                last_output_arr = None
                continue
            # v8: some models (mobilenet_v3_fp16 first CONV) store a placeholder
            # output shape [1,1,1,C].  Re-derive OH/OW from H/W/stride/padding so
            # we don't trust suspicious values.
            if pad_enum == fb.Padding.SAME:
                OH_calc = (H + s_h - 1) // s_h
                OW_calc = (W + s_w - 1) // s_w
            else:                                              # VALID
                OH_calc = max(0, (H - Kh)) // s_h + 1
                OW_calc = max(0, (W - Kw)) // s_w + 1
            OH_meta, OW_meta, _ = to_hwc(osh)
            OH = OH_calc if (OH_meta <= 1 and OH_calc > 1) else OH_meta
            OW = OW_calc if (OW_meta <= 1 and OW_calc > 1) else OW_meta
            pT, pB = _padding_to_pad(fb, pad_enum, Kh, H, OH, s_h)
            pL, pR = _padding_to_pad(fb, pad_enum, Kw, W, OW, s_w)

        # v8: FP CONV/DWCONV path. The whole quant chain (zp_in/zp_w/MBQM, corr
        # map) doesn't apply — just sum(in*w) + bias, then activation clip.
        # Reference is FP32 vectorised; sim does FP32 sum in nested loops.
        # Self-consistency relaxes from byte-true to ~1e-3 abs FP tolerance.
        if opname in ("CONV_2D", "DEPTHWISE_CONV_2D") and is_fp_layer:
            # v8.10: store FP16 in DRAM/L1 (2 B/elem) to match spec §3A.2 — the
            # earlier FP32 storage was a sim simplification that doubled udma_r/w
            # bandwidth for FP layers.  Internal compute still runs in FP32
            # (matches HW with FP32 accumulator on the FP cluster).
            wgt_h16 = wgt.astype(np.float16)
            # v8.12: chain mode for FP — reuse previous layer's FP16 ref output
            # if shapes match.  Same fusion benefit as the int paths.
            in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
            in_i8 = in_arr

            # Bias stays FP32 in the params blob: it's tiny (4·OC bytes) and
            # FP32 lets the bias-add not lose precision compared to the
            # FP32-accumulator MAC chain.
            bias_arr = np.zeros(OC, dtype=np.float32)
            if op.InputsLength() >= 3 and op.Inputs(2) >= 0:
                bt = sg.Tensors(op.Inputs(2))
                barr = _tensor_array(bt)
                if barr is None:
                    prod, prod_name = _find_op_producer(fb, model, sg, op.Inputs(2))
                    if prod is not None and prod_name == "DEQUANTIZE":
                        barr = _tensor_array(sg.Tensors(prod.Inputs(0)))
                if barr is not None:
                    barr_f32 = barr.astype(np.float32).flatten()
                    bias_arr[:barr_f32.size] = barr_f32

            fused = int(co.FusedActivationFunction())
            INF = float("inf")
            if   fused == 1: act_min, act_max =  0.0, INF
            elif fused == 3: act_min, act_max =  0.0, 6.0
            elif fused == 2: act_min, act_max = -1.0, 1.0
            else:            act_min, act_max = -INF, INF

            # Reference compute in FP32 (lossless wrt the sim's FP32 accumulator);
            # final output cast back to FP16 for storage compare.
            in_f32  = in_arr.astype(np.float32)
            wgt_f32 = wgt_h16.astype(np.float32)
            ref_f32 = conv_fp_ref(in_f32, wgt_f32, s_h, s_w,
                                  (pT, pB, pL, pR), group,
                                  bias_arr, act_min, act_max)
            ref = ref_f32.astype(np.float16)
            ref_b = ref.tobytes(order="C")

            # FP params blob: [ f32 act_min | f32 act_max | f32 bias[OC] ]
            amin = -3.4e38 if not np.isfinite(act_min) else float(act_min)
            amax =  3.4e38 if not np.isfinite(act_max) else float(act_max)
            params_b = struct.pack("<ff", amin, amax) + bias_arr.astype("<f4").tobytes()
            conv_wgt_payload = wgt_h16.astype("<f2").tobytes(order="C") + params_b
            layer_dtype = DT_FP16
            is_int16    = False
            elem_size   = 2
            if force_materialize_native_reason:
                H, W, Cin = int(OH), int(OW), int(OC)
                in_arr = ref.reshape(H, W, Cin).astype(np.float16, copy=False)
                in_i8 = in_arr
                ref_b = in_arr.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                conv_wgt_payload = b""
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0

        if materialize_op:
            pass
        elif opname in ("CONV_2D", "DEPTHWISE_CONV_2D") and not is_fp_layer:
            # ---- v1.2: extract per-tensor / per-channel quant params ----
            in_q = in_t.Quantization()
            wq   = wgt_t.Quantization()
            oq   = out_t.Quantization()
            scale_in  = _qscale(in_q)
            scale_out = _qscale(oq)
            zp_out    = int  (oq.ZeroPoint(0)) if oq and oq.ZeroPointLength() else 0
            zp_in     = int  (in_q.ZeroPoint(0)) if in_q and in_q.ZeroPointLength() else 0
            # validate_tflite.py pre-shifts uint8 synth by +128 before feeding TFLite,
            # so the sim and TFLite agree on signed activations and zp_in is already
            # absorbed; only subtract zp_in for INT8 input dtype.
            zp_in_eff = zp_in if in_t.Type() == fb.TensorType.INT8 else 0
            # weight scales: per-channel array (TFLite v2) or single (v1).
            if wq and wq.ScaleLength() == OC:
                scales_w = np.array([
                    (float(wq.Scale(i)) if np.isfinite(float(wq.Scale(i))) and float(wq.Scale(i)) > 0.0 else 1.0)
                    for i in range(OC)
                ], dtype=np.float64)
            elif wq and wq.ScaleLength() == 1:
                scales_w = np.full(OC, _qscale(wq), dtype=np.float64)
            else:
                scales_w = np.ones(OC, dtype=np.float64)
            # v7: weight zero-point. For uint8 we centered the bytes by -128 above,
            # so zp_w_eff = zp_w_uint8 - 128. For int8 it's the raw zp (typically 0
            # for per-channel quant). Per-channel zp_w arrays are extracted but only
            # the case where all entries collapse to a single scalar is fully wired.
            zp_w_uint8_shift = 128 if wgt_t.Type() == fb.TensorType.UINT8 else 0
            if wq and wq.ZeroPointLength() == OC:
                zp_w_arr = np.array([int(wq.ZeroPoint(i)) for i in range(OC)],
                                    dtype=np.int32) - zp_w_uint8_shift
            elif wq and wq.ZeroPointLength() == 1:
                zp_w_arr = np.full(OC, int(wq.ZeroPoint(0)) - zp_w_uint8_shift,
                                   dtype=np.int32)
            else:
                zp_w_arr = np.zeros(OC, dtype=np.int32)
            zp_w_uniform = bool(np.all(zp_w_arr == zp_w_arr[0]))
            zp_w_scalar  = int(zp_w_arr[0]) if zp_w_uniform else 0
            eff = scales_w * scale_in / scale_out
            mult_arr  = np.zeros(OC, dtype=np.int32)
            shift_arr = np.zeros(OC, dtype=np.int8)
            for c in range(OC):
                mq, sh = quantize_multiplier(float(eff[c]))
                mult_arr[c]  = mq
                shift_arr[c] = sh

            # ---- v3.5: TFLite fused activation (dtype-aware in v4.1) ----
            # ---- v6: shift uint8 output to int8 representation -------
            # FusedActivationFunction enum:
            #   0=NONE  1=RELU  2=RELU_N1_TO_1  3=RELU6
            fused = int(co.FusedActivationFunction())
            # Compute clamps in TFLite's native quant range first.
            is_uint8_out = (out_t.Type() == fb.TensorType.UINT8)
            if is_uint8_out:
                OUT_MIN, OUT_MAX = 0, 255
            elif is_int16:
                OUT_MIN, OUT_MAX = -32768, 32767
            else:                                       # int8
                OUT_MIN, OUT_MAX = -128, 127
            act_min, act_max = OUT_MIN, OUT_MAX
            if fused == 1:           # ReLU
                act_min = max(OUT_MIN, zp_out)
            elif fused == 3:         # ReLU6
                act_min = max(OUT_MIN, zp_out)
                act_max = min(OUT_MAX, zp_out + int(round(6.0 / scale_out)))
            elif fused == 2:         # ReLU_N1_TO_1
                act_min = max(OUT_MIN, zp_out + int(round(-1.0 / scale_out)))
                act_max = min(OUT_MAX, zp_out + int(round( 1.0 / scale_out)))
            # Sim writes int8 bytes — for uint8-output models we represent the
            # output as `tflite_uint8 - 128`, so shift zp/clamps by -128 here.
            shift_out = 128 if is_uint8_out else 0
            zp_out  -= shift_out
            act_min -= shift_out
            act_max -= shift_out
            # After the shift, clip to int8 range so downstream byte-storage works.
            DT_MIN = -32768 if is_int16 else -128
            DT_MAX =  32767 if is_int16 else  127
            act_min = max(DT_MIN, act_min)
            act_max = min(DT_MAX, act_max)

            # ---- v6: bias + zp_in folding;  v7: + zp_w-aware folding ----
            # TFLite math:
            #   acc = sum_{kh,kw,ic} (in - zp_in) * (w - zp_w) + bias
            #       = sum(in*w)
            #         - zp_w * S_window(in)              [per-pixel correction]
            #         - zp_in * sum_w[oc]                [per-channel constant]
            #         + zp_in * zp_w * window_size       [per-channel constant]
            #         + bias[oc]
            # The two per-channel constants fold into bias_eff[oc]; the per-pixel
            # term is handled via an OH×OW (or OH×OW×OC for dwconv) correction
            # map appended to the params blob and added by RequantEngine.
            bias = np.zeros(OC, dtype=np.int64)
            if op.InputsLength() >= 3 and op.Inputs(2) >= 0:
                bt = sg.Tensors(op.Inputs(2))
                barr = _tensor_array(bt)
                if barr is not None:
                    bias[:barr.size] = barr.astype(np.int64).flatten()
            sum_w        = wgt.astype(np.int64).sum(axis=(1, 2, 3))        # [OC]
            window_size  = int(Kh) * int(Kw) * int(wgt.shape[3])           # Kh*Kw*(Cin/group)
            bias_eff = (bias
                        - zp_in_eff * sum_w
                        + zp_in_eff * zp_w_arr.astype(np.int64) * window_size
                       ).astype(np.int64)
            bias_eff_i32 = np.clip(bias_eff, -(1 << 31), (1 << 31) - 1).astype(np.int32)

            # int32 partial sum. Pad-aware so boundary kernel positions
            # contribute zp_in*w (matches TFLite's q_in=zp_in padding).
            psum = conv_int8_ref(in_arr, wgt, s_h, s_w,
                                 (pT, pB, pL, pR), group=group,
                                 pad_value=zp_in_eff)

            # v7: build per-pixel correction map for non-zero zp_w. Two cases:
            #   non-DW (group==1): corr[oh,ow]    = -zp_w * Σ_kernel(in_eff)
            #   DW    (group==Cin): corr[oh,ow,oc] = -zp_w * Σ_kernel(in_eff[..,oc])
            # in_eff uses zp_in at OOB positions to stay consistent with the CONV
            # engine; this matters when both zp_in_eff and zp_w are non-zero.
            corr_arr = None       # int32 array, None = no correction needed
            corr_per_oc = False
            if zp_w_uniform and zp_w_scalar != 0:
                if op_kind == OP_DWCONV:
                    corr_arr = _conv_window_sum_dw(in_arr, Kh, Kw, s_h, s_w,
                                                   (pT, pB, pL, pR), OH, OW,
                                                   pad_value=zp_in_eff)
                    corr_arr = (-zp_w_scalar * corr_arr).astype(np.int32)
                    corr_per_oc = True
                else:
                    corr_arr = _conv_window_sum(in_arr, Kh, Kw, s_h, s_w,
                                                (pT, pB, pL, pR), OH, OW,
                                                in_per_group=Cin // group,
                                                pad_value=zp_in_eff)
                    corr_arr = (-zp_w_scalar * corr_arr).astype(np.int32)
                    corr_per_oc = False
            elif (not zp_w_uniform):
                raise SystemExit(f"layer {li}: per-channel non-zero zp_w not yet supported")

            psum_with_bias = (psum.astype(np.int64)
                              + bias_eff_i32.reshape(1, 1, OC).astype(np.int64))
            if corr_arr is not None:
                if corr_per_oc:
                    psum_with_bias = psum_with_bias + corr_arr.astype(np.int64)
                else:
                    psum_with_bias = psum_with_bias + corr_arr.astype(np.int64)[..., None]
            psum_with_bias = np.clip(psum_with_bias,
                                     -(1 << 31), (1 << 31) - 1).astype(np.int32)
            scaled = multiply_by_quantized_multiplier_np(
                psum_with_bias,
                mult_arr.reshape(1, 1, OC), shift_arr.reshape(1, 1, OC))
            ref_int = np.clip(scaled + zp_out, act_min, act_max)
            if is_int16:
                ref = ref_int.astype(np.int16)
            else:
                ref = ref_int.astype(np.int8)
            ref_b = ref.tobytes(order="C")

            # v7 params blob (compatible with v6 when corr is empty):
            #   [ i32 zp_out | i32 act_min | i32 act_max
            #   | i32 mult[OC] | i8 shift[OC] | i32 bias_eff[OC]
            #   | i32 corr[...]                                    (empty if no corr) ]
            #
            # corr layout when present:
            #   non-DW: shape [OH, OW]      → 4 * OH * OW bytes
            #   DW    : shape [OH, OW, OC]  → 4 * OH * OW * OC bytes
            params_b = (struct.pack("<iii", zp_out, act_min, act_max)
                        + mult_arr.astype("<i4").tobytes()
                        + shift_arr.astype(np.int8).tobytes()
                        + bias_eff_i32.tobytes())
            if corr_arr is not None:
                params_b += corr_arr.astype("<i4").tobytes(order="C")
            conv_wgt_payload = wgt.tobytes(order="C") + params_b
            if force_materialize_native_reason:
                H, W, Cin = int(OH), int(OW), int(OC)
                in_arr = ref.reshape(H, W, Cin).astype(ref.dtype, copy=False)
                in_i8 = in_arr
                ref_b = in_arr.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                conv_wgt_payload = b""
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1

        elif opname == "FULLY_CONNECTED":
            # Map FC → 1x1 CONV_2D so the chain (CONV → Requant) handles it.
            # TFLite FC tensors:
            #   input  [N, in_features]
            #   weight [out_features, in_features]
            #   output [N, out_features]
            wgt_t  = sg.Tensors(op.Inputs(1))
            wgt    = _tensor_array(wgt_t)
            # v8.24: walk DEQUANTIZE for FP-quantized FC weights, mirroring CONV.
            if wgt is None:
                prod, prod_name = _find_op_producer(fb, model, sg, op.Inputs(1))
                if prod is not None and prod_name == "DEQUANTIZE":
                    wgt = _tensor_array(sg.Tensors(prod.Inputs(0)))
            if wgt is None:
                # Transformer-style "FC" where the weight is a runtime tensor
                # (e.g. ALBERT/BERT attention's Q×Kᵀ via FC with weight = output
                # of a RESHAPE/CAST chain on activations). The sim pre-loads
                # weights into DRAM at startup, so runtime-matmul FC lowers to
                # a materialized tensor fallback until a real matmul datapath
                # is modeled.
                ref = _synth_output_array(out_t, OH, OW, OC)
                H, W, Cin = OH, OW, OC
                in_arr = ref
                in_i8 = in_arr
                ref_b = ref.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                layer_dtype = TYPE_TO_DTYPE.get(out_t.Type(), layer_dtype)
                is_int16 = (layer_dtype == DT_INT16x16)
                is_fp_layer = layer_dtype in (DT_FP16, DT_BFP16, DT_FP8)
                elem_size = _elem_size_for_layer_dtype(layer_dtype)
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1
            is_fp_fc = False
            if op_kind != OP_MATERIALIZE:
                is_fp_fc = (in_t.Type() in FP_TFLITE_TYPES) \
                           or (wgt_t.Type() in FP_TFLITE_TYPES) \
                           or wgt.dtype in (np.float16, np.float32)

            if op_kind != OP_MATERIALIZE and is_fp_fc:
                # ---- v8.24: FP FULLY_CONNECTED. Treat as 1x1 conv where the
                # input's spatial dims (H, W) are the FC batch axis — handles
                # both single-vector FC (input shape [1, in_features]) and
                # batched FC (e.g. audio_yamnet's [96, 257] → [96, 64]) without
                # forcing H=W=1 the way the int path does. Compute mirrors FP
                # CONV: FP32 mul-add into running sum, FP16 storage.
                FC_out, FC_in = wgt.shape
                if Cin != FC_in:
                    raise SystemExit(
                        f"layer {li}: FC in_features ({FC_in}) "
                        f"!= input Cin ({Cin})")
                OH, OW, OC = H, W, FC_out
                Kh = Kw = 1
                s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1
                wgt_h16 = wgt.astype(np.float16).reshape(OC, 1, 1, Cin)
                # Chain mode: prefer the previous layer's FP16 ref output if
                # its shape matches (H, W, Cin); otherwise synth a fresh draw.
                in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
                in_i8 = in_arr
                op_kind = OP_FC
                fc_label = True

                bias_arr = np.zeros(OC, dtype=np.float32)
                if op.InputsLength() >= 3 and op.Inputs(2) >= 0:
                    bt = sg.Tensors(op.Inputs(2))
                    barr = _tensor_array(bt)
                    if barr is None:
                        prod, prod_name = _find_op_producer(fb, model, sg, op.Inputs(2))
                        if prod is not None and prod_name == "DEQUANTIZE":
                            barr = _tensor_array(sg.Tensors(prod.Inputs(0)))
                    if barr is not None:
                        barr_f32 = barr.astype(np.float32).flatten()
                        bias_arr[:barr_f32.size] = barr_f32

                fc_opts = fb.FullyConnectedOptions()
                fc_opts.Init(opt_table.Bytes, opt_table.Pos)
                fused = int(fc_opts.FusedActivationFunction())
                INF = float("inf")
                if   fused == 1: act_min, act_max =  0.0, INF
                elif fused == 3: act_min, act_max =  0.0, 6.0
                elif fused == 2: act_min, act_max = -1.0, 1.0
                else:            act_min, act_max = -INF, INF

                in_f32  = in_arr.astype(np.float32)
                wgt_f32 = wgt_h16.astype(np.float32)
                ref_f32 = conv_fp_ref(in_f32, wgt_f32, s_h, s_w,
                                      (pT, pB, pL, pR), group,
                                      bias_arr, act_min, act_max)
                ref = ref_f32.astype(np.float16)
                ref_b = ref.tobytes(order="C")

                amin = -3.4e38 if not np.isfinite(act_min) else float(act_min)
                amax =  3.4e38 if not np.isfinite(act_max) else float(act_max)
                params_b = (struct.pack("<ff", amin, amax)
                            + bias_arr.astype("<f4").tobytes())
                conv_wgt_payload = wgt_h16.astype("<f2").tobytes(order="C") + params_b
                layer_dtype = DT_FP16
                is_int16    = False
                is_fp_layer = True
                elem_size   = 2
                zp_in_eff   = 0
                # v8.30: transformer-style batched FC (e.g. swin_float layer 19:
                # `1×3136×384 → 1×3136×96` FP16) has H=1 and a huge W. The sim's
                # OH-tiling can't help (out_h=1) and per_oh_in = W*Cin*elem ≈
                # 2.4 MB busts L1. Swap the descriptor's (H, W) so the big axis
                # becomes OH and OH-tiling kicks in. Bytes are unchanged because
                # row-major (H, W, C) layout with one of {H, W}=1 has the same
                # serialisation as (W, H, C); ref/last_output_arr keep the
                # original (H, W) shape so downstream chain-mode stays valid.
                if H == 1 and W > 1:
                    H, W   = W, H
                    OH, OW = OW, OH
                if force_materialize_native_reason:
                    H, W, Cin = int(OH), int(OW), int(OC)
                    in_arr = ref.reshape(H, W, Cin).astype(np.float16, copy=False)
                    in_i8 = in_arr
                    ref_b = in_arr.tobytes(order="C")
                    op_kind = OP_MATERIALIZE
                    fc_label = False
                    wgt = np.zeros((0,), dtype=np.int8)
                    conv_wgt_payload = b""
                    Kh = Kw = s_h = s_w = 1
                    pT = pB = pL = pR = 0
                    group = 1
            if op_kind != OP_MATERIALIZE and not is_fp_fc:
                # v7: same uint8 -> centered int8 mapping as conv path.
                if wgt.dtype == np.uint8:
                    wgt = (wgt.astype(np.int16) - 128).astype(np.int8)
                FC_out, FC_in = wgt.shape
                # Reshape inputs / weights to a 1×1 conv.  Keep the leading
                # batch/sequence product as H so transformer FC such as
                # [1,77,768] -> [1,77,2304] can feed row-aware layout tails.
                native_in_shape = list(_tensor_shape(in_t))
                fc_rows = 1
                if len(native_in_shape) >= 2 and native_in_shape[-1] == FC_in:
                    for dim in native_in_shape[:-1]:
                        fc_rows *= max(1, int(dim))
                H = max(1, fc_rows)
                W = 1
                Cin = FC_in
                OH = H
                OW = 1
                OC = FC_out
                Kh = Kw = 1
                s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1
                wgt = wgt.reshape(OC, 1, 1, Cin)            # OHWI for our compute
                # v8.13/v8.41: respect the chain — the top-of-loop in_arr is
                # already shape (H,1,FC_in) for row-aware FC. Only fall back to
                # rng if the chain isn't a match.
                expected_dtype_fc = np.int16 if is_int16 else np.int8
                in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), expected_dtype_fc)
                in_i8 = in_arr
                op_kind = OP_FC     # dispatched through CONV engine, labelled "fc"
                fc_label = True

                fc_opts = fb.FullyConnectedOptions()
                fc_opts.Init(opt_table.Bytes, opt_table.Pos)

                # Quant params (mirrors CONV path; FC uses per-tensor weight scale).
                in_q = in_t.Quantization()
                wq   = wgt_t.Quantization()
                oq   = out_t.Quantization()
                scale_in  = _qscale(in_q)
                scale_out = _qscale(oq)
                zp_out    = int  (oq.ZeroPoint(0)) if oq and oq.ZeroPointLength() else 0
                zp_in     = int  (in_q.ZeroPoint(0)) if in_q and in_q.ZeroPointLength() else 0
                zp_in_eff = zp_in if in_t.Type() == fb.TensorType.INT8 else 0
                if wq and wq.ScaleLength() == OC:
                    scales_w = np.array([
                        (float(wq.Scale(i)) if np.isfinite(float(wq.Scale(i))) and float(wq.Scale(i)) > 0.0 else 1.0)
                        for i in range(OC)
                    ], dtype=np.float64)
                elif wq and wq.ScaleLength() == 1:
                    scales_w = np.full(OC, _qscale(wq), dtype=np.float64)
                else:
                    scales_w = np.ones(OC, dtype=np.float64)
                # v7: zp_w extraction (mirrors conv path).
                zp_w_uint8_shift = 128 if wgt_t.Type() == fb.TensorType.UINT8 else 0
                if wq and wq.ZeroPointLength() == OC:
                    zp_w_arr = np.array([int(wq.ZeroPoint(i)) for i in range(OC)],
                                        dtype=np.int32) - zp_w_uint8_shift
                elif wq and wq.ZeroPointLength() == 1:
                    zp_w_arr = np.full(OC, int(wq.ZeroPoint(0)) - zp_w_uint8_shift,
                                       dtype=np.int32)
                else:
                    zp_w_arr = np.zeros(OC, dtype=np.int32)
                zp_w_uniform = bool(np.all(zp_w_arr == zp_w_arr[0]))
                zp_w_scalar  = int(zp_w_arr[0]) if zp_w_uniform else 0
                eff = scales_w * scale_in / scale_out
                mult_arr  = np.zeros(OC, dtype=np.int32)
                shift_arr = np.zeros(OC, dtype=np.int8)
                for c in range(OC):
                    mq, sh = quantize_multiplier(float(eff[c]))
                    mult_arr[c]  = mq;  shift_arr[c] = sh

                fused = int(fc_opts.FusedActivationFunction())
                is_uint8_out = (out_t.Type() == fb.TensorType.UINT8)
                if is_uint8_out:
                    OUT_MIN, OUT_MAX = 0, 255
                elif is_int16:
                    OUT_MIN, OUT_MAX = -32768, 32767
                else:
                    OUT_MIN, OUT_MAX = -128, 127
                act_min, act_max = OUT_MIN, OUT_MAX
                if fused == 1:
                    act_min = max(OUT_MIN, zp_out)
                elif fused == 3:
                    act_min = max(OUT_MIN, zp_out)
                    act_max = min(OUT_MAX, zp_out + int(round(6.0 / scale_out)))
                elif fused == 2:
                    act_min = max(OUT_MIN, zp_out + int(round(-1.0 / scale_out)))
                    act_max = min(OUT_MAX, zp_out + int(round( 1.0 / scale_out)))
                shift_out = 128 if is_uint8_out else 0
                zp_out  -= shift_out
                act_min -= shift_out
                act_max -= shift_out
                DT_MIN = -32768 if is_int16 else -128
                DT_MAX =  32767 if is_int16 else  127
                act_min = max(DT_MIN, act_min)
                act_max = min(DT_MAX, act_max)

                # FC bias + zp_in/zp_w folding (mirrors CONV path; v7 adds zp_w).
                bias = np.zeros(OC, dtype=np.int64)
                if op.InputsLength() >= 3 and op.Inputs(2) >= 0:
                    bt = sg.Tensors(op.Inputs(2))
                    barr = _tensor_array(bt)
                    if barr is not None:
                        bias[:barr.size] = barr.astype(np.int64).flatten()
                sum_w        = wgt.astype(np.int64).sum(axis=(1, 2, 3))   # [OC]
                window_size  = int(Kh) * int(Kw) * int(wgt.shape[3])      # 1*1*Cin for FC
                bias_eff = (bias
                            - zp_in_eff * sum_w
                            + zp_in_eff * zp_w_arr.astype(np.int64) * window_size
                           ).astype(np.int64)
                bias_eff_i32 = np.clip(bias_eff, -(1 << 31), (1 << 31) - 1).astype(np.int32)

                psum = conv_int8_ref(in_arr, wgt, s_h, s_w,
                                     (pT, pB, pL, pR), group=group,
                                     pad_value=zp_in_eff)

                # v7: per-pixel correction (FC has 1×1 spatial → scalar per "pixel").
                corr_arr = None
                if zp_w_uniform and zp_w_scalar != 0:
                    corr_arr = _conv_window_sum(in_arr, Kh, Kw, s_h, s_w,
                                                (pT, pB, pL, pR), OH, OW,
                                                in_per_group=Cin // group,
                                                pad_value=zp_in_eff)
                    corr_arr = (-zp_w_scalar * corr_arr).astype(np.int32)
                elif (not zp_w_uniform):
                    raise SystemExit(f"layer {li}: per-channel non-zero zp_w not yet supported (FC)")

                psum_with_bias = (psum.astype(np.int64)
                                  + bias_eff_i32.reshape(1, 1, OC).astype(np.int64))
                if corr_arr is not None:
                    psum_with_bias = psum_with_bias + corr_arr.astype(np.int64)[..., None]
                psum_with_bias = np.clip(psum_with_bias,
                                         -(1 << 31), (1 << 31) - 1).astype(np.int32)
                scaled = multiply_by_quantized_multiplier_np(
                    psum_with_bias,
                    mult_arr.reshape(1, 1, OC), shift_arr.reshape(1, 1, OC))
                ref = (np.clip(scaled + zp_out, act_min, act_max).astype(np.int16)
                       if is_int16 else
                       np.clip(scaled + zp_out, act_min, act_max).astype(np.int8))
                ref_b = ref.tobytes(order="C")
                params_b = (struct.pack("<iii", zp_out, act_min, act_max)
                            + mult_arr.astype("<i4").tobytes()
                            + shift_arr.astype(np.int8).tobytes()
                            + bias_eff_i32.tobytes())
                if corr_arr is not None:
                    params_b += corr_arr.astype("<i4").tobytes(order="C")
                conv_wgt_payload = wgt.tobytes(order="C") + params_b

        elif opname == "ADD" and in_t.Type() in FP_TFLITE_TYPES:
            # ---- v8.17: FP element-wise ADD (mobilenet_v3 residuals) ----
            # Match EweEngine::run_add_fp: a + b in FP32, clamp, store FP16.
            # Input-A is the chained FP16 ref output of the previous layer (so
            # the simulator's L1 holds bit-identical values when fusion runs);
            # input-B is freshly synthesized — sim and ref both see the same
            # rng tensor at this layer's dram_in slot.
            # v8.31: mdla7_model_runner.cpp can tile binary EWE ops over flat
            # contiguous chunks when a single row is wider than the H-tiler's
            # L1 budget, so large transformer-style FP ADDs no longer need a
            # compiler-side skip.
            in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
            in_b_arr = _input_array_for_tensor(int(op.Inputs(1)), (H, W, Cin), np.float16)

            try:
                ao = fb.AddOptions(); ao.Init(opt_table.Bytes, opt_table.Pos)
                fused = int(ao.FusedActivationFunction())
            except Exception:
                fused = 0
            INF = float("inf")
            if   fused == 1: act_min, act_max =  0.0, INF
            elif fused == 3: act_min, act_max =  0.0, 6.0
            elif fused == 2: act_min, act_max = -1.0, 1.0
            else:            act_min, act_max = -INF, INF
            amin_sent = np.float32(-3.4e38 if not np.isfinite(act_min) else float(act_min))
            amax_sent = np.float32( 3.4e38 if not np.isfinite(act_max) else float(act_max))

            a_f32 = in_arr.astype(np.float32)
            b_f32 = in_b_arr.astype(np.float32)
            out_f32 = a_f32 + b_f32
            out_f32 = np.maximum(out_f32, amin_sent)
            out_f32 = np.minimum(out_f32, amax_sent)
            ref = out_f32.astype(np.float16)
            op_kind = OP_ADD
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")

            # Wgt payload = input-B FP16 || 48 bytes (first 8 = act_min/max
            # sentinels, remaining 40 padded zeros to keep the same layout
            # mdla7_model_runner.cpp uses for INT ADD — `params_l1 = wgt_size - 48`).
            add_params_b = _fp_binary_params(amin_sent, amax_sent)
            add_wgt_payload = _fp_binary_b_payload(in_b_arr, add_params_b)
            layer_dtype  = DT_FP16
            is_int16     = False
            is_fp_layer  = True
            elem_size    = 2

        elif opname == "ADD":
            # ---- v6: TFLite int8 element-wise ADD (residual / SE merge) ----
            # Math (per TFLite reference):
            #   a' = (a - zp_a) << ls;  b' = (b - zp_b) << ls
            #   sa = MBQM(a', mult_a, shift_a)  ;  sb = MBQM(b', mult_b, shift_b)
            #   raw = sa + sb
            #   out = clip( MBQM(raw, mult_o, shift_o) + zp_out, [act_min, act_max] )
            # left_shift = 20 gives gemmlowp's typical int8 ADD headroom.
            in_a_t = in_t                                       # alias
            in_b_t = sg.Tensors(op.Inputs(1))
            qa = in_a_t.Quantization(); qb = in_b_t.Quantization(); qo = out_t.Quantization()
            scale_a = _qscale(qa)
            scale_b = _qscale(qb)
            scale_o = _qscale(qo)
            zp_a    = int  (qa.ZeroPoint(0)) if qa and qa.ZeroPointLength() else 0
            zp_b    = int  (qb.ZeroPoint(0)) if qb and qb.ZeroPointLength() else 0
            zp_o    = int  (qo.ZeroPoint(0)) if qo and qo.ZeroPointLength() else 0
            # Asymmetric uint8 needs a +128 shift (sim stores int8) — same trick
            # as the conv path. zp values then collapse symmetrically.
            shift_uint8_a = 128 if in_a_t.Type() == fb.TensorType.UINT8 else 0
            shift_uint8_b = 128 if in_b_t.Type() == fb.TensorType.UINT8 else 0
            shift_uint8_o = 128 if out_t.Type()  == fb.TensorType.UINT8 else 0
            zp_a_eff = zp_a - shift_uint8_a
            zp_b_eff = zp_b - shift_uint8_b
            zp_o_eff = zp_o - shift_uint8_o

            # ADD options for fused activation + clamp range.
            try:
                ao = fb.AddOptions(); ao.Init(opt_table.Bytes, opt_table.Pos)
                fused = int(ao.FusedActivationFunction())
            except Exception:
                fused = 0
            DT_MIN = -32768 if is_int16 else -128
            DT_MAX =  32767 if is_int16 else  127
            if shift_uint8_o:
                OUT_MIN, OUT_MAX = 0, 255
            else:
                OUT_MIN, OUT_MAX = DT_MIN, DT_MAX
            act_min, act_max = OUT_MIN, OUT_MAX
            if fused == 1:
                act_min = max(OUT_MIN, zp_o)
            elif fused == 3:
                act_min = max(OUT_MIN, zp_o)
                act_max = min(OUT_MAX, zp_o + int(round(6.0 / scale_o)))
            elif fused == 2:
                act_min = max(OUT_MIN, zp_o + int(round(-1.0 / scale_o)))
                act_max = min(OUT_MAX, zp_o + int(round( 1.0 / scale_o)))
            act_min -= shift_uint8_o
            act_max -= shift_uint8_o
            act_min = max(DT_MIN, act_min)
            act_max = min(DT_MAX, act_max)

            # Multipliers — gemmlowp-style left_shift 20.
            left_shift = 20
            twice_max = 2.0 * max(scale_a, scale_b)
            r_mult_a = scale_a / twice_max
            r_mult_b = scale_b / twice_max
            r_mult_o = twice_max / ((1 << left_shift) * scale_o)
            mq_a, sh_a = quantize_multiplier(r_mult_a)
            mq_b, sh_b = quantize_multiplier(r_mult_b)
            mq_o, sh_o = quantize_multiplier(r_mult_o)

            # Synthesize input B (input A reuses the per-layer in_arr).
            in_b_arr = _input_array_for_tensor(
                int(op.Inputs(1)), (H, W, Cin), np.int16 if is_int16 else np.int8)
            # Numpy reference (matches ewe_pool.h::run_add).
            a_v = (in_arr.astype(np.int32) - zp_a_eff) << left_shift
            b_v = (in_b_arr.astype(np.int32) - zp_b_eff) << left_shift
            sa  = multiply_by_quantized_multiplier_np(a_v, mq_a, sh_a)
            sb  = multiply_by_quantized_multiplier_np(b_v, mq_b, sh_b)
            raw = sa.astype(np.int64) + sb.astype(np.int64)
            raw = np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int32)
            out = multiply_by_quantized_multiplier_np(raw, mq_o, sh_o) + zp_o_eff
            ref = np.clip(out, act_min, act_max).astype(np.int16 if is_int16 else np.int8)
            op_kind = OP_ADD
            OH, OW, OC = H, W, Cin              # ADD preserves shape (no broadcast yet)
            ref_b = ref.tobytes(order="C")

            add_params_b = struct.pack("<iiiiiiiiiiii",
                                        zp_a_eff, zp_b_eff, zp_o_eff,
                                        mq_a, sh_a, mq_b, sh_b,
                                        mq_o, sh_o,
                                        left_shift, act_min, act_max)
            add_wgt_payload = in_b_arr.tobytes(order="C") + add_params_b

        elif opname in ("MUL", "SUB") and in_t.Type() in FP_TFLITE_TYPES:
            # ---- v8.30: FP element-wise MUL / SUB ----
            # Mirrors the ADD FP path. Sim runs run_binary_fp(op=1 for MUL,
            # 2 for SUB) which reads two FP16 operands, computes a*b or a-b in
            # FP32, clamps with the ±sentinel pair, writes FP16 to L1.
            # v8.31: large rows are handled by the sim's flat binary-EWE tiler.
            in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
            in_b_arr = _input_array_for_tensor(int(op.Inputs(1)), (H, W, Cin), np.float16)
            try:
                # Both MulOptions and SubOptions expose FusedActivationFunction.
                if opname == "MUL":
                    bo = fb.MulOptions(); bo.Init(opt_table.Bytes, opt_table.Pos)
                else:
                    bo = fb.SubOptions(); bo.Init(opt_table.Bytes, opt_table.Pos)
                fused = int(bo.FusedActivationFunction())
            except Exception:
                fused = 0
            INF = float("inf")
            if   fused == 1: act_min, act_max =  0.0, INF
            elif fused == 3: act_min, act_max =  0.0, 6.0
            elif fused == 2: act_min, act_max = -1.0, 1.0
            else:            act_min, act_max = -INF, INF
            amin_sent = np.float32(-3.4e38 if not np.isfinite(act_min) else float(act_min))
            amax_sent = np.float32( 3.4e38 if not np.isfinite(act_max) else float(act_max))
            a_f32 = in_arr.astype(np.float32)
            b_f32 = in_b_arr.astype(np.float32)
            out_f32 = (a_f32 * b_f32) if opname == "MUL" else (a_f32 - b_f32)
            out_f32 = np.maximum(out_f32, amin_sent)
            out_f32 = np.minimum(out_f32, amax_sent)
            ref = out_f32.astype(np.float16)
            op_kind = OP_MUL if opname == "MUL" else OP_SUB
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")
            add_params_b = _fp_binary_params(amin_sent, amax_sent)
            add_wgt_payload = _fp_binary_b_payload(in_b_arr, add_params_b)
            layer_dtype  = DT_FP16
            is_int16     = False
            is_fp_layer  = True
            elem_size    = 2

        elif opname == "MUL":
            # ---- v8.30: TFLite int8 MUL ----
            #   raw   = (a - zp_a) * (b - zp_b)
            #   out_q = MBQM(raw, mq, sh) + zp_o
            #   out   = clip(out_q, [act_min, act_max])
            in_a_t = in_t
            in_b_t = sg.Tensors(op.Inputs(1))
            qa = in_a_t.Quantization(); qb = in_b_t.Quantization(); qo = out_t.Quantization()
            scale_a = _qscale(qa)
            scale_b = _qscale(qb)
            scale_o = _qscale(qo)
            zp_a    = int  (qa.ZeroPoint(0)) if qa and qa.ZeroPointLength() else 0
            zp_b    = int  (qb.ZeroPoint(0)) if qb and qb.ZeroPointLength() else 0
            zp_o    = int  (qo.ZeroPoint(0)) if qo and qo.ZeroPointLength() else 0
            shift_uint8_a = 128 if in_a_t.Type() == fb.TensorType.UINT8 else 0
            shift_uint8_b = 128 if in_b_t.Type() == fb.TensorType.UINT8 else 0
            shift_uint8_o = 128 if out_t.Type()  == fb.TensorType.UINT8 else 0
            zp_a_eff = zp_a - shift_uint8_a
            zp_b_eff = zp_b - shift_uint8_b
            zp_o_eff = zp_o - shift_uint8_o
            try:
                bo = fb.MulOptions(); bo.Init(opt_table.Bytes, opt_table.Pos)
                fused = int(bo.FusedActivationFunction())
            except Exception:
                fused = 0
            DT_MIN = -32768 if is_int16 else -128
            DT_MAX =  32767 if is_int16 else  127
            if shift_uint8_o:
                OUT_MIN, OUT_MAX = 0, 255
            else:
                OUT_MIN, OUT_MAX = DT_MIN, DT_MAX
            act_min, act_max = OUT_MIN, OUT_MAX
            if fused == 1:
                act_min = max(OUT_MIN, zp_o)
            elif fused == 3:
                act_min = max(OUT_MIN, zp_o)
                act_max = min(OUT_MAX, zp_o + int(round(6.0 / scale_o)))
            elif fused == 2:
                act_min = max(OUT_MIN, zp_o + int(round(-1.0 / scale_o)))
                act_max = min(OUT_MAX, zp_o + int(round( 1.0 / scale_o)))
            act_min -= shift_uint8_o
            act_max -= shift_uint8_o
            act_min = max(DT_MIN, act_min)
            act_max = min(DT_MAX, act_max)
            r_mult_o = scale_a * scale_b / scale_o
            mq_o, sh_o = quantize_multiplier(r_mult_o)
            in_b_arr = _input_array_for_tensor(
                int(op.Inputs(1)), (H, W, Cin), np.int16 if is_int16 else np.int8)
            a_v = in_arr.astype(np.int32) - zp_a_eff
            b_v = in_b_arr.astype(np.int32) - zp_b_eff
            raw = a_v * b_v
            out = multiply_by_quantized_multiplier_np(raw, mq_o, sh_o) + zp_o_eff
            ref = np.clip(out, act_min, act_max).astype(np.int16 if is_int16 else np.int8)
            op_kind = OP_MUL
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")
            # Same 12-int32 layout as ADD; mq_a/mq_b/left_shift unused here so
            # set to 0/1 (multiplier 0 would be a divide-by-zero in MBQM but
            # run_mul reads only p[7]/p[8]/p[10]/p[11] anyway).
            add_params_b = struct.pack("<iiiiiiiiiiii",
                                        zp_a_eff, zp_b_eff, zp_o_eff,
                                        0, 0, 0, 0,
                                        mq_o, sh_o,
                                        0, act_min, act_max)
            add_wgt_payload = in_b_arr.tobytes(order="C") + add_params_b

        elif opname == "SUB":
            # ---- v8.30: TFLite int8 SUB ----
            # Same gemmlowp 3-multiplier shape as ADD but with `sa - sb` in raw.
            in_a_t = in_t
            in_b_t = sg.Tensors(op.Inputs(1))
            qa = in_a_t.Quantization(); qb = in_b_t.Quantization(); qo = out_t.Quantization()
            scale_a = _qscale(qa)
            scale_b = _qscale(qb)
            scale_o = _qscale(qo)
            zp_a    = int  (qa.ZeroPoint(0)) if qa and qa.ZeroPointLength() else 0
            zp_b    = int  (qb.ZeroPoint(0)) if qb and qb.ZeroPointLength() else 0
            zp_o    = int  (qo.ZeroPoint(0)) if qo and qo.ZeroPointLength() else 0
            shift_uint8_a = 128 if in_a_t.Type() == fb.TensorType.UINT8 else 0
            shift_uint8_b = 128 if in_b_t.Type() == fb.TensorType.UINT8 else 0
            shift_uint8_o = 128 if out_t.Type()  == fb.TensorType.UINT8 else 0
            zp_a_eff = zp_a - shift_uint8_a
            zp_b_eff = zp_b - shift_uint8_b
            zp_o_eff = zp_o - shift_uint8_o
            try:
                bo = fb.SubOptions(); bo.Init(opt_table.Bytes, opt_table.Pos)
                fused = int(bo.FusedActivationFunction())
            except Exception:
                fused = 0
            DT_MIN = -32768 if is_int16 else -128
            DT_MAX =  32767 if is_int16 else  127
            if shift_uint8_o:
                OUT_MIN, OUT_MAX = 0, 255
            else:
                OUT_MIN, OUT_MAX = DT_MIN, DT_MAX
            act_min, act_max = OUT_MIN, OUT_MAX
            if fused == 1:
                act_min = max(OUT_MIN, zp_o)
            elif fused == 3:
                act_min = max(OUT_MIN, zp_o)
                act_max = min(OUT_MAX, zp_o + int(round(6.0 / scale_o)))
            elif fused == 2:
                act_min = max(OUT_MIN, zp_o + int(round(-1.0 / scale_o)))
                act_max = min(OUT_MAX, zp_o + int(round( 1.0 / scale_o)))
            act_min -= shift_uint8_o
            act_max -= shift_uint8_o
            act_min = max(DT_MIN, act_min)
            act_max = min(DT_MAX, act_max)
            left_shift = 20
            twice_max = 2.0 * max(scale_a, scale_b)
            r_mult_a = scale_a / twice_max
            r_mult_b = scale_b / twice_max
            r_mult_o = twice_max / ((1 << left_shift) * scale_o)
            mq_a, sh_a = quantize_multiplier(r_mult_a)
            mq_b, sh_b = quantize_multiplier(r_mult_b)
            mq_o, sh_o = quantize_multiplier(r_mult_o)
            in_b_arr = _input_array_for_tensor(
                int(op.Inputs(1)), (H, W, Cin), np.int16 if is_int16 else np.int8)
            a_v = (in_arr.astype(np.int32) - zp_a_eff) << left_shift
            b_v = (in_b_arr.astype(np.int32) - zp_b_eff) << left_shift
            sa = multiply_by_quantized_multiplier_np(a_v, mq_a, sh_a)
            sb = multiply_by_quantized_multiplier_np(b_v, mq_b, sh_b)
            raw = sa.astype(np.int64) - sb.astype(np.int64)
            raw = np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int32)
            out = multiply_by_quantized_multiplier_np(raw, mq_o, sh_o) + zp_o_eff
            ref = np.clip(out, act_min, act_max).astype(np.int16 if is_int16 else np.int8)
            op_kind = OP_SUB
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")
            add_params_b = struct.pack("<iiiiiiiiiiii",
                                        zp_a_eff, zp_b_eff, zp_o_eff,
                                        mq_a, sh_a, mq_b, sh_b,
                                        mq_o, sh_o,
                                        left_shift, act_min, act_max)
            add_wgt_payload = in_b_arr.tobytes(order="C") + add_params_b

        elif (opname in ("RSQRT", "TANH", "LOGISTIC", "HARD_SWISH", "GELU")
              and in_t.Type() not in FP_TFLITE_TYPES
              and not is_int16):
            # ---- v11: INT8 unary EWE via 256-byte LUT ----
            # All 256 possible input bytes are pre-evaluated through the
            # nonlinearity using TFLite quant params (dequant -> f(x) ->
            # requant) and packed into wgt_b. Runtime EWE engine becomes a
            # single byte-indexed table lookup, bit-exact against TFLite.
            qin  = in_t.Quantization()
            qout = out_t.Quantization()
            scale_in  = _qscale(qin)
            scale_out = _qscale(qout, 1.0 / 256.0 if opname == "LOGISTIC" else 1.0)
            zp_in     = int(qin.ZeroPoint(0))  if qin  and qin.ZeroPointLength()  else 0
            zp_out    = int(qout.ZeroPoint(0)) if qout and qout.ZeroPointLength() else 0
            shift_in  = 128 if in_t.Type()  == fb.TensorType.UINT8 else 0
            shift_out = 128 if out_t.Type() == fb.TensorType.UINT8 else 0
            zp_in_eff  = zp_in  - shift_in
            zp_out_eff = zp_out - shift_out

            # Build LUT[idx_uint8] = output_int8. Index by storage byte
            # (signed int8 reinterpreted as uint8): -128 -> 128, 0 -> 0,
            # 127 -> 127. Output is clipped to int8 range.
            q_idx = np.arange(0, 256, dtype=np.int32)         # 0..255 = uint8 idx
            q_signed = np.where(q_idx >= 128, q_idx - 256, q_idx)  # -128..127
            x_real = (q_signed - zp_in_eff).astype(np.float32) * np.float32(scale_in)
            if opname == "RSQRT":
                # Guard against domain errors: input <= 0 is undefined for RSQRT.
                # TFLite reference clamps to a tiny positive epsilon (1e-12).
                x_safe = np.maximum(x_real, np.float32(1.0e-12))
                y_real = np.float32(1.0) / np.sqrt(x_safe)
            elif opname == "TANH":
                y_real = np.tanh(x_real).astype(np.float32)
            elif opname == "LOGISTIC":
                y_real = np.float32(1.0) / (np.float32(1.0) + np.exp(-x_real, dtype=np.float32))
            elif opname == "HARD_SWISH":
                # y = x * relu6(x + 3) / 6 (TFLite spec)
                r = np.minimum(np.maximum(x_real + np.float32(3.0), np.float32(0.0)), np.float32(6.0))
                y_real = (x_real * r / np.float32(6.0)).astype(np.float32)
            else:  # GELU - tanh approximation
                k = np.float32(0.7978845608028654)        # sqrt(2/pi)
                c = np.float32(0.044715)
                u = k * (x_real + c * x_real * x_real * x_real)
                y_real = (np.float32(0.5) * x_real * (np.float32(1.0) + np.tanh(u))).astype(np.float32)
            q_out = np.floor(y_real / np.float32(scale_out) + np.float32(zp_out_eff) + np.float32(0.5)).astype(np.int32)
            q_out = np.clip(q_out, -128, 127).astype(np.int8)
            lut_b = q_out.tobytes(order="C")     # 256 bytes, indexed by uint8(input)

            # Apply the LUT to the actual input tensor to get the byte-identical
            # reference output. Use the same uint8 indexing convention.
            in_idx = in_arr.astype(np.int32)
            in_idx = np.where(in_idx < 0, in_idx + 256, in_idx).astype(np.uint8)
            ref = q_out[in_idx].astype(np.int8)
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")
            add_wgt_payload = lut_b
            op_kind = {
                "RSQRT":     OP_RSQRT,
                "TANH":      OP_TANH,
                "LOGISTIC":  OP_LOGISTIC,
                "HARD_SWISH": OP_HARD_SWISH,
                "GELU":      OP_GELU,
            }[opname]

        elif (opname in ("RSQRT", "TANH", "LOGISTIC", "HARD_SWISH", "GELU")
              and in_t.Type() not in FP_TFLITE_TYPES
              and is_int16):
            # ---- v11: INT16 unary EWE via 64K-entry LUT (128 KB / layer) ----
            # Same lowering pattern as INT8 but with 65536 input bins and INT16
            # output entries. compile-time only; runtime is still a single
            # indexed read per element, just 16-bit wide.
            qin  = in_t.Quantization()
            qout = out_t.Quantization()
            scale_in  = _qscale(qin)
            scale_out = _qscale(qout, 1.0 / 65536.0 if opname == "LOGISTIC" else 1.0)
            zp_in     = int(qin.ZeroPoint(0))  if qin  and qin.ZeroPointLength()  else 0
            zp_out    = int(qout.ZeroPoint(0)) if qout and qout.ZeroPointLength() else 0
            # INT16 has no UINT alias in TFLite; uint8 shifts collapse to 0.

            # Build LUT[idx_uint16] = output_int16. Index by storage word
            # (signed int16 reinterpreted as uint16): -32768 -> 32768, 0 -> 0,
            # 32767 -> 32767. Output is clipped to int16 range.
            q_idx = np.arange(0, 65536, dtype=np.int64)
            q_signed = np.where(q_idx >= 32768, q_idx - 65536, q_idx)
            x_real = (q_signed - zp_in).astype(np.float32) * np.float32(scale_in)
            if opname == "RSQRT":
                x_safe = np.maximum(x_real, np.float32(1.0e-12))
                y_real = np.float32(1.0) / np.sqrt(x_safe)
            elif opname == "TANH":
                y_real = np.tanh(x_real).astype(np.float32)
            elif opname == "LOGISTIC":
                y_real = np.float32(1.0) / (np.float32(1.0) + np.exp(-x_real, dtype=np.float32))
            elif opname == "HARD_SWISH":
                r = np.minimum(np.maximum(x_real + np.float32(3.0), np.float32(0.0)), np.float32(6.0))
                y_real = (x_real * r / np.float32(6.0)).astype(np.float32)
            else:  # GELU
                k = np.float32(0.7978845608028654)
                c = np.float32(0.044715)
                u = k * (x_real + c * x_real * x_real * x_real)
                y_real = (np.float32(0.5) * x_real * (np.float32(1.0) + np.tanh(u))).astype(np.float32)
            q_out = np.floor(y_real / np.float32(scale_out) + np.float32(zp_out) + np.float32(0.5)).astype(np.int64)
            q_out = np.clip(q_out, -32768, 32767).astype(np.int16)
            lut_b = q_out.tobytes(order="C")     # 131072 bytes (64K * 2)

            # Apply LUT to actual input tensor for ref bytes.
            in_idx = in_arr.astype(np.int64)
            in_idx = np.where(in_idx < 0, in_idx + 65536, in_idx).astype(np.uint16)
            ref = q_out[in_idx].astype(np.int16)
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")
            add_wgt_payload = lut_b
            op_kind = {
                "RSQRT":     OP_RSQRT,
                "TANH":      OP_TANH,
                "LOGISTIC":  OP_LOGISTIC,
                "HARD_SWISH": OP_HARD_SWISH,
                "GELU":      OP_GELU,
            }[opname]

        elif opname in ("HARD_SWISH", "GELU", "LOGISTIC") and in_t.Type() in FP_TFLITE_TYPES:
            # ---- v8.30: FP unary activation ----
            # HARD_SWISH: x * relu6(x + 3) / 6
            # GELU: tanh-approximation, 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            # LOGISTIC: sigmoid(x), used by EfficientNet swish gates.
            # Sim runs run_unary_fp(subtype=ES_HARD_SWISH/ES_GELU/ES_LOGISTIC) which mirrors
            # the same FP32 expression order, so output is FP16 byte-identical.
            in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
            x = in_arr.astype(np.float32)
            if opname == "GELU":
                k = np.float32(0.7978845608028654)        # sqrt(2/pi)
                c = np.float32(0.044715)
                tanhf = _libm_tanhf()
                y = np.empty_like(x, dtype=np.float32)
                x_flat = x.reshape(-1)
                y_flat = y.reshape(-1)
                for j, xv in enumerate(x_flat):
                    xv = np.float32(xv)
                    u = np.float32(k * np.float32(xv + np.float32(c * xv * xv * xv)))
                    tv = np.float32(tanhf(float(u))) if tanhf else np.float32(math.tanh(float(u)))
                    y_flat[j] = np.float32(np.float32(0.5) * xv * np.float32(np.float32(1.0) + tv))
                op_kind = OP_GELU
            elif opname == "LOGISTIC":
                expf = _libm_expf()
                y = np.empty_like(x, dtype=np.float32)
                x_flat = x.reshape(-1)
                y_flat = y.reshape(-1)
                for j, xv in enumerate(x_flat):
                    xv = np.float32(xv)
                    ev = np.float32(expf(float(np.float32(-xv)))) if expf else np.float32(math.exp(float(np.float32(-xv))))
                    y_flat[j] = np.float32(np.float32(1.0) / np.float32(np.float32(1.0) + ev))
                op_kind = OP_LOGISTIC
            else:
                r = np.minimum(np.maximum(x + 3.0, 0.0), 6.0)
                y = x * r / np.float32(6.0)
                op_kind = OP_HARD_SWISH
            amin_sent = np.float32(-3.4e38)
            amax_sent = np.float32( 3.4e38)
            y = np.maximum(y, amin_sent)
            y = np.minimum(y, amax_sent)
            ref = y.astype(np.float16)
            OH, OW, OC = H, W, Cin
            ref_b = ref.tobytes(order="C")
            # 8-byte clamp params blob (loaded into L1 PARAMS slot at addr 0).
            add_wgt_payload = struct.pack("<ff", float(amin_sent), float(amax_sent))
            layer_dtype = DT_FP16
            is_int16    = False
            is_fp_layer = True
            elem_size   = 2

        elif opname == "MEAN":
            # ---- v8.30: MEAN reducing over (H, W) — emit as global avg pool ----
            # Read axes from input(1) (constant int32 tensor). Skip non-spatial.
            # mobilenet_v3 has 8 such MEANs in its SE blocks.
            try:
                axes_t  = sg.Tensors(op.Inputs(1))
                axes_np = _tensor_array(axes_t)
                axes    = sorted(int(a) % 4 for a in (axes_np.flatten() if axes_np is not None else []))
            except Exception:
                axes = []
            if axes != [1, 2]:
                # Non-spatial MEAN (sequence/channel reductions used by
                # transformers and SD) is not representable as PoolEngine's
                # H/W-only window. Materialize it as a compiler fallback.
                OH, OW, OC = to_hwc(osh)
                ref = _synth_output_array(out_t, OH, OW, OC)
                H, W, Cin = OH, OW, OC
                in_arr = ref
                in_i8 = in_arr
                ref_b = ref.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                layer_dtype = TYPE_TO_DTYPE.get(out_t.Type(), layer_dtype)
                is_int16 = (layer_dtype == DT_INT16x16)
                is_fp_layer = layer_dtype in (DT_FP16, DT_BFP16, DT_FP8)
                elem_size = _elem_size_for_layer_dtype(layer_dtype)
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1
            else:
                Kh, Kw = H, W
                s_h, s_w = 1, 1
                pT = pB = pL = pR = 0
                OH, OW, OC = 1, 1, Cin
                op_kind = OP_AVG_POOL                    # routed through avg_pool path
                count_include_pad = False
                if in_t.Type() in FP_TFLITE_TYPES:
                    in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
                    ref = pool_fp_ref(in_arr, Kh, Kw, s_h, s_w,
                                      (pT, pB, pL, pR), op_kind, count_include_pad)
                    ref_b = ref.tobytes(order="C")
                    layer_dtype  = DT_FP16
                    is_int16     = False
                    is_fp_layer  = True
                    elem_size    = 2
                else:
                    ref = pool_int8_ref(in_i8, Kh, Kw, s_h, s_w,
                                        (pT, pB, pL, pR), op_kind, count_include_pad)
                    ref_b = ref.astype(np.int16 if is_int16 else np.int8).tobytes(order="C")
                if force_materialize_native_reason:
                    H, W, Cin = int(OH), int(OW), int(OC)
                    in_arr = ref.reshape(H, W, Cin).astype(ref.dtype, copy=False)
                    in_i8 = in_arr
                    ref_b = in_arr.tobytes(order="C")
                    op_kind = OP_MATERIALIZE
                    wgt = np.zeros((0,), dtype=np.int8)
                    Kh = Kw = s_h = s_w = 1
                    pT = pB = pL = pR = 0
                    group = 1

        elif opname in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            po = fb.Pool2DOptions(); po.Init(opt_table.Bytes, opt_table.Pos)
            Kh = int(po.FilterHeight()); Kw = int(po.FilterWidth())
            s_h = int(po.StrideH());     s_w = int(po.StrideW())
            OH, OW, _ = to_hwc(osh)
            OC = Cin
            pT, pB = _padding_to_pad(fb, po.Padding(), Kh, H, OH, s_h)
            pL, pR = _padding_to_pad(fb, po.Padding(), Kw, W, OW, s_w)
            op_kind = OP_AVG_POOL if opname.startswith("AVERAGE") else OP_MAX_POOL
            count_include_pad = False                 # TFLite default
            if in_t.Type() in FP_TFLITE_TYPES:
                # v8.17: FP path. Pull a chained FP16 input or synth fresh.
                in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
                ref = pool_fp_ref(in_arr, Kh, Kw, s_h, s_w,
                                  (pT, pB, pL, pR), op_kind, count_include_pad)
                ref_b = ref.tobytes(order="C")
                layer_dtype  = DT_FP16
                is_int16     = False
                is_fp_layer  = True
                elem_size    = 2
            else:
                ref = pool_int8_ref(in_i8, Kh, Kw, s_h, s_w,
                                    (pT, pB, pL, pR), op_kind, count_include_pad)
                ref_b = ref.astype(np.int16 if is_int16 else np.int8).tobytes(order="C")
            if force_materialize_native_reason:
                H, W, Cin = int(OH), int(OW), int(OC)
                in_arr = ref.reshape(H, W, Cin).astype(ref.dtype, copy=False)
                in_i8 = in_arr
                ref_b = in_arr.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1

        elif opname == "SOFTMAX":
            op_kind = OP_SOFTMAX
            OH, OW, OC = H, W, Cin                    # softmax preserves shape
            if in_t.Type() in FP_TFLITE_TYPES:
                # v8.28: FP softmax. Pull a chained FP16 input or synth fresh
                # (mirrors the FP CONV/POOL path). compute_fp in EWE engine
                # uses the same loop order as softmax_fp_ref so output is
                # FP16 byte-identical.
                in_arr = _input_or_last(int(op.Inputs(0)), (H, W, Cin), np.float16)
                in_i8 = in_arr
                ref = softmax_fp_ref(in_arr)
                ref_b = ref.tobytes(order="C")
                layer_dtype = DT_FP16
                is_int16    = False
                is_fp_layer = True
                elem_size   = 2
            else:
                # v13: when MDLA7_DECOMPOSE_SOFTMAX=1 the runner emits a
                # dequant -> FP16 chain -> requant descriptor sequence per row;
                # the reference must follow the same FP16-storage trajectory or
                # the layer-end byte-compare reports spurious mismatches.
                in_q = in_t.Quantization()
                zp_in = (int(in_q.ZeroPoint(0))
                         if in_q and in_q.ZeroPointLength() else 0)
                if os.environ.get("MDLA7_DECOMPOSE_SOFTMAX"):
                    ref = softmax_int8_decomp_ref(in_i8, zp_in)
                else:
                    ref = softmax_int8_ref(in_i8)      # v1: real LUT-based
                ref_b = ref.astype(np.int8).tobytes(order="C")

        elif opname == "CONCATENATION":
            # ---- v6/v8.31: concat as a byte-preserving copy layer ----
            # Sim implements concat as a pure DRAM->DRAM copy of pre-arranged
            # bytes (compile_model concatenates the synth inputs in numpy and
            # the byte order already matches the NHWC output stream).
            # Asserts all inputs share the output's scale/zp; otherwise per-input
            # requant would be needed and is not yet supported.
            try:
                copts = fb.ConcatenationOptions(); copts.Init(opt_table.Bytes, opt_table.Pos)
                axis = int(copts.Axis())
            except Exception:
                axis = -1
            n_in = op.InputsLength()
            osh_full = list(_tensor_shape(out_t))
            concat_rank = len(osh_full) if osh_full else len(_tensor_shape(sg.Tensors(op.Inputs(0))))
            if concat_rank <= 0:
                concat_rank = 1
            if axis < 0:
                axis += concat_rank
            if axis < 0 or axis >= concat_rank:
                print(f"  layer {li:>2d}   concat  in={H}x{W}x{Cin} "
                      f"skipped (axis={axis} out of rank {concat_rank})")
                last_output_arr = None
                continue
            # Verify all inputs share output's scale/zp (else skip — requant TBD).
            oq = out_t.Quantization()
            scale_o = _qscale(oq)
            zp_o    = int  (oq.ZeroPoint(0)) if oq and oq.ZeroPointLength() else 0
            requant_needed = False
            for k in range(n_in):
                tk = sg.Tensors(op.Inputs(k))
                qk = tk.Quantization()
                sk = _qscale(qk)
                zk = int  (qk.ZeroPoint(0)) if qk and qk.ZeroPointLength() else 0
                if abs(sk - scale_o) > 1e-9 or zk != zp_o:
                    requant_needed = True
                    break
            if requant_needed:
                print(f"  layer {li:>2d}   concat  in={H}x{W}x{Cin} "
                      f"skipped (input requant on concat not yet implemented)")
                continue

            def _concat_synth(tk, k):
                shk = list(_tensor_shape(tk))
                if not shk:
                    shk = [1]
                # Match TFLite storage policy: FP source tensors are lowered to
                # FP16 bytes, INT16 stays 2B, INT8/UINT8 use byte storage.
                if tk.Type() in FP_TFLITE_TYPES:
                    dt = np.float16
                elif tk.Type() == fb.TensorType.INT16:
                    dt = np.int16
                else:
                    dt = np.int8
                size = int(np.prod(shk))
                tensor_idx = int(op.Inputs(k))
                arr = tensor_values.get(tensor_idx)
                if arr is not None and arr.size == size and arr.dtype == dt:
                    return arr.reshape(shk)
                if (k == 0
                    and _last_output_matches(tensor_idx, last_output_arr.shape if last_output_arr is not None else (), dt)
                    and last_output_arr.size == size):
                    return last_output_arr.reshape(shk)
                return _input_array_for_tensor(tensor_idx, shk, dt)

            # Build synthesised input slices in their native ranks, concatenate
            # along TFLite's axis, then canonicalise the result back to HWC for
            # the MDLA descriptor. This covers both image channel concat and
            # rank-2 sequence concat used by ETHZ_v5 lstm_float.
            slices = []
            for k in range(n_in):
                tk = sg.Tensors(op.Inputs(k))
                slices.append(_concat_synth(tk, k))
            try:
                ref_full = np.concatenate(slices, axis=axis)
            except ValueError as e:
                print(f"  layer {li:>2d}   concat  in={H}x{W}x{Cin} "
                      f"skipped (shape mismatch on axis={axis}: {e})")
                last_output_arr = None
                continue
            OH, OW, OC = to_hwc(osh_full)
            if ref_full.size != OH * OW * OC:
                print(f"  layer {li:>2d}   concat  in={H}x{W}x{Cin} "
                      f"skipped (output size mismatch: ref={ref_full.size} "
                      f"!= {OH*OW*OC})")
                last_output_arr = None
                continue
            ref = ref_full.reshape(OH, OW, OC)
            op_kind = OP_CONCAT
            ref_b = ref.tobytes(order="C")
            # Override in_arr/in_b so post-loop accounting stores the concat blob
            # as the layer "input" (sim will dram->dram copy it).
            in_arr = ref
            H, W, Cin = OH, OW, OC            # so descriptor dims match bytes
            if ref.dtype == np.float16:
                layer_dtype = DT_FP16
                is_int16 = False
                is_fp_layer = True
                elem_size = 2

        elif opname == "GATHER":
            # ---- v6: indexed lookup (embedding tables / mel-bin tables) ----
            # data:    [..., D]    (in_t, op.Inputs(0))      — int8/uint8 only
            # indices: [...]       (op.Inputs(1))            — int32
            # output:  data[indices]
            # We require the data tensor (params) to be int8/uint8; FP gather
            # is out of scope for the int8 sim path.
            data_t = in_t                                     # alias
            if data_t.Type() not in (fb.TensorType.INT8, fb.TensorType.UINT8):
                print(f"  layer {li:>2d}   gather  in={H}x{W}x{Cin} "
                      f"skipped (data dtype={data_t.Type()} non-int8)")
                continue
            try:
                go = fb.GatherOptions(); go.Init(opt_table.Bytes, opt_table.Pos)
                axis = int(go.Axis())
            except Exception:
                axis = 0
            data_arr = _tensor_array(data_t)
            if data_arr is None:
                print(f"  layer {li:>2d}   gather  skipped (data has no buffer)")
                continue
            if data_arr.dtype == np.uint8:
                data_arr = data_arr.view(np.int8)
            idx_t = sg.Tensors(op.Inputs(1))
            idx_arr = _tensor_array(idx_t)
            if idx_arr is None:
                # synthesise a small index tensor
                idx_shape = list(_tensor_shape(idx_t))
                if not idx_shape:
                    idx_shape = [16]
                idx_count = int(np.prod(idx_shape))
                idx_arr = rng.integers(0, data_arr.shape[axis],
                                       size=idx_count, dtype=np.int32).reshape(idx_shape)
            else:
                idx_arr = idx_arr.astype(np.int32)
            ref = np.take(data_arr, idx_arr, axis=axis).astype(np.int8)
            op_kind = OP_GATHER
            # Canonicalise output to (H, W, C) for the print line.
            osh_full = list(_tensor_shape(out_t))
            OH, OW, OC = to_hwc(osh_full)
            assert ref.size == OH * OW * OC, \
                f"gather size mismatch: ref={ref.size} osh={OH}x{OW}x{OC}"
            ref_b = ref.tobytes(order="C")
            in_arr = ref                                      # passthrough copy
            Cin = OC

        elif opname == "RESHAPE":
            OH, OW, OC = to_hwc(osh)
            # v8.29: transformer-class models (llama2 / mobilenet_v3_b4 /
            # mobilevit_v2 / sam / swin / sd_decoder/encoder) hit
            # `H*W*Cin != OH*OW*OC` here because compile_model's shape
            # propagation doesn't follow attention's reshape patterns
            # (sequence-length collapse, multi-head split, batch broadcast).
            # The pre-v8.29 assertion killed the whole compile; downgrade to
            # a graceful skip so the rest of the graph still compiles +
            # downstream layers fall back to fresh-rng synth (no chain).
            if H * W * Cin != OH * OW * OC:
                ref = _synth_output_array(out_t, OH, OW, OC)
                H, W, Cin = OH, OW, OC
                in_arr = ref
                in_i8 = in_arr
                ref_b = ref.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                layer_dtype = TYPE_TO_DTYPE.get(out_t.Type(), layer_dtype)
                is_int16 = (layer_dtype == DT_INT16x16)
                is_fp_layer = layer_dtype in (DT_FP16, DT_BFP16, DT_FP8)
                elem_size = _elem_size_for_layer_dtype(layer_dtype)
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1
            else:
                op_kind = OP_RESHAPE
                # RESHAPE is byte-preserving, not int8-only. Keep FP16/INT16
                # storage width intact so L.ref_size matches the UDMA passthrough
                # length that mdla7_model_runner emits from L.in_size.
                ref = in_arr.reshape(OH * OW * OC).astype(in_arr.dtype, copy=False)
                ref_b = ref.tobytes(order="C")
        elif opname in ("SPACE_TO_DEPTH", "TRANSPOSE",
                        "SQUEEZE", "EXPAND_DIMS",
                        "SLICE", "STRIDED_SLICE", "SPLIT", "SPLIT_V",
                        "PAD", "PADV2", "PACK", "UNPACK", "TILE"):
            osh_full = list(_tensor_shape(out_t))
            OH, OW, OC = to_hwc(osh_full)
            src_for_tnps = np.asarray(in_arr)
            materialized_tnps = False
            try:
                if opname == "SPACE_TO_DEPTH":
                    try:
                        opt = fb.SpaceToDepthOptions(); opt.Init(opt_table.Bytes, opt_table.Pos)
                        block = int(opt.BlockSize())
                    except Exception:
                        block = max(1, H // max(OH, 1))
                    ref_full = in_arr.reshape(H // block, block, W // block, block,
                                              Cin).transpose(0, 2, 1, 3, 4)
                    ref_full = ref_full.reshape(H // block, W // block,
                                                Cin * block * block).astype(in_arr.dtype)
                    op_kind = OP_S2SPACE
                    Kh = Kw = block
                elif opname == "TRANSPOSE":
                    perm_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                    perm = _tensor_array(perm_t).astype(np.int32).tolist() if perm_t else None
                    src = np.asarray(in_arr)
                    if perm is None:
                        perm = list(reversed(range(src.ndim)))
                    if len(perm) != src.ndim:
                        raise ValueError("axes don't match array")
                    ref_full = np.transpose(src, axes=perm)
                    layout_wgt_payload = _pack_tnps_meta(
                        src.ndim, elem_size, src.shape, ref_full.shape, perm)
                    op_kind = OP_TRANSPOSE
                elif opname == "SQUEEZE":
                    ref_full = np.squeeze(np.asarray(in_arr))
                    op_kind = OP_SQUEEZE
                elif opname == "EXPAND_DIMS":
                    axis_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                    axis_a = _tensor_array(axis_t) if axis_t else None
                    axis = int(axis_a.reshape(-1)[0]) if axis_a is not None and axis_a.size else 0
                    ref_full = np.expand_dims(np.asarray(in_arr), axis=axis)
                    op_kind = OP_EXPAND_DIMS
                elif opname in ("SLICE", "STRIDED_SLICE"):
                    src = np.asarray(in_arr)
                    begin_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                    second_t = sg.Tensors(op.Inputs(2)) if op.InputsLength() > 2 else None
                    begin_a = _tensor_array(begin_t).astype(np.int32) if begin_t else None
                    second_a = _tensor_array(second_t).astype(np.int32) if second_t else None
                    stride_a = None
                    begin_mask = end_mask = 0
                    if opname == "STRIDED_SLICE":
                        stride_t = sg.Tensors(op.Inputs(3)) if op.InputsLength() > 3 else None
                        stride_a = _tensor_array(stride_t).astype(np.int32) if stride_t else None
                        try:
                            opt = fb.StridedSliceOptions(); opt.Init(opt_table.Bytes, opt_table.Pos)
                            begin_mask = int(opt.BeginMask())
                            end_mask = int(opt.EndMask())
                        except Exception:
                            pass
                    if begin_a is None or second_a is None:
                        ref_full = _synth_output_array(out_t, OH, OW, OC)
                        materialized_tnps = True
                    else:
                        begin_v = begin_a.reshape(-1).tolist()
                        second_v = second_a.reshape(-1).tolist()
                        if opname == "STRIDED_SLICE":
                            stride_v = (stride_a.reshape(-1).tolist()
                                        if stride_a is not None else [1] * len(begin_v))
                        else:
                            stride_v = [1] * len(begin_v)
                        while len(begin_v) > src.ndim:
                            begin_v = begin_v[1:]
                            second_v = second_v[1:]
                            stride_v = stride_v[1:]
                            begin_mask >>= 1
                            end_mask >>= 1
                        slices_np = []
                        meta_begin = []
                        meta_stride = []
                        for ax, b in enumerate(begin_v):
                            st = stride_v[ax] if ax < len(stride_v) else 1
                            dim = src.shape[ax]
                            if begin_mask & (1 << ax):
                                b = 0 if st > 0 else dim - 1
                            if b < 0:
                                b += dim
                            if opname == "STRIDED_SLICE":
                                e = second_v[ax] if ax < len(second_v) else dim
                                if end_mask & (1 << ax):
                                    e = dim if st > 0 else -1
                                elif e < 0:
                                    e += dim
                            else:
                                n = second_v[ax] if ax < len(second_v) else -1
                                e = dim if n < 0 else b + n
                            slices_np.append(slice(b, e, st))
                            meta_begin.append(b)
                            meta_stride.append(st)
                        ref_full = src[tuple(slices_np)]
                        layout_wgt_payload = _pack_tnps_meta(
                            src.ndim, elem_size, src.shape, ref_full.shape,
                            meta_begin, meta_stride)
                    op_kind = OP_STRIDED_SLICE if opname == "STRIDED_SLICE" else OP_SLICE
                elif opname in ("SPLIT", "SPLIT_V"):
                    src = np.asarray(in_arr)
                    axis_input = 0 if opname == "SPLIT" else 2
                    axis_t = sg.Tensors(op.Inputs(axis_input)) if op.InputsLength() > axis_input else None
                    axis_a = _tensor_array(axis_t) if axis_t else None
                    axis = int(axis_a.reshape(-1)[0]) if axis_a is not None and axis_a.size else 0
                    if axis < 0:
                        axis += src.ndim
                    if axis < 0 or axis >= src.ndim:
                        axis = 0
                    first_size = None
                    if opname == "SPLIT_V" and op.InputsLength() > 1:
                        size_t = sg.Tensors(op.Inputs(1))
                        size_a = _tensor_array(size_t)
                        if size_a is not None and size_a.size:
                            sizes = [int(x) for x in size_a.reshape(-1).tolist()]
                            first_size = sizes[0] if sizes else None
                    if first_size is None:
                        first_size = src.shape[axis] // max(1, op.OutputsLength())
                    slices_np = [slice(None)] * src.ndim
                    slices_np[axis] = slice(0, first_size, 1)
                    ref_full = src[tuple(slices_np)]
                    meta_begin = [0] * src.ndim
                    meta_stride = [1] * src.ndim
                    layout_wgt_payload = _pack_tnps_meta(
                        src.ndim, elem_size, src.shape, ref_full.shape,
                        meta_begin, meta_stride)
                    op_kind = OP_SPLIT
                elif opname in ("PAD", "PADV2"):
                    pad_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                    pad_a = _tensor_array(pad_t).astype(np.int32) if pad_t else None
                    pad_value = 0
                    if opname == "PADV2" and op.InputsLength() > 2:
                        val_a = _tensor_array(sg.Tensors(op.Inputs(2)))
                        if val_a is not None and val_a.size:
                            pad_value = val_a.reshape(-1)[0]
                    if pad_a is None:
                        ref_full = _synth_output_array(out_t, OH, OW, OC)
                    else:
                        pad_pairs = pad_a.reshape(-1, 2).tolist()
                        # Preserve spatial pad metadata for runtime PAD->CONV
                        # fusion. Canonical HWC uses the last three non-batch
                        # axes; channel padding remains materialized.
                        hwc_pad = pad_pairs[1:] if len(pad_pairs) >= 4 else pad_pairs
                        while len(hwc_pad) < 3:
                            hwc_pad = [[0, 0]] + hwc_pad
                        if len(hwc_pad) > 3:
                            hwc_pad = hwc_pad[-3:]
                        pT, pB = int(hwc_pad[0][0]), int(hwc_pad[0][1])
                        pL, pR = int(hwc_pad[1][0]), int(hwc_pad[1][1])
                        ref_full = np.pad(np.asarray(in_arr), hwc_pad,
                                          mode="constant", constant_values=pad_value)
                    op_kind = OP_PAD
                elif opname == "PACK":
                    axis = 0
                    try:
                        opt = fb.PackOptions(); opt.Init(opt_table.Bytes, opt_table.Pos)
                        axis = int(opt.Axis())
                    except Exception:
                        pass
                    parts = []
                    for k in range(op.InputsLength()):
                        tk = sg.Tensors(op.Inputs(k))
                        arr = _tensor_array(tk)
                        if arr is None:
                            if tk.Type() in FP_TFLITE_TYPES:
                                dt = np.float16
                            elif tk.Type() == fb.TensorType.INT16:
                                dt = np.int16
                            else:
                                dt = np.int8
                            arr = _input_or_last(int(op.Inputs(k)), _tensor_shape(tk), dt)
                        parts.append(np.asarray(arr))
                    ref_full = np.stack(parts, axis=axis)
                    op_kind = OP_PACK
                elif opname == "UNPACK":
                    axis = 0
                    try:
                        opt = fb.UnpackOptions(); opt.Init(opt_table.Bytes, opt_table.Pos)
                        axis = int(opt.Axis())
                    except Exception:
                        pass
                    # The current program format has one output tensor per layer.
                    # Model the first output; multi-output graph wiring remains
                    # represented by compiler-synthesized downstream inputs.
                    ref_full = np.take(np.asarray(in_arr), 0, axis=axis)
                    op_kind = OP_UNPACK
                elif opname == "TILE":
                    mult_t = sg.Tensors(op.Inputs(1)) if op.InputsLength() > 1 else None
                    mult_a = _tensor_array(mult_t).astype(np.int32) if mult_t else None
                    ref_full = (np.tile(np.asarray(in_arr), mult_a.reshape(-1).tolist())
                                if mult_a is not None else
                                _synth_output_array(out_t, OH, OW, OC))
                    op_kind = OP_TILE
                if np.asarray(ref_full).size != OH * OW * OC:
                    ref_full = np.asarray(ref_full).reshape(-1)
                    if ref_full.size != OH * OW * OC:
                        ref_full = _synth_output_array(out_t, OH, OW, OC)
                ref = np.asarray(ref_full).reshape(OH, OW, OC).astype(in_arr.dtype, copy=False)
            except Exception as e:
                print(f"  layer {li:>2d}  {opname.lower():>7s}  in={H}x{W}x{Cin} "
                      f"materialized by TNPS fallback ({e})")
                ref = _synth_output_array(out_t, OH, OW, OC)
                materialized_tnps = True
                layout_wgt_payload = None
                op_kind = {
                    "SPACE_TO_DEPTH": OP_S2SPACE,
                    "TRANSPOSE": OP_TRANSPOSE,
                    "SQUEEZE": OP_SQUEEZE,
                    "EXPAND_DIMS": OP_EXPAND_DIMS,
                    "SLICE": OP_SLICE,
                    "STRIDED_SLICE": OP_STRIDED_SLICE,
                    "SPLIT": OP_SPLIT,
                    "SPLIT_V": OP_SPLIT,
                    "PAD": OP_PAD,
                    "PADV2": OP_PAD,
                    "PACK": OP_PACK,
                    "UNPACK": OP_UNPACK,
                    "TILE": OP_TILE,
                }[opname]
            if materialized_tnps or op_kind not in (OP_S2SPACE, OP_TRANSPOSE, OP_SLICE, OP_STRIDED_SLICE, OP_SPLIT):
                H, W, Cin = OH, OW, OC
                in_arr = ref
            else:
                in_arr = src_for_tnps
                if op_kind == OP_SPLIT:
                    H, W, Cin = to_hwc(list(np.asarray(src_for_tnps).shape))
            in_i8 = in_arr
            ref_b = ref.tobytes(order="C")
            if force_materialize_native_reason:
                H, W, Cin = int(OH), int(OW), int(OC)
                in_arr = ref.reshape(H, W, Cin).astype(ref.dtype, copy=False)
                in_i8 = in_arr
                ref_b = in_arr.tobytes(order="C")
                op_kind = OP_MATERIALIZE
                wgt = np.zeros((0,), dtype=np.int8)
                layout_wgt_payload = None
                Kh = Kw = s_h = s_w = 1
                pT = pB = pL = pR = 0
                group = 1
        elif opname == "DEPTH_TO_SPACE":
            try:
                d2s = fb.DepthToSpaceOptions(); d2s.Init(opt_table.Bytes, opt_table.Pos)
                block = int(d2s.BlockSize())
            except Exception:
                block = max(1, OH // max(H, 1))
            if block < 1 or Cin % (block * block) != 0:
                print(f"  layer {li:>2d}  d2spac  in={H}x{W}x{Cin} "
                      f"skipped (invalid block={block})")
                last_output_arr = None
                continue
            OH, OW, OC = H * block, W * block, Cin // (block * block)
            reshaped = in_arr.reshape(H, W, block, block, OC)
            ref = reshaped.transpose(0, 2, 1, 3, 4).reshape(OH, OW, OC).astype(in_arr.dtype)
            op_kind = OP_D2SPACE
            Kh = Kw = block
            ref_b = ref.tobytes(order="C")
        elif opname in ("CONV_2D", "DEPTHWISE_CONV_2D") and is_fp_layer:
            pass    # v8: already handled by the FP block earlier in this loop
        else:
            raise SystemExit(f"unhandled op: {opname}")

        # Preserve native dtype bytes (int8 → 1B/elem, int16 → 2B/elem).
        in_b  = in_arr.tobytes(order="C")
        if layout_wgt_payload is not None:
            wgt_b = layout_wgt_payload
        elif op_kind in (OP_CONV, OP_DWCONV, OP_FC, OP_FC_BMM):
            wgt_b = conv_wgt_payload                      # weights + params blob
        elif op_kind in (OP_ADD, OP_MUL, OP_SUB,
                         OP_HARD_SWISH, OP_GELU, OP_LOGISTIC,
                         OP_RSQRT, OP_TANH):
            wgt_b = add_wgt_payload                       # input-B + params (binary)
                                                          # or 8-byte clamp params (FP unary)
                                                          # or 256-byte LUT (INT8 unary)
        elif wgt.size:
            wgt_b = wgt.tobytes(order="C")
        else:
            wgt_b = b""

        # zp_in_eff is meaningful only for conv-class ops (CONV/DWCONV/FC); 0 elsewhere.
        zp_in_eff_local = 0
        if op_kind in (OP_CONV, OP_DWCONV, OP_FC, OP_FC_BMM):
            zp_in_eff_local = int(zp_in_eff)
        # v8.29: descriptor LAYER_FMT stores each spatial / channel dim as
        # ushort (max 65535). sd_encoder_quant attention has a 65536-row
        # sequence (256×256 spatial→sequence flatten) which busts this on
        # downstream ADD/RESHAPE/CONV layers. Pre-skip any layer whose dims
        # would overflow the descriptor schema rather than die during
        # struct.pack at file-write time.
        DIM_MAX = 65535
        if (max(H, W, Cin) > DIM_MAX or max(OH, OW, OC) > DIM_MAX):
            flat_ref = np.asarray(ref).reshape(-1)
            packed_hwc = _pack_hwc_for_elems(flat_ref.size)
            if packed_hwc[0] * packed_hwc[1] * packed_hwc[2] != flat_ref.size:
                print(f"  layer {li:>2d}  {OP_NAME[op_kind]}  in={H}x{W}x{Cin} "
                      f"skipped (shape exceeds descriptor's ushort dim limit "
                      f"{DIM_MAX}; out={OH}x{OW}x{OC})")
                last_output_arr = None
                continue
            H, W, Cin = packed_hwc
            OH, OW, OC = packed_hwc
            in_arr = flat_ref.reshape(H, W, Cin).astype(flat_ref.dtype, copy=False)
            in_i8 = in_arr
            ref = in_arr
            ref_b = ref.tobytes(order="C")
            wgt = np.zeros((0,), dtype=np.int8)
            op_kind = OP_MATERIALIZE
            Kh = Kw = s_h = s_w = 1
            pT = pB = pL = pR = 0
            group = 1
        # LayerMeta stores k_h/k_w as uint8_t. For global spatial reductions
        # (MEAN routed as AVG_POOL, and TFLite global-ish pools), use 255 as a
        # pool-only sentinel meaning "full input dimension"; mdla7_model_runner.cpp
        # expands it before planning tiles and PoolEngine expands it at run time.
        k_h_meta, k_w_meta = Kh, Kw
        if op_kind in (OP_AVG_POOL, OP_MAX_POOL):
            if Kh > 255:
                if Kh != H:
                    print(f"  layer {li:>2d}  {OP_NAME[op_kind]}  in={H}x{W}x{Cin} "
                          f"skipped (pool k_h={Kh} exceeds descriptor byte and is not global)")
                    last_output_arr = None
                    continue
                k_h_meta = 255
            if Kw > 255:
                if Kw != W:
                    print(f"  layer {li:>2d}  {OP_NAME[op_kind]}  in={H}x{W}x{Cin} "
                          f"skipped (pool k_w={Kw} exceeds descriptor byte and is not global)")
                    last_output_arr = None
                    continue
                k_w_meta = 255
        input0_tensor = int(op.Inputs(primary_input_slot)) if op.InputsLength() > primary_input_slot else -1
        input_alias_layer = -1
        if (
            layers and input0_tensor >= 0 and last_output_tensor == input0_tensor
            and last_output_arr is not None
            and last_output_arr.shape == (int(H), int(W), int(Cin))
            and last_output_arr.dtype == np.asarray(in_arr).dtype
            and last_output_arr.tobytes(order="C") == in_b
        ):
            input_alias_layer = len(layers) - 1
        stored_in_b = b"" if input_alias_layer >= 0 else in_b

        budget_error = _program_budget_error(len(stored_in_b), len(wgt_b), len(ref_b))
        if budget_error:
            print(f"  layer {li:>2d}  {OP_NAME[op_kind]}  in={H}x{W}x{Cin} "
                  f"skipped ({budget_error}; stopping compile here to preserve "
                  f"downstream chain consistency)")
            last_output_arr = None
            break
        input1_tensor = -1
        for _slot in range(op.InputsLength()):
            if _slot != primary_input_slot:
                input1_tensor = int(op.Inputs(_slot))
                break
        output_tensor = int(op.Outputs(0)) if op.OutputsLength() > 0 else -1
        compiled_layer_idx = len(layers)
        layers.append(dict(
            in_h=H, in_w=W, in_c=Cin, out_h=OH, out_w=OW, out_c=OC,
            k_h=k_h_meta, k_w=k_w_meta, s_h=s_h, s_w=s_w,
            p_t=pT, p_b=pB, p_l=pL, p_r=pR,
            dram_in=DRAM_IN  + cur_i, dram_wgt=DRAM_WGT + cur_w,
            dram_out=DRAM_OUT + cur_o,
            in_size=len(in_b), wgt_size=len(wgt_b), ref_size=len(ref_b),
            in_off=in_off, wgt_off=wgt_off, ref_off=ref_off,
            in_alias_layer=input_alias_layer,
            group=group,
            op_kind=op_kind,
            dtype=layer_dtype,
            zp_in_eff=zp_in_eff_local,
            orig_op_index=orig_op_index,
            input0_tensor=input0_tensor,
            input1_tensor=input1_tensor,
            output_tensor=output_tensor,
        ))
        op_to_compiled[orig_op_index] = compiled_layer_idx
        cur_w  += len(wgt_b); cur_i  += len(stored_in_b); cur_o  += len(ref_b)
        wgt_off += len(wgt_b); in_off += len(stored_in_b); ref_off += len(ref_b)
        if input_alias_layer < 0:
            in_blobs.append(in_b)
        wgt_blobs.append(wgt_b); ref_blobs.append(ref_b)

        # v8.12: remember this layer's reference output for the next layer's
        # chain.  Reshape to (H, W, C) since `ref` may be flat (RESHAPE) or
        # multidim depending on op.
        try:
            if ref.ndim == 1 and OH * OW * OC == ref.size:
                last_output_arr = ref.reshape(OH, OW, OC)
            elif ref.ndim == 3:
                last_output_arr = ref
            else:
                last_output_arr = None      # break chain at unusual shapes
        except Exception:
            last_output_arr = None
        last_output_tensor = output_tensor if last_output_arr is not None else None
        if output_tensor >= 0:
            try:
                tensor_values[int(output_tensor)] = ref.reshape(-1).copy()
                if last_output_arr is not None:
                    tensor_values[int(output_tensor)] = last_output_arr.copy()
            except Exception:
                pass

        # Canonical line — byte-identical with mdla7_model_runner.cpp.
        nelem = OH * OW * OC
        if layer_dtype in (DT_FP16, DT_BFP16, DT_FP8):
            unit = "FP16"
        elif is_int16:
            unit = "INT16"
        else:
            unit = "INT8"
        # Materialized fallbacks lose their TFLite op identity in the op_kind
        # label (everything collapses to "matrlz"); preserve the original
        # opname as a parseable trailing tag so reports can show it.
        from_tag = f"  from={opname}" if op_kind == OP_MATERIALIZE else ""
        print(f"  layer {li:>2d}  {OP_NAME[op_kind]}  in={H}x{W}x{Cin}  k={Kh}x{Kw}  "
              f"s={s_h}x{s_w}  g={group}  out={OH}x{OW}x{OC}  "
              f"({nelem} {unit})  ready{from_tag}")

    # v8.23: now that we know cur_w / cur_i / cur_o totals, place each region
    # at a base that leaves no overlap with the next. 64 KB alignment between
    # regions keeps DRAM-row accounting tidy. Each layer's dram_* address gets
    # rewritten from its placeholder absolute (built around the old +64MB
    # constants) to placeholder_base + (placeholder_addr - placeholder_base)
    # under the new base.
    new_DRAM_WGT = DRAM_BASE
    new_DRAM_IN  = new_DRAM_WGT + _round_up(cur_w, REGION_ALIGN)
    new_DRAM_OUT = new_DRAM_IN  + _round_up(cur_i, REGION_ALIGN)
    for L in layers:
        L["dram_wgt"] = new_DRAM_WGT + (L["dram_wgt"] - DRAM_WGT)
        if L.get("in_alias_layer", -1) < 0:
            L["dram_in"]  = new_DRAM_IN  + (L["dram_in"]  - DRAM_IN)
        L["dram_out"] = new_DRAM_OUT + (L["dram_out"] - DRAM_OUT)
    for L in layers:
        alias_layer = int(L.get("in_alias_layer", -1))
        if 0 <= alias_layer < len(layers):
            L["dram_in"] = layers[alias_layer]["dram_out"]

    # v3 graph sidecar: tensor-level provenance.  Unsupported/compiled-away
    # unary ops such as YOLO LOGISTIC keep graph identity by resolving their
    # output producer back through input(0).  Consumers are likewise chased
    # through uncompiled ops so last-use reflects the original TFLite graph,
    # not just the reduced MDLA layer list.
    def _resolve_producer_layer(tensor_idx, seen_ops=None):
        if tensor_idx is None or tensor_idx < 0:
            return -1
        if seen_ops is None:
            seen_ops = set()
        prod_op_idx = producer_op_by_tensor.get(int(tensor_idx))
        if prod_op_idx is None:
            return -1
        if prod_op_idx in op_to_compiled:
            return int(op_to_compiled[prod_op_idx])
        if prod_op_idx in seen_ops:
            return -1
        seen_ops.add(prod_op_idx)
        prod_op = sg.Operators(prod_op_idx)
        if prod_op.InputsLength() <= 0:
            return -1
        return _resolve_producer_layer(int(prod_op.Inputs(0)), seen_ops)

    def _compiled_consumers(tensor_idx, seen_ops=None):
        if tensor_idx is None or tensor_idx < 0:
            return []
        if seen_ops is None:
            seen_ops = set()
        out = []
        for consumer_op_idx in consumers_by_tensor.get(int(tensor_idx), []):
            if consumer_op_idx in op_to_compiled:
                out.append(int(op_to_compiled[consumer_op_idx]))
                continue
            if consumer_op_idx in seen_ops:
                continue
            seen_ops.add(consumer_op_idx)
            consumer_op = sg.Operators(consumer_op_idx)
            for k in range(consumer_op.OutputsLength()):
                out.extend(_compiled_consumers(int(consumer_op.Outputs(k)), seen_ops))
        return out

    for L in layers:
        consumers = sorted(set(c for c in _compiled_consumers(L["output_tensor"])
                               if c >= 0 and c != op_to_compiled.get(L["orig_op_index"], -1)))
        L["producer0_layer"] = _resolve_producer_layer(L["input0_tensor"])
        L["producer1_layer"] = _resolve_producer_layer(L["input1_tensor"])
        L["consumer_count"] = len(consumers)
        L["first_consumer_layer"] = consumers[0] if consumers else -1
        L["last_consumer_layer"] = consumers[-1] if consumers else -1

    # Concatenated data section: inputs first, then weights, then refs.
    inputs_section = b"".join(in_blobs)
    weights_section = b"".join(wgt_blobs)
    refs_section    = b"".join(ref_blobs)
    # Adjust offsets to be relative to start of data section (inputs come first).
    base_w = len(inputs_section)
    base_r = base_w + len(weights_section)

    def _abs_layer_offsets(L, data_offset):
        alias_layer = int(L.get("in_alias_layer", -1))
        in_file_off = (
            data_offset + base_r + layers[alias_layer]["ref_off"]
            if 0 <= alias_layer < len(layers)
            else data_offset + L["in_off"]
        )
        return (
            in_file_off,
            data_offset + base_w + L["wgt_off"],
            data_offset + base_r + L["ref_off"],
        )

    graph_meta_offset_v3 = HEADER_SIZE + LAYER_SIZE_V3 * len(layers)
    data_offset_v3 = graph_meta_offset_v3 + GRAPH_META_SIZE * len(layers)
    max_file_end_v3 = data_offset_v3 + len(inputs_section) + len(weights_section) + len(refs_section)
    for L in layers:
        in_file_off, wgt_file_off, ref_file_off = _abs_layer_offsets(L, data_offset_v3)
        max_file_end_v3 = max(
            max_file_end_v3,
            in_file_off + L["in_size"],
            wgt_file_off + L["wgt_size"],
            ref_file_off + L["ref_size"],
        )

    image_version = VERSION_V4 if max_file_end_v3 > UINT32_MAX else VERSION_V3
    layer_fmt = LAYER_FMT_V4 if image_version >= VERSION_V4 else LAYER_FMT_V3
    layer_size = LAYER_SIZE_V4 if image_version >= VERSION_V4 else LAYER_SIZE_V3
    graph_meta_offset = HEADER_SIZE + layer_size * len(layers)
    data_offset = graph_meta_offset + GRAPH_META_SIZE * len(layers)
    if data_offset > UINT32_MAX:
        raise RuntimeError(f"program metadata {data_offset} bytes exceeds uint32 data_offset field")

    with open(args.output, "wb") as f:
        # header
        f.write(struct.pack(HEADER_FMT, MAGIC, image_version, len(layers), data_offset))
        # layer metas — store ABSOLUTE file offsets so C++ can use them as-is.
        for L in layers:
            in_file_off, wgt_file_off, ref_file_off = _abs_layer_offsets(L, data_offset)
            f.write(struct.pack(
                layer_fmt,
                L["in_h"], L["in_w"], L["in_c"], L["out_h"], L["out_w"], L["out_c"],
                L["k_h"], L["k_w"], L["s_h"], L["s_w"],
                L["p_t"], L["p_b"], L["p_l"], L["p_r"],
                L["dram_in"], L["dram_wgt"], L["dram_out"],
                L["in_size"], L["wgt_size"], L["ref_size"],
                in_file_off,
                wgt_file_off,
                ref_file_off,
                L["group"], L["op_kind"], L["dtype"],
                L["zp_in_eff"],
            ))
        for L in layers:
            f.write(struct.pack(
                GRAPH_META_FMT,
                L["input0_tensor"], L["input1_tensor"], L["output_tensor"],
                L["producer0_layer"], L["producer1_layer"],
                L["first_consumer_layer"], L["last_consumer_layer"],
                L["consumer_count"],
            ))
        # data
        f.write(inputs_section)
        f.write(weights_section)
        f.write(refs_section)

    sz = os.path.getsize(args.output)
    print(f"  → {args.output}  ({sz / 1024:.1f} KB,  {len(layers)} layers)")


if __name__ == "__main__":
    main()
