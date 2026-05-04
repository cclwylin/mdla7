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
    uint32 version = 2
    uint32 num_layers
    uint32 data_offset           -- byte offset to start of data section

  LayerMeta[num_layers] (64 byte each):
    uint16 in_h, in_w, in_c
    uint16 out_h, out_w, out_c
    uint8  k_h, k_w, s_h, s_w
    uint8  p_t, p_b, p_l, p_r
    uint32 dram_in,  dram_wgt,  dram_out   -- DRAM placement
    uint32 in_size,  wgt_size,  ref_size   -- bytes
    uint32 in_off,   wgt_off,   ref_off    -- offsets in DATA section
    uint32 _reserved[2]

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


MAGIC, VERSION = 0x374C444D, 2
HEADER_FMT     = "<IIII"                # 16 byte
HEADER_SIZE    = struct.calcsize(HEADER_FMT)
LAYER_FMT      = "<HHHHHHBBBBBBBBIIIIIIIIIHHHh"   # 64 byte (last short = zp_in_eff)
LAYER_SIZE     = struct.calcsize(LAYER_FMT)
assert LAYER_SIZE == 64, f"LayerMeta size mismatch: {LAYER_SIZE}"

# op_kind enum — must mirror C++ in test_model.cpp
OP_CONV     = 0
OP_DWCONV   = 1
OP_AVG_POOL = 2
OP_MAX_POOL = 3
OP_SOFTMAX  = 4
OP_RESHAPE  = 5
OP_FC       = 6           # FC mapped to 1x1 conv but displayed separately
OP_ADD      = 7           # element-wise binary add (residual / SE)
OP_CONCAT   = 8           # channel concat (Inception/DenseNet branch merge)
OP_GATHER   = 9           # indexed lookup (BERT embeddings, audio mel-bins)

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
           OP_FC:     "     fc",
           OP_ADD:    "    add",
           OP_CONCAT: " concat",
           OP_GATHER: " gather"}


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
                 "ADD", "CONCATENATION", "GATHER")


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
    if exponent <= 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + np.where(x < 0, 1, 0).astype(x.dtype)
    return (x >> exponent) + np.where(remainder > threshold, 1, 0).astype(x.dtype)


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
    boundaries even when zp_in != 0."""
    H, W, Cin = act_i8.shape
    OC, K_h, K_w, in_per_group = wgt_i8.shape
    out_per_group = OC // group
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - K_h) // s_h + 1
    OW = (W + pL + pR - K_w) // s_w + 1
    out = np.zeros((OH, OW, OC), dtype=np.int64)
    a = act_i8.astype(np.int64)
    w = wgt_i8.astype(np.int64).reshape(group, out_per_group, K_h, K_w, in_per_group)
    pad_const = int(pad_value)
    pad_tile_per_kk = None
    if pad_const != 0:
        # Precompute the kernel-position contribution from a fully-padded slot:
        #   pad_const * Σ_ic w[g, oc, kh, kw, ic]   (per (group, out_per_group, kh, kw))
        pad_w_per_kk = pad_const * w.sum(axis=4)            # [g, out_per_group, K_h, K_w]
    for oh in range(OH):
        for ow in range(OW):
            tile = np.zeros((group, out_per_group), dtype=np.int64)
            for kh in range(K_h):
                ih = oh * s_h + kh - pT
                ih_ok = (0 <= ih < H)
                for kw in range(K_w):
                    iw = ow * s_w + kw - pL
                    iw_ok = (0 <= iw < W)
                    if ih_ok and iw_ok:
                        a_in = a[ih, iw].reshape(group, in_per_group)
                        tile += np.einsum("gi,goi->go", a_in, w[:, :, kh, kw, :])
                    elif pad_const != 0:
                        tile += pad_w_per_kk[:, :, kh, kw]
            out[oh, ow] = tile.flatten()
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
            elif op_name == "CONCATENATION":
                # Concat along channel dim by default.
                total_c = 0
                for k in range(op.InputsLength()):
                    sh = shapes.get(op.Inputs(k))
                    if sh is None:
                        sh = _to_hwc(_tensor_shape(sg.Tensors(op.Inputs(k))))
                    total_c += sh[2]
                out_shape = (H, W, total_c)
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
    """Deterministic INT8 softmax — must match softmax_int8() in C++."""
    flat = logits_i8.reshape(-1).astype(np.int32)
    max_v = int(flat.max())
    diff = np.clip(max_v - flat, 0, 255)
    exp_q = SOFTMAX_LUT[diff]
    sum_q = int(exp_q.sum())
    if sum_q == 0: sum_q = 1
    out = (exp_q.astype(np.int64) * 127) // sum_q
    return np.clip(out, 0, 127).astype(np.int8).reshape(logits_i8.shape)


def pool_int8_ref(in_i8, k_h, k_w, s_h, s_w, pad, mode, count_include_pad):
    """Reference avg/max pool — must match POOL engine implementation byte-for-byte."""
    H, W, C = in_i8.shape
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - k_h) // s_h + 1
    OW = (W + pL + pR - k_w) // s_w + 1
    out = np.zeros((OH, OW, C), dtype=np.int8)
    for oh in range(OH):
        for ow in range(OW):
            for c in range(C):
                if mode == OP_MAX_POOL:
                    best = -128
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
                    if q > 127: q = 127
                    if q < -128: q = -128
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

    # v8: propagate tensor shapes through the graph so FP-quant models with
    # placeholder [1,1,1,C] static shapes get real (H, W, C) at every op.
    shape_dict = _infer_shapes(fb, model, sg)

    # Collect every supported op in execution order.
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        name = _opcode_name(fb, model, op)
        if name not in SUPPORTED_OPS:
            continue
        ops.append((name, op))
    if args.max_layers:
        ops = ops[: args.max_layers]
    if not ops:
        sys.exit("no supported op found in model")

    print(f"compile_model: {args.model}")
    op_counts = {}
    for name, _ in ops: op_counts[name] = op_counts.get(name, 0) + 1
    print(f"  {len(ops)} ops: " + ", ".join(f"{k}={v}" for k, v in op_counts.items()))

    # DRAM bump allocator: weights → inputs → outputs in disjoint regions.
    DRAM_BASE = 0x10000000
    DRAM_WGT  = DRAM_BASE + 0x00000000     # weights region
    DRAM_IN   = DRAM_BASE + 0x04000000     # +64 MB
    DRAM_OUT  = DRAM_BASE + 0x08000000     # +128 MB
    cur_w = cur_i = cur_o = 0
    in_off = wgt_off = ref_off = 0

    layers = []
    in_blobs, wgt_blobs, ref_blobs = [], [], []

    # v8.12: chain mode — when layer N+1's expected input shape & dtype match
    # layer N's reference output, reuse N's output as N+1's input (instead of a
    # fresh rng draw).  This makes the per-layer rng-synth align with what the
    # actual model would feed forward, AND lets test_model.cpp detect fusable
    # adjacent pairs and skip layer N+1's udma_r (the input is already in L1
    # at layer N's L1_OUT slot).  Models where the chain breaks (skipped FP
    # ADD / HARD_SWISH / etc.) just get a fresh rng draw at the break.
    last_output_arr = None
    rng = np.random.default_rng(0)
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

    for li, (opname, op) in enumerate(ops):
        in_t   = sg.Tensors(op.Inputs(0))
        out_t  = sg.Tensors(op.Outputs(0))

        # ---- shape extraction ----
        ish = list(_tensor_shape(in_t))
        osh = list(_tensor_shape(out_t))
        # Canonicalise to NHWC: pad with 1s on the left if rank<4, else take last 3 spatial dims.
        def to_hwc(shape):
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
        H, W, Cin   = _pick(to_hwc(ish), shape_dict.get(op.Inputs(0)))
        OH, OW, OC  = _pick(to_hwc(osh), shape_dict.get(op.Outputs(0)))

        # Defaults (overwritten per-op below)
        Kh = Kw = 1; s_h = s_w = 1
        pT = pB = pL = pR = 0
        group = 1
        op_kind = OP_CONV
        wgt = np.zeros((0,), dtype=np.int8)            # weights buffer (may be empty)
        fc_label = False                                # only set true for FC
        is_fp_layer = False                             # v8: set true inside CONV/DWCONV FP path
        zp_in_eff = 0                                   # int default; FP path keeps it at 0

        # v4.1: pick element width from the input tensor's TFLite dtype.
        layer_dtype = TYPE_TO_DTYPE.get(in_t.Type(), DT_INT8x8)
        is_int16    = (layer_dtype == DT_INT16x16)
        elem_size   = 2 if is_int16 else 1
        np_in_dt    = np.int16 if is_int16 else np.int8

        # v4.1 limitation: only CONV / DEPTHWISE_CONV in int16 path so far.
        if is_int16 and opname not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            print(f"  layer {li:>2d}  {opname.lower():>7s}  in={H}x{W}x{Cin} "
                  f"skipped (int16 {opname} not yet supported in v4.1)")
            continue
        # v8.17: FP path now also handles ADD (FP32 sum + clamp) and AVG/MAX
        # POOL (FP32 reduction in (kh,kw) order). RESHAPE/CONCAT/GATHER are
        # byte-passthrough. SOFTMAX and FULLY_CONNECTED are still int-only.
        if in_t.Type() in FP_TFLITE_TYPES and opname in ("SOFTMAX",
                                                          "FULLY_CONNECTED"):
            print(f"  layer {li:>2d}  {opname.lower():>7s}  in={H}x{W}x{Cin} "
                  f"skipped (FP {opname} not yet supported)")
            continue

        # v8.12: prefer the previous layer's reference output when its shape +
        # dtype match the current layer's expected input — enables L1-resident
        # fusion in test_model.cpp.  Falls back to fresh rng on first layer or
        # whenever the chain breaks (skipped op, shape mismatch).
        expected_dtype = (np.int16 if is_int16 else np.int8)   # FP path overrides this later
        if (last_output_arr is not None
            and last_output_arr.shape == (H, W, Cin)
            and last_output_arr.dtype == expected_dtype):
            in_arr = last_output_arr
        elif is_int16:
            in_arr = rng.integers(-128, 128, size=(H, W, Cin), dtype=np.int16)
        else:
            in_arr = rng.integers(-8, 8, size=(H, W, Cin), dtype=np.int8)
        in_i8 = in_arr        # legacy name kept for downstream readability

        opt_table = op.BuiltinOptions()  # may be None for some ops

        if opname in ("CONV_2D", "DEPTHWISE_CONV_2D"):
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
            if (last_output_arr is not None
                and last_output_arr.shape == (H, W, Cin)
                and last_output_arr.dtype == np.float16):
                in_arr = last_output_arr
            else:
                in_arr = (rng.standard_normal((H, W, Cin)) * 0.5).astype(np.float16)
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

        if opname in ("CONV_2D", "DEPTHWISE_CONV_2D") and not is_fp_layer:
            # ---- v1.2: extract per-tensor / per-channel quant params ----
            in_q = in_t.Quantization()
            wq   = wgt_t.Quantization()
            oq   = out_t.Quantization()
            scale_in  = float(in_q.Scale(0))   if in_q and in_q.ScaleLength() else 1.0
            scale_out = float(oq.Scale(0))     if oq and oq.ScaleLength()     else 1.0
            zp_out    = int  (oq.ZeroPoint(0)) if oq and oq.ZeroPointLength() else 0
            zp_in     = int  (in_q.ZeroPoint(0)) if in_q and in_q.ZeroPointLength() else 0
            # validate_tflite.py pre-shifts uint8 synth by +128 before feeding TFLite,
            # so the sim and TFLite agree on signed activations and zp_in is already
            # absorbed; only subtract zp_in for INT8 input dtype.
            zp_in_eff = zp_in if in_t.Type() == fb.TensorType.INT8 else 0
            # weight scales: per-channel array (TFLite v2) or single (v1).
            if wq and wq.ScaleLength() == OC:
                scales_w = np.array([wq.Scale(i) for i in range(OC)], dtype=np.float64)
            elif wq and wq.ScaleLength() == 1:
                scales_w = np.full(OC, float(wq.Scale(0)), dtype=np.float64)
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

        elif opname == "FULLY_CONNECTED":
            # Map FC → 1x1 CONV_2D so the chain (CONV → Requant) handles it.
            # TFLite FC tensors:
            #   input  [N, in_features]
            #   weight [out_features, in_features]
            #   output [N, out_features]
            wgt_t  = sg.Tensors(op.Inputs(1))
            wgt    = _tensor_array(wgt_t)
            if wgt is None:
                raise SystemExit(f"layer {li}: FC weights tensor has no buffer")
            # v7: same uint8 -> centered int8 mapping as conv path.
            if wgt.dtype == np.uint8:
                wgt = (wgt.astype(np.int16) - 128).astype(np.int8)
            FC_out, FC_in = wgt.shape
            # Reshape inputs / weights to a 1×1 conv.
            H = W = 1
            Cin = FC_in
            OH = OW = 1
            OC = FC_out
            Kh = Kw = 1
            s_h = s_w = 1
            pT = pB = pL = pR = 0
            group = 1
            wgt = wgt.reshape(OC, 1, 1, Cin)            # OHWI for our compute
            # v8.13: respect the chain — the top-of-loop in_arr is already
            # shape (1, 1, FC_in). Only fall back to rng if the chain isn't
            # a (1,1,Cin) match (e.g. shape changed via RESHAPE just before).
            expected_dtype_fc = np.int16 if is_int16 else np.int8
            if (last_output_arr is not None
                and last_output_arr.shape == (1, 1, Cin)
                and last_output_arr.dtype == expected_dtype_fc):
                in_arr = last_output_arr
            else:
                in_arr = (rng.integers(-128, 128, size=(1, 1, Cin), dtype=np.int16)
                          if is_int16
                          else rng.integers(-8, 8, size=(1, 1, Cin), dtype=np.int8))
            in_i8 = in_arr
            op_kind = OP_FC     # dispatched through CONV engine, labelled "fc"
            fc_label = True

            fc_opts = fb.FullyConnectedOptions()
            fc_opts.Init(opt_table.Bytes, opt_table.Pos)

            # Quant params (mirrors CONV path; FC uses per-tensor weight scale).
            in_q = in_t.Quantization()
            wq   = wgt_t.Quantization()
            oq   = out_t.Quantization()
            scale_in  = float(in_q.Scale(0))   if in_q and in_q.ScaleLength() else 1.0
            scale_out = float(oq.Scale(0))     if oq and oq.ScaleLength()     else 1.0
            zp_out    = int  (oq.ZeroPoint(0)) if oq and oq.ZeroPointLength() else 0
            zp_in     = int  (in_q.ZeroPoint(0)) if in_q and in_q.ZeroPointLength() else 0
            zp_in_eff = zp_in if in_t.Type() == fb.TensorType.INT8 else 0
            if wq and wq.ScaleLength() == OC:
                scales_w = np.array([wq.Scale(i) for i in range(OC)], dtype=np.float64)
            elif wq and wq.ScaleLength() == 1:
                scales_w = np.full(OC, float(wq.Scale(0)), dtype=np.float64)
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
            if (last_output_arr is not None
                and last_output_arr.shape == (H, W, Cin)
                and last_output_arr.dtype == np.float16):
                in_arr = last_output_arr
            else:
                in_arr = (rng.standard_normal((H, W, Cin)) * 0.5).astype(np.float16)
            in_b_arr = (rng.standard_normal((H, W, Cin)) * 0.5).astype(np.float16)

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
            # test_model.cpp uses for INT ADD — `params_l1 = wgt_size - 48`).
            add_params_b = (struct.pack("<ff", float(amin_sent), float(amax_sent))
                            + b"\x00" * 40)
            add_wgt_payload = in_b_arr.tobytes(order="C") + add_params_b
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
            scale_a = float(qa.Scale(0))     if qa and qa.ScaleLength()     else 1.0
            scale_b = float(qb.Scale(0))     if qb and qb.ScaleLength()     else 1.0
            scale_o = float(qo.Scale(0))     if qo and qo.ScaleLength()     else 1.0
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
            if is_int16:
                in_b_arr = rng.integers(-128, 128, size=(H, W, Cin), dtype=np.int16)
            else:
                in_b_arr = rng.integers(-8, 8, size=(H, W, Cin), dtype=np.int8)
            # Numpy reference (matches ewe_pool.h::run_add).
            a_v = (in_arr.astype(np.int32) - zp_a_eff) << left_shift
            b_v = (in_b_arr.astype(np.int32) - zp_b_eff) << left_shift
            sa  = multiply_by_quantized_multiplier_np(a_v, mq_a, sh_a)
            sb  = multiply_by_quantized_multiplier_np(b_v, mq_b, sh_b)
            raw = sa.astype(np.int64) + sb.astype(np.int64)
            raw = np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int32)
            out = multiply_by_quantized_multiplier_np(raw, mq_o, sh_o) + zp_o_eff
            ref = np.clip(out, act_min, act_max).astype(np.int8)
            op_kind = OP_ADD
            OH, OW, OC = H, W, Cin              # ADD preserves shape (no broadcast yet)
            ref_b = ref.tobytes(order="C")

            add_params_b = struct.pack("<iiiiiiiiiiii",
                                        zp_a_eff, zp_b_eff, zp_o_eff,
                                        mq_a, sh_a, mq_b, sh_b,
                                        mq_o, sh_o,
                                        left_shift, act_min, act_max)
            add_wgt_payload = in_b_arr.tobytes(order="C") + add_params_b

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
                if (last_output_arr is not None
                    and last_output_arr.shape == (H, W, Cin)
                    and last_output_arr.dtype == np.float16):
                    in_arr = last_output_arr
                else:
                    in_arr = (rng.standard_normal((H, W, Cin)) * 0.5).astype(np.float16)
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
                ref_b = ref.astype(np.int8).tobytes(order="C")

        elif opname == "SOFTMAX":
            op_kind = OP_SOFTMAX
            OH, OW, OC = H, W, Cin                    # softmax preserves shape
            ref = softmax_int8_ref(in_i8)              # v1: real LUT-based
            ref_b = ref.astype(np.int8).tobytes(order="C")

        elif opname == "CONCATENATION":
            # ---- v6: channel-axis concat (most common Inception case) ----
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
            # Canonicalise axis to NHWC channel axis = 3 (or -1).
            # Most TFLite models concat along the last axis (channel).
            if axis < 0:
                axis = axis + 4   # rank 4 typical
            channel_concat = (axis in (3, -1))
            if not channel_concat:
                print(f"  layer {li:>2d}   concat  in={H}x{W}x{Cin} "
                      f"skipped (axis={axis} non-channel concat)")
                continue
            # Verify all inputs share output's scale/zp (else skip — requant TBD).
            oq = out_t.Quantization()
            scale_o = float(oq.Scale(0))     if oq and oq.ScaleLength()     else 1.0
            zp_o    = int  (oq.ZeroPoint(0)) if oq and oq.ZeroPointLength() else 0
            requant_needed = False
            for k in range(n_in):
                tk = sg.Tensors(op.Inputs(k))
                qk = tk.Quantization()
                sk = float(qk.Scale(0))     if qk and qk.ScaleLength()     else 1.0
                zk = int  (qk.ZeroPoint(0)) if qk and qk.ZeroPointLength() else 0
                if abs(sk - scale_o) > 1e-9 or zk != zp_o:
                    requant_needed = True
                    break
            if requant_needed:
                print(f"  layer {li:>2d}   concat  in={H}x{W}x{Cin} "
                      f"skipped (input requant on concat not yet implemented)")
                continue

            # Build synthesised input slices — each (H, W, Cin_k); concat to (H, W, OC).
            slices = []
            for k in range(n_in):
                tk = sg.Tensors(op.Inputs(k))
                shk = list(_tensor_shape(tk))
                if len(shk) >= 4:
                    Hk, Wk, Ck = int(shk[1]), int(shk[2]), int(shk[3])
                else:
                    Hk, Wk, Ck = H, W, int(shk[-1])
                if (Hk, Wk) != (H, W):
                    raise SystemExit(f"layer {li}: concat slice {k} shape mismatch")
                if k == 0:
                    s_arr = in_arr           # reuse the first per-layer rng draw
                    if s_arr.shape != (H, W, Ck):
                        s_arr = (rng.integers(-128, 128, size=(H, W, Ck), dtype=np.int16)
                                 if is_int16
                                 else rng.integers(-8, 8, size=(H, W, Ck), dtype=np.int8))
                else:
                    s_arr = (rng.integers(-128, 128, size=(H, W, Ck), dtype=np.int16)
                             if is_int16
                             else rng.integers(-8, 8, size=(H, W, Ck), dtype=np.int8))
                slices.append(s_arr)
            ref = np.concatenate(slices, axis=-1)
            op_kind = OP_CONCAT
            OH, OW, OC = H, W, ref.shape[-1]
            ref_b = ref.tobytes(order="C")
            # Override in_arr/in_b so post-loop accounting stores the concat blob
            # as the layer "input" (sim will dram->dram copy it).
            in_arr = ref
            Cin = OC                          # so in_size matches ref_size

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
            op_kind = OP_RESHAPE
            OH, OW, OC = to_hwc(osh)
            assert H * W * Cin == OH * OW * OC, \
                f"reshape size mismatch: {H*W*Cin} -> {OH*OW*OC}"
            ref = in_i8.reshape(OH * OW * OC).astype(np.int8)
            ref_b = ref.tobytes(order="C")
        elif opname in ("CONV_2D", "DEPTHWISE_CONV_2D") and is_fp_layer:
            pass    # v8: already handled by the FP block earlier in this loop
        else:
            raise SystemExit(f"unhandled op: {opname}")

        # Preserve native dtype bytes (int8 → 1B/elem, int16 → 2B/elem).
        in_b  = in_arr.tobytes(order="C")
        if op_kind in (OP_CONV, OP_DWCONV, OP_FC):
            wgt_b = conv_wgt_payload                      # weights + params blob
        elif op_kind == OP_ADD:
            wgt_b = add_wgt_payload                       # input-B + ADD params blob
        elif wgt.size:
            wgt_b = wgt.tobytes(order="C")
        else:
            wgt_b = b""

        # zp_in_eff is meaningful only for conv-class ops (CONV/DWCONV/FC); 0 elsewhere.
        zp_in_eff_local = 0
        if op_kind in (OP_CONV, OP_DWCONV, OP_FC):
            zp_in_eff_local = int(zp_in_eff)
        layers.append(dict(
            in_h=H, in_w=W, in_c=Cin, out_h=OH, out_w=OW, out_c=OC,
            k_h=Kh, k_w=Kw, s_h=s_h, s_w=s_w,
            p_t=pT, p_b=pB, p_l=pL, p_r=pR,
            dram_in=DRAM_IN  + cur_i, dram_wgt=DRAM_WGT + cur_w,
            dram_out=DRAM_OUT + cur_o,
            in_size=len(in_b), wgt_size=len(wgt_b), ref_size=len(ref_b),
            in_off=in_off, wgt_off=wgt_off, ref_off=ref_off,
            group=group,
            op_kind=op_kind,
            dtype=layer_dtype,
            zp_in_eff=zp_in_eff_local,
        ))
        cur_w  += len(wgt_b); cur_i  += len(in_b); cur_o  += len(ref_b)
        wgt_off += len(wgt_b); in_off += len(in_b); ref_off += len(ref_b)
        in_blobs.append(in_b); wgt_blobs.append(wgt_b); ref_blobs.append(ref_b)

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

        # Canonical line — byte-identical with test_model.cpp.
        nelem = OH * OW * OC
        if is_fp_layer:
            unit = "FP16"
        elif is_int16:
            unit = "INT16"
        else:
            unit = "INT8"
        print(f"  layer {li:>2d}  {OP_NAME[op_kind]}  in={H}x{W}x{Cin}  k={Kh}x{Kw}  "
              f"s={s_h}x{s_w}  g={group}  out={OH}x{OW}x{OC}  "
              f"({nelem} {unit})  ready")

    # Concatenated data section: inputs first, then weights, then refs.
    inputs_section = b"".join(in_blobs)
    weights_section = b"".join(wgt_blobs)
    refs_section    = b"".join(ref_blobs)
    # Adjust offsets to be relative to start of data section (inputs come first).
    base_w = len(inputs_section)
    base_r = base_w + len(weights_section)

    data_offset = HEADER_SIZE + LAYER_SIZE * len(layers)

    with open(args.output, "wb") as f:
        # header
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, len(layers), data_offset))
        # layer metas — store ABSOLUTE file offsets so C++ can use them as-is.
        for L in layers:
            f.write(struct.pack(
                LAYER_FMT,
                L["in_h"], L["in_w"], L["in_c"], L["out_h"], L["out_w"], L["out_c"],
                L["k_h"], L["k_w"], L["s_h"], L["s_w"],
                L["p_t"], L["p_b"], L["p_l"], L["p_r"],
                L["dram_in"], L["dram_wgt"], L["dram_out"],
                L["in_size"], L["wgt_size"], L["ref_size"],
                data_offset + L["in_off"],
                data_offset + base_w + L["wgt_off"],
                data_offset + base_r + L["ref_off"],
                L["group"], L["op_kind"], L["dtype"],
                L["zp_in_eff"],
            ))
        # data
        f.write(inputs_section)
        f.write(weights_section)
        f.write(refs_section)

    sz = os.path.getsize(args.output)
    print(f"  → {args.output}  ({sz / 1024:.1f} KB,  {len(layers)} layers)")


if __name__ == "__main__":
    main()
