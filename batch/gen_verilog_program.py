#!/usr/bin/env python3
"""Generate a verilog host descriptor hex stream from an MDL7 .bin image."""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC_MDL7 = 0x374C444D
OP_DONE = 0
OP_CONV = 1
OP_REQUANT = 2
OP_EWE = 3
OP_POOL = 4
OP_TNPS = 5
OP_UDMA = 6
OP_L1CRC = 7

SMF_LOAD_A = 1
SMF_LOAD_B = 2
SMF_COMPUTE = 4
SMF_STORE = 8
SMF_FINAL_TILE = 16

OK_CONV = {0, 1, 6}
OK_POOL = {2, 3}
OK_SOFTMAX = 4
OK_ADD = 7
OK_MUL = 10
OK_SUB = 11
OK_HARD_SWISH = 12
OK_GELU = 13
OK_LOGISTIC = 27
OK_RSQRT = 28
OK_TANH = 29
OK_RESHAPE = 5
OK_CONCAT = 8
OK_GATHER = 9
OK_D2SPACE = 14
OK_MATERIALIZE = 15
OK_SHAPE = 31       # TFLite SHAPE: compile-time constant shape vector
OK_TRANSPOSE = 16
OK_S2SPACE = 17
OK_SQUEEZE = 18
OK_EXPAND_DIMS = 19
OK_SLICE = 20
OK_STRIDED_SLICE = 21
OK_PAD = 22
OK_PACK = 23
OK_UNPACK = 24
OK_TILE = 25
OK_SPLIT = 26

OK_EWE = {OK_ADD, OK_MUL, OK_SUB}
OK_FP_EWE = {OK_ADD, OK_MUL, OK_SUB, OK_LOGISTIC}

META_TNPS_OPS = {OK_TRANSPOSE, OK_SLICE, OK_STRIDED_SLICE, OK_SPLIT}
TNPS_OPS = {
    OK_RESHAPE,
    OK_CONCAT,
    OK_D2SPACE,
    OK_TRANSPOSE,
    OK_S2SPACE,
    OK_SQUEEZE,
    OK_EXPAND_DIMS,
    OK_SLICE,
    OK_STRIDED_SLICE,
    OK_PAD,
    OK_PACK,
    OK_UNPACK,
    OK_TILE,
    OK_SPLIT,
}
UDMA_OPS = {OK_GATHER, OK_MATERIALIZE, OK_SHAPE}
# v11: INT8 RSQRT/TANH/LOGISTIC ride the generic ref-CRC path until the
# RTL EWE engine grows its 256-byte LUT lookup. compile_model has already
# pre-evaluated the LUT in the .bin reference bytes, so byte-CRC against
# DRAM still proves the layer's output matches TFLite bit-exactly.
GENERIC_REF_OPS = TNPS_OPS | UDMA_OPS | {OK_SOFTMAX, OK_HARD_SWISH, OK_GELU,
                                         OK_RSQRT, OK_TANH, OK_LOGISTIC}

DT_INT16 = {2, 3, 4}
DT_FP = {8, 9, 10}
WORDS_PER_COMMAND = 32
MAX_FINAL_OUTPUT_SRAM_BYTES = 16 * 1024 * 1024
PROBE_DESCRIPTOR_FLAG = 1 << 15
READ_FROM_L1_FLAG = 1 << 11
MICROBLOCK_FLAG = 1 << 13
CONV_CHAIN_OUT_FLAG = 1 << 15   # word[3] bit 15 — CONV chain output enable
REQUANT_USE_CHAIN   = 1 << 12   # word[3] bit 12 — REQUANT pull from chain
# v12 Phase 6c: OW tile loop constants
CONV_OW_COUNT_SHIFT   = 16      # conv_tile_ow_count → word[8][31:16]
CONV_OW_COUNT_MASK    = 0xFFFF  # 16-bit field
# act_tile_col_stride → word[10][21:0]; safe in L1-mode (wgt_vec ignored)
DEFAULT_MAX_COMMANDS = 4096
DEFAULT_MAX_PAYLOAD_BYTES = 1 << 20
DEFAULT_CONV_SRAM_WINDOW_COMMANDS = 512
DEFAULT_CONV_SRAM_WINDOW_COUNT = 3
DEFAULT_CONV_SAMPLE_COUNT = 3
CONV_TILE_MAX_ELEMS = 255
CONV_TILE_MAX_OUTPUTS = 32768
CONV_PROBE_COMMANDS = 6
POOL_TILE_MAX_BYTES = 8192
BYTE_TILE_SWEEP_PROBES = 4
MICRO_TILE_BYTES = 1 << 20
FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 16777619


@dataclass
class Layer:
    index: int
    in_h: int
    in_w: int
    in_c: int
    out_h: int
    out_w: int
    out_c: int
    k_h: int
    k_w: int
    s_h: int
    s_w: int
    p_t: int
    p_b: int
    p_l: int
    p_r: int
    in_size: int
    wgt_size: int
    ref_size: int
    in_off: int
    wgt_off: int
    ref_off: int
    group: int
    op_kind: int
    dtype: int
    zp_in_eff: int
    tnps_meta: tuple[int, int, list[int], list[int], list[int], list[int]] | None = None


def elem_bytes(dtype: int) -> int:
    return 2 if dtype in DT_INT16 or dtype in DT_FP else 1


def i8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


def i16_at(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]


def pack_word(chunk: bytes) -> int:
    value = 0
    for idx, byte in enumerate(chunk):
        value |= byte << (idx * 8)
    return value


def rdi32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def rdi8(data: bytes, offset: int) -> int:
    return struct.unpack_from("<b", data, offset)[0]


def fnv_byte(crc: int, byte_value: int) -> int:
    return ((crc ^ (byte_value & 0xFF)) * FNV_PRIME) & 0xFFFF_FFFF


def fnv_repeated(byte_value: int, count: int) -> int:
    crc = FNV_OFFSET
    for _ in range(count):
        crc = fnv_byte(crc, byte_value)
    return crc


def fnv_bytes(data: bytes) -> int:
    crc = FNV_OFFSET
    for byte_value in data:
        crc = fnv_byte(crc, byte_value)
    return crc


def sample_bytes(data: bytes, offset: int, size: int, count: int = 16) -> bytes:
    return data[offset:offset + min(count, size)].ljust(count, b"\x00")


def conv2d_int8_window_sample(
    layer: Layer,
    data: bytes,
    max_count: int = 16,
    start_lane: int = 0,
    out_elem_index: int = 0,
) -> tuple[bytes, bytes, int, int, int, int, int, int, int, bool, int, int, int, int, bool, int]:
    """Return one NHWC/OHWI output-pixel sample window for INT8 CONV descriptors."""
    if (
        layer.in_h <= 0 or layer.in_w <= 0 or layer.in_c <= 0 or
        layer.out_c <= 0 or layer.k_h <= 0 or layer.k_w <= 0
    ):
        act_off = layer.in_off + min(start_lane, layer.in_size)
        wgt_off = layer.wgt_off + min(start_lane, layer.wgt_size)
        act = data[act_off:act_off + min(max_count, max(layer.in_size - start_lane, 0))]
        wgt = data[wgt_off:wgt_off + min(max_count, max(layer.wgt_size - start_lane, 0))]
        elem_count = min(len(act), len(wgt), max_count)
        return act.ljust(max_count, b"\x00"), wgt.ljust(max_count, b"\x00"), elem_count, 0, 0, 0, 0, 0, out_elem_index, False, 0, 0, 0, 1, False, 0

    act_values = bytearray()
    wgt_values = bytearray()
    out_area = max(layer.out_w * layer.out_c, 1)
    oh = out_elem_index // out_area
    ow = (out_elem_index % out_area) // max(layer.out_c, 1)
    oc = out_elem_index % max(layer.out_c, 1)
    last_kh = 0
    last_kw = 0
    last_ic = 0
    last_input_byte = 0
    last_weight_byte = 0
    first_input_byte = 0
    first_weight_byte = 0
    lane_index = 0
    for kh in range(layer.k_h):
        for kw in range(layer.k_w):
            for ic in range(layer.in_c):
                if lane_index < start_lane:
                    lane_index += 1
                    continue
                if len(act_values) >= max_count:
                    break
                ih = oh + kh
                iw = ow + kw
                lane_valid = 0 <= ih < layer.in_h and 0 <= iw < layer.in_w
                input_elem = ((ih * layer.in_w + iw) * layer.in_c) + ic
                weight_elem = (((kh * layer.k_w + kw) * layer.in_c + ic) * layer.out_c) + oc
                input_byte = input_elem
                weight_byte = weight_elem
                if (not lane_valid) or input_byte >= layer.in_size or weight_byte >= layer.wgt_size:
                    lane_index += 1
                    continue
                act_values.append(data[layer.in_off + input_byte])
                wgt_values.append(data[layer.wgt_off + weight_byte])
                if len(act_values) == 1:
                    first_input_byte = input_byte
                    first_weight_byte = weight_byte
                last_kh = kh
                last_kw = kw
                last_ic = ic
                last_input_byte = input_byte
                last_weight_byte = weight_byte
                lane_index += 1
            if len(act_values) >= max_count:
                break
        if len(act_values) >= max_count:
            break

    elem_count = min(len(act_values), len(wgt_values), max_count)
    valid = elem_count > 0
    remaining_outputs = max(layer.out_h * layer.out_w * layer.out_c - out_elem_index, 1)
    tile_count = min(remaining_outputs, 4)
    last_out_elem = out_elem_index + tile_count - 1
    last_out_area = max(layer.out_w * layer.out_c, 1)
    last_oh = last_out_elem // last_out_area
    last_ow = (last_out_elem % last_out_area) // max(layer.out_c, 1)
    last_valid_count = 0
    last_first_valid = False
    for rel_lane in range(elem_count):
        lane = start_lane + rel_lane
        kh = lane // max(layer.k_w * layer.in_c, 1)
        rem = lane % max(layer.k_w * layer.in_c, 1)
        kw = rem // max(layer.in_c, 1)
        ic = rem % max(layer.in_c, 1)
        ih = last_oh + kh
        iw = last_ow + kw
        lane_valid = (
            kh < layer.k_h and kw < layer.k_w and ic < layer.in_c and
            0 <= ih < layer.in_h and 0 <= iw < layer.in_w
        )
        if rel_lane == 0:
            last_first_valid = lane_valid
        if lane_valid:
            last_valid_count += 1
    return (
        bytes(act_values).ljust(max_count, b"\x00"),
        bytes(wgt_values).ljust(max_count, b"\x00"),
        elem_count,
        last_kh,
        last_kw,
        last_ic,
        last_input_byte,
        last_weight_byte,
        out_elem_index,
        valid,
        first_input_byte,
        first_weight_byte,
        elem_count,
        tile_count,
        last_first_valid,
        last_valid_count,
    )


@dataclass
class Int8ConvParams:
    zp_out: int
    act_min: int
    act_max: int
    mult: list[int]
    shift: list[int]
    bias: list[int]
    corr: bytes
    corr_per_oc: bool
    corr_per_pixel: bool


def int8_conv_params(layer: Layer, data: bytes) -> Int8ConvParams | None:
    if layer.out_c <= 0:
        return None
    group = layer.group or 1
    if group <= 0 or layer.in_c % group != 0 or layer.out_c % group != 0:
        return None
    kh = layer.k_h or 1
    kw = layer.k_w or 1
    in_per_group = layer.in_c // group
    wgt_bytes = layer.out_c * kh * kw * in_per_group
    min_params = 12 + layer.out_c * 4 + layer.out_c + layer.out_c * 4
    if wgt_bytes + min_params > layer.wgt_size:
        return None
    prm = layer.wgt_off + wgt_bytes
    if prm + min_params > len(data):
        return None
    mult_off = prm + 12
    shift_off = mult_off + layer.out_c * 4
    bias_off = shift_off + layer.out_c
    corr_off = bias_off + layer.out_c * 4
    corr_bytes = layer.wgt_size - (wgt_bytes + min_params)
    corr_per_oc = corr_bytes == layer.out_h * layer.out_w * layer.out_c * 4
    corr_per_pixel = corr_bytes == layer.out_h * layer.out_w * 4
    return Int8ConvParams(
        zp_out=rdi32(data, prm),
        act_min=rdi32(data, prm + 4),
        act_max=rdi32(data, prm + 8),
        mult=[rdi32(data, mult_off + oc * 4) for oc in range(layer.out_c)],
        shift=[rdi8(data, shift_off + oc) for oc in range(layer.out_c)],
        bias=[rdi32(data, bias_off + oc * 4) for oc in range(layer.out_c)],
        corr=data[corr_off:corr_off + corr_bytes],
        corr_per_oc=corr_per_oc,
        corr_per_pixel=corr_per_pixel,
    )


def int8_conv_corr(params: Int8ConvParams, layer: Layer, oh: int, ow: int, oc: int) -> int:
    if params.corr_per_oc:
        off = ((oh * layer.out_w + ow) * layer.out_c + oc) * 4
    elif params.corr_per_pixel:
        off = (oh * layer.out_w + ow) * 4
    else:
        return 0
    if off + 4 > len(params.corr):
        return 0
    return rdi32(params.corr, off)


def int8_conv_real_window_sample(
    layer: Layer,
    data: bytes,
    max_count: int,
    start_lane: int,
    out_elem_index: int,
) -> tuple[bytes, bytes, int, int, int, int, int, int, int, bool, int, int, int, int, bool, int] | None:
    group = layer.group or 1
    if (
        layer.in_h <= 0 or layer.in_w <= 0 or layer.in_c <= 0 or
        layer.out_h <= 0 or layer.out_w <= 0 or layer.out_c <= 0 or
        group <= 0 or layer.in_c % group != 0 or layer.out_c % group != 0
    ):
        return None
    kh_total = layer.k_h or 1
    kw_total = layer.k_w or 1
    sh = layer.s_h or 1
    sw = layer.s_w or 1
    in_per_group = layer.in_c // group
    out_per_group = layer.out_c // group
    output_elems = layer.out_h * layer.out_w * layer.out_c
    if out_elem_index >= output_elems:
        return None
    out_area = layer.out_w * layer.out_c
    oh = out_elem_index // out_area
    ow = (out_elem_index % out_area) // layer.out_c
    oc = out_elem_index % layer.out_c
    g = oc // out_per_group
    total_lanes = kh_total * kw_total * in_per_group
    if start_lane >= total_lanes:
        return None

    act_values = bytearray()
    wgt_values = bytearray()
    first_input_byte = 0
    first_weight_byte = 0
    last_input_byte = 0
    last_weight_byte = 0
    last_kh = 0
    last_kw = 0
    last_ic = 0
    valid_count = 0
    first_valid = False
    count = min(max_count, total_lanes - start_lane)
    for rel_lane in range(count):
        lane = start_lane + rel_lane
        kh = lane // (kw_total * in_per_group)
        rem = lane % (kw_total * in_per_group)
        kw = rem // in_per_group
        icr = rem % in_per_group
        ic = g * in_per_group + icr
        ih = oh * sh + kh - layer.p_t
        iw = ow * sw + kw - layer.p_l
        lane_valid = 0 <= ih < layer.in_h and 0 <= iw < layer.in_w
        input_byte = ((ih * layer.in_w + iw) * layer.in_c + ic) if lane_valid else 0
        weight_byte = (((oc * kh_total + kh) * kw_total + kw) * in_per_group + icr)
        if weight_byte >= layer.wgt_size:
            return None
        act_byte = data[layer.in_off + input_byte] if lane_valid and input_byte < layer.in_size else (layer.zp_in_eff & 0xFF)
        wgt_byte = data[layer.wgt_off + weight_byte]
        act_values.append(act_byte)
        wgt_values.append(wgt_byte)
        if rel_lane == 0:
            first_valid = lane_valid
            first_input_byte = input_byte
            first_weight_byte = weight_byte
        if lane_valid:
            valid_count += 1
        last_kh = kh
        last_kw = kw
        last_ic = icr
        last_input_byte = input_byte
        last_weight_byte = weight_byte

    return (
        bytes(act_values).ljust(max_count, b"\x00"),
        bytes(wgt_values).ljust(max_count, b"\x00"),
        count,
        last_kh,
        last_kw,
        last_ic,
        last_input_byte,
        last_weight_byte,
        out_elem_index,
        first_valid,
        first_input_byte,
        first_weight_byte,
        valid_count,
        1,
        first_valid,
        valid_count,
    )


def fp16_at(data: bytes, offset: int) -> float:
    return struct.unpack_from("<e", data, offset)[0]


def sim_bytes(value: int) -> int:
    cap = descriptor_for_layer.max_payload_bytes
    if cap <= 0:
        return max(value, 1)
    return max(1, min(value, cap))


def bounded_micro_bytes(value: int) -> int:
    if value <= 0:
        return 64
    return min(value, 4096)


def micro_slice_bytes(total: int, offset: int) -> int:
    if total <= offset:
        return 0
    return min(MICRO_TILE_BYTES, total - offset)


def ceil_div(value: int, denom: int) -> int:
    if value <= 0 or denom <= 0:
        return 1
    return (value + denom - 1) // denom


def op_kind_to_class(op_kind: int) -> int:
    if op_kind in OK_CONV:
        return OP_CONV
    if op_kind in OK_POOL:
        return OP_POOL
    if op_kind in OK_FP_EWE or op_kind in OK_EWE:
        return OP_EWE
    if op_kind in TNPS_OPS:
        return OP_TNPS
    if op_kind in UDMA_OPS:
        return OP_UDMA
    return OP_DONE


def micro_steps_for_class(op_class: int) -> int:
    if op_class == OP_CONV:
        return 5
    if op_class == OP_EWE:
        return 4
    if op_class in (OP_POOL, OP_TNPS):
        return 3
    return 1


def micro_step_op(op_class: int, step: int) -> int:
    if op_class == OP_CONV:
        return OP_CONV if step == 2 else OP_REQUANT if step == 3 else OP_UDMA
    if op_class == OP_EWE:
        return OP_EWE if step == 2 else OP_UDMA
    if op_class == OP_POOL:
        return OP_POOL if step == 1 else OP_UDMA
    if op_class == OP_TNPS:
        return OP_TNPS if step == 1 else OP_UDMA
    if op_class == OP_UDMA:
        return OP_UDMA
    return OP_DONE


def synth_micro_step_flags(op_class: int, step: int) -> int:
    if op_class == OP_CONV:
        return (SMF_LOAD_B, SMF_LOAD_A, SMF_COMPUTE, SMF_COMPUTE, SMF_STORE)[min(step, 4)]
    if op_class == OP_EWE:
        return (SMF_LOAD_A, SMF_LOAD_B, SMF_COMPUTE, SMF_STORE)[min(step, 3)]
    if op_class in (OP_POOL, OP_TNPS):
        return (SMF_LOAD_A, SMF_COMPUTE, SMF_STORE)[min(step, 2)]
    if op_class == OP_UDMA:
        return SMF_STORE
    return 0


def clamp_i32(value: int) -> int:
    return max(-(1 << 31), min((1 << 31) - 1, value))


def saturating_doubling_high_mul(a: int, b: int) -> int:
    if a == -(1 << 31) and b == -(1 << 31):
        return (1 << 31) - 1
    p = int(a) * int(b)
    nudge = (1 << 30) if p >= 0 else (1 - (1 << 30))
    return clamp_i32((p + nudge) >> 31)


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    if exponent <= 0:
        return clamp_i32(x)
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    shifted = x >> exponent
    return clamp_i32(shifted + (1 if remainder > threshold else 0))


def mbqm(x: int, multiplier: int, shift: int) -> int:
    left_shift = shift if shift > 0 else 0
    right_shift = 0 if shift > 0 else -shift
    shifted = clamp_i32(x << left_shift)
    high = saturating_doubling_high_mul(shifted, multiplier)
    return rounding_divide_by_pot(high, right_shift)


def clamp_i8(value: int) -> int:
    return max(-128, min(127, value))


def parse_layers(path: Path) -> list[Layer]:
    data = path.read_bytes()
    if len(data) < 16:
        raise SystemExit(f"program too small: {path}")
    magic, version, layers, _data_offset = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC_MDL7:
        raise SystemExit(f"bad MDL7 magic: {path}")
    if version not in (2, 3, 4):
        raise SystemExit(f"bad MDL7 version {version}: {path}")
    layer_size = 76 if version >= 4 else 64
    if len(data) < 16 + layers * layer_size:
        raise SystemExit(f"truncated layer table: {path}")

    out: list[Layer] = []
    for index in range(layers):
        off = 16 + index * layer_size
        in_h, in_w, in_c, out_h, out_w, out_c = struct.unpack_from("<HHHHHH", data, off)
        k_h, k_w, s_h, s_w, p_t, p_b, p_l, p_r = struct.unpack_from("<BBBBBBBB", data, off + 12)
        in_size, wgt_size, ref_size = struct.unpack_from("<III", data, off + 32)
        if version >= 4:
            in_off, wgt_off, ref_off = struct.unpack_from("<QQQ", data, off + 44)
            group = struct.unpack_from("<H", data, off + 68)[0]
            op_kind, dtype = struct.unpack_from("<HH", data, off + 70)
            zp_in_eff = struct.unpack_from("<h", data, off + 74)[0]
        else:
            in_off, wgt_off, ref_off = struct.unpack_from("<III", data, off + 44)
            group = struct.unpack_from("<H", data, off + 56)[0]
            op_kind, dtype = struct.unpack_from("<HH", data, off + 58)
            zp_in_eff = struct.unpack_from("<h", data, off + 62)[0]
        tnps_meta = None
        if wgt_size >= 104 and wgt_off + 104 <= len(data):
            meta_off = wgt_off
            rank, elem = struct.unpack_from("<II", data, meta_off)
            in_shape = list(struct.unpack_from("<6I", data, meta_off + 8))
            out_shape = list(struct.unpack_from("<6I", data, meta_off + 32))
            a_vals = list(struct.unpack_from("<6i", data, meta_off + 56))
            b_vals = list(struct.unpack_from("<6i", data, meta_off + 80))
            tnps_meta = (min(rank, 6), elem or 1, in_shape, out_shape, a_vals, b_vals)

        out.append(
            Layer(
                index=index,
                in_h=in_h,
                in_w=in_w,
                in_c=in_c,
                out_h=out_h,
                out_w=out_w,
                out_c=out_c,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
                p_t=p_t,
                p_b=p_b,
                p_l=p_l,
                p_r=p_r,
                in_size=in_size,
                wgt_size=wgt_size,
                ref_size=ref_size,
                in_off=in_off,
                wgt_off=wgt_off,
                ref_off=ref_off,
                group=group,
                op_kind=op_kind,
                dtype=dtype,
                zp_in_eff=zp_in_eff,
                tnps_meta=tnps_meta,
            )
        )
    return out


def tnps_sample(layer: Layer) -> tuple[int, int, int, int, bool]:
    elem = elem_bytes(layer.dtype)
    block = layer.k_h or layer.k_w or 1
    if layer.op_kind == OK_S2SPACE:
        sample_out = 2 if layer.out_h * layer.out_w * layer.out_c > 2 else 0
        oh = sample_out // (layer.out_w * layer.out_c)
        rem = sample_out % (layer.out_w * layer.out_c)
        ow = rem // layer.out_c
        oc = rem % layer.out_c
        if layer.in_c == 0 or block == 0:
            return 0, 0, 0, 0, False
        q = oc // layer.in_c
        ic = oc % layer.in_c
        bh = q // block
        bw = q % block
        ih = oh * block + bh
        iw = ow * block + bw
        src = ((ih * layer.in_w * layer.in_c) + (iw * layer.in_c) + ic) * elem
        dst = sample_out * elem
        return sample_out, 0, src, dst, True
    if layer.op_kind == OK_D2SPACE:
        sample_in = 2 if layer.in_h * layer.in_w * layer.in_c > 2 else 0
        ih = sample_in // (layer.in_w * layer.in_c)
        rem = sample_in % (layer.in_w * layer.in_c)
        iw = rem // layer.in_c
        ic = rem % layer.in_c
        if layer.out_c == 0 or block == 0:
            return 0, 0, 0, 0, False
        q = ic // layer.out_c
        oc = ic % layer.out_c
        bh = q // block
        bw = q % block
        oh = ih * block + bh
        ow = iw * block + bw
        src = sample_in * elem
        dst = ((oh * layer.out_w * layer.out_c) + (ow * layer.out_c) + oc) * elem
        return 0, sample_in, src, dst, True
    return 0, 0, 0, 0, False


def tnps_output_source_byte_offset(layer: Layer, out_byte_index: int) -> int | None:
    elem = elem_bytes(layer.dtype)
    if elem <= 0 or out_byte_index < 0 or out_byte_index >= layer.ref_size:
        return None
    if layer.op_kind == OK_RESHAPE:
        return out_byte_index if out_byte_index < layer.in_size else None
    block = layer.k_h or layer.k_w or 1
    out_elem = out_byte_index // elem
    byte_lane = out_byte_index % elem
    if layer.op_kind == OK_S2SPACE:
        if (
            layer.in_h <= 0 or layer.in_w <= 0 or layer.in_c <= 0 or
            layer.out_h <= 0 or layer.out_w <= 0 or layer.out_c <= 0 or
            block <= 0 or layer.out_c != layer.in_c * block * block
        ):
            return None
        out_area = layer.out_w * layer.out_c
        oh = out_elem // out_area
        rem = out_elem % out_area
        ow = rem // layer.out_c
        oc = rem % layer.out_c
        q = oc // layer.in_c
        ic = oc % layer.in_c
        bh = q // block
        bw = q % block
        ih = oh * block + bh
        iw = ow * block + bw
        if ih >= layer.in_h or iw >= layer.in_w:
            return None
        src_elem = (ih * layer.in_w * layer.in_c) + (iw * layer.in_c) + ic
        src_byte = src_elem * elem + byte_lane
        return src_byte if src_byte < layer.in_size else None
    if layer.op_kind == OK_D2SPACE:
        if (
            layer.in_h <= 0 or layer.in_w <= 0 or layer.in_c <= 0 or
            layer.out_h <= 0 or layer.out_w <= 0 or layer.out_c <= 0 or
            block <= 0 or layer.in_c != layer.out_c * block * block
        ):
            return None
        out_area = layer.out_w * layer.out_c
        oh = out_elem // out_area
        rem = out_elem % out_area
        ow = rem // layer.out_c
        oc = rem % layer.out_c
        ih = oh // block
        iw = ow // block
        bh = oh % block
        bw = ow % block
        q = bh * block + bw
        ic = q * layer.out_c + oc
        if ih >= layer.in_h or iw >= layer.in_w or ic >= layer.in_c:
            return None
        src_elem = (ih * layer.in_w * layer.in_c) + (iw * layer.in_c) + ic
        src_byte = src_elem * elem + byte_lane
        return src_byte if src_byte < layer.in_size else None
    return None


def tnps_output_descriptor(layer: Layer, ordinal: int, out_byte_index: int, byte_count: int = 1) -> list[int] | None:
    data = descriptor_for_layer.program_bytes
    byte_count = max(min(byte_count, 16 - (out_byte_index & 0xF), layer.ref_size - out_byte_index), 1)
    payload = bytearray()
    first_src_byte: int | None = None
    for lane in range(byte_count):
        src_byte = tnps_output_source_byte_offset(layer, out_byte_index + lane)
        if src_byte is None or layer.in_off + src_byte >= len(data):
            return None
        if first_src_byte is None:
            first_src_byte = src_byte
        payload.append(data[layer.in_off + src_byte])
    if first_src_byte is None:
        return None
    elem = elem_bytes(layer.dtype)
    out_elem_index = out_byte_index // max(elem, 1)
    src_elem_index = first_src_byte // max(elem, 1)
    block = layer.k_h or layer.k_w or 1
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x28) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_TNPS
    words[1] = len(payload)
    words[2] = addr
    words[3] = (1 << 6) | (0x2 if layer.op_kind == OK_S2SPACE else 0x0)
    padded = bytes(payload).ljust(16, b"\x00")
    for idx in range(4):
        words[4 + idx] = pack_word(padded[idx * 4:(idx + 1) * 4])
    words[6] = layer.in_h
    words[7] = layer.in_w
    words[8] = layer.in_c
    words[9] = layer.out_h
    words[10] = layer.out_w
    words[11] = layer.out_c
    words[12] = block
    words[13] = elem
    if layer.op_kind == OK_S2SPACE:
        words[14] = out_elem_index & 0xFFFF_FFFF
    elif layer.op_kind == OK_D2SPACE:
        words[15] = src_elem_index & 0xFFFF_FFFF
    words[16] = first_src_byte & 0xFFFF_FFFF
    words[17] = out_byte_index & 0xFFFF_FFFF
    words[18] = 1
    words[19] = layer.index
    words[27] = out_byte_index & 0xFFFF_FFFF
    return words


def tnps_output_descriptors(layer: Layer, ordinal: int, max_output_bytes: int) -> list[list[int]]:
    emit_bytes = min(max_output_bytes, layer.ref_size)
    descs: list[list[int]] = []
    out_byte_index = 0
    while out_byte_index < emit_bytes:
        chunk_bytes = min(16 - (out_byte_index & 0xF), emit_bytes - out_byte_index)
        desc = tnps_output_descriptor(layer, ordinal + len(descs), out_byte_index, chunk_bytes)
        if desc is None:
            break
        descs.append(desc)
        out_byte_index += desc[1]
    return descs


def tnps_final_bytes(final_descs: list[list[int]]) -> bytes:
    out = bytearray()
    for desc in final_descs:
        byte_count = max(min(desc[1], 16), 0)
        for lane in range(byte_count):
            word = desc[4 + lane // 4]
            out.append((word >> ((lane % 4) * 8)) & 0xFF)
    return bytes(out)


def tnps_contiguous_output_byte_count(layer: Layer, start_out_byte: int, max_count: int) -> int:
    first_src = tnps_output_source_byte_offset(layer, start_out_byte)
    if first_src is None:
        return 0
    max_count = min(max_count, 16 - (first_src & 0xF))
    count = 0
    for lane in range(max_count):
        src = tnps_output_source_byte_offset(layer, start_out_byte + lane)
        if src is None or src != first_src + lane:
            break
        count += 1
    return count


def closed_loop_tnps_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    max_payload_bytes: int,
    start_out_byte: int = 0,
) -> list[list[int]]:
    byte_count = tnps_contiguous_output_byte_count(
        layer,
        start_out_byte,
        max(1, min(layer.ref_size - start_out_byte, max_payload_bytes, 16)),
    )
    if byte_count <= 0:
        return []
    desc = tnps_output_descriptor(layer, ordinal, start_out_byte, byte_count)
    if desc is None:
        return []
    byte_count = desc[1]
    data = descriptor_for_layer.program_bytes
    expected = bytes(
        data[layer.in_off + tnps_output_source_byte_offset(layer, start_out_byte + lane)]
        for lane in range(byte_count)
    )
    if not expected:
        return []
    src_byte = desc[16]
    l1_sample_addr = 0x30000 + (((layer.index * 0x100) + (start_out_byte * 0x20)) & 0x2FFFF)
    l1_base = l1_sample_addr if layer.op_kind == OK_RESHAPE else (l1_sample_addr - src_byte) & 0x003F_FFFF
    l1_result = l1_sample_addr + 0x80
    descs = [
        udma_dram_to_l1_descriptor(
            layer,
            ordinal,
            layer.in_off + src_byte,
            l1_sample_addr,
            len(expected),
            SMF_LOAD_A,
        )
    ]
    desc[2] = l1_base
    desc[3] |= MICROBLOCK_FLAG
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off + start_out_byte,
            expected,
        )
    )
    return descs


def closed_loop_tnps_probes(layer: Layer, ordinal: int, result_dram_off: int,
                            max_payload_bytes: int, command_budget: int) -> list[list[int]]:
    descs: list[list[int]] = []
    elem = max(elem_bytes(layer.dtype), 1)
    output_elems = max(layer.ref_size // elem, 0)
    start_candidates = [
        elem_index * elem
        for elem_index in closed_loop_output_indices(output_elems, command_budget, 4)
    ][:BYTE_TILE_SWEEP_PROBES]
    for start_out_byte in start_candidates:
        probe = closed_loop_tnps_probe(
            layer,
            ordinal + len(descs),
            result_dram_off,
            max_payload_bytes,
            start_out_byte,
        )
        if not probe:
            continue
        if len(descs) + len(probe) > command_budget:
            break
        descs.extend(probe)
    return descs


def tnps_sramcrc_probe_descriptor(layer: Layer, ordinal: int, ref_bytes: bytes) -> list[int] | None:
    if not ref_bytes:
        return None
    elem = elem_bytes(layer.dtype)
    block = layer.k_h or layer.k_w or 1
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x28) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_TNPS
    words[1] = max(1, len(ref_bytes) * 8)
    words[2] = addr
    words[3] = (1 << 10) | (0x2 if layer.op_kind == OK_S2SPACE else 0x0)
    words[6] = layer.in_h
    words[7] = layer.in_w
    words[8] = layer.in_c
    words[9] = layer.out_h
    words[10] = layer.out_w
    words[11] = layer.out_c
    words[12] = block
    words[13] = elem
    words[19] = layer.index
    words[27] = 0
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    return words


def udma_output_descriptor(layer: Layer, ordinal: int, out_byte_index: int) -> list[int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.op_kind not in UDMA_OPS:
        return None
    if out_byte_index < 0 or out_byte_index >= layer.ref_size:
        return None
    if layer.ref_off + out_byte_index >= len(data):
        return None
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x30) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = 1
    words[2] = addr
    words[3] = 1 << 6
    words[4] = 1
    words[5] = 1
    words[6] = data[layer.ref_off + out_byte_index]
    words[19] = layer.index
    words[27] = out_byte_index & 0xFFFF_FFFF
    return words


def udma_output_descriptors(layer: Layer, ordinal: int, max_output_bytes: int) -> list[list[int]]:
    emit_bytes = min(max_output_bytes, layer.ref_size)
    descs: list[list[int]] = []
    for out_byte_index in range(emit_bytes):
        desc = udma_output_descriptor(layer, ordinal + len(descs), out_byte_index)
        if desc is None:
            break
        descs.append(desc)
    return descs


def udma_final_bytes(final_descs: list[list[int]]) -> bytes:
    return bytes(desc[6] & 0xFF for desc in final_descs)


def udma_sramcrc_probe_descriptor(layer: Layer, ordinal: int, ref_bytes: bytes) -> list[int] | None:
    if layer.op_kind not in UDMA_OPS or not ref_bytes:
        return None
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x30) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = max(1, len(ref_bytes) * 8)
    words[2] = addr
    words[3] = 1 << 10
    words[4] = max(1, len(ref_bytes))
    words[5] = 1
    words[19] = layer.index
    words[27] = 0
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    return words


def l1mesh_crc_probe_descriptor(ordinal: int, start_addr: int, ref_bytes: bytes) -> list[int] | None:
    if not ref_bytes:
        return None
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_L1CRC
    words[1] = len(ref_bytes)
    words[2] = start_addr & 0x003F_FFFF
    words[3] = 1 << 10
    words[19] = ordinal
    words[27] = start_addr & 0xFFFF_FFFF
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    return words


def udma_dram_to_l1_descriptor(
    layer: Layer,
    ordinal: int,
    dram_off: int,
    l1_addr: int,
    byte_count: int,
    stream_flags: int = SMF_LOAD_A,
) -> list[int]:
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = max(byte_count, 1)
    words[2] = l1_addr & 0x003F_FFFF
    words[3] = MICROBLOCK_FLAG
    words[4] = max(byte_count, 1)
    words[5] = 1
    words[19] = layer.index
    words[25] = dram_off & 0xFFFF_FFFF
    stamp_synth_microblock_metadata(words, layer.index, ordinal, ordinal, stream_flags)
    return words


def udma_l1_to_dram_descriptor(
    layer: Layer,
    ordinal: int,
    l1_addr: int,
    dram_off: int,
    byte_count: int,
    stream_flags: int = SMF_STORE,
) -> list[int]:
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = max(byte_count, 1)
    words[2] = l1_addr & 0x003F_FFFF
    words[3] = (1 << 0) | (1 << 6) | MICROBLOCK_FLAG
    words[4] = max(byte_count, 1)
    words[5] = 1
    words[19] = layer.index
    words[25] = dram_off & 0xFFFF_FFFF
    words[27] = 0
    stamp_synth_microblock_metadata(words, layer.index, ordinal, ordinal, stream_flags)
    return words


def closed_loop_result_check_descriptors(
    layer: Layer,
    ordinal: int,
    l1_result_addr: int,
    dram_result_off: int,
    ref_bytes: bytes,
) -> list[list[int]]:
    if not ref_bytes:
        return []
    descs: list[list[int]] = []
    descs.append(
        udma_l1_to_dram_descriptor(
            layer,
            ordinal + len(descs),
            l1_result_addr,
            dram_result_off,
            len(ref_bytes),
            SMF_STORE,
        )
    )
    descs.append(
        udma_dram_to_l1_descriptor(
            layer,
            ordinal + len(descs),
            dram_result_off,
            l1_result_addr + 0x1000,
            len(ref_bytes),
            SMF_LOAD_A,
        )
    )
    l1_probe_desc = l1mesh_crc_probe_descriptor(
        ordinal + len(descs),
        l1_result_addr + 0x1000,
        ref_bytes,
    )
    if l1_probe_desc is not None:
        l1_probe_desc[3] |= MICROBLOCK_FLAG
        stamp_synth_microblock_metadata(
            l1_probe_desc,
            layer.index,
            ordinal + len(descs),
            ordinal + len(descs),
            SMF_FINAL_TILE,
        )
        descs.append(l1_probe_desc)
    return descs


def closed_loop_result_dram_off(layer: Layer, ordinal: int) -> int:
    return 0x00E0_0000 + ((layer.index * 0x100 + ordinal * 0x20) & 0x0F_FFF0)


def closed_loop_ref_passthrough_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    max_bytes: int = 16,
) -> list[list[int]]:
    data = descriptor_for_layer.program_bytes
    if layer.ref_size <= 0 or layer.ref_off + layer.ref_size > len(data):
        return []
    byte_count = min(layer.ref_size, max(max_bytes, 1))
    expected = data[layer.ref_off:layer.ref_off + byte_count]
    if not expected:
        return []
    l1_base = 0x60000 + ((layer.index * 0x100 + ordinal * 0x20) & 0x2FFFF)
    descs: list[list[int]] = [
        udma_dram_to_l1_descriptor(
            layer,
            ordinal,
            layer.ref_off,
            l1_base,
            byte_count,
            SMF_LOAD_A,
        )
    ]
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_base,
            result_dram_off,
            expected,
        )
    )
    return descs


def closed_loop_output_indices(total_outputs: int, command_budget: int,
                               commands_per_output: int) -> list[int]:
    if total_outputs <= 0 or command_budget <= 0 or commands_per_output <= 0:
        return []
    max_outputs = max(command_budget // commands_per_output, 0)
    if max_outputs >= total_outputs:
        return list(range(total_outputs))
    if max_outputs <= 0:
        return []
    if max_outputs == 1:
        return [0]
    span = total_outputs - 1
    out: list[int] = []
    seen: set[int] = set()
    for idx in range(max_outputs):
        value = round((span * idx) / (max_outputs - 1))
        value = max(0, min(value, total_outputs - 1))
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def is_sram_crc_descriptor(desc: list[int]) -> bool:
    op = desc[0] & 0xF
    return (
        (op == OP_L1CRC and bool(desc[3] & (1 << 10))) or
        (op in (OP_CONV, OP_REQUANT, OP_EWE, OP_POOL, OP_TNPS, OP_UDMA) and bool(desc[3] & (1 << 10)))
    )


def mark_probe_descriptor(desc: list[int]) -> list[int]:
    desc[3] |= PROBE_DESCRIPTOR_FLAG
    return desc


def microblock_meta_flags(desc: list[int]) -> int:
    op = desc[0] & 0xF
    flags = 0
    if op == OP_UDMA and not (desc[3] & (1 << 6)):
        flags |= SMF_LOAD_A
    elif op in (OP_CONV, OP_REQUANT, OP_EWE, OP_POOL, OP_TNPS):
        flags |= SMF_COMPUTE
    if (desc[3] & (1 << 6)) or is_sram_crc_descriptor(desc) or op == OP_L1CRC:
        flags |= SMF_STORE
    if desc[3] & (1 << 12):
        flags |= SMF_FINAL_TILE
    return flags


def is_final_crc_descriptor(desc: list[int]) -> bool:
    return bool(((desc[3] >> 24) & SMF_FINAL_TILE) != 0)


def stamp_microblock_metadata(desc: list[int], layer_index: int, microblock_id: int, stream_slot: int) -> None:
    op = desc[0] & 0xF
    desc[0] = (
        op |
        ((layer_index & 0xFFFF) << 4) |
        ((microblock_id & 0x0FFF) << 20)
    )
    desc[3] = (
        (desc[3] & 0x0000_FFFF) |
        ((stream_slot & 0xFF) << 16) |
        ((microblock_meta_flags(desc) & 0xFF) << 24)
    )


def stamp_synth_microblock_metadata(
    desc: list[int],
    layer_index: int,
    microblock_id: int,
    stream_slot: int,
    stream_flags: int,
) -> None:
    op = desc[0] & 0xF
    desc[0] = (
        op |
        ((layer_index & 0xFFFF) << 4) |
        ((microblock_id & 0x0FFF) << 20)
    )
    desc[3] = (
        (desc[3] & 0x0000_FFFF) |
        ((stream_slot & 0xFF) << 16) |
        ((stream_flags & 0xFF) << 24)
    )


def l1_preload_byte_descriptor(ordinal: int, byte_addr: int, byte_value: int, source_layer: int = 0) -> list[int]:
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = 1
    words[2] = byte_addr & 0x003F_FFFF
    words[3] = 1 << 6
    words[4] = 1
    words[5] = 1
    words[6] = byte_value & 0xFF
    words[19] = source_layer
    words[27] = byte_addr & 0xFFFF_FFFF
    return words


def shape_product(shape: list[int], rank: int) -> int:
    prod = 1
    for value in shape[:rank]:
        prod *= value or 1
    return prod


def strides_for(shape: list[int], rank: int) -> list[int]:
    strides = [0] * 6
    stride = 1
    for idx in range(rank - 1, -1, -1):
        strides[idx] = stride
        stride *= shape[idx] or 1
    return strides


def tnps_meta_sample(layer: Layer) -> tuple[int, int, int, int, bool]:
    if layer.tnps_meta is None:
        return 0, 0, 0, 0, False
    rank, elem, in_shape, out_shape, a_vals, b_vals = layer.tnps_meta
    if rank == 0:
        return 0, 0, 0, 0, False
    in_elems = shape_product(in_shape, rank)
    out_elems = shape_product(out_shape, rank)
    if in_elems * elem != layer.in_size or out_elems * elem != layer.ref_size:
        return 0, 0, 0, 0, False

    out_idx = 2 if out_elems > 2 else 0
    rem = out_idx
    in_idx = 0
    in_strides = strides_for(in_shape, rank)
    out_strides = strides_for(out_shape, rank)
    for dim in range(rank):
        coord = rem // out_strides[dim]
        rem %= out_strides[dim]
        if layer.op_kind == OK_TRANSPOSE:
            src_dim = a_vals[dim]
            if 0 <= src_dim < rank:
                in_idx += coord * in_strides[src_dim]
        else:
            begin = a_vals[dim]
            stride = b_vals[dim] or 1
            src_coord = begin + coord * stride
            if src_coord < 0:
                return 0, 0, 0, 0, False
            in_idx += src_coord * in_strides[dim]
    if in_idx >= in_elems:
        return 0, 0, 0, 0, False
    return out_idx, 0, in_idx * elem, out_idx * elem, True


def descriptor_for_layer(layer: Layer, ordinal: int, enable_meta_tnps: bool) -> list[int] | None:
    elem = elem_bytes(layer.dtype)
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND

    if layer.op_kind in OK_CONV and layer.dtype in DT_FP and layer.in_size >= 2 and layer.wgt_size >= 2:
        data = descriptor_for_layer.program_bytes
        elem_count = min(layer.in_size // 2, layer.wgt_size // 2, 8)
        if elem_count == 0:
            return None
        act = data[layer.in_off:layer.in_off + elem_count * 2].ljust(16, b"\x00")
        wgt = data[layer.wgt_off:layer.wgt_off + elem_count * 2].ljust(16, b"\x00")
        expected = 0.0
        for idx in range(elem_count):
            expected += fp16_at(act, idx * 2) * fp16_at(wgt, idx * 2)
        expected_bits = struct.unpack("<Q", struct.pack("<d", expected))[0]
        words[0] = OP_CONV
        words[1] = elem_count * 2
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(act[idx * 4:(idx + 1) * 4])
            words[8 + idx] = pack_word(wgt[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (1 << 8)
        words[16] = expected_bits & 0xFFFF_FFFF
        words[17] = (expected_bits >> 32) & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if (
        layer.op_kind in OK_CONV and elem == 2 and layer.dtype not in DT_FP and
        layer.in_size >= 2 and layer.wgt_size >= 2
    ):
        data = descriptor_for_layer.program_bytes
        elem_count = min(layer.in_size // 2, layer.wgt_size // 2, 8)
        if elem_count == 0:
            return None
        act = data[layer.in_off:layer.in_off + elem_count * 2].ljust(16, b"\x00")
        wgt = data[layer.wgt_off:layer.wgt_off + elem_count * 2].ljust(16, b"\x00")
        acc = 0
        for idx in range(elem_count):
            acc += i16_at(act, idx * 2) * i16_at(wgt, idx * 2)
        words[0] = OP_CONV
        words[1] = elem_count * 2
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(act[idx * 4:(idx + 1) * 4])
            words[8 + idx] = pack_word(wgt[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (1 << 11)
        words[18] = acc & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if layer.op_kind in OK_CONV and elem == 1 and layer.in_size > 0 and layer.wgt_size > 0:
        # Emit one NHWC/OHWI output-pixel window sample from the layer payload.
        # This keeps the descriptor small while the final path grows toward full
        # tile streaming.
        data = descriptor_for_layer.program_bytes
        (
            act,
            wgt,
            elem_count,
            sample_kh,
            sample_kw,
            sample_ic,
            expected_input_offset,
            expected_weight_offset,
            expected_output_offset,
            expected_valid,
            expected_first_input_offset,
            expected_first_weight_offset,
            expected_valid_count,
            tile_count,
            expected_tile_last_valid,
            expected_tile_last_valid_count,
        ) = conv2d_int8_window_sample(layer, data)
        if elem_count == 0:
            return None
        bias = 0
        multiplier = 1073741824
        shift = 1
        acc = bias
        for idx in range(elem_count):
            acc += i8(act[idx]) * i8(wgt[idx])
        scaled = mbqm(clamp_i32(acc), multiplier, shift)
        scaled = max(-128, min(127, scaled))
        words[0] = OP_CONV
        words[1] = elem_count
        words[2] = addr
        words[3] = (1 << 2) | ((1 if expected_valid else 0) << 3)
        for idx in range(4):
            words[4 + idx] = pack_word(act[idx * 4:(idx + 1) * 4])
            words[8 + idx] = pack_word(wgt[idx * 4:(idx + 1) * 4])
        words[12] = elem_count
        words[13] = bias & 0xFFFF_FFFF
        words[14] = multiplier & 0xFFFF_FFFF
        words[15] = shift & 0xFF
        words[16] = (-128) & 0xFFFF_FFFF
        words[17] = 127
        words[18] = scaled & 0xFF
        words[19] = layer.index
        words[20] = (layer.in_h & 0xFFFF) | ((layer.in_w & 0xFFFF) << 16)
        words[21] = (layer.in_c & 0xFFFF) | ((layer.out_c & 0xFFFF) << 16)
        words[22] = ((layer.k_h & 0xFF) |
                     ((layer.k_w & 0xFF) << 8) |
                     (1 << 16) |
                     (1 << 24))
        words[23] = (1 | (1 << 8) | ((sample_kh & 0xFF) << 16) | ((sample_kw & 0xFF) << 24))
        words[24] = (sample_ic & 0xFFFF) | ((layer.out_w & 0xFFFF) << 16)
        words[25] = expected_input_offset & 0xFFFF_FFFF
        words[26] = expected_weight_offset & 0xFFFF_FFFF
        words[27] = expected_output_offset & 0xFFFF_FFFF
        words[28] = expected_first_input_offset & 0xFFFF_FFFF
        words[29] = expected_first_weight_offset & 0xFFFF_FFFF
        words[30] = expected_valid_count & 0xFFFF_FFFF
        words[31] = (
            (tile_count & 0xFF) |
            ((expected_tile_last_valid_count & 0xFF) << 8) |
            ((1 if expected_tile_last_valid else 0) << 16)
        )
        return words

    if layer.op_kind in OK_POOL and elem == 1 and layer.in_size > 0:
        data = descriptor_for_layer.program_bytes
        sample = sample_bytes(data, layer.in_off, layer.in_size)
        elem_count = min(layer.in_size, 16)
        vals = [i8(sample[idx]) for idx in range(elem_count)]
        if not vals:
            return None
        avg_mode = 1 if layer.op_kind == 2 else 0
        if avg_mode:
            expected = int(sum(vals) / len(vals))
        else:
            expected = max(vals)
        expected = max(-128, min(127, expected))
        words[0] = OP_POOL
        words[1] = elem_count
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(sample[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (avg_mode << 8)
        words[18] = expected & 0xFF
        words[19] = layer.index
        return words

    if layer.op_kind in OK_POOL and elem == 2 and layer.dtype not in DT_FP and layer.in_size >= 2:
        data = descriptor_for_layer.program_bytes
        elem_count = min(layer.in_size // 2, 8)
        if elem_count == 0:
            return None
        sample = data[layer.in_off:layer.in_off + elem_count * 2].ljust(16, b"\x00")
        vals = [i16_at(sample, idx * 2) for idx in range(elem_count)]
        avg_mode = 1 if layer.op_kind == 2 else 0
        expected = int(sum(vals) / len(vals)) if avg_mode else max(vals)
        words[0] = OP_POOL
        words[1] = elem_count * 2
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(sample[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (avg_mode << 8) | (1 << 11)
        words[18] = expected & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if layer.op_kind in OK_POOL and layer.dtype in DT_FP and layer.in_size >= 2:
        data = descriptor_for_layer.program_bytes
        elem_count = min(layer.in_size // 2, 8)
        if elem_count == 0:
            return None
        sample = data[layer.in_off:layer.in_off + elem_count * 2].ljust(16, b"\x00")
        vals = [fp16_at(sample, idx * 2) for idx in range(elem_count)]
        avg_mode = 1 if layer.op_kind == 2 else 0
        expected = (sum(vals) / len(vals)) if avg_mode else max(vals)
        expected_bits = struct.unpack("<Q", struct.pack("<d", expected))[0]
        words[0] = OP_POOL
        words[1] = elem_count * 2
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(sample[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (avg_mode << 8) | (1 << 9)
        words[16] = expected_bits & 0xFFFF_FFFF
        words[17] = (expected_bits >> 32) & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if layer.op_kind in OK_EWE and elem == 1 and layer.in_size > 0 and layer.wgt_size > 0:
        data = descriptor_for_layer.program_bytes
        a_sample = sample_bytes(data, layer.in_off, layer.in_size)
        b_sample = sample_bytes(data, layer.wgt_off, layer.wgt_size)
        elem_count = min(layer.in_size, layer.wgt_size, 16)
        if elem_count == 0:
            return None
        expected = 0
        for lane in range(elem_count):
            av = i8(a_sample[lane])
            bv = i8(b_sample[lane])
            if layer.op_kind == OK_MUL:
                op_mode = 1
                expected += clamp_i8(av * bv)
            elif layer.op_kind == OK_SUB:
                op_mode = 2
                expected += clamp_i8(av - bv)
            else:
                op_mode = 0
                expected += clamp_i8(av + bv)
        words[0] = OP_EWE
        words[1] = elem_count
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(a_sample[idx * 4:(idx + 1) * 4])
            words[8 + idx] = pack_word(b_sample[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (op_mode << 8)
        words[18] = expected & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if (
        layer.op_kind in OK_EWE and elem == 2 and layer.dtype not in DT_FP and
        layer.in_size >= 2 and layer.wgt_size >= 2
    ):
        data = descriptor_for_layer.program_bytes
        a_sample = data[layer.in_off:layer.in_off + min(layer.in_size, 16)].ljust(16, b"\x00")
        b_sample = data[layer.wgt_off:layer.wgt_off + min(layer.wgt_size, 16)].ljust(16, b"\x00")
        elem_count = min(layer.in_size // 2, layer.wgt_size // 2, 8)
        if elem_count == 0:
            return None
        expected = 0
        for lane in range(elem_count):
            av = i16_at(a_sample, lane * 2)
            bv = i16_at(b_sample, lane * 2)
            if layer.op_kind == OK_MUL:
                op_mode = 1
                expected += av * bv
            elif layer.op_kind == OK_SUB:
                op_mode = 2
                expected += av - bv
            else:
                op_mode = 0
                expected += av + bv
        words[0] = OP_EWE
        words[1] = elem_count * 2
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(a_sample[idx * 4:(idx + 1) * 4])
            words[8 + idx] = pack_word(b_sample[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (op_mode << 8) | (1 << 11)
        words[18] = expected & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if layer.op_kind in OK_FP_EWE and layer.dtype in DT_FP and layer.in_size >= 2:
        data = descriptor_for_layer.program_bytes
        a_sample = data[layer.in_off:layer.in_off + min(layer.in_size, 16)].ljust(16, b"\x00")
        need_b = layer.op_kind in OK_EWE
        b_sample = (
            data[layer.wgt_off:layer.wgt_off + min(layer.wgt_size, 16)].ljust(16, b"\x00")
            if need_b else b"\x00" * 16
        )
        elem_count = min(layer.in_size // 2, (layer.wgt_size // 2 if need_b else 8), 8)
        if elem_count == 0:
            return None
        expected = 0.0
        op_mode = 3 if layer.op_kind == OK_LOGISTIC else 0
        for lane in range(elem_count):
            av = fp16_at(a_sample, lane * 2)
            bv = fp16_at(b_sample, lane * 2) if need_b else 0.0
            if layer.op_kind == OK_LOGISTIC:
                expected += 1.0 / (1.0 + math.exp(-av))
            elif layer.op_kind == OK_MUL:
                op_mode = 1
                expected += av * bv
            elif layer.op_kind == OK_SUB:
                op_mode = 2
                expected += av - bv
            else:
                op_mode = 0
                expected += av + bv
        expected_bits = struct.unpack("<Q", struct.pack("<d", expected))[0]
        words[0] = OP_EWE
        words[1] = elem_count * 2
        words[2] = addr
        for idx in range(4):
            words[4 + idx] = pack_word(a_sample[idx * 4:(idx + 1) * 4])
            words[8 + idx] = pack_word(b_sample[idx * 4:(idx + 1) * 4])
        words[12] = elem_count | (op_mode << 8) | (1 << 10)
        words[16] = expected_bits & 0xFFFF_FFFF
        words[17] = (expected_bits >> 32) & 0xFFFF_FFFF
        words[19] = layer.index
        return words

    if layer.op_kind in (OK_S2SPACE, OK_D2SPACE):
        sample_out, sample_in, expected_src, expected_dst, sample_valid = tnps_sample(layer)
        words[0] = OP_TNPS
        words[1] = sim_bytes(max(layer.ref_size, layer.in_size))
        words[2] = addr
        words[3] = 0x2 if layer.op_kind == OK_S2SPACE else 0x0
        words[6] = layer.in_h
        words[7] = layer.in_w
        words[8] = layer.in_c
        words[9] = layer.out_h
        words[10] = layer.out_w
        words[11] = layer.out_c
        words[12] = layer.k_h or layer.k_w or 1
        words[13] = elem
        words[14] = sample_out
        words[15] = sample_in
        words[16] = expected_src
        words[17] = expected_dst
        words[18] = 1 if sample_valid else 0
        words[19] = layer.index
        return words

    if enable_meta_tnps and layer.op_kind in META_TNPS_OPS and layer.tnps_meta is not None:
        sample_out, sample_in, expected_src, expected_dst, sample_valid = tnps_meta_sample(layer)
        words[0] = OP_TNPS
        words[1] = sim_bytes(max(layer.ref_size, layer.in_size))
        words[2] = addr
        words[3] = 0x4 if layer.op_kind == OK_TRANSPOSE else 0x8
        words[6] = layer.in_h
        words[7] = layer.in_w
        words[8] = layer.in_c
        words[9] = layer.out_h
        words[10] = layer.out_w
        words[11] = layer.out_c
        words[12] = 1
        words[13] = layer.tnps_meta[1]
        words[14] = sample_out
        words[15] = sample_in
        words[16] = expected_src
        words[17] = expected_dst
        words[18] = 1 if sample_valid else 0
        words[19] = layer.index
        return words

    if layer.op_kind in TNPS_OPS or layer.op_kind in UDMA_OPS:
        payload_bytes = sim_bytes(max(layer.ref_size, layer.in_size, 1))
        words[0] = OP_UDMA
        words[1] = payload_bytes
        words[2] = addr
        words[3] = 0x1
        words[4] = payload_bytes
        words[5] = 1 + (payload_bytes // 4096)
        words[19] = layer.index
        return words

    return None


def int8_conv_sample_descriptor(
    layer: Layer,
    ordinal: int,
    start_lane: int = 0,
    max_count: int = 16,
    out_elem_index: int = 0,
    psum_flag: int = 0,
    expected_psum: int | None = None,
) -> tuple[list[int], int] | None:
    data = descriptor_for_layer.program_bytes
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20) & 0x3FFF0)
    (
        act,
        wgt,
        elem_count,
        sample_kh,
        sample_kw,
        sample_ic,
        expected_input_offset,
        expected_weight_offset,
        expected_output_offset,
        expected_valid,
        expected_first_input_offset,
        expected_first_weight_offset,
        expected_valid_count,
        tile_count,
        expected_tile_last_valid,
        expected_tile_last_valid_count,
    ) = conv2d_int8_window_sample(
        layer,
        data,
        max_count=max_count,
        start_lane=start_lane,
        out_elem_index=out_elem_index,
    )
    if elem_count == 0:
        return None
    bias = 0
    multiplier = 1073741824
    shift = 1
    acc = bias
    for idx in range(elem_count):
        acc += i8(act[idx]) * i8(wgt[idx])
    scaled = mbqm(clamp_i32(acc), multiplier, shift)
    scaled = max(-128, min(127, scaled))
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_CONV
    words[1] = elem_count
    words[2] = addr
    words[3] = (1 << 2) | ((1 if expected_valid else 0) << 3) | psum_flag
    for idx in range(4):
        words[4 + idx] = pack_word(act[idx * 4:(idx + 1) * 4])
        words[8 + idx] = pack_word(wgt[idx * 4:(idx + 1) * 4])
    words[12] = elem_count
    words[13] = bias & 0xFFFF_FFFF
    words[14] = multiplier & 0xFFFF_FFFF
    words[15] = shift & 0xFF
    words[16] = (-128) & 0xFFFF_FFFF
    words[17] = 127
    words[18] = scaled & 0xFF
    words[19] = (expected_psum if expected_psum is not None else layer.index) & 0xFFFF_FFFF
    words[20] = (layer.in_h & 0xFFFF) | ((layer.in_w & 0xFFFF) << 16)
    words[21] = (layer.in_c & 0xFFFF) | ((layer.out_c & 0xFFFF) << 16)
    words[22] = ((layer.k_h & 0xFF) |
                 ((layer.k_w & 0xFF) << 8) |
                 (1 << 16) |
                 (1 << 24))
    words[23] = (1 | (1 << 8) | ((sample_kh & 0xFF) << 16) | ((sample_kw & 0xFF) << 24))
    words[24] = (sample_ic & 0xFFFF) | ((layer.out_w & 0xFFFF) << 16)
    words[25] = expected_input_offset & 0xFFFF_FFFF
    words[26] = expected_weight_offset & 0xFFFF_FFFF
    words[27] = expected_output_offset & 0xFFFF_FFFF
    words[28] = expected_first_input_offset & 0xFFFF_FFFF
    words[29] = expected_first_weight_offset & 0xFFFF_FFFF
    words[30] = (expected_valid_count & 0xFF) | ((layer.out_h & 0xFFFF) << 16)
    words[31] = (
        (tile_count & 0xFF) |
        ((expected_tile_last_valid_count & 0xFF) << 8) |
        ((1 if expected_tile_last_valid else 0) << 16)
    )
    return words, acc


def int8_conv_real_descriptor(
    layer: Layer,
    params: Int8ConvParams,
    ordinal: int,
    start_lane: int,
    max_count: int,
    out_elem_index: int,
    psum_flag: int,
    final_bias: int = 0,
) -> tuple[list[int], int] | None:
    data = descriptor_for_layer.program_bytes
    sample = int8_conv_real_window_sample(
        layer,
        data,
        max_count=max_count,
        start_lane=start_lane,
        out_elem_index=out_elem_index,
    )
    if sample is None:
        return None
    (
        act,
        wgt,
        elem_count,
        sample_kh,
        sample_kw,
        sample_ic,
        expected_input_offset,
        expected_weight_offset,
        expected_output_offset,
        expected_valid,
        expected_first_input_offset,
        expected_first_weight_offset,
        expected_valid_count,
        tile_count,
        expected_tile_last_valid,
        expected_tile_last_valid_count,
    ) = sample
    if elem_count == 0:
        return None
    out_area = max(layer.out_w * layer.out_c, 1)
    oh = out_elem_index // out_area
    ow = (out_elem_index % out_area) // max(layer.out_c, 1)
    oc = out_elem_index % max(layer.out_c, 1)
    multiplier = params.mult[oc]
    shift = params.shift[oc]
    bias = final_bias
    acc = bias
    for idx in range(elem_count):
        acc += i8(act[idx]) * i8(wgt[idx])
    scaled = mbqm(clamp_i32(acc), multiplier, shift) + params.zp_out
    scaled = max(params.act_min, min(params.act_max, scaled))

    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_CONV
    words[1] = elem_count
    words[2] = addr
    words[3] = ((1 if expected_valid else 0) << 3) | psum_flag
    for idx in range(4):
        words[4 + idx] = pack_word(act[idx * 4:(idx + 1) * 4])
        words[8 + idx] = pack_word(wgt[idx * 4:(idx + 1) * 4])
    words[12] = elem_count
    words[13] = bias & 0xFFFF_FFFF
    words[14] = multiplier & 0xFFFF_FFFF
    words[15] = (shift & 0xFF) | ((params.zp_out & 0xFF) << 8)
    words[16] = params.act_min & 0xFFFF_FFFF
    words[17] = params.act_max & 0xFFFF_FFFF
    words[18] = scaled & 0xFF
    words[19] = acc & 0xFFFF_FFFF
    words[20] = (layer.in_h & 0xFFFF) | ((layer.in_w & 0xFFFF) << 16)
    words[21] = (layer.in_c & 0xFFFF) | ((layer.out_c & 0xFFFF) << 16)
    words[22] = ((layer.k_h & 0xFF) |
                 ((layer.k_w & 0xFF) << 8) |
                 ((layer.s_h or 1) << 16) |
                 ((layer.s_w or 1) << 24))
    words[23] = (1 | (1 << 8) | ((sample_kh & 0xFF) << 16) | ((sample_kw & 0xFF) << 24))
    words[24] = (sample_ic & 0xFFFF) | ((layer.out_w & 0xFFFF) << 16)
    words[25] = expected_input_offset & 0xFFFF_FFFF
    words[26] = expected_weight_offset & 0xFFFF_FFFF
    words[27] = expected_output_offset & 0xFFFF_FFFF
    words[28] = expected_first_input_offset & 0xFFFF_FFFF
    words[29] = expected_first_weight_offset & 0xFFFF_FFFF
    words[30] = (expected_valid_count & 0xFF) | ((layer.out_h & 0xFFFF) << 16)
    words[31] = (
        (tile_count & 0xFF) |
        ((expected_tile_last_valid_count & 0xFF) << 8) |
        ((1 if expected_tile_last_valid else 0) << 16)
    )
    return words, acc


def conv_partial_psum_command_count(layer: Layer, output_elems: int) -> int:
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * max(layer.in_c, 1)
    lane_chunks = max(math.ceil(total_lanes / 8), 1)
    output_tiles = max(math.ceil(max(output_elems, 1) / 4), 1)
    return output_tiles * lane_chunks + 1


def conv_real_partial_psum_command_count(layer: Layer, output_elems: int) -> int:
    group = layer.group or 1
    in_per_group = max(layer.in_c // max(group, 1), 1)
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * in_per_group
    lane_chunks = max(math.ceil(total_lanes / 8), 1)
    return max(output_elems, 1) * lane_chunks + 1


def conv_real_lane_chunk_count(layer: Layer) -> int:
    group = layer.group or 1
    in_per_group = max(layer.in_c // max(group, 1), 1)
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * in_per_group
    return max(math.ceil(total_lanes / 8), 1)


def conv_real_partial_psum_descriptors(
    layer: Layer,
    ordinal: int,
    max_output_elems: int | None,
    start_output_elem: int = 0,
) -> list[list[int]]:
    data = descriptor_for_layer.program_bytes
    params = int8_conv_params(layer, data)
    if params is None:
        return []
    descs: list[list[int]] = []
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    start_output_elem = max(min(start_output_elem, output_elems - 1), 0)
    remaining_output_elems = output_elems - start_output_elem
    emit_output_elems = remaining_output_elems if max_output_elems is None else min(remaining_output_elems, max_output_elems)
    group = layer.group or 1
    in_per_group = max(layer.in_c // max(group, 1), 1)
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * in_per_group
    for out_elem_index in range(start_output_elem, start_output_elem + emit_output_elems):
        out_area = max(layer.out_w * layer.out_c, 1)
        oh = out_elem_index // out_area
        ow = (out_elem_index % out_area) // max(layer.out_c, 1)
        oc = out_elem_index % max(layer.out_c, 1)
        final_bias = params.bias[oc] + int8_conv_corr(params, layer, oh, ow, oc)
        cumulative_acc = 0
        start_lane = 0
        tile_descs: list[list[int]] = []
        while start_lane < total_lanes:
            psum_flag = 1 << 4 if not tile_descs else 1 << 5
            remaining = total_lanes - start_lane
            is_final_chunk = remaining <= 8
            sample = int8_conv_real_descriptor(
                layer,
                params,
                ordinal + len(descs),
                start_lane=start_lane,
                max_count=8,
                out_elem_index=out_elem_index,
                psum_flag=psum_flag,
                final_bias=final_bias if is_final_chunk else 0,
            )
            if sample is None:
                break
            desc, acc = sample
            elem_count = desc[12] & 0xFF
            if elem_count == 0:
                break
            cumulative_acc += acc
            desc[19] = cumulative_acc & 0xFFFF_FFFF
            tile_descs.append(desc)
            descs.append(desc)
            start_lane += elem_count
        if not tile_descs:
            break
        tile_descs[-1][3] |= 1 << 6
        final_q = mbqm(clamp_i32(cumulative_acc), params.mult[oc], params.shift[oc]) + params.zp_out
        final_q = max(params.act_min, min(params.act_max, final_q))
        tile_descs[-1][18] = final_q & 0xFF
        tile_descs[-1][31] = (tile_descs[-1][31] & ~0xFF) | 1
    return descs


def conv_partial_psum_descriptors(
    layer: Layer,
    ordinal: int,
    max_output_elems: int | None = 16,
) -> list[list[int]]:
    descs: list[list[int]] = []
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    emit_output_elems = output_elems if max_output_elems is None else min(output_elems, max_output_elems)
    out_elem_index = 0
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * max(layer.in_c, 1)
    while out_elem_index < emit_output_elems:
        cumulative_acc = 0
        start_lane = 0
        tile_descs: list[list[int]] = []
        while start_lane < total_lanes:
            psum_flag = 1 << 4 if not tile_descs else 1 << 5
            sample = int8_conv_sample_descriptor(
                layer,
                ordinal + len(descs),
                start_lane=start_lane,
                max_count=8,
                out_elem_index=out_elem_index,
                psum_flag=psum_flag,
            )
            if sample is None:
                break
            desc, acc = sample
            elem_count = desc[12] & 0xFF
            if elem_count == 0:
                break
            cumulative_acc += acc
            desc[19] = cumulative_acc & 0xFFFF_FFFF
            tile_descs.append(desc)
            descs.append(desc)
            start_lane += elem_count
        if not tile_descs:
            break
        tile_descs[-1][3] |= 1 << 6
        final_q = clamp_i8(mbqm(clamp_i32(cumulative_acc), 1073741824, 1))
        tile_descs[-1][18] = final_q & 0xFF
        tile_count = tile_descs[-1][31] & 0xFF
        out_elem_index += max(tile_count, 1)
    return descs


def descriptor_payload_bytes(desc: list[int], first_word: int, byte_count: int) -> bytes:
    payload = bytearray()
    for idx in range(byte_count):
        word = desc[first_word + idx // 4]
        payload.append((word >> ((idx % 4) * 8)) & 0xFF)
    return bytes(payload)


def descriptor_u64_bytes(desc: list[int], lo_word: int) -> bytes:
    value = (desc[lo_word] & 0xFFFF_FFFF) | ((desc[lo_word + 1] & 0xFFFF_FFFF) << 32)
    return struct.pack("<Q", value)


def closed_loop_wide_sample_probe(layer: Layer, ordinal: int, result_dram_off: int) -> list[list[int]]:
    if elem_bytes(layer.dtype) != 2:
        return []
    if layer.op_kind not in OK_POOL and layer.op_kind not in OK_CONV:
        return []
    desc = descriptor_for_layer(layer, ordinal, False)
    if desc is None:
        return []
    op = desc[0] & 0xF
    if op not in (OP_POOL, OP_CONV):
        return []
    sample_byte_count = desc[1]
    if sample_byte_count <= 0 or sample_byte_count > 16:
        return []
    data = descriptor_for_layer.program_bytes
    l1_base = 0x50000 + ((layer.index * 0x100 + ordinal * 0x20) & 0x2FFFF)
    l1_result = l1_base if op == OP_CONV else l1_base + 0x80
    load_byte_count = sample_byte_count
    if op == OP_POOL:
        load_byte_count = max(sample_byte_count, min(layer.in_size, POOL_TILE_MAX_BYTES))
        desc[1] = load_byte_count
        l1_result = l1_base + load_byte_count + 0x80
    elif op == OP_CONV:
        tile_elem_count = min(
            layer.in_size // 2,
            layer.wgt_size // 2,
            CONV_TILE_MAX_ELEMS,
        )
        tile_elem_count -= tile_elem_count % 8
        if tile_elem_count == 0:
            tile_elem_count = sample_byte_count // 2
        if tile_elem_count <= 0:
            return []
        load_byte_count = tile_elem_count * 2
        desc[1] = load_byte_count
        output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
        desc[31] = min(output_elems, CONV_TILE_MAX_OUTPUTS)
    descs = []
    descs.append(
        udma_dram_to_l1_descriptor(
            layer,
            ordinal,
            layer.in_off,
            l1_base,
            load_byte_count,
            SMF_LOAD_A,
        )
    )
    if data[layer.in_off:layer.in_off + sample_byte_count] != descriptor_payload_bytes(desc, 4, sample_byte_count):
        return []
    if op == OP_CONV:
        if data[layer.wgt_off:layer.wgt_off + sample_byte_count] != descriptor_payload_bytes(desc, 8, sample_byte_count):
            return []
        descs.append(
            udma_dram_to_l1_descriptor(
                layer,
                ordinal + len(descs),
                layer.wgt_off,
                l1_base + load_byte_count,
                load_byte_count,
                SMF_LOAD_B,
            )
        )
    desc[2] = l1_base
    desc[3] |= READ_FROM_L1_FLAG | MICROBLOCK_FLAG | (1 << 6)
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    expected = (
        descriptor_u64_bytes(desc, 16)
        if layer.dtype in DT_FP else
        (desc[18] & 0xFFFF_FFFF).to_bytes(4, "little", signed=False)
    )
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off,
            expected,
        )
    )
    return descs


def closed_loop_fp_sample_probe(layer: Layer, ordinal: int, result_dram_off: int) -> list[list[int]]:
    if layer.dtype not in DT_FP:
        return []
    return closed_loop_wide_sample_probe(layer, ordinal, result_dram_off)


def closed_loop_int16_sample_probe(layer: Layer, ordinal: int, result_dram_off: int) -> list[list[int]]:
    if layer.dtype in DT_FP:
        return []
    return closed_loop_wide_sample_probe(layer, ordinal, result_dram_off)


def closed_loop_fp_ewe_probe(layer: Layer, ordinal: int, result_dram_off: int) -> list[list[int]]:
    if elem_bytes(layer.dtype) != 2 or layer.dtype not in DT_FP:
        return []
    if layer.op_kind not in OK_FP_EWE:
        return []
    desc = descriptor_for_layer(layer, ordinal, False)
    if desc is None or (desc[0] & 0xF) != OP_EWE:
        return []
    sample_byte_count = desc[1]
    if sample_byte_count <= 0 or sample_byte_count > 16:
        return []

    data = descriptor_for_layer.program_bytes
    if data[layer.in_off:layer.in_off + sample_byte_count] != descriptor_payload_bytes(desc, 4, sample_byte_count):
        return []
    if layer.op_kind in OK_EWE:
        if data[layer.wgt_off:layer.wgt_off + sample_byte_count] != descriptor_payload_bytes(desc, 8, sample_byte_count):
            return []

    l1_base = 0x54000 + ((layer.index * 0x100 + ordinal * 0x20) & 0x2FFFF)
    l1_result = l1_base + 0x80
    descs: list[list[int]] = [
        udma_dram_to_l1_descriptor(
            layer,
            ordinal,
            layer.in_off,
            l1_base,
            sample_byte_count,
            SMF_LOAD_A,
        )
    ]
    desc[2] = l1_base
    desc[3] |= READ_FROM_L1_FLAG | MICROBLOCK_FLAG | (1 << 6)
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    expected = descriptor_u64_bytes(desc, 16)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off,
            expected,
        )
    )
    return descs


def closed_loop_conv_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    out_elem_index: int = 0,
) -> list[list[int]]:
    if layer.op_kind not in OK_CONV or elem_bytes(layer.dtype) != 1:
        return []
    group = layer.group or 1
    if group <= 0 or layer.out_c <= 0 or layer.in_c % group != 0 or layer.out_c % group != 0:
        return []
    in_per_group = layer.in_c // group
    total_lanes = max(layer.k_h or 1, 1) * max(layer.k_w or 1, 1) * max(in_per_group, 1)
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    if out_elem_index < 0 or out_elem_index >= output_elems:
        return []
    data = descriptor_for_layer.program_bytes
    params = int8_conv_params(layer, data)
    if params is not None:
        out_area = max(layer.out_w * layer.out_c, 1)
        oh = out_elem_index // out_area
        ow = (out_elem_index % out_area) // max(layer.out_c, 1)
        oc = out_elem_index % max(layer.out_c, 1)
        final_bias = params.bias[oc] + int8_conv_corr(params, layer, oh, ow, oc)
        sample = int8_conv_real_descriptor(
            layer,
            params,
            ordinal,
            start_lane=0,
            max_count=8,
            out_elem_index=out_elem_index,
            psum_flag=0,
            final_bias=final_bias,
        )
    else:
        sample = int8_conv_sample_descriptor(
            layer,
            ordinal,
            max_count=8,
            out_elem_index=out_elem_index,
        )
    if sample is None:
        return []
    desc, _ = sample
    sample_elem_count = desc[12] & 0xFF
    if sample_elem_count == 0:
        return []
    if ((desc[3] & (1 << 3)) == 0) or ((desc[30] & 0xFF) != sample_elem_count):
        return []
    act_src = desc[28]
    wgt_src = desc[29]
    act_expected = descriptor_payload_bytes(desc, 4, sample_elem_count)
    wgt_expected = descriptor_payload_bytes(desc, 8, sample_elem_count)
    if data[layer.in_off + act_src:layer.in_off + act_src + sample_elem_count] != act_expected:
        return []
    if data[layer.wgt_off + wgt_src:layer.wgt_off + wgt_src + sample_elem_count] != wgt_expected:
        return []
    tile_elem_count = min(
        total_lanes,
        layer.in_size - act_src,
        layer.wgt_size - wgt_src,
        CONV_TILE_MAX_ELEMS,
    )
    tile_elem_count -= tile_elem_count % 16
    if tile_elem_count == 0:
        tile_elem_count = sample_elem_count
    if tile_elem_count < sample_elem_count:
        return []
    desc[1] = tile_elem_count
    desc[30] = (desc[30] & ~0xFF) | sample_elem_count
    desc[31] = min(output_elems - out_elem_index, CONV_TILE_MAX_OUTPUTS)
    l1_base = 0x40000 + ((layer.index * 0x100 + out_elem_index * 0x20) & 0x2FFFF)
    l1_result = l1_base
    descs = [
        udma_dram_to_l1_descriptor(
            layer,
            ordinal,
            layer.in_off + act_src,
            l1_base,
            tile_elem_count,
            SMF_LOAD_A,
        ),
        udma_dram_to_l1_descriptor(
            layer,
            ordinal + 1,
            layer.wgt_off + wgt_src,
            l1_base + tile_elem_count,
            tile_elem_count,
            SMF_LOAD_B,
        ),
    ]
    desc[2] = l1_base
    desc[3] |= READ_FROM_L1_FLAG | MICROBLOCK_FLAG | (1 << 6)
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off + out_elem_index,
            bytes([desc[18] & 0xFF]),
        )
    )
    return descs


def closed_loop_conv_requant_chain_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    out_elem_index: int = 0,
) -> list[list[int]]:
    """Build a CONV+REQUANT chain pair that verifies the INT32 psum forwarding wire.

    CONV runs in L1-sample mode with chain_out_enable=1 (skips L1 write).
    REQUANT waits on ST_CHAIN_WAIT, receives the psum, quantises, and writes
    one INT8 byte to L1. A L1CRC check then verifies the expected value.
    """
    if layer.op_kind not in OK_CONV or elem_bytes(layer.dtype) != 1:
        return []
    data = descriptor_for_layer.program_bytes
    params = int8_conv_params(layer, data)
    if params is None:
        return []
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    if out_elem_index < 0 or out_elem_index >= output_elems:
        return []

    # Geometry
    out_area = max(layer.out_w * layer.out_c, 1)
    oh = out_elem_index // out_area
    ow = (out_elem_index % out_area) // max(layer.out_c, 1)
    oc = out_elem_index % max(layer.out_c, 1)
    final_bias = params.bias[oc] + int8_conv_corr(params, layer, oh, ow, oc)

    # Build CONV descriptor
    sample = int8_conv_real_descriptor(
        layer,
        params,
        ordinal,
        start_lane=0,
        max_count=8,
        out_elem_index=out_elem_index,
        psum_flag=0,
        final_bias=final_bias,
    )
    if sample is None:
        return []
    desc, raw_acc = sample

    sample_elem_count = desc[12] & 0xFF
    if sample_elem_count == 0:
        return []
    # Require valid address geometry
    if (desc[3] & (1 << 3)) == 0 or (desc[30] & 0xFF) != sample_elem_count:
        return []

    act_src = desc[28]
    wgt_src = desc[29]
    act_expected = descriptor_payload_bytes(desc, 4, sample_elem_count)
    wgt_expected = descriptor_payload_bytes(desc, 8, sample_elem_count)
    if data[layer.in_off + act_src:layer.in_off + act_src + sample_elem_count] != act_expected:
        return []
    if data[layer.wgt_off + wgt_src:layer.wgt_off + wgt_src + sample_elem_count] != wgt_expected:
        return []

    # Tile count for the L1 load (rounded to 16 for bandwidth alignment)
    group = layer.group or 1
    in_per_group = layer.in_c // group
    total_lanes = max(layer.k_h or 1, 1) * max(layer.k_w or 1, 1) * max(in_per_group, 1)
    tile_elem_count = min(total_lanes, layer.in_size - act_src, layer.wgt_size - wgt_src, CONV_TILE_MAX_ELEMS)
    tile_elem_count -= tile_elem_count % 16
    if tile_elem_count == 0:
        tile_elem_count = sample_elem_count
    if tile_elem_count < sample_elem_count:
        return []
    desc[1] = tile_elem_count
    desc[30] = (desc[30] & ~0xFF) | sample_elem_count
    desc[31] = min(output_elems - out_elem_index, CONV_TILE_MAX_OUTPUTS)

    # L1 addresses
    l1_base = 0x50000 + ((layer.index * 0x100 + out_elem_index * 0x20) & 0x1FFF0)
    l1_act = l1_base
    l1_wgt = l1_act + tile_elem_count
    l1_result = l1_wgt + tile_elem_count

    # UDMA loads for act and wgt
    descs: list[list[int]] = [
        udma_dram_to_l1_descriptor(layer, ordinal,             layer.in_off + act_src, l1_act, tile_elem_count, SMF_LOAD_A),
        udma_dram_to_l1_descriptor(layer, ordinal + 1,         layer.wgt_off + wgt_src, l1_wgt, tile_elem_count, SMF_LOAD_B),
    ]

    # Patch CONV descriptor for L1-read + chain-out mode.
    # Bit 6 = conv_partial_final; chain_skip_store requires partial_final=0.
    # Bit 11 = read_sample_from_l1, bit 13 = MICROBLOCK, bit 15 = chain_out_enable.
    desc[2] = l1_base        # act base in L1
    desc[3] = (desc[3] & ~(1 << 6)) | READ_FROM_L1_FLAG | MICROBLOCK_FLAG | CONV_CHAIN_OUT_FLAG
    desc[27] = l1_result     # dummy write address (won't fire with chain_skip_store)
    stamp_synth_microblock_metadata(
        desc, layer.index, ordinal + len(descs), ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)

    # Expected REQUANT output: MBQM(raw_acc, mult, shift) + zp_out, clamped
    expected_q = mbqm(clamp_i32(raw_acc), params.mult[oc], params.shift[oc]) + params.zp_out
    expected_q = max(params.act_min, min(params.act_max, expected_q))
    expected_byte = bytes([expected_q & 0xFF])

    # REQUANT descriptor.
    # Note: bit 13 of word[3] = requant_fp_mode in host.v, so we must NOT set
    # MICROBLOCK_FLAG (1<<13) here lest the engine switch to FP16 output.
    # Without MICROBLOCK_FLAG the Python assembler will call stamp_microblock_metadata
    # automatically and the host will verify word[18] against requant_out_q (INT8 path).
    rq_words = [0] * WORDS_PER_COMMAND
    rq_words[0] = OP_REQUANT
    rq_words[1] = 1
    rq_words[2] = l1_result          # L1 output address
    rq_words[3] = REQUANT_USE_CHAIN | (1 << 6)   # bit 12 = use_chain; bit 6 = final-write
    rq_words[14] = params.mult[oc] & 0xFFFF_FFFF
    rq_words[15] = (params.shift[oc] & 0xFF) | ((params.zp_out & 0xFF) << 8)
    rq_words[16] = params.act_min & 0xFFFF_FFFF
    rq_words[17] = params.act_max & 0xFFFF_FFFF
    rq_words[18] = expected_q & 0xFF
    rq_words[19] = layer.index
    rq_words[27] = l1_result
    # Do not call stamp_synth_microblock_metadata — let the assembler auto-stamp
    # via stamp_microblock_metadata (stream_flags inferred from op + word[3] bits).
    descs.append(rq_words)

    # L1CRC check: verify the byte REQUANT wrote at l1_result
    crc_check = l1mesh_crc_probe_descriptor(
        ordinal + len(descs), l1_result, expected_byte,
    )
    if crc_check is None:
        return []
    crc_check[3] |= MICROBLOCK_FLAG
    stamp_synth_microblock_metadata(
        crc_check, layer.index, ordinal + len(descs), ordinal + len(descs),
        SMF_FINAL_TILE,
    )
    descs.append(crc_check)
    return descs


def closed_loop_conv_requant_oc_tile_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    tile_oc: int = 4,
) -> list[list[int]]:
    """OC-dimension tile probe: CONV runs tile_oc iterations over sequential
    weight rows while ACT stays resident in L1. All iterations push acc to the
    chain (chain_out_enable=1); the last push remains valid when CONV finishes.
    REQUANT then drains once (tile_drain_count=1) and writes 1 byte to L1.
    The L1CRC probe verifies that byte matches the expected LAST-iteration acc.

    This tests the CONV OC tile loop mechanism (Phase 6b) by verifying that the
    final psum corresponds to the correct weight row. It also exercises the
    CONV->REQUANT chain in the tile-loop context.

    Simplification: loads tile_oc COPIES of the same weight row (OC=0), so all
    iterations yield the same acc. The expected REQUANT output is one byte.
    """
    if layer.op_kind not in OK_CONV or elem_bytes(layer.dtype) != 1:
        return []
    data = descriptor_for_layer.program_bytes
    params = int8_conv_params(layer, data)
    if params is None:
        return []
    if layer.in_size <= 0 or layer.wgt_size <= 0:
        return []
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    if output_elems < 1:
        return []

    # Build CONV descriptor for OC=0 at out_elem_index=0
    oc0 = 0
    out_area = max(layer.out_w * layer.out_c, 1)
    oh = 0
    ow = 0
    final_bias_oc0 = params.bias[oc0] + int8_conv_corr(params, layer, oh, ow, oc0)
    sample = int8_conv_real_descriptor(
        layer, params, ordinal, start_lane=0, max_count=8,
        out_elem_index=0, psum_flag=0, final_bias=final_bias_oc0,
    )
    if sample is None:
        return []
    desc, raw_acc_oc0 = sample

    sample_elem_count = desc[12] & 0xFF
    if sample_elem_count == 0:
        return []
    if (desc[3] & (1 << 3)) == 0 or (desc[30] & 0xFF) != sample_elem_count:
        return []

    act_src = desc[28]
    wgt_src = desc[29]
    if data[layer.in_off + act_src:layer.in_off + act_src + sample_elem_count] != \
       descriptor_payload_bytes(desc, 4, sample_elem_count):
        return []
    if data[layer.wgt_off + wgt_src:layer.wgt_off + wgt_src + sample_elem_count] != \
       descriptor_payload_bytes(desc, 8, sample_elem_count):
        return []

    # Tile element count
    group = layer.group or 1
    in_per_group = layer.in_c // group
    total_lanes = max(layer.k_h or 1, 1) * max(layer.k_w or 1, 1) * max(in_per_group, 1)
    tile_elem_count = min(total_lanes, layer.in_size - act_src, layer.wgt_size - wgt_src, CONV_TILE_MAX_ELEMS)
    tile_elem_count -= tile_elem_count % 16
    if tile_elem_count == 0:
        tile_elem_count = sample_elem_count
    if tile_elem_count < sample_elem_count:
        return []

    # Determine how many iterations fit in L1 without overlap
    l1_base = 0x58000 + ((layer.index * 0x100 + ordinal * 0x20) & 0x7FF0)
    l1_act = l1_base
    l1_wgt = l1_act + tile_elem_count
    # Use at most tile_oc copies, limited by L1 address space
    actual_tile_oc = 1
    for oc_i in range(1, tile_oc):
        if l1_wgt + (oc_i + 1) * tile_elem_count < (1 << 22):
            actual_tile_oc = oc_i + 1
        else:
            break
    if actual_tile_oc < 2:
        return []   # Need >= 2 iterations to prove the loop
    l1_result = l1_wgt + actual_tile_oc * tile_elem_count

    desc[1] = tile_elem_count
    desc[30] = (desc[30] & ~0xFF) | sample_elem_count
    desc[31] = min(output_elems, CONV_TILE_MAX_OUTPUTS)

    # UDMA: load ACT
    descs: list[list[int]] = [
        udma_dram_to_l1_descriptor(layer, ordinal, layer.in_off + act_src, l1_act, tile_elem_count, SMF_LOAD_A),
    ]
    # UDMA: load actual_tile_oc copies of the same weight row
    for oc_i in range(actual_tile_oc):
        descs.append(udma_dram_to_l1_descriptor(
            layer, ordinal + len(descs),
            layer.wgt_off + wgt_src,
            l1_wgt + oc_i * tile_elem_count,
            tile_elem_count, SMF_LOAD_B,
        ))

    # CONV descriptor: L1-read, chain_out_enable=1, conv_tile_oc_count=actual_tile_oc
    # chain_psum_data from the last iteration stays valid when CONV completes.
    desc[2] = l1_base
    desc[3] = (desc[3] & ~(1 << 6)) | READ_FROM_L1_FLAG | MICROBLOCK_FLAG | CONV_CHAIN_OUT_FLAG
    desc[27] = l1_result   # dummy (chain_skip_store skips write)
    stamp_synth_microblock_metadata(
        desc, layer.index, ordinal + len(descs), ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    # conv_tile_oc_count in word[31][31:16] (upper 16 bits unused by non-microblock checks)
    desc[31] = (desc[31] & 0x0000_FFFF) | ((actual_tile_oc & 0xFFFF) << 16)
    descs.append(desc)

    # Expected: REQUANT gets chain_psum_data = acc from LAST iteration.
    # Since all iterations use the same act+wgt (copies of same row), acc is the same.
    eq = mbqm(clamp_i32(raw_acc_oc0), params.mult[0], params.shift[0]) + params.zp_out
    eq = max(params.act_min, min(params.act_max, eq))
    expected_byte = bytes([eq & 0xFF])

    # REQUANT: use_chain_input=1, tile_drain_count=1 (drain once from chain).
    rq_words = [0] * WORDS_PER_COMMAND
    rq_words[0] = OP_REQUANT
    rq_words[1] = 1
    rq_words[2] = l1_result
    rq_words[3] = REQUANT_USE_CHAIN | (1 << 6)
    # rq_words[8] = 0 (tile_drain_count=0 → 1, single drain)
    rq_words[14] = params.mult[0] & 0xFFFF_FFFF
    rq_words[15] = (params.shift[0] & 0xFF) | ((params.zp_out & 0xFF) << 8)
    rq_words[16] = params.act_min & 0xFFFF_FFFF
    rq_words[17] = params.act_max & 0xFFFF_FFFF
    rq_words[18] = expected_byte[0] & 0xFF
    rq_words[19] = layer.index
    rq_words[27] = l1_result
    descs.append(rq_words)

    # L1CRC: verify the 1 byte at l1_result
    crc_check = l1mesh_crc_probe_descriptor(
        ordinal + len(descs), l1_result, expected_byte,
    )
    if crc_check is None:
        return []
    crc_check[3] |= MICROBLOCK_FLAG
    stamp_synth_microblock_metadata(
        crc_check, layer.index, ordinal + len(descs), ordinal + len(descs),
        SMF_FINAL_TILE,
    )
    descs.append(crc_check)
    return descs


def closed_loop_conv_requant_ow_tile_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    tile_ow: int = 2,
) -> list[list[int]]:
    """OW spatial tile probe: CONV iterates tile_ow output columns, re-loading ACT
    from a different L1 address each time while keeping the weight resident.
    Both iterations push one psum to chain (OC=1 per column). REQUANT drains
    tile_ow times (tile_drain_count=tile_ow). L1CRC verifies both output bytes.

    Tests v12 Phase 6c: the OW spatial tile loop and the act_tile_col_stride
    pointer advancement.

    Simplification: both OW positions load the SAME data (two identical copies
    of the ACT vector) so both drains produce the same expected output byte.
    """
    if layer.op_kind not in OK_CONV or elem_bytes(layer.dtype) != 1:
        return []
    data = descriptor_for_layer.program_bytes
    params = int8_conv_params(layer, data)
    if params is None:
        return []
    if layer.in_size <= 0 or layer.wgt_size <= 0:
        return []

    # Build base CONV descriptor for OC=0, ow=0
    oc0 = 0
    oh, ow = 0, 0
    final_bias_oc0 = params.bias[oc0] + int8_conv_corr(params, layer, oh, ow, oc0)
    sample = int8_conv_real_descriptor(
        layer, params, ordinal, start_lane=0, max_count=8,
        out_elem_index=0, psum_flag=0, final_bias=final_bias_oc0,
    )
    if sample is None:
        return []
    desc, raw_acc_oc0 = sample

    sample_elem_count = desc[12] & 0xFF
    if sample_elem_count == 0:
        return []
    if (desc[3] & (1 << 3)) == 0 or (desc[30] & 0xFF) != sample_elem_count:
        return []

    act_src = desc[28]
    wgt_src = desc[29]
    if data[layer.in_off + act_src:layer.in_off + act_src + sample_elem_count] != \
       descriptor_payload_bytes(desc, 4, sample_elem_count):
        return []
    if data[layer.wgt_off + wgt_src:layer.wgt_off + wgt_src + sample_elem_count] != \
       descriptor_payload_bytes(desc, 8, sample_elem_count):
        return []

    tile_elem_count = sample_elem_count

    # L1 layout: tile_ow copies of ACT (stride = tile_elem_count), then WGT
    l1_base = 0x58000 + ((layer.index * 0x100 + ordinal * 0x20) & 0x7FF0)
    l1_act = l1_base
    l1_wgt = l1_act + tile_ow * tile_elem_count
    l1_result = l1_wgt + tile_elem_count

    if l1_result + tile_ow >= (1 << 22):
        return []

    desc[1] = tile_elem_count
    desc[30] = (desc[30] & ~0xFF) | sample_elem_count
    desc[31] = min(layer.ref_size // max(elem_bytes(layer.dtype), 1), CONV_TILE_MAX_OUTPUTS)

    # UDMA: load tile_ow identical copies of ACT into consecutive L1 slots
    descs: list[list[int]] = []
    for ow_i in range(tile_ow):
        descs.append(udma_dram_to_l1_descriptor(
            layer, ordinal + len(descs),
            layer.in_off + act_src,
            l1_act + ow_i * tile_elem_count,
            tile_elem_count, SMF_LOAD_A,
        ))
    # UDMA: load WGT (one weight row, shared by all OW positions)
    descs.append(udma_dram_to_l1_descriptor(
        layer, ordinal + len(descs),
        layer.wgt_off + wgt_src,
        l1_wgt,
        tile_elem_count, SMF_LOAD_B,
    ))

    # CONV descriptor: L1-read, chain_out_enable=1,
    # conv_tile_ow_count=tile_ow in word[8][31:16],
    # act_tile_col_stride=tile_elem_count in word[10][21:0].
    desc[2] = l1_base         # L1 base addr (ACT starts here)
    desc[3] = (desc[3] & ~(1 << 6)) | READ_FROM_L1_FLAG | MICROBLOCK_FLAG | CONV_CHAIN_OUT_FLAG
    # word[8][31:16] = conv_tile_ow_count; word[8][15:0] = 0 (no single-descriptor
    # drain — drain is handled by the separate REQUANT descriptor below).
    desc[8] = (tile_ow & CONV_OW_COUNT_MASK) << CONV_OW_COUNT_SHIFT
    # word[10][21:0] = act_col_stride = tile_elem_count (bytes between ACT copies)
    desc[10] = tile_elem_count & 0x3FFFFF
    # WGT starts at l1_wgt = l1_act + tile_ow * tile_elem_count; encode in
    # the standard wgt-relative-to-base field (workload_sample_bytes offset).
    # The RTL adds l1_req_base_addr + workload_sample_bytes + wgt_tile_l1_offset.
    # wgt_tile_l1_offset=0, so wgt addr = l1_base + workload_sample_bytes.
    # We need l1_wgt = l1_base + tile_ow * tile_elem_count, so set workload_bytes
    # such that workload_sample_bytes = tile_ow * tile_elem_count.
    desc[1] = tile_ow * tile_elem_count   # workload_bytes → wgt offset from base
    desc[27] = l1_result  # output slot (chain_skip_store skips write anyway)
    stamp_synth_microblock_metadata(
        desc, layer.index, ordinal + len(descs), ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    # conv_tile_oc_count=0 (=1, single OC per OW position) in word[31][31:16]
    desc[31] = (desc[31] & 0x0000_FFFF) | (0 << 16)
    descs.append(desc)

    # Expected: same ACT + same WGT → same acc for every OW position
    eq = mbqm(clamp_i32(raw_acc_oc0), params.mult[0], params.shift[0]) + params.zp_out
    eq = max(params.act_min, min(params.act_max, eq))
    expected_byte = bytes([eq & 0xFF])
    expected_bytes = expected_byte * tile_ow

    # REQUANT: tile_drain_count=tile_ow drains consecutive bytes
    rq_words = [0] * WORDS_PER_COMMAND
    rq_words[0] = OP_REQUANT
    rq_words[1] = 1
    rq_words[2] = l1_result
    rq_words[3] = REQUANT_USE_CHAIN | (1 << 6)
    rq_words[8] = tile_ow & 0xFFFF   # tile_drain_count in word[8][15:0]
    rq_words[14] = params.mult[0] & 0xFFFF_FFFF
    rq_words[15] = (params.shift[0] & 0xFF) | ((params.zp_out & 0xFF) << 8)
    rq_words[16] = params.act_min & 0xFFFF_FFFF
    rq_words[17] = params.act_max & 0xFFFF_FFFF
    rq_words[18] = expected_byte[0] & 0xFF
    rq_words[19] = layer.index
    rq_words[27] = l1_result
    descs.append(rq_words)

    # L1CRC: verify tile_ow bytes starting at l1_result
    crc_check = l1mesh_crc_probe_descriptor(
        ordinal + len(descs), l1_result, expected_bytes,
    )
    if crc_check is None:
        return []
    crc_check[3] |= MICROBLOCK_FLAG
    stamp_synth_microblock_metadata(
        crc_check, layer.index, ordinal + len(descs), ordinal + len(descs),
        SMF_FINAL_TILE,
    )
    descs.append(crc_check)
    return descs


def closed_loop_conv_probes(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    command_budget: int,
    sample_count: int = DEFAULT_CONV_SAMPLE_COUNT,
) -> list[list[int]]:
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    descs: list[list[int]] = []

    def nearest_valid_probe(target_index: int) -> tuple[int, list[list[int]]] | None:
        for delta in range(output_elems):
            for candidate in (target_index - delta, target_index + delta):
                if candidate < 0 or candidate >= output_elems:
                    continue
                if candidate in sampled_indices:
                    continue
                probe = closed_loop_conv_probe(
                    layer,
                    ordinal + len(descs),
                    result_dram_off,
                    out_elem_index=candidate,
                )
                if probe:
                    return candidate, probe
        return None

    sampled_indices: set[int] = set()
    if sample_count > 0:
        targets = closed_loop_output_indices(
            output_elems,
            max(sample_count, 1) * CONV_PROBE_COMMANDS,
            CONV_PROBE_COMMANDS,
        )
    else:
        targets = closed_loop_output_indices(output_elems, command_budget, CONV_PROBE_COMMANDS)

    for out_elem_index in targets:
        remaining = command_budget - len(descs)
        if remaining < CONV_PROBE_COMMANDS:
            break
        found = nearest_valid_probe(out_elem_index) if sample_count > 0 else None
        if found is None:
            probe = closed_loop_conv_probe(
                layer,
                ordinal + len(descs),
                result_dram_off,
                out_elem_index=out_elem_index,
            )
            if not probe:
                continue
            found = (out_elem_index, probe)
        valid_index, probe = found
        if len(probe) > remaining:
            break
        sampled_indices.add(valid_index)
        descs.extend(probe)

    # v12 Phase 4: emit one CONV->REQUANT chain probe to exercise the psum
    # forwarding wire. Only for INT8 CONV layers; limit to 1 per call so we
    # don't blow the command budget.
    CHAIN_PROBE_COMMANDS = 5   # 2 UDMA + 1 CONV + 1 REQUANT + 1 L1CRC
    if command_budget - len(descs) >= CHAIN_PROBE_COMMANDS:
        chain_probes = closed_loop_conv_requant_chain_probe(
            layer, ordinal + len(descs), result_dram_off, out_elem_index=0,
        )
        if chain_probes and len(chain_probes) <= command_budget - len(descs):
            descs.extend(chain_probes)

    # v12 Phase 6c: emit one OC tile probe to exercise conv_tile_oc_count and
    # requant tile_drain_count together. Only for INT8 CONV with >= 4 OC.
    OC_TILE_PROBE_COMMANDS = 6   # 2 UDMA + 1 CONV + 1 REQUANT + 1 L1CRC + margin
    if command_budget - len(descs) >= OC_TILE_PROBE_COMMANDS:
        oc_tile_probes = closed_loop_conv_requant_oc_tile_probe(
            layer, ordinal + len(descs), result_dram_off, tile_oc=4,
        )
        if oc_tile_probes and len(oc_tile_probes) <= command_budget - len(descs):
            descs.extend(oc_tile_probes)

    # v12 Phase 6c (OW spatial): emit one OW tile probe to exercise the outer
    # OW spatial loop in CONV (act_tile_col_stride, conv_tile_ow_count) and
    # multi-element drain in REQUANT.
    OW_TILE_PROBE_COMMANDS = 8   # tile_ow UDMAs + 1 WGT UDMA + CONV + REQUANT + L1CRC + margin
    if command_budget - len(descs) >= OW_TILE_PROBE_COMMANDS:
        ow_tile_probes = closed_loop_conv_requant_ow_tile_probe(
            layer, ordinal + len(descs), result_dram_off, tile_ow=2,
        )
        if ow_tile_probes and len(ow_tile_probes) <= command_budget - len(descs):
            descs.extend(ow_tile_probes)

    return descs


def conv_shadow_readback_descriptor(
    layer: Layer,
    ordinal: int,
    final_descs: list[list[int]],
    ref_bytes: bytes | None = None,
    sram_start_offset: int = 0,
) -> list[int] | None:
    if not final_descs:
        return None
    final_desc = final_descs[-1]
    tile_count = final_desc[31] & 0xFF
    if tile_count == 0:
        tile_count = 1
    first_out_offset = final_desc[27]
    final_q_bytes: list[int] = []
    for desc in final_descs:
        tile_outputs = desc[31] & 0xFF
        if tile_outputs == 0:
            tile_outputs = 1
        final_q_bytes.extend([desc[18] & 0xFF] * tile_outputs)
    read_elem_index = first_out_offset + tile_count - 1
    sample = int8_conv_sample_descriptor(
        layer,
        ordinal,
        start_lane=0,
        max_count=8,
        out_elem_index=read_elem_index,
    )
    if sample is None:
        return None
    desc, _ = sample
    if ref_bytes is not None:
        desc[3] = (desc[3] & ~(1 << 2)) | (1 << 6) | (1 << 10)
        desc[27] = sram_start_offset & 0xFFFF_FFFF
    else:
        desc[3] = (desc[3] & ~(1 << 2)) | (1 << 7) | (1 << 8)
    expected_last = (ref_bytes[-1] if ref_bytes else final_desc[18]) & 0xFF
    desc[19] = (expected_last if expected_last < 0x80 else expected_last - 0x100) & 0xFFFF_FFFF
    crc = FNV_OFFSET
    crc_source = ref_bytes if ref_bytes is not None else bytes(final_q_bytes)
    for byte_value in crc_source:
        crc = fnv_byte(crc, byte_value)
    desc[28] = crc
    desc[29] = len(crc_source)
    desc[31] = (desc[31] & ~0xFF) | 1
    return desc


def conv_final_q_bytes(final_descs: list[list[int]]) -> bytes:
    final_q_bytes: list[int] = []
    for desc in final_descs:
        tile_outputs = desc[31] & 0xFF
        if tile_outputs == 0:
            tile_outputs = 1
        final_q_bytes.extend([desc[18] & 0xFF] * tile_outputs)
    return bytes(final_q_bytes)


def conv_sramcrc_window_starts(output_elems: int, window_elems: int, window_count: int) -> list[int]:
    if output_elems <= 0 or window_elems <= 0 or window_count <= 0:
        return []
    window_elems = min(window_elems, output_elems)
    if window_count == 1:
        candidates = [0]
    else:
        span = output_elems - window_elems
        candidates = [
            round((span * idx) / (window_count - 1))
            for idx in range(window_count)
        ]
    starts: list[int] = []
    for start in candidates:
        start = max(min(start, output_elems - window_elems), 0)
        if start not in starts:
            starts.append(start)
    return starts


def int8_pool_output_sample(layer: Layer, out_elem_index: int) -> tuple[bytes, int, int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.op_kind not in OK_POOL or elem_bytes(layer.dtype) != 1:
        return None
    if (
        layer.in_h <= 0 or layer.in_w <= 0 or layer.in_c <= 0 or
        layer.out_h <= 0 or layer.out_w <= 0 or layer.out_c <= 0
    ):
        return None
    output_elems = layer.out_h * layer.out_w * layer.out_c
    if out_elem_index < 0 or out_elem_index >= output_elems:
        return None
    out_area = max(layer.out_w * layer.out_c, 1)
    oh = out_elem_index // out_area
    ow = (out_elem_index % out_area) // max(layer.out_c, 1)
    oc = out_elem_index % max(layer.out_c, 1)
    kh_size = max(layer.k_h, 1)
    kw_size = max(layer.k_w, 1)
    sh = layer.s_h or 1
    sw = layer.s_w or 1
    values: list[int] = []
    sample = bytearray()
    for kh in range(kh_size):
        for kw in range(kw_size):
            ih = oh * sh + kh - layer.p_t
            iw = ow * sw + kw - layer.p_l
            if 0 <= ih < layer.in_h and 0 <= iw < layer.in_w:
                input_elem = ((ih * layer.in_w + iw) * layer.in_c) + oc
                if 0 <= input_elem < layer.in_size:
                    byte_value = data[layer.in_off + input_elem]
                    values.append(i8(byte_value))
                    if len(sample) < 16:
                        sample.append(byte_value)
    if not values or len(values) > 16:
        return None
    if layer.op_kind == 2:
        expected = int(sum(values) / len(values))
    else:
        expected = max(values)
    expected = max(-128, min(127, expected))
    return bytes(sample).ljust(16, b"\x00"), len(values), expected & 0xFF


def pool_int8_output_descriptor(
    layer: Layer,
    ordinal: int,
    out_elem_index: int,
    read_sample_from_l1: bool = False,
) -> list[int] | None:
    sample = int8_pool_output_sample(layer, out_elem_index)
    if sample is None:
        return None
    sample_bytes_value, elem_count, expected_q = sample
    avg_mode = 1 if layer.op_kind == 2 else 0
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_POOL
    words[1] = elem_count
    words[2] = addr
    words[3] = (1 << 6) | ((1 << 11) if read_sample_from_l1 else 0)
    for idx in range(4):
        words[4 + idx] = pack_word(sample_bytes_value[idx * 4:(idx + 1) * 4])
    words[12] = elem_count | (avg_mode << 8)
    words[18] = expected_q
    words[19] = layer.index
    words[27] = out_elem_index & 0xFFFF_FFFF
    return words


def pool_int8_output_descriptors(
    layer: Layer,
    ordinal: int,
    max_output_elems: int,
    start_output_elem: int = 0,
    read_sample_from_l1: bool = False,
    max_commands: int | None = None,
) -> list[list[int]]:
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    start_output_elem = max(min(start_output_elem, output_elems - 1), 0)
    emit_output_elems = min(max_output_elems, output_elems - start_output_elem)
    descs: list[list[int]] = []
    for out_elem_index in range(start_output_elem, start_output_elem + emit_output_elems):
        sample = int8_pool_output_sample(layer, out_elem_index)
        if sample is None:
            break
        sample_bytes_value, elem_count, _ = sample
        if read_sample_from_l1:
            if (out_elem_index & 0xF) + elem_count > 16:
                break
            if max_commands is not None and len(descs) + elem_count + 1 > max_commands:
                break
            for sample_idx in range(elem_count):
                descs.append(
                    l1_preload_byte_descriptor(
                        ordinal + len(descs),
                        out_elem_index + sample_idx,
                        sample_bytes_value[sample_idx],
                        layer.index,
                    )
                )
        desc = pool_int8_output_descriptor(
            layer,
            ordinal + len(descs),
            out_elem_index,
            read_sample_from_l1=read_sample_from_l1,
        )
        if desc is None:
            break
        descs.append(desc)
    return descs


def pool_final_q_bytes(final_descs: list[list[int]]) -> bytes:
    return bytes(
        desc[18] & 0xFF
        for desc in final_descs
        if (desc[0] & 0xF) == OP_POOL and (desc[3] & (1 << 6))
    )


def int8_pool_source_offsets(layer: Layer, out_elem_index: int) -> list[int] | None:
    if layer.op_kind not in OK_POOL or elem_bytes(layer.dtype) != 1:
        return None
    if (
        layer.in_h <= 0 or layer.in_w <= 0 or layer.in_c <= 0 or
        layer.out_h <= 0 or layer.out_w <= 0 or layer.out_c <= 0
    ):
        return None
    output_elems = layer.out_h * layer.out_w * layer.out_c
    if out_elem_index < 0 or out_elem_index >= output_elems:
        return None
    out_area = max(layer.out_w * layer.out_c, 1)
    oh = out_elem_index // out_area
    ow = (out_elem_index % out_area) // max(layer.out_c, 1)
    oc = out_elem_index % max(layer.out_c, 1)
    offsets: list[int] = []
    for kh in range(max(layer.k_h, 1)):
        for kw in range(max(layer.k_w, 1)):
            ih = oh * (layer.s_h or 1) + kh - layer.p_t
            iw = ow * (layer.s_w or 1) + kw - layer.p_l
            if 0 <= ih < layer.in_h and 0 <= iw < layer.in_w:
                input_elem = ((ih * layer.in_w + iw) * layer.in_c) + oc
                if 0 <= input_elem < layer.in_size:
                    offsets.append(input_elem)
    return offsets if offsets else None


def closed_loop_pool_probe(layer: Layer, ordinal: int, result_dram_off: int,
                           out_elem_index: int = 0) -> list[list[int]]:
    sample = int8_pool_output_sample(layer, out_elem_index)
    offsets = int8_pool_source_offsets(layer, out_elem_index)
    if sample is None or offsets is None:
        return []
    _, elem_count, expected_q = sample
    if elem_count <= 0 or elem_count > 16 or len(offsets) < elem_count:
        return []
    source_offsets = offsets[:elem_count]
    l1_base = 0x10000 + (((layer.index * 0x100) + (out_elem_index * 0x20)) & 0x2FFFF)
    l1_result = l1_base + 0x80
    descs: list[list[int]] = []
    for idx, src_off in enumerate(source_offsets):
        descs.append(
            udma_dram_to_l1_descriptor(
                layer,
                ordinal + len(descs),
                layer.in_off + src_off,
                l1_base + idx,
                1,
                SMF_LOAD_A,
            )
        )
    desc = pool_int8_output_descriptor(
        layer, ordinal + len(descs), out_elem_index, read_sample_from_l1=True)
    if desc is None:
        return []
    desc[2] = l1_base
    desc[3] |= MICROBLOCK_FLAG
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off + out_elem_index,
            bytes([expected_q]),
        )
    )
    return descs


def closed_loop_pool_probes(layer: Layer, ordinal: int, result_dram_off: int,
                            command_budget: int) -> list[list[int]]:
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 0)
    descs: list[list[int]] = []
    indices = closed_loop_output_indices(output_elems, command_budget, 20)[:BYTE_TILE_SWEEP_PROBES]
    for out_elem_index in indices:
        probe = closed_loop_pool_probe(
            layer, ordinal + len(descs), result_dram_off, out_elem_index)
        if not probe:
            continue
        if len(descs) + len(probe) > command_budget:
            break
        descs.extend(probe)
    return descs


def pool_sramcrc_probe_descriptor(layer: Layer, ordinal: int, start_output_elem: int, ref_bytes: bytes) -> list[int] | None:
    if not ref_bytes:
        return None
    desc = descriptor_for_layer(layer, ordinal, False)
    if desc is None or (desc[0] & 0xF) != OP_POOL:
        return None
    desc[3] = (desc[3] & ~((1 << 9) | (1 << 6))) | (1 << 10)
    desc[25] = (layer.ref_off + start_output_elem) & 0xFFFF_FFFF
    desc[27] = start_output_elem & 0xFFFF_FFFF
    desc[28] = fnv_bytes(ref_bytes)
    desc[29] = len(ref_bytes)
    return desc


def int8_ewe_output_value(layer: Layer, out_elem_index: int) -> int | None:
    data = descriptor_for_layer.program_bytes
    if layer.op_kind not in OK_EWE or elem_bytes(layer.dtype) != 1:
        return None
    if out_elem_index < 0 or out_elem_index >= layer.ref_size:
        return None
    if out_elem_index >= layer.in_size or out_elem_index >= layer.wgt_size:
        return None
    params = int8_ewe_params(layer)
    if params is None:
        return None
    av = i8(data[layer.in_off + out_elem_index])
    bv = i8(data[layer.wgt_off + out_elem_index])
    zp_a, zp_b, zp_out, mult_a, shift_a, mult_b, shift_b, mult_o, shift_o, left_shift, act_min, act_max = params
    if layer.op_kind == OK_MUL:
        value = mbqm(clamp_i32((av - zp_a) * (bv - zp_b)), mult_o, shift_o) + zp_out
    elif layer.op_kind == OK_SUB:
        a = mbqm(clamp_i32((av - zp_a) << left_shift), mult_a, shift_a)
        b = mbqm(clamp_i32((bv - zp_b) << left_shift), mult_b, shift_b)
        value = mbqm(clamp_i32(a - b), mult_o, shift_o) + zp_out
    else:
        a = mbqm(clamp_i32((av - zp_a) << left_shift), mult_a, shift_a)
        b = mbqm(clamp_i32((bv - zp_b) << left_shift), mult_b, shift_b)
        value = mbqm(clamp_i32(a + b), mult_o, shift_o) + zp_out
    return max(act_min, min(act_max, value)) & 0xFF


def int8_ewe_params(layer: Layer) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.wgt_size < 48:
        return None
    param_off = layer.wgt_off + layer.wgt_size - 48
    if param_off < layer.wgt_off or param_off + 48 > len(data):
        return None
    return tuple(rdi32(data, param_off + idx * 4) for idx in range(12))  # type: ignore[return-value]


def int8_ewe_output_descriptor(
    layer: Layer,
    ordinal: int,
    out_elem_index: int,
    read_a_from_l1: bool = False,
) -> list[int] | None:
    expected = int8_ewe_output_value(layer, out_elem_index)
    if expected is None:
        return None
    data = descriptor_for_layer.program_bytes
    params = int8_ewe_params(layer)
    if params is None:
        return None
    op_mode = 1 if layer.op_kind == OK_MUL else 2 if layer.op_kind == OK_SUB else 0
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x20) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_EWE
    words[1] = 1
    words[2] = addr
    words[3] = (1 << 6) | ((1 << 11) if read_a_from_l1 else 0)
    words[4] = data[layer.in_off + out_elem_index]
    words[8] = data[layer.wgt_off + out_elem_index]
    words[12] = 1 | (op_mode << 8)
    words[13] = params[0] & 0xFFFF_FFFF
    words[14] = params[1] & 0xFFFF_FFFF
    words[15] = params[2] & 0xFFFF_FFFF
    words[16] = params[3] & 0xFFFF_FFFF
    words[17] = params[4] & 0xFFFF_FFFF
    words[18] = expected
    words[19] = layer.index
    words[20] = params[5] & 0xFFFF_FFFF
    words[21] = params[6] & 0xFFFF_FFFF
    words[22] = params[7] & 0xFFFF_FFFF
    words[23] = params[8] & 0xFFFF_FFFF
    words[24] = params[9] & 0xFFFF_FFFF
    words[25] = params[10] & 0xFFFF_FFFF
    words[26] = params[11] & 0xFFFF_FFFF
    words[27] = out_elem_index & 0xFFFF_FFFF
    return words


def int8_ewe_output_descriptors(
    layer: Layer,
    ordinal: int,
    max_output_elems: int,
    start_output_elem: int = 0,
    read_a_from_l1: bool = False,
) -> list[list[int]]:
    output_elems = min(layer.ref_size, layer.in_size, layer.wgt_size)
    if output_elems <= 0:
        return []
    start_output_elem = max(min(start_output_elem, output_elems - 1), 0)
    emit_output_elems = min(max_output_elems, output_elems - start_output_elem)
    descs: list[list[int]] = []
    for out_elem_index in range(start_output_elem, start_output_elem + emit_output_elems):
        if read_a_from_l1:
            byte_value = descriptor_for_layer.program_bytes[layer.in_off + out_elem_index]
            descs.append(l1_preload_byte_descriptor(ordinal + len(descs), out_elem_index, byte_value, layer.index))
        desc = int8_ewe_output_descriptor(
            layer,
            ordinal + len(descs),
            out_elem_index,
            read_a_from_l1=read_a_from_l1,
        )
        if desc is None:
            break
        descs.append(desc)
    return descs


def ewe_final_q_bytes(final_descs: list[list[int]]) -> bytes:
    return bytes(
        desc[18] & 0xFF
        for desc in final_descs
        if (desc[0] & 0xF) == OP_EWE and (desc[3] & (1 << 6))
    )


def closed_loop_ewe_probe(layer: Layer, ordinal: int, result_dram_off: int,
                          out_elem_index: int = 0) -> list[list[int]]:
    if out_elem_index < 0 or out_elem_index >= layer.ref_size:
        return []
    l1_base = 0x20000 + (((layer.index * 0x100) + (out_elem_index * 0x20)) & 0x2FFFF)
    l1_result = l1_base + 0x80
    descs: list[list[int]] = [
        udma_dram_to_l1_descriptor(
            layer, ordinal, layer.in_off + out_elem_index, l1_base, 1, SMF_LOAD_A)
    ]
    desc = int8_ewe_output_descriptor(
        layer, ordinal + len(descs), out_elem_index, read_a_from_l1=True)
    if desc is None:
        return []
    expected = desc[18] & 0xFF
    desc[2] = l1_base
    desc[3] |= MICROBLOCK_FLAG
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off + out_elem_index,
            bytes([expected]),
        )
    )
    return descs


def lut_l1_addr_for_layer(layer: Layer) -> int:
    """Reserve an L1 region for the layer's 256-byte unary LUT, kept clear of
    the per-probe input/result slots used by closed_loop_lut_unary_probe."""
    return 0x30000 + ((layer.index * 0x100) & 0x0FFFF)


def int8_lut_unary_descriptor(
    layer: Layer,
    ordinal: int,
    out_elem_index: int,
    l1_input_addr: int,
    l1_result_addr: int,
) -> list[int] | None:
    """v11: build a single OP_EWE descriptor in LUT mode for INT8 unary ops.
    Engine reads one input byte from l1_input_addr, indexes its preloaded
    lut_mem (filled in ST_LUT_LOAD from lut_addr), writes lut_mem[input] to
    l1_result_addr. Input bytes and the 256-byte LUT must both already live
    in L1 (separate UDMA descriptors)."""
    if layer.wgt_size < 256:
        return None
    if out_elem_index < 0 or out_elem_index >= layer.in_size or out_elem_index >= layer.ref_size:
        return None
    data = descriptor_for_layer.program_bytes
    in_byte = data[layer.in_off + out_elem_index]
    # Compute the expected output by indexing the 256-byte LUT in wgt_b with
    # the unsigned-byte view of the input. Engine should produce the same byte.
    lut_byte = data[layer.wgt_off + in_byte]
    expected = lut_byte & 0xFF
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_EWE
    words[1] = 1
    words[2] = l1_input_addr & 0x003F_FFFF
    # final_q_mode bit (1<<6) is unused under lut_mode but kept for parity
    # with the binary EWE descriptor layout. read_a_from_l1 = 1 means a_vec[0]
    # comes from the L1 input we just preloaded.
    words[3] = (1 << 6) | (1 << 11)
    words[4] = in_byte                      # a_vec[0] (also fetched from L1)
    words[8] = 0                            # b_vec unused in LUT mode
    # elem_count=1, op_mode=0 (unused), lut_mode bit 12 = 1
    words[12] = 1 | (0 << 8) | (1 << 12)
    words[18] = expected
    words[19] = layer.index
    words[27] = l1_result_addr & 0xFFFF_FFFF  # output store offset in L1
    words[30] = lut_l1_addr_for_layer(layer) & 0xFFFF_FFFF
    return words


def closed_loop_lut_unary_probe(layer: Layer, ordinal: int, result_dram_off: int,
                                out_elem_index: int = 0) -> list[list[int]]:
    if out_elem_index < 0 or out_elem_index >= layer.ref_size:
        return []
    l1_input  = 0x20000 + (((layer.index * 0x100) + (out_elem_index * 0x20)) & 0x2FFFF)
    l1_result = l1_input + 0x80
    descs: list[list[int]] = [
        udma_dram_to_l1_descriptor(
            layer, ordinal, layer.in_off + out_elem_index, l1_input, 1, SMF_LOAD_A)
    ]
    desc = int8_lut_unary_descriptor(
        layer, ordinal + len(descs), out_elem_index, l1_input, l1_result)
    if desc is None:
        return []
    expected = desc[18] & 0xFF
    desc[3] |= MICROBLOCK_FLAG
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off + out_elem_index,
            bytes([expected]),
        )
    )
    return descs


def closed_loop_lut_unary_probes(layer: Layer, ordinal: int, result_dram_off: int,
                                 command_budget: int) -> list[list[int]]:
    if layer.wgt_size < 256:
        return []
    output_elems = min(layer.ref_size, layer.in_size)
    descs: list[list[int]] = []
    # Load the 256-byte LUT once per layer. The DRAM model returns only 16 B
    # per request, so split the 256-byte LUT into 16 sequential UDMA loads of
    # 16 B each. Engine reloads from L1 on every descriptor entry into
    # ST_LUT_LOAD, but the L1 contents stay valid across all probes.
    lut_l1 = lut_l1_addr_for_layer(layer)
    for beat in range(16):
        descs.append(
            udma_dram_to_l1_descriptor(
                layer, ordinal + len(descs),
                layer.wgt_off + beat * 16,
                lut_l1 + beat * 16,
                16, SMF_LOAD_B)
        )
    indices = closed_loop_output_indices(output_elems, command_budget, 5)[:BYTE_TILE_SWEEP_PROBES]
    for out_elem_index in indices:
        probe = closed_loop_lut_unary_probe(
            layer, ordinal + len(descs), result_dram_off, out_elem_index)
        if not probe:
            continue
        if len(descs) + len(probe) > command_budget:
            break
        descs.extend(probe)
    return descs


def closed_loop_ewe_probes(layer: Layer, ordinal: int, result_dram_off: int,
                           command_budget: int) -> list[list[int]]:
    output_elems = min(layer.ref_size, layer.in_size, layer.wgt_size)
    descs: list[list[int]] = []
    indices = closed_loop_output_indices(output_elems, command_budget, 5)[:BYTE_TILE_SWEEP_PROBES]
    for out_elem_index in indices:
        probe = closed_loop_ewe_probe(
            layer, ordinal + len(descs), result_dram_off, out_elem_index)
        if not probe:
            continue
        if len(descs) + len(probe) > command_budget:
            break
        descs.extend(probe)
    return descs


def int16_ewe_output_descriptor(
    layer: Layer,
    ordinal: int,
    out_elem_index: int,
    read_a_from_l1: bool = False,
) -> list[int] | None:
    if layer.op_kind not in OK_EWE or elem_bytes(layer.dtype) != 2:
        return None
    byte_index = out_elem_index * 2
    data = descriptor_for_layer.program_bytes
    if byte_index + 2 > layer.in_size or byte_index + 2 > layer.wgt_size:
        return None
    op_mode = 1 if layer.op_kind == OK_MUL else 2 if layer.op_kind == OK_SUB else 0
    a_bytes = data[layer.in_off + byte_index:layer.in_off + byte_index + 2]
    b_bytes = data[layer.wgt_off + byte_index:layer.wgt_off + byte_index + 2]
    av = i16_at(a_bytes.ljust(2, b"\x00"), 0)
    bv = i16_at(b_bytes.ljust(2, b"\x00"), 0)
    if layer.op_kind == OK_MUL:
        expected = av * bv
    elif layer.op_kind == OK_SUB:
        expected = av - bv
    else:
        expected = av + bv
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x24) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_EWE
    words[1] = 2
    words[2] = addr
    words[3] = (1 << 6) | ((1 << 11) if read_a_from_l1 else 0)
    words[4] = pack_word(a_bytes)
    words[8] = pack_word(b_bytes)
    words[12] = 1 | (op_mode << 8) | (1 << 11)
    words[18] = expected & 0xFFFF_FFFF
    words[19] = layer.index
    words[27] = byte_index & 0xFFFF_FFFF
    return words


def closed_loop_int16_ewe_probe(
    layer: Layer,
    ordinal: int,
    result_dram_off: int,
    out_elem_index: int = 0,
) -> list[list[int]]:
    byte_index = out_elem_index * 2
    desc = int16_ewe_output_descriptor(
        layer,
        ordinal + 1,
        out_elem_index,
        read_a_from_l1=True,
    )
    if desc is None:
        return []
    expected = (desc[18] & 0xFFFF).to_bytes(2, "little", signed=False)
    l1_base = 0x24000 + (((layer.index * 0x100) + (out_elem_index * 0x20)) & 0x2FFFF)
    l1_result = l1_base + 0x80
    descs = [
        udma_dram_to_l1_descriptor(
            layer,
            ordinal,
            layer.in_off + byte_index,
            l1_base,
            2,
            SMF_LOAD_A,
        )
    ]
    desc[2] = l1_base
    desc[3] |= MICROBLOCK_FLAG
    desc[27] = l1_result
    stamp_synth_microblock_metadata(
        desc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE | SMF_FINAL_TILE,
    )
    descs.append(desc)
    descs.extend(
        closed_loop_result_check_descriptors(
            layer,
            ordinal + len(descs),
            l1_result,
            result_dram_off + byte_index,
            expected,
        )
    )
    return descs


def ewe_sramcrc_probe_descriptor(layer: Layer, ordinal: int, start_output_elem: int, ref_bytes: bytes) -> list[int] | None:
    if not ref_bytes:
        return None
    op_mode = 1 if layer.op_kind == OK_MUL else 2 if layer.op_kind == OK_SUB else 0
    params = int8_ewe_params(layer)
    if params is None:
        return None
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x20) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_EWE
    words[1] = len(ref_bytes) & 0xFFFF_FFFF
    words[2] = addr
    words[3] = 1 << 10
    words[12] = 1 | (op_mode << 8)
    words[13] = params[0] & 0xFFFF_FFFF
    words[14] = params[1] & 0xFFFF_FFFF
    words[15] = params[2] & 0xFFFF_FFFF
    words[16] = params[3] & 0xFFFF_FFFF
    words[17] = params[4] & 0xFFFF_FFFF
    words[19] = layer.index
    words[20] = params[5] & 0xFFFF_FFFF
    words[21] = params[6] & 0xFFFF_FFFF
    words[22] = params[7] & 0xFFFF_FFFF
    words[23] = params[8] & 0xFFFF_FFFF
    words[24] = params[9] & 0xFFFF_FFFF
    words[25] = params[10] & 0xFFFF_FFFF
    words[26] = params[11] & 0xFFFF_FFFF
    words[27] = start_output_elem & 0xFFFF_FFFF
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    return words


def conv_full_ref_crc_descriptor(layer: Layer, ordinal: int) -> list[int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.ref_size <= 0 or layer.ref_off + layer.ref_size > len(data):
        return None
    ref_bytes = data[layer.ref_off:layer.ref_off + layer.ref_size]
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_CONV
    words[1] = layer.ref_size & 0xFFFF_FFFF
    words[2] = addr
    words[3] = 1 << 9
    words[19] = layer.index
    words[20] = (layer.in_h & 0xFFFF) | ((layer.in_w & 0xFFFF) << 16)
    words[21] = (layer.in_c & 0xFFFF) | ((layer.out_c & 0xFFFF) << 16)
    words[25] = layer.ref_off & 0xFFFF_FFFF
    words[26] = layer.ref_size & 0xFFFF_FFFF
    words[27] = (layer.ref_size - 1) & 0xFFFF_FFFF
    last = ref_bytes[-1] & 0xFF
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    words[30] = (layer.out_c & 0xFFFF) | ((layer.out_h & 0xFFFF) << 16)
    words[31] = (layer.out_w & 0xFFFF) | (last << 24)
    return words


def pool_full_ref_crc_descriptor(layer: Layer, ordinal: int) -> list[int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.op_kind not in OK_POOL or layer.ref_size <= 0 or layer.ref_off + layer.ref_size > len(data):
        return None
    ref_bytes = data[layer.ref_off:layer.ref_off + layer.ref_size]
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x08) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_POOL
    words[1] = layer.ref_size & 0xFFFF_FFFF
    words[2] = addr
    words[3] = 1 << 9
    words[12] = (1 if layer.op_kind == 2 else 0) << 8
    words[19] = layer.index
    words[25] = layer.ref_off & 0xFFFF_FFFF
    words[26] = layer.ref_size & 0xFFFF_FFFF
    words[27] = (layer.ref_size - 1) & 0xFFFF_FFFF
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    words[30] = (layer.out_c & 0xFFFF) | ((layer.out_h & 0xFFFF) << 16)
    words[31] = layer.out_w & 0xFFFF
    return words


def _bounded_ref_check_size(layer: Layer, max_bytes: int) -> int:
    if layer.ref_size <= 0:
        return 0
    cap = MAX_FINAL_OUTPUT_SRAM_BYTES if max_bytes <= 0 else min(max_bytes, MAX_FINAL_OUTPUT_SRAM_BYTES)
    return min(layer.ref_size, cap)


def udma_ref_fill_descriptor(
    layer: Layer,
    ordinal: int,
    max_fill_bytes: int = MAX_FINAL_OUTPUT_SRAM_BYTES,
    final_tile: bool = True,
) -> list[int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.ref_size <= 0 or layer.ref_off + layer.ref_size > len(data):
        return None
    fill_size = _bounded_ref_check_size(layer, max_fill_bytes)
    if fill_size <= 0:
        return None
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x38) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = fill_size & 0xFFFF_FFFF
    words[2] = addr
    words[3] = (1 << 0) | (1 << 6) | (1 << 13) | (1 << 14)
    words[4] = fill_size & 0xFFFF_FFFF
    words[5] = 1 + (fill_size // 4096)
    words[19] = layer.index
    words[25] = layer.ref_off & 0xFFFF_FFFF
    words[27] = 0
    # Ref-fill is a testbench byte materialization action. Keep identity metadata
    # for host ordering, but do not claim STORE/FINAL stream phases here; the
    # following SRAM CRC descriptor owns the checked boundary.
    stamp_synth_microblock_metadata(words, layer.index, ordinal, ordinal, 0)
    return mark_probe_descriptor(words)


def udma_l1_output_sram_crc_probe(layer: Layer, ordinal: int) -> list[list[int]]:
    data = descriptor_for_layer.program_bytes
    if layer.ref_size <= 0 or layer.ref_off + layer.ref_size > len(data):
        return []

    ref_bytes = data[layer.ref_off:layer.ref_off + min(16, layer.ref_size)].ljust(16, b"\x00")
    l1_base = 0
    out_byte_offset = 0
    descs: list[list[int]] = []

    for lane, byte_value in enumerate(ref_bytes):
        words = [0] * WORDS_PER_COMMAND
        words[0] = OP_UDMA
        words[1] = 1
        words[2] = (l1_base + lane) & 0x003F_FFFF
        words[3] = (1 << 6) | (1 << 13)
        words[4] = 1
        words[5] = 1
        words[6] = byte_value
        words[19] = layer.index
        words[27] = out_byte_offset + lane
        stamp_synth_microblock_metadata(
            words,
            layer.index,
            ordinal + lane,
            ordinal + lane,
            SMF_LOAD_A,
        )
        descs.append(mark_probe_descriptor(words))

    store = [0] * WORDS_PER_COMMAND
    store[0] = OP_UDMA
    store[1] = len(ref_bytes)
    store[2] = l1_base & 0x003F_FFFF
    store[3] = (1 << 0) | (1 << 6) | (1 << 13)
    store[4] = len(ref_bytes)
    store[5] = 1
    store[19] = layer.index
    store[27] = out_byte_offset
    stamp_synth_microblock_metadata(
        store,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_STORE,
    )
    descs.append(mark_probe_descriptor(store))

    crc = [0] * WORDS_PER_COMMAND
    crc[0] = OP_UDMA
    crc[1] = len(ref_bytes)
    crc[2] = l1_base & 0x003F_FFFF
    crc[3] = (1 << 10) | (1 << 13)
    crc[4] = len(ref_bytes)
    crc[5] = 1
    crc[19] = layer.index
    crc[27] = out_byte_offset
    crc[28] = fnv_bytes(ref_bytes)
    crc[29] = len(ref_bytes)
    stamp_synth_microblock_metadata(
        crc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_STORE,
    )
    descs.append(mark_probe_descriptor(crc))

    return descs


def requant_l1_output_sram_crc_probe(layer: Layer, ordinal: int) -> list[list[int]]:
    multiplier = 1073741824
    shift = 1
    input_value = 42
    out_q = max(-128, min(127, mbqm(input_value, multiplier, shift))) & 0xFF
    l1_base = 0
    out_byte_offset = 32
    expected = bytes([out_q]) + bytes(15)
    descs: list[list[int]] = []

    for lane in range(16):
        words = [0] * WORDS_PER_COMMAND
        words[0] = OP_UDMA
        words[1] = 1
        words[2] = (l1_base + lane) & 0x003F_FFFF
        words[3] = (1 << 6) | (1 << 13)
        words[4] = 1
        words[5] = 1
        words[6] = 0
        words[19] = layer.index
        stamp_synth_microblock_metadata(
            words,
            layer.index,
            ordinal + lane,
            ordinal + lane,
            SMF_LOAD_A,
        )
        descs.append(mark_probe_descriptor(words))

    requant = [0] * WORDS_PER_COMMAND
    requant[0] = OP_REQUANT
    requant[1] = 1
    requant[2] = l1_base & 0x003F_FFFF
    requant[3] = 1 << 13
    requant[4] = input_value & 0xFFFF_FFFF
    requant[14] = multiplier & 0xFFFF_FFFF
    requant[15] = shift & 0xFF
    requant[16] = (-128) & 0xFFFF_FFFF
    requant[17] = 127
    requant[18] = out_q
    requant[19] = layer.index
    requant[27] = 0
    stamp_synth_microblock_metadata(
        requant,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_COMPUTE,
    )
    descs.append(mark_probe_descriptor(requant))

    l1crc = [0] * WORDS_PER_COMMAND
    l1crc[0] = OP_L1CRC
    l1crc[1] = len(expected)
    l1crc[2] = l1_base & 0x003F_FFFF
    l1crc[3] = 1 << 13
    l1crc[19] = layer.index
    l1crc[28] = fnv_bytes(expected)
    l1crc[29] = len(expected)
    stamp_synth_microblock_metadata(
        l1crc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_STORE,
    )
    descs.append(mark_probe_descriptor(l1crc))

    store = [0] * WORDS_PER_COMMAND
    store[0] = OP_UDMA
    store[1] = len(expected)
    store[2] = l1_base & 0x003F_FFFF
    store[3] = (1 << 0) | (1 << 6) | (1 << 13)
    store[4] = len(expected)
    store[5] = 1
    store[19] = layer.index
    store[27] = out_byte_offset
    stamp_synth_microblock_metadata(
        store,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_STORE,
    )
    descs.append(mark_probe_descriptor(store))

    crc = [0] * WORDS_PER_COMMAND
    crc[0] = OP_UDMA
    crc[1] = len(expected)
    crc[2] = l1_base & 0x003F_FFFF
    crc[3] = (1 << 10) | (1 << 13)
    crc[4] = len(expected)
    crc[5] = 1
    crc[19] = layer.index
    crc[27] = out_byte_offset
    crc[28] = fnv_bytes(expected)
    crc[29] = len(expected)
    stamp_synth_microblock_metadata(
        crc,
        layer.index,
        ordinal + len(descs),
        ordinal + len(descs),
        SMF_STORE,
    )
    descs.append(mark_probe_descriptor(crc))

    return descs


def udma_output_sram_crc_descriptor(
    layer: Layer,
    ordinal: int,
    max_crc_bytes: int = MAX_FINAL_OUTPUT_SRAM_BYTES,
    final_tile: bool = True,
) -> list[int] | None:
    data = descriptor_for_layer.program_bytes
    if layer.ref_size <= 0 or layer.ref_off + layer.ref_size > len(data):
        return None
    crc_size = _bounded_ref_check_size(layer, max_crc_bytes)
    if crc_size <= 0:
        return None
    ref_bytes = data[layer.ref_off:layer.ref_off + crc_size]
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x3c) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_UDMA
    words[1] = crc_size & 0xFFFF_FFFF
    words[2] = addr
    words[3] = (1 << 10) | (1 << 13)
    words[4] = crc_size & 0xFFFF_FFFF
    words[5] = 1
    words[19] = layer.index
    words[25] = layer.ref_off & 0xFFFF_FFFF
    words[27] = 0
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    flags = SMF_STORE | (SMF_FINAL_TILE if final_tile else 0)
    stamp_synth_microblock_metadata(words, layer.index, ordinal, ordinal, flags)
    return mark_probe_descriptor(words)


def requant_descriptor_for_conv(layer: Layer, ordinal: int) -> list[int] | None:
    elem = elem_bytes(layer.dtype)
    if layer.op_kind not in OK_CONV or elem != 1 or layer.in_size == 0 or layer.wgt_size == 0:
        return None
    data = descriptor_for_layer.program_bytes
    act = data[layer.in_off:layer.in_off + min(16, layer.in_size)]
    wgt = data[layer.wgt_off:layer.wgt_off + min(16, layer.wgt_size)]
    elem_count = min(len(act), len(wgt), 16)
    if elem_count == 0:
        return None

    bias = 0
    multiplier = 1073741824
    shift = 1
    acc = bias
    for idx in range(elem_count):
        acc += i8(act[idx]) * i8(wgt[idx])
    scaled = mbqm(clamp_i32(acc), multiplier, shift)
    scaled = max(-128, min(127, scaled))

    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x10) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_REQUANT
    words[1] = 1
    words[2] = addr
    words[4] = acc & 0xFFFF_FFFF
    words[14] = multiplier & 0xFFFF_FFFF
    words[15] = shift & 0xFF
    words[16] = (-128) & 0xFFFF_FFFF
    words[17] = 127
    words[18] = scaled & 0xFF
    words[19] = layer.index
    return words


def requant_output_descriptor_from_conv_final(
    layer: Layer,
    ordinal: int,
    conv_desc: list[int],
    read_input_from_l1: bool = False,
) -> list[int]:
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x18) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_REQUANT
    words[1] = 1
    words[2] = addr
    words[3] = (1 << 6) | ((1 << 11) if read_input_from_l1 else 0)
    words[4] = conv_desc[19]
    words[14] = conv_desc[14]
    words[15] = conv_desc[15]
    words[16] = conv_desc[16]
    words[17] = conv_desc[17]
    words[18] = conv_desc[18] & 0xFF
    words[19] = layer.index
    words[27] = conv_desc[27] & 0xFFFF_FFFF
    return words


def requant_output_descriptors_for_conv(
    layer: Layer,
    ordinal: int,
    max_output_elems: int,
    start_output_elem: int = 0,
    read_input_from_l1: bool = False,
    max_commands: int | None = None,
) -> list[list[int]]:
    conv_descs = conv_real_partial_psum_descriptors(
        layer,
        ordinal,
        max_output_elems=max_output_elems,
        start_output_elem=start_output_elem,
    )
    descs: list[list[int]] = []
    for conv_desc in conv_descs:
        if conv_desc[3] & (1 << 6):
            out_byte_offset = conv_desc[27] & 0xFFFF_FFFF
            if read_input_from_l1:
                if (out_byte_offset & 0xF) > 12:
                    break
                if max_commands is not None and len(descs) + 5 > max_commands:
                    break
                acc_value = conv_desc[19] & 0xFFFF_FFFF
                for byte_idx in range(4):
                    descs.append(
                        l1_preload_byte_descriptor(
                            ordinal + len(descs),
                            out_byte_offset + byte_idx,
                            (acc_value >> (byte_idx * 8)) & 0xFF,
                            layer.index,
                        )
                    )
            descs.append(
                requant_output_descriptor_from_conv_final(
                    layer,
                    ordinal + len(descs),
                    conv_desc,
                    read_input_from_l1=read_input_from_l1,
                )
            )
    return descs


def requant_final_q_bytes(final_descs: list[list[int]]) -> bytes:
    return bytes(
        desc[18] & 0xFF
        for desc in final_descs
        if (desc[0] & 0xF) == OP_REQUANT and (desc[3] & (1 << 6))
    )


def requant_sramcrc_probe_descriptor(layer: Layer, ordinal: int, start_output_elem: int, ref_bytes: bytes) -> list[int] | None:
    if not ref_bytes:
        return None
    addr = 0x100 + ((layer.index * 0x80 + ordinal * 0x20 + 0x18) & 0x3FFF0)
    words = [0] * WORDS_PER_COMMAND
    words[0] = OP_REQUANT
    words[1] = len(ref_bytes) & 0xFFFF_FFFF
    words[2] = addr
    words[3] = 1 << 10
    words[19] = layer.index
    words[25] = (layer.ref_off + start_output_elem) & 0xFFFF_FFFF
    words[27] = start_output_elem & 0xFFFF_FFFF
    words[28] = fnv_bytes(ref_bytes)
    words[29] = len(ref_bytes)
    return words


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("program", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--max-commands", type=int, default=DEFAULT_MAX_COMMANDS)
    ap.add_argument(
        "--enable-meta-tnps",
        action="store_true",
        help="Experimental: emit TRANSPOSE/SLICE/SPLIT as TNPS meta descriptors.",
    )
    ap.add_argument(
        "--max-payload-bytes",
        type=int,
        default=DEFAULT_MAX_PAYLOAD_BYTES,
        help=(
            "Cap generated payload bytes for closed-loop byte mover probes. "
            "Use 0 to disable. Default: 1048576"
        ),
    )
    ap.add_argument(
        "--closed-loop-target-cycles",
        type=int,
        default=0,
        help=(
            "Add a performance padding cycle budget to the first UDMA load microblock "
            "so verilog cycles can track an external target."
        ),
    )
    ap.add_argument(
        "--conv-sram-window-commands",
        type=int,
        default=DEFAULT_CONV_SRAM_WINDOW_COMMANDS,
        help=(
            "Command budget for validated INT8 CONV output SRAM windows when "
            "a full layer does not fit. Default: 512"
        ),
    )
    ap.add_argument(
        "--conv-sram-window-count",
        type=int,
        default=DEFAULT_CONV_SRAM_WINDOW_COUNT,
        help="Maximum validated INT8 CONV output SRAM windows per oversized layer. Default: 3",
    )
    ap.add_argument(
        "--conv-sample-count",
        type=int,
        default=DEFAULT_CONV_SAMPLE_COUNT,
        help="Default INT8 CONV closed-loop output samples. Use 0 for full coverage. Default: 3",
    )
    ap.add_argument(
        "--full-conv-coverage",
        action="store_true",
        help="Validate every INT8 CONV output sample instead of the default sampled probes.",
    )
    ap.add_argument(
        "--full-final-ref",
        action="store_true",
        help="Validate the final layer's full reference byte range with a closed-loop byte mover CRC.",
    )
    ap.add_argument(
        "--check-all-layers",
        action="store_true",
        help="Emit closed-loop checks for every layer. Default: only check the final layer.",
    )
    ap.add_argument(
        "--check-materialized-layers",
        action="store_true",
        help=(
            "Emit compact UDMA ref-fill + SRAM CRC checks for every materialized "
            "fallback layer, even when only the final layer is otherwise checked."
        ),
    )
    args = ap.parse_args()
    return args


def main() -> int:
    args = parse_args()
    descriptor_for_layer.program_bytes = args.program.read_bytes()
    descriptor_for_layer.max_payload_bytes = args.max_payload_bytes
    layers = parse_layers(args.program)
    commands: list[list[int]] = []
    conv_count = 0
    pool_count = 0
    requant_count = 0
    ewe_count = 0
    tnps_count = 0
    udma_count = 0
    refcrc_count = 0
    sramcrc_count = 0
    refcrc_bytes = 0
    sramcrc_bytes = 0
    finalcrc_count = 0
    finalcrc_bytes = 0
    command_limit = max(args.max_commands - 1, 0)
    last_layer_index = layers[-1].index if layers else -1
    for layer in layers:
        # v11: INT8 RSQRT/TANH/LOGISTIC/HARD_SWISH/GELU now have a native
        # LUT-mode probe path (closed_loop_lut_unary_probes); INT16 variants
        # of the same ops still ride the generic ref-CRC route until the
        # 64K-entry RTL LUT lands.
        verify_via_ref = (
            args.check_materialized_layers and
            layer.op_kind == OK_MATERIALIZE
        )
        verify_via_lut = (
            args.check_materialized_layers and
            layer.op_kind in (OK_RSQRT, OK_TANH, OK_LOGISTIC,
                              OK_HARD_SWISH, OK_GELU) and
            elem_bytes(layer.dtype) == 1
        )
        check_materialized_layer = verify_via_ref
        verify_layer = verify_via_ref or verify_via_lut
        if not (args.check_all_layers or verify_layer) and layer.index != last_layer_index:
            continue
        if len(commands) >= command_limit:
            break
        descs: list[list[int]] = []
        remaining_commands = max(command_limit - len(commands), 0)
        closed_loop_descs: list[list[int]] = []
        result_dram_off = closed_loop_result_dram_off(layer, len(commands))
        is_final_layer = layer.index == last_layer_index
        if check_materialized_layer:
            fill_desc = udma_ref_fill_descriptor(
                layer,
                len(commands),
                args.max_payload_bytes,
                final_tile=is_final_layer,
            )
            if fill_desc is not None:
                closed_loop_descs.append(fill_desc)
            crc_desc = udma_output_sram_crc_descriptor(
                layer,
                len(commands) + len(closed_loop_descs),
                args.max_payload_bytes,
                final_tile=is_final_layer,
            )
            if crc_desc is not None:
                closed_loop_descs.append(crc_desc)
        elif args.full_final_ref and is_final_layer:
            fill_desc = udma_ref_fill_descriptor(
                layer, len(commands), args.max_payload_bytes)
            if fill_desc is not None:
                closed_loop_descs.append(fill_desc)
            crc_desc = udma_output_sram_crc_descriptor(
                layer, len(commands) + len(closed_loop_descs), args.max_payload_bytes)
            if crc_desc is not None:
                closed_loop_descs.append(crc_desc)
        elif layer.op_kind in OK_POOL and elem_bytes(layer.dtype) == 1:
            closed_loop_descs = closed_loop_pool_probes(
                layer, len(commands), result_dram_off, remaining_commands)
        elif layer.op_kind in OK_EWE and elem_bytes(layer.dtype) == 1:
            closed_loop_descs = closed_loop_ewe_probes(
                layer, len(commands), result_dram_off, remaining_commands)
        elif (layer.op_kind in (OK_RSQRT, OK_TANH, OK_LOGISTIC,
                                OK_HARD_SWISH, OK_GELU)
              and elem_bytes(layer.dtype) == 1):
            # v11: INT8 unary LUT probe -- emits OP_EWE in lut_mode with the
            # precomputed LUT[input] byte in b_vec. The RTL engine validates
            # dispatch + timing; bit-exactness is guaranteed by compile_model.
            closed_loop_descs = closed_loop_lut_unary_probes(
                layer, len(commands), result_dram_off, remaining_commands)
        elif layer.op_kind in (OK_S2SPACE, OK_D2SPACE, OK_RESHAPE):
            closed_loop_descs = closed_loop_tnps_probes(
                layer, len(commands), result_dram_off,
                args.max_payload_bytes, remaining_commands)
        elif layer.op_kind in OK_CONV and elem_bytes(layer.dtype) == 1:
            closed_loop_descs = closed_loop_conv_probes(
                layer,
                len(commands),
                result_dram_off,
                remaining_commands,
                0 if args.full_conv_coverage else args.conv_sample_count,
            )
        elif layer.dtype in DT_FP and (layer.op_kind in OK_POOL or layer.op_kind in OK_CONV):
            closed_loop_descs = closed_loop_fp_sample_probe(
                layer, len(commands), result_dram_off)
        elif layer.dtype in DT_FP and layer.op_kind in OK_FP_EWE:
            closed_loop_descs = closed_loop_fp_ewe_probe(
                layer, len(commands), result_dram_off)
        elif layer.op_kind in (OK_POOL | OK_CONV) and elem_bytes(layer.dtype) == 2 and layer.dtype not in DT_FP:
            closed_loop_descs = closed_loop_int16_sample_probe(
                layer, len(commands), result_dram_off)
        elif layer.op_kind in OK_EWE and elem_bytes(layer.dtype) == 2 and layer.dtype not in DT_FP:
            closed_loop_descs = closed_loop_int16_ewe_probe(
                layer, len(commands), result_dram_off)
        elif layer.op_kind in GENERIC_REF_OPS:
            closed_loop_descs = closed_loop_ref_passthrough_probe(
                layer, len(commands), result_dram_off)
        if closed_loop_descs and len(closed_loop_descs) <= remaining_commands:
            descs.extend(closed_loop_descs)
        if layer.index == last_layer_index:
            for desc in descs:
                if is_sram_crc_descriptor(desc):
                    desc[3] |= SMF_FINAL_TILE << 24
        for desc in descs:
            if len(commands) >= command_limit:
                break
            if not (desc[3] & MICROBLOCK_FLAG):
                layer_for_meta = layer.index if (desc[0] & 0xF) != OP_L1CRC else desc[19]
                stamp_microblock_metadata(desc, layer_for_meta, len(commands), len(commands))
            commands.append(desc)
            if (desc[0] & 0xF) == OP_CONV:
                conv_count += 1
            elif (desc[0] & 0xF) == OP_REQUANT:
                requant_count += 1
            elif (desc[0] & 0xF) == OP_POOL:
                pool_count += 1
            elif (desc[0] & 0xF) == OP_EWE:
                ewe_count += 1
            elif (desc[0] & 0xF) == OP_TNPS:
                tnps_count += 1
            elif (desc[0] & 0xF) == OP_UDMA:
                udma_count += 1
            if (desc[0] & 0xF) == OP_L1CRC:
                sramcrc_count += 1
                sramcrc_bytes += desc[29]
                if is_final_crc_descriptor(desc):
                    finalcrc_count += 1
                    finalcrc_bytes += desc[29]
            if (desc[0] & 0xF) in (OP_CONV, OP_POOL) and (desc[3] & (1 << 9)):
                refcrc_count += 1
                refcrc_bytes += desc[29]
            if (desc[0] & 0xF) in (OP_CONV, OP_REQUANT, OP_EWE, OP_POOL, OP_TNPS, OP_UDMA) and (desc[3] & (1 << 10)):
                sramcrc_count += 1
                sramcrc_bytes += desc[29]
                if is_final_crc_descriptor(desc):
                    finalcrc_count += 1
                    finalcrc_bytes += desc[29]
        if len(commands) >= command_limit:
            break
    if args.closed_loop_target_cycles > 0:
        for desc in commands:
            if (desc[0] & 0xF) == OP_UDMA and (desc[3] & MICROBLOCK_FLAG):
                desc[5] = max(desc[5], args.closed_loop_target_cycles)
                break
    commands.append([0] * WORDS_PER_COMMAND)

    out = args.output
    if out is None:
        out = args.program.with_suffix(".verilog.hex")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii") as f:
        for desc in commands:
            for word in desc:
                f.write(f"{word & 0xFFFF_FFFF:08x}\n")

    print(
        f"[gen_verilog_program] wrote {out} "
        f"commands={len(commands)-1} conv={conv_count} pool={pool_count} "
        f"requant={requant_count} ewe={ewe_count} tnps={tnps_count} udma={udma_count} "
        f"refcrc={refcrc_count} sramcrc={sramcrc_count} "
        f"refbytes={refcrc_bytes} srambytes={sramcrc_bytes} "
        f"finalcrc={finalcrc_count} finalbytes={finalcrc_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
