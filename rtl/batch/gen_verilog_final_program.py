#!/usr/bin/env python3
"""Generate a verilog_final host descriptor hex stream from an MDL7 .bin image."""

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

OK_CONV = {0, 1, 6}
OK_POOL = {2, 3}
OK_ADD = 7
OK_MUL = 10
OK_SUB = 11
OK_LOGISTIC = 27
OK_RESHAPE = 5
OK_CONCAT = 8
OK_GATHER = 9
OK_D2SPACE = 14
OK_MATERIALIZE = 15
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
UDMA_OPS = {OK_GATHER, OK_MATERIALIZE}

DT_INT16 = {2, 3, 4}
DT_FP = {8, 9, 10}
WORDS_PER_COMMAND = 32
DEFAULT_MAX_COMMANDS = 4096
DEFAULT_MAX_PAYLOAD_BYTES = 1 << 20
MAX_REFCRC_PREFIX_COMMANDS = 512
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
    magic, _version, layers, _data_offset = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC_MDL7:
        raise SystemExit(f"bad MDL7 magic: {path}")
    if len(data) < 16 + layers * 64:
        raise SystemExit(f"truncated layer table: {path}")

    out: list[Layer] = []
    for index in range(layers):
        off = 16 + index * 64
        in_h, in_w, in_c, out_h, out_w, out_c = struct.unpack_from("<HHHHHH", data, off)
        k_h, k_w, s_h, s_w, p_t, p_b, p_l, p_r = struct.unpack_from("<BBBBBBBB", data, off + 12)
        in_size, wgt_size, ref_size = struct.unpack_from("<III", data, off + 32)
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

    if layer.op_kind in OK_CONV and elem == 2 and layer.in_size >= 2 and layer.wgt_size >= 2:
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

    if layer.op_kind in OK_POOL and elem == 2 and layer.in_size >= 2:
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

    if layer.op_kind in OK_EWE and elem == 2 and layer.in_size >= 2 and layer.wgt_size >= 2:
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
        words[0] = OP_UDMA
        words[1] = sim_bytes(max(layer.ref_size, layer.in_size, 1))
        words[2] = addr
        words[3] = 0x1
        words[4] = sim_bytes(max(layer.in_size, layer.ref_size, 1))
        words[5] = 1 + (max(layer.ref_size, layer.in_size) // 4096)
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
) -> list[list[int]]:
    data = descriptor_for_layer.program_bytes
    params = int8_conv_params(layer, data)
    if params is None:
        return []
    descs: list[list[int]] = []
    output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
    emit_output_elems = output_elems if max_output_elems is None else min(output_elems, max_output_elems)
    group = layer.group or 1
    in_per_group = max(layer.in_c // max(group, 1), 1)
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * in_per_group
    for out_elem_index in range(emit_output_elems):
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


def conv_shadow_readback_descriptor(
    layer: Layer,
    ordinal: int,
    final_descs: list[list[int]],
    ref_bytes: bytes | None = None,
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
        desc[27] = 0
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
            "Cap generated payload bytes for sample-path simulation. "
            "Use 0 to disable. Default: 1048576"
        ),
    )
    ap.add_argument(
        "--emit-conv-partial-psum",
        action="store_true",
        help=(
            "Experimental: split generated INT8 CONV sample descriptors into "
            "psum first/accumulate pairs."
        ),
    )
    return ap.parse_args()


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
    command_limit = max(args.max_commands - 1, 0)
    for layer in layers:
        if len(commands) >= command_limit:
            break
        desc = descriptor_for_layer(layer, len(commands), args.enable_meta_tnps)
        if (
            args.emit_conv_partial_psum and
            desc is not None and
            (desc[0] & 0xF) == OP_CONV and
            (desc[12] & ((1 << 8) | (1 << 11))) == 0
        ):
            output_elems = max(layer.ref_size // max(elem_bytes(layer.dtype), 1), 1)
            remaining_commands = max(command_limit - len(commands), 0)
            real_full_command_count = conv_real_partial_psum_command_count(layer, output_elems)
            ref_bytes: bytes | None = None
            if int8_conv_params(layer, descriptor_for_layer.program_bytes) is not None and real_full_command_count <= remaining_commands:
                max_output_elems = None
                descs = conv_real_partial_psum_descriptors(
                    layer,
                    len(commands),
                    max_output_elems=max_output_elems,
                )
                ref_bytes = descriptor_for_layer.program_bytes[layer.ref_off:layer.ref_off + layer.ref_size]
            elif int8_conv_params(layer, descriptor_for_layer.program_bytes) is not None:
                lane_chunks = conv_real_lane_chunk_count(layer)
                prefix_budget = max(min(remaining_commands - 3, MAX_REFCRC_PREFIX_COMMANDS), 0)
                max_prefix_outputs = prefix_budget // lane_chunks
                descs = []
                if max_prefix_outputs > 0:
                    descs = conv_real_partial_psum_descriptors(
                        layer,
                        len(commands),
                        max_output_elems=max_prefix_outputs,
                    )
                    generated_prefix = conv_final_q_bytes([d for d in descs if d[3] & (1 << 6)])
                    if generated_prefix:
                        ref_prefix = descriptor_for_layer.program_bytes[
                            layer.ref_off:layer.ref_off + min(len(generated_prefix), layer.ref_size)
                        ]
                        if generated_prefix == ref_prefix:
                            ref_bytes = ref_prefix
                        else:
                            descs = []
                compact_desc = conv_full_ref_crc_descriptor(layer, len(commands) + len(descs))
                if compact_desc is not None:
                    descs.append(compact_desc)
            else:
                full_command_count = conv_partial_psum_command_count(layer, output_elems)
                max_output_elems = None if full_command_count <= remaining_commands else 16
                descs = conv_partial_psum_descriptors(
                    layer,
                    len(commands),
                    max_output_elems=max_output_elems,
                )
            if descs:
                probe_desc = conv_shadow_readback_descriptor(
                    layer,
                    len(commands) + len(descs),
                    [d for d in descs if (d[3] & (1 << 6))],
                    ref_bytes=ref_bytes,
                )
                if probe_desc is not None:
                    descs.append(probe_desc)
        else:
            descs = [desc] if desc is not None else []
        req_desc = requant_descriptor_for_conv(layer, len(commands) + len(descs))
        if req_desc is not None:
            descs.append(req_desc)
        for desc in descs:
            if len(commands) >= command_limit:
                break
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
            if (desc[0] & 0xF) == OP_CONV and (desc[3] & (1 << 9)):
                refcrc_count += 1
            if (desc[0] & 0xF) == OP_CONV and (desc[3] & (1 << 10)):
                sramcrc_count += 1
        if len(commands) >= command_limit:
            break
    commands.append([0] * WORDS_PER_COMMAND)

    out = args.output
    if out is None:
        out = args.program.with_suffix(".verilog_final.hex")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii") as f:
        for desc in commands:
            for word in desc:
                f.write(f"{word & 0xFFFF_FFFF:08x}\n")

    print(
        f"[gen_verilog_final_program] wrote {out} "
        f"commands={len(commands)-1} conv={conv_count} pool={pool_count} "
        f"requant={requant_count} ewe={ewe_count} tnps={tnps_count} udma={udma_count} "
        f"refcrc={refcrc_count} sramcrc={sramcrc_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
