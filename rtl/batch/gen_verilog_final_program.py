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
DEFAULT_MAX_PAYLOAD_BYTES = 1 << 20


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
    in_size: int
    wgt_size: int
    ref_size: int
    in_off: int
    wgt_off: int
    ref_off: int
    op_kind: int
    dtype: int
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


def sample_bytes(data: bytes, offset: int, size: int, count: int = 16) -> bytes:
    return data[offset:offset + min(count, size)].ljust(count, b"\x00")


def conv2d_int8_window_sample(
    layer: Layer,
    data: bytes,
    max_count: int = 16,
    start_lane: int = 0,
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
        return act.ljust(max_count, b"\x00"), wgt.ljust(max_count, b"\x00"), elem_count, 0, 0, 0, 0, 0, 0, False, 0, 0, 0, 1, False, 0

    act_values = bytearray()
    wgt_values = bytearray()
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
                input_elem = ((kh * layer.in_w + kw) * layer.in_c) + ic
                weight_elem = (((kh * layer.k_w + kw) * layer.in_c + ic) * layer.out_c)
                input_byte = input_elem
                weight_byte = weight_elem
                if input_byte >= layer.in_size or weight_byte >= layer.wgt_size:
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
    tile_count = min(max(layer.out_w * layer.out_c, 1), 4)
    last_out_elem = tile_count - 1
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
        0,
        valid,
        first_input_byte,
        first_weight_byte,
        elem_count,
        tile_count,
        last_first_valid,
        last_valid_count,
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
        k_h, k_w = struct.unpack_from("<BB", data, off + 12)
        in_size, wgt_size, ref_size = struct.unpack_from("<III", data, off + 32)
        in_off, wgt_off, ref_off = struct.unpack_from("<III", data, off + 44)
        op_kind, dtype = struct.unpack_from("<HH", data, off + 58)
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
                in_size=in_size,
                wgt_size=wgt_size,
                ref_size=ref_size,
                in_off=in_off,
                wgt_off=wgt_off,
                ref_off=ref_off,
                op_kind=op_kind,
                dtype=dtype,
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
    ) = conv2d_int8_window_sample(layer, data, max_count=max_count, start_lane=start_lane)
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
    words[30] = expected_valid_count & 0xFFFF_FFFF
    words[31] = (
        (tile_count & 0xFF) |
        ((expected_tile_last_valid_count & 0xFF) << 8) |
        ((1 if expected_tile_last_valid else 0) << 16)
    )
    return words, acc


def conv_partial_psum_descriptors(layer: Layer, ordinal: int) -> list[list[int]]:
    descs: list[list[int]] = []
    cumulative_acc = 0
    start_lane = 0
    total_lanes = max(layer.k_h, 1) * max(layer.k_w, 1) * max(layer.in_c, 1)
    while start_lane < total_lanes:
        psum_flag = 1 << 4 if not descs else 1 << 5
        sample = int8_conv_sample_descriptor(
            layer,
            ordinal + len(descs),
            start_lane=start_lane,
            max_count=8,
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
        descs.append(desc)
        start_lane += elem_count
    if descs:
        descs[-1][3] |= 1 << 6
        final_q = clamp_i8(mbqm(clamp_i32(cumulative_acc), 1073741824, 1))
        descs[-1][18] = final_q & 0xFF
    return descs


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
    ap.add_argument("--max-commands", type=int, default=64)
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
            descs = conv_partial_psum_descriptors(layer, len(commands))
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
        f"requant={requant_count} ewe={ewe_count} tnps={tnps_count} udma={udma_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
