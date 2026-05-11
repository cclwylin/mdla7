#!/usr/bin/env python3
"""Generate a verilog_final host descriptor hex stream from an MDL7 .bin image."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC_MDL7 = 0x374C444D
OP_DONE = 0
OP_TNPS = 5
OP_UDMA = 6

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
WORDS_PER_COMMAND = 20


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

    if layer.op_kind in (OK_S2SPACE, OK_D2SPACE):
        sample_out, sample_in, expected_src, expected_dst, sample_valid = tnps_sample(layer)
        words[0] = OP_TNPS
        words[1] = max(layer.ref_size, layer.in_size)
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
        words[1] = max(layer.ref_size, layer.in_size)
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
        words[1] = max(layer.ref_size, layer.in_size, 1)
        words[2] = addr
        words[3] = 0x1
        words[4] = max(layer.in_size, layer.ref_size, 1)
        words[5] = 1 + (max(layer.ref_size, layer.in_size) // 4096)
        words[19] = layer.index
        return words

    return None


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
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    layers = parse_layers(args.program)
    commands: list[list[int]] = []
    tnps_count = 0
    udma_count = 0
    for layer in layers:
        desc = descriptor_for_layer(layer, len(commands), args.enable_meta_tnps)
        if desc is not None:
            commands.append(desc)
            if (desc[0] & 0xF) == OP_TNPS:
                tnps_count += 1
            elif (desc[0] & 0xF) == OP_UDMA:
                udma_count += 1
            if len(commands) >= args.max_commands:
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
        f"commands={len(commands)-1} tnps={tnps_count} udma={udma_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
