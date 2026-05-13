#!/usr/bin/env python3
"""Generate tiny MDL7 .bin programs covering ETHZ_v6 op kinds.

The files are synthetic one-layer programs. They are intentionally small and
deduplicated by op kind so fast/CX/Verilog bring-up can exercise the operation
surface without pulling in full ETHZ_v6 models.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = 0x374C444D
VERSION = 3
HEADER_FMT = "<IIII"
LAYER_FMT = "<HHHHHHBBBBBBBBIIIIIIIIIHHHh"
GRAPH_META_FMT = "<iiiiiiii"
DRAM_BASE = 0x10000000
REGION_ALIGN = 64 * 1024 * 1024

OP_CONV = 0
OP_DWCONV = 1
OP_AVG_POOL = 2
OP_MAX_POOL = 3
OP_SOFTMAX = 4
OP_RESHAPE = 5
OP_FC = 6
OP_ADD = 7
OP_CONCAT = 8
OP_GATHER = 9
OP_MUL = 10
OP_SUB = 11
OP_HARD_SWISH = 12
OP_GELU = 13
OP_D2SPACE = 14
OP_MATERIALIZE = 15
OP_TRANSPOSE = 16
OP_S2SPACE = 17
OP_SLICE = 20
OP_STRIDED_SLICE = 21
OP_PAD = 22
OP_PACK = 23
OP_UNPACK = 24
OP_SPLIT = 26
OP_LOGISTIC = 27

DT_INT8x8 = 1
DT_INT16x16 = 4
DT_FP16 = 9


@dataclass(frozen=True)
class Spec:
    name: str
    op: int
    dtype: int = DT_INT8x8
    in_shape: tuple[int, int, int] = (2, 2, 4)
    out_shape: tuple[int, int, int] = (2, 2, 4)
    kernel: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    group: int = 1
    wgt_bytes: int = 0


def round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def elem_size(dtype: int) -> int:
    return 2 if dtype in (DT_INT16x16, DT_FP16) else 1


def zeros(count: int) -> bytes:
    return bytes(max(count, 0))


def conv_params(out_c: int) -> bytes:
    return (
        struct.pack("<iii", 0, -128, 127)
        + struct.pack(f"<{out_c}i", *([0] * out_c))
        + bytes(out_c)
        + struct.pack(f"<{out_c}i", *([0] * out_c))
    )


def ewe_params() -> bytes:
    return struct.pack(
        "<iiiiiiiiiiii",
        0, 0, 0,
        0, 0, 0, 0,
        0, 0,
        0, -128, 127,
    )


def fp16_zeros(elems: int) -> bytes:
    return struct.pack(f"<{elems}e", *([0.0] * elems))


def tnps_meta(rank: int, elem: int, in_shape: tuple[int, ...], out_shape: tuple[int, ...],
              a_vals: tuple[int, ...] = (), b_vals: tuple[int, ...] = ()) -> bytes:
    def pad_u(vals: tuple[int, ...]) -> list[int]:
        out = list(vals[:6])
        return out + [0] * (6 - len(out))

    def pad_i(vals: tuple[int, ...]) -> list[int]:
        out = list(vals[:6])
        return out + [0] * (6 - len(out))

    words = [rank, elem]
    words += pad_u(in_shape)
    words += pad_u(out_shape)
    return struct.pack("<14I6i6i", *words, *pad_i(a_vals), *pad_i(b_vals))


def payload_for(spec: Spec) -> tuple[bytes, bytes, bytes]:
    in_h, in_w, in_c = spec.in_shape
    out_h, out_w, out_c = spec.out_shape
    in_elems = in_h * in_w * in_c
    out_elems = out_h * out_w * out_c
    es = elem_size(spec.dtype)
    input_blob = fp16_zeros(in_elems) if spec.dtype == DT_FP16 else zeros(in_elems * es)
    ref_blob = fp16_zeros(out_elems) if spec.dtype == DT_FP16 else zeros(out_elems * es)
    weight_blob = zeros(spec.wgt_bytes)

    if spec.op in (OP_CONV, OP_DWCONV, OP_FC):
        k_h, k_w = spec.kernel
        if spec.op == OP_DWCONV:
            weight_elems = k_h * k_w * in_c
        else:
            weight_elems = k_h * k_w * in_c * out_c
        weight_blob = zeros(weight_elems * es) + conv_params(out_c)
    elif spec.op in (OP_ADD, OP_MUL, OP_SUB):
        weight_blob = zeros(in_elems * es) + ewe_params()
    elif spec.op in (OP_TRANSPOSE, OP_SLICE, OP_STRIDED_SLICE, OP_SPLIT):
        if spec.op == OP_TRANSPOSE:
            weight_blob = tnps_meta(4, es, (1, *spec.in_shape), (1, *spec.out_shape), (0, 2, 1, 3))
        elif spec.op in (OP_SLICE, OP_STRIDED_SLICE, OP_SPLIT):
            weight_blob = tnps_meta(4, es, (1, *spec.in_shape), (1, *spec.out_shape), (0, 0, 0, 0), (1, 1, 1, 1))
    elif spec.op == OP_GATHER:
        weight_blob = bytes([0, 0, 0, 0])
    return input_blob, weight_blob, ref_blob


SPECS = [
    Spec("g1op_ethz_conv_int8", OP_CONV, in_shape=(4, 4, 2), out_shape=(4, 4, 2), kernel=(3, 3), wgt_bytes=0),
    Spec("g1op_ethz_dwconv_int8", OP_DWCONV, in_shape=(4, 4, 2), out_shape=(4, 4, 2), kernel=(3, 3), group=2),
    Spec("g1op_ethz_avgpool_int8", OP_AVG_POOL, in_shape=(4, 4, 2), out_shape=(2, 2, 2), kernel=(2, 2), stride=(2, 2)),
    Spec("g1op_ethz_maxpool_int8", OP_MAX_POOL, in_shape=(4, 4, 2), out_shape=(2, 2, 2), kernel=(2, 2), stride=(2, 2)),
    Spec("g1op_ethz_softmax_int8", OP_SOFTMAX, in_shape=(1, 1, 8), out_shape=(1, 1, 8)),
    Spec("g1op_ethz_reshape_int8", OP_RESHAPE, in_shape=(2, 2, 4), out_shape=(4, 4, 1)),
    Spec("g1op_ethz_fc_int8", OP_FC, in_shape=(1, 1, 8), out_shape=(1, 1, 4)),
    Spec("g1op_ethz_add_int8", OP_ADD),
    Spec("g1op_ethz_concat_int8", OP_CONCAT, in_shape=(2, 2, 4), out_shape=(2, 2, 4)),
    Spec("g1op_ethz_gather_int8", OP_GATHER, in_shape=(1, 4, 4), out_shape=(1, 2, 4)),
    Spec("g1op_ethz_mul_int8", OP_MUL),
    Spec("g1op_ethz_sub_int8", OP_SUB),
    Spec("g1op_ethz_hard_swish_fp16", OP_HARD_SWISH, dtype=DT_FP16),
    Spec("g1op_ethz_gelu_fp16", OP_GELU, dtype=DT_FP16),
    Spec("g1op_ethz_d2space_int8", OP_D2SPACE, in_shape=(2, 2, 8), out_shape=(4, 4, 2), kernel=(2, 2)),
    Spec("g1op_ethz_materialize_int8", OP_MATERIALIZE),
    Spec("g1op_ethz_transpose_int8", OP_TRANSPOSE, in_shape=(2, 3, 4), out_shape=(3, 2, 4)),
    Spec("g1op_ethz_s2space_int8", OP_S2SPACE, in_shape=(4, 4, 2), out_shape=(2, 2, 8), kernel=(2, 2)),
    Spec("g1op_ethz_slice_int8", OP_SLICE, in_shape=(4, 4, 2), out_shape=(2, 4, 2)),
    Spec("g1op_ethz_strided_slice_int8", OP_STRIDED_SLICE, in_shape=(4, 4, 2), out_shape=(2, 4, 2)),
    Spec("g1op_ethz_pad_int8", OP_PAD, in_shape=(2, 2, 2), out_shape=(4, 4, 2)),
    Spec("g1op_ethz_pack_int8", OP_PACK, in_shape=(2, 2, 2), out_shape=(2, 2, 4)),
    Spec("g1op_ethz_unpack_int8", OP_UNPACK, in_shape=(2, 2, 4), out_shape=(2, 2, 2)),
    Spec("g1op_ethz_split_int8", OP_SPLIT, in_shape=(2, 2, 4), out_shape=(2, 2, 2)),
    Spec("g1op_ethz_logistic_fp16", OP_LOGISTIC, dtype=DT_FP16),
]


def write_program(path: Path, spec: Spec) -> None:
    input_blob, weight_blob, ref_blob = payload_for(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_size = struct.calcsize(HEADER_FMT)
    layer_size = struct.calcsize(LAYER_FMT)
    graph_size = struct.calcsize(GRAPH_META_FMT)
    data_offset = header_size + layer_size + graph_size
    base_w = len(input_blob)
    base_r = base_w + len(weight_blob)
    dram_wgt_base = DRAM_BASE
    dram_in_base = dram_wgt_base + round_up(len(weight_blob), REGION_ALIGN)
    dram_out_base = dram_in_base + round_up(len(input_blob), REGION_ALIGN)
    in_h, in_w, in_c = spec.in_shape
    out_h, out_w, out_c = spec.out_shape
    k_h, k_w = spec.kernel
    s_h, s_w = spec.stride
    with path.open("wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, 1, data_offset))
        f.write(struct.pack(
            LAYER_FMT,
            in_h, in_w, in_c, out_h, out_w, out_c,
            k_h, k_w, s_h, s_w, 0, 0, 0, 0,
            dram_in_base, dram_wgt_base, dram_out_base,
            len(input_blob), len(weight_blob), len(ref_blob),
            data_offset,
            data_offset + base_w,
            data_offset + base_r,
            spec.group, spec.op, spec.dtype, 0,
        ))
        f.write(struct.pack(GRAPH_META_FMT, 0, -1, 1, -1, -1, -1, -1, 0))
        f.write(input_blob)
        f.write(weight_blob)
        f.write(ref_blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--out-dir", type=Path, default=root / "rtl" / "bin" / "UnitTest" / "g1op_ethz")
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args()
    filters = tuple(x.lower() for x in args.only)
    generated = 0
    for spec in SPECS:
        if filters and not any(f in spec.name.lower() for f in filters):
            continue
        out = args.out_dir / f"{spec.name}.bin"
        write_program(out, spec)
        print(f"bin: {out}")
        generated += 1
    if generated == 0:
        raise SystemExit("no g1op_ethz bins matched")
    print(f"generated {generated} g1op_ethz bin(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
