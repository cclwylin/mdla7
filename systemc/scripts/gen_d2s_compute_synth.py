#!/usr/bin/env python3
"""Generate a synthetic MDL7 program for DEPTH_TO_SPACE -> CONV coverage.

The graph is:

  DEPTH_TO_SPACE block=2 -> CONV 3x3 SAME

All payloads are zero-valued, so references are byte-stable while still
exercising the real TNPS D2S tile -> CONV/Requant microblock handoff.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


MAGIC = 0x374C444D
VERSION = 3
HEADER_FMT = "<IIII"
LAYER_FMT = "<HHHHHHBBBBBBBBIIIIIIIIIHHHh"
GRAPH_META_FMT = "<iiiiiiii"

OP_CONV = 0
OP_D2SPACE = 14
DT_INT8x8 = 1
DRAM_BASE = 0x10000000
REGION_ALIGN = 64 * 1024 * 1024


def round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def conv_params(out_c: int) -> bytes:
    # Requant params layout:
    # [i32 zp_out, i32 act_min, i32 act_max,
    #  i32 mult[out_c], i8 shift[out_c], i32 bias_eff[out_c]]
    return (
        struct.pack("<iii", 0, -128, 127)
        + struct.pack(f"<{out_c}i", *([0] * out_c))
        + bytes(out_c)
        + struct.pack(f"<{out_c}i", *([0] * out_c))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="batch/output/d2s_compute_synth.bin",
        help="output MDL7 program path",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    block = 2
    d2s_h = 256
    d2s_w = 256
    d2s_in_c = 32
    conv_h = d2s_h * block
    conv_w = d2s_w * block
    conv_in_c = d2s_in_c // (block * block)
    conv_out_c = 8

    d2s_in_bytes = d2s_h * d2s_w * d2s_in_c
    d2s_out_bytes = conv_h * conv_w * conv_in_c
    conv_out_bytes = conv_h * conv_w * conv_out_c

    d2s_wgt = b""
    conv_wgt = bytes(conv_out_c * 3 * 3 * conv_in_c) + conv_params(conv_out_c)

    inputs = [
        bytes(d2s_in_bytes),
        bytes(d2s_out_bytes),
    ]
    weights = [d2s_wgt, conv_wgt]
    refs = [
        bytes(d2s_out_bytes),
        bytes(conv_out_bytes),
    ]

    input_offsets = []
    weight_offsets = []
    ref_offsets = []
    cur = 0
    for blob in inputs:
        input_offsets.append(cur)
        cur += len(blob)
    cur = 0
    for blob in weights:
        weight_offsets.append(cur)
        cur += len(blob)
    cur = 0
    for blob in refs:
        ref_offsets.append(cur)
        cur += len(blob)

    total_inputs = sum(map(len, inputs))
    total_weights = sum(map(len, weights))
    dram_wgt_base = DRAM_BASE
    dram_in_base = dram_wgt_base + round_up(total_weights, REGION_ALIGN)
    dram_out_base = dram_in_base + round_up(total_inputs, REGION_ALIGN)

    layers = [
        dict(
            in_h=d2s_h, in_w=d2s_w, in_c=d2s_in_c,
            out_h=conv_h, out_w=conv_w, out_c=conv_in_c,
            k_h=block, k_w=block, s_h=1, s_w=1,
            p_t=0, p_b=0, p_l=0, p_r=0,
            group=1, op_kind=OP_D2SPACE,
            dram_in=dram_in_base + input_offsets[0],
            in_off=input_offsets[0], in_size=d2s_in_bytes,
            input0_tensor=0, input1_tensor=-1, output_tensor=1,
            producer0_layer=-1, producer1_layer=-1,
            first_consumer_layer=1, last_consumer_layer=1, consumer_count=1,
        ),
        dict(
            in_h=conv_h, in_w=conv_w, in_c=conv_in_c,
            out_h=conv_h, out_w=conv_w, out_c=conv_out_c,
            k_h=3, k_w=3, s_h=1, s_w=1,
            p_t=1, p_b=1, p_l=1, p_r=1,
            group=1, op_kind=OP_CONV,
            dram_in=dram_in_base + input_offsets[1],
            in_off=input_offsets[1], in_size=d2s_out_bytes,
            input0_tensor=1, input1_tensor=-1, output_tensor=2,
            producer0_layer=0, producer1_layer=-1,
            first_consumer_layer=-1, last_consumer_layer=-1, consumer_count=0,
        ),
    ]

    header_size = struct.calcsize(HEADER_FMT)
    layer_size = struct.calcsize(LAYER_FMT)
    graph_size = struct.calcsize(GRAPH_META_FMT)
    data_offset = header_size + len(layers) * layer_size + len(layers) * graph_size
    base_w = total_inputs
    base_r = total_inputs + total_weights

    with out_path.open("wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, len(layers), data_offset))
        for idx, layer in enumerate(layers):
            f.write(struct.pack(
                LAYER_FMT,
                layer["in_h"], layer["in_w"], layer["in_c"],
                layer["out_h"], layer["out_w"], layer["out_c"],
                layer["k_h"], layer["k_w"], layer["s_h"], layer["s_w"],
                layer["p_t"], layer["p_b"], layer["p_l"], layer["p_r"],
                layer["dram_in"],
                dram_wgt_base + weight_offsets[idx],
                dram_out_base + ref_offsets[idx],
                layer["in_size"], len(weights[idx]), len(refs[idx]),
                data_offset + layer["in_off"],
                data_offset + base_w + weight_offsets[idx],
                data_offset + base_r + ref_offsets[idx],
                layer["group"], layer["op_kind"], DT_INT8x8,
                0,
            ))
        for layer in layers:
            f.write(struct.pack(
                GRAPH_META_FMT,
                layer["input0_tensor"], layer["input1_tensor"], layer["output_tensor"],
                layer["producer0_layer"], layer["producer1_layer"],
                layer["first_consumer_layer"], layer["last_consumer_layer"],
                layer["consumer_count"],
            ))
        for blob in inputs:
            f.write(blob)
        for blob in weights:
            f.write(blob)
        for blob in refs:
            f.write(blob)

    print(f"wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
