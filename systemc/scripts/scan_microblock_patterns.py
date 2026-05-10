#!/usr/bin/env python3
"""Scan TFLite models for microblock pattern candidates.

This is a corpus-discovery helper, not a TFLite graph slicer.  It writes a CSV
manifest under model/MB_Path_Slice/ so implementation work can start from concrete
model/layer ranges before we add a real FlatBuffer subgraph cutter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import struct


COMPUTE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}
BINARY_EWE_OPS = {"ADD", "MUL", "SUB"}
UNARY_EWE_OPS = {"HARD_SWISH", "GELU", "LOGISTIC"}
POOL_OPS = {"AVERAGE_POOL_2D", "MAX_POOL_2D"}
LAYOUT_OPS = {
    "SLICE", "STRIDED_SLICE", "TRANSPOSE", "PACK", "UNPACK", "SPLIT",
    "SPLIT_V", "PAD", "PADV2", "TILE", "RESHAPE", "SQUEEZE",
    "EXPAND_DIMS", "DEPTH_TO_SPACE", "SPACE_TO_DEPTH", "CONCATENATION",
}

BUILTIN_OPS = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    9: "FULLY_CONNECTED",
    14: "LOGISTIC",
    17: "MAX_POOL_2D",
    18: "MUL",
    22: "RESHAPE",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    34: "PAD",
    36: "GATHER",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    43: "SQUEEZE",
    45: "STRIDED_SLICE",
    49: "SPLIT",
    60: "PADV2",
    65: "SLICE",
    69: "TILE",
    70: "EXPAND_DIMS",
    83: "PACK",
    88: "UNPACK",
    102: "SPLIT_V",
    117: "HARD_SWISH",
}


class FB:
    def __init__(self, data: bytes):
        self.b = data

    def i8(self, o: int) -> int:
        return struct.unpack_from("<b", self.b, o)[0]

    def u8(self, o: int) -> int:
        return self.b[o]

    def i32(self, o: int) -> int:
        return struct.unpack_from("<i", self.b, o)[0]

    def u32(self, o: int) -> int:
        return struct.unpack_from("<I", self.b, o)[0]

    def u16(self, o: int) -> int:
        return struct.unpack_from("<H", self.b, o)[0]

    def root(self) -> int:
        return self.u32(0)

    def field(self, tbl: int, fid: int) -> int:
        vt = tbl - self.i32(tbl)
        vsz = self.u16(vt)
        pos = 4 + fid * 2
        if pos >= vsz:
            return 0
        off = self.u16(vt + pos)
        return tbl + off if off else 0

    def ref(self, off: int) -> int:
        return off + self.u32(off)

    def vec_len(self, vf: int) -> int:
        if not vf:
            return 0
        return self.u32(self.ref(vf))

    def vec_table(self, vf: int, i: int) -> int:
        v = self.ref(vf)
        elem = v + 4 + i * 4
        return elem + self.u32(elem)

    def vec_i32(self, vf: int) -> list[int]:
        if not vf:
            return []
        v = self.ref(vf)
        n = self.u32(v)
        return [self.i32(v + 4 + i * 4) for i in range(n)]

    def scalar_i32(self, tbl: int, fid: int, default: int = 0) -> int:
        f = self.field(tbl, fid)
        return self.i32(f) if f else default

    def scalar_i8(self, tbl: int, fid: int, default: int = 0) -> int:
        f = self.field(tbl, fid)
        return self.i8(f) if f else default


@dataclass
class OpInfo:
    idx: int
    name: str
    inputs: list[int]
    outputs: list[int]
    in_shape: str = ""
    out_shape: str = ""
    k: str = ""
    stride: str = ""


def shape_str(shape: Iterable[int]) -> str:
    vals = [str(int(x)) for x in shape]
    return "[" + ",".join(vals) + "]"


def load_ops(path: Path) -> list[OpInfo]:
    fb = FB(path.read_bytes())
    model = fb.root()
    opcodes_f = fb.field(model, 1)
    subgraphs_f = fb.field(model, 2)
    if not subgraphs_f or fb.vec_len(subgraphs_f) < 1:
        return []
    sg = fb.vec_table(subgraphs_f, 0)
    tensors_f = fb.field(sg, 0)
    operators_f = fb.field(sg, 3)

    opcode_names: list[str] = []
    for i in range(fb.vec_len(opcodes_f)):
        oc = fb.vec_table(opcodes_f, i)
        code = fb.scalar_i32(oc, 3, -1)
        if code < 0:
            code = fb.scalar_i8(oc, 0, -1)
        opcode_names.append(BUILTIN_OPS.get(code, f"opcode_{code}"))

    def tensor_table(tidx: int) -> int:
        if tidx < 0 or not tensors_f or tidx >= fb.vec_len(tensors_f):
            return 0
        return fb.vec_table(tensors_f, tidx)

    def tensor_shape(tidx: int) -> str:
        t = tensor_table(tidx)
        return shape_str(fb.vec_i32(fb.field(t, 0))) if t else ""

    def tensor_shape_list(tidx: int) -> list[int]:
        t = tensor_table(tidx)
        return fb.vec_i32(fb.field(t, 0)) if t else []

    ops: list[OpInfo] = []
    for i in range(fb.vec_len(operators_f)):
        op = fb.vec_table(operators_f, i)
        opcode_idx = fb.scalar_i32(op, 0, -1)
        name = opcode_names[opcode_idx] if 0 <= opcode_idx < len(opcode_names) else f"opcode_index_{opcode_idx}"
        inputs = fb.vec_i32(fb.field(op, 1))
        outputs = fb.vec_i32(fb.field(op, 2))
        info = OpInfo(
            idx=i,
            name=name,
            inputs=inputs,
            outputs=outputs,
            in_shape=tensor_shape(inputs[0]) if inputs else "",
            out_shape=tensor_shape(outputs[0]) if outputs else "",
        )
        if name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(inputs) >= 2:
            w = tensor_shape_list(inputs[1])
            if len(w) == 4:
                info.k = f"{w[1]}x{w[2]}"
        ops.append(info)
    return ops


def build_consumers(ops: list[OpInfo]) -> dict[int, list[int]]:
    consumers: dict[int, list[int]] = {}
    for op in ops:
        for t in op.inputs:
            if t >= 0:
                consumers.setdefault(t, []).append(op.idx)
    return consumers


def feeds(producer: OpInfo, consumer: OpInfo) -> bool:
    produced = {t for t in producer.outputs if t >= 0}
    if not produced:
        return False
    return any(t in produced for t in consumer.inputs if t >= 0)


def add_row(rows: list[dict[str, str]], *, pattern: str, model: Path,
            ops: list[OpInfo], start: int, end: int, notes: str) -> None:
    seq = ops[start:end + 1]
    rows.append({
        "pattern": pattern,
        "model": model.name,
        "start_op": str(start),
        "end_op": str(end),
        "op_sequence": " -> ".join(op.name for op in seq),
        "input_shape": seq[0].in_shape if seq else "",
        "output_shape": seq[-1].out_shape if seq else "",
        "suggested_slice": f"{model.stem}_L{start}_L{end}_{pattern}.tflite",
        "notes": notes,
        "source_path": str(model),
    })


def scan_model(model: Path) -> list[dict[str, str]]:
    ops = load_ops(model)
    consumers = build_consumers(ops)
    rows: list[dict[str, str]] = []

    for i, op in enumerate(ops):
        if (op.name in COMPUTE_OPS or op.name in BINARY_EWE_OPS or
                op.name in UNARY_EWE_OPS or op.name in POOL_OPS or op.name in LAYOUT_OPS):
            add_row(rows, pattern="producer_compute", model=model,
                    ops=ops, start=i, end=i,
                    notes="single producer engine candidate")
        if op.name in COMPUTE_OPS and op.in_shape == op.out_shape and op.in_shape:
            add_row(rows, pattern="streaming_preload", model=model,
                    ops=ops, start=i, end=i,
                    notes="large/tiled compute may use UDMA_R head/tail preload overlap")

    for i in range(len(ops) - 1):
        a, b = ops[i], ops[i + 1]

        if not feeds(a, b):
            if b.name in COMPUTE_OPS:
                add_row(rows, pattern="udma_as_engine", model=model,
                        ops=ops, start=i + 1, end=i + 1,
                        notes="compute consumer normally needs UDMA_R preload and optional UDMA_W store tail")
            continue

        if a.name in COMPUTE_OPS and b.name in BINARY_EWE_OPS:
            add_row(rows, pattern="consumer_tail", model=model,
                    ops=ops, start=i, end=i + 1,
                    notes="compute producer to binary EWE consumer tail")

        if a.name in COMPUTE_OPS and b.name in POOL_OPS:
            add_row(rows, pattern="consumer_tail", model=model,
                    ops=ops, start=i, end=i + 1,
                    notes="compute producer to POOL consumer tail")

        if a.name in COMPUTE_OPS and b.name in LAYOUT_OPS:
            add_row(rows, pattern="consumer_tail", model=model,
                    ops=ops, start=i, end=i + 1,
                    notes="compute producer to layout/TNPS consumer tail")

        if a.name in LAYOUT_OPS and b.name in COMPUTE_OPS:
            add_row(rows, pattern="layout_bridge", model=model,
                    ops=ops, start=i, end=i + 1,
                    notes="layout producer directly feeds compute consumer candidate")

        if a.name in POOL_OPS and b.name in COMPUTE_OPS:
            add_row(rows, pattern="layout_bridge", model=model,
                    ops=ops, start=i, end=i + 1,
                    notes="POOL producer to compute consumer candidate")

        if a.name == "CONCATENATION" and b.name == "CONV_2D":
            add_row(rows, pattern="layout_bridge", model=model,
                    ops=ops, start=i, end=i + 1,
                    notes="concat -> conv candidate; check k=1x1 for direct packed pointwise")

        if b.name in COMPUTE_OPS:
            add_row(rows, pattern="udma_as_engine", model=model,
                    ops=ops, start=i + 1, end=i + 1,
                    notes="compute consumer normally needs UDMA_R preload and optional UDMA_W store tail")

    for i in range(len(ops) - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        if not feeds(a, b) or not feeds(b, c):
            continue
        if a.name in COMPUTE_OPS and b.name in BINARY_EWE_OPS and c.name in POOL_OPS:
            add_row(rows, pattern="consumer_tail", model=model,
                    ops=ops, start=i, end=i + 2,
                    notes="compute -> binary EWE -> POOL tail candidate")
        if a.name in COMPUTE_OPS and b.name in BINARY_EWE_OPS and c.name in LAYOUT_OPS:
            add_row(rows, pattern="consumer_tail", model=model,
                    ops=ops, start=i, end=i + 2,
                    notes="compute -> binary EWE -> TNPS/layout tail candidate")

    for op in ops:
        for out_t in op.outputs:
            cs = consumers.get(out_t, [])
            if len(cs) >= 2:
                end = max(cs[:4])
                add_row(rows, pattern="fanout_live_range", model=model,
                        ops=ops, start=op.idx, end=end,
                        notes=f"producer tensor {out_t} has {len(cs)} consumers: {cs[:8]}")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("model/ETHZ_v6"))
    ap.add_argument("--out-dir", type=Path, default=Path("model/MB_Path_Slice"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-ops", type=int, default=32,
                    help="drop candidate ranges longer than this; 0 keeps all")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(p for p in args.model_dir.glob("*.tflite") if not p.name.startswith("._"))
    if args.limit:
        models = models[:args.limit]

    rows: list[dict[str, str]] = []
    for model in models:
        try:
            rows.extend(scan_model(model))
        except Exception as exc:
            rows.append({
                "pattern": "scan_error",
                "model": model.name,
                "start_op": "",
                "end_op": "",
                "op_sequence": "",
                "input_shape": "",
                "output_shape": "",
                "suggested_slice": "",
                "notes": str(exc),
                "source_path": str(model),
            })

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["pattern"] == "scan_error":
            deduped.append(row)
            continue
        key = (row["pattern"], row["op_sequence"], row["input_shape"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = deduped

    if args.max_ops:
        rows = [
            row for row in rows
            if row["pattern"] == "scan_error"
            or int(row["end_op"]) - int(row["start_op"]) + 1 <= args.max_ops
        ]

    out_csv = args.out_dir / "microblock_pattern_candidates.csv"
    fields = [
        "pattern", "model", "start_op", "end_op", "op_sequence",
        "input_shape", "output_shape", "suggested_slice", "notes", "source_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"scanned {len(models)} models")
    print(f"wrote {len(rows)} candidates -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
