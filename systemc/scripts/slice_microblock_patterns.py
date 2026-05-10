#!/usr/bin/env python3
"""Cut representative TFLite operator ranges from MB pattern candidates.

The slicer uses flatc JSON round-tripping.  It keeps the original tensor,
buffer, and operator_code tables intact, then replaces SubGraph[0].operators
with the selected operator range and rewrites SubGraph[0].inputs/outputs to the
range boundary tensors.  This conservative approach avoids index remapping and
is enough to create small pattern models for compile/sim regression.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PATTERNS = [
    "producer_compute",
    "consumer_tail",
    "layout_bridge",
    "fanout_live_range",
    "udma_as_engine",
    "streaming_preload",
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def buffer_has_data(model_json: dict, tensor: dict) -> bool:
    bidx = tensor.get("buffer", 0)
    if not isinstance(bidx, int) or bidx < 0 or bidx >= len(model_json.get("buffers", [])):
        return False
    return "data" in model_json["buffers"][bidx]


def boundary_tensors(model_json: dict, start: int, end: int) -> tuple[list[int], list[int]]:
    sg = model_json["subgraphs"][0]
    ops = sg["operators"]
    selected = ops[start:end + 1]
    selected_outputs = {t for op in selected for t in op.get("outputs", []) if t >= 0}
    selected_inputs = [t for op in selected for t in op.get("inputs", []) if t >= 0]
    tensors = sg["tensors"]

    inputs: list[int] = []
    seen_in: set[int] = set()
    for t in selected_inputs:
        if t in selected_outputs:
            continue
        if t >= len(tensors) or buffer_has_data(model_json, tensors[t]):
            continue
        if t not in seen_in:
            seen_in.add(t)
            inputs.append(t)

    consumed_inside = {t for op in selected for t in op.get("inputs", []) if t >= 0}
    outputs: list[int] = []
    seen_out: set[int] = set()
    for op in selected:
        for t in op.get("outputs", []):
            if t < 0:
                continue
            if t not in consumed_inside or op is selected[-1]:
                if t not in seen_out:
                    seen_out.add(t)
                    outputs.append(t)
    if not outputs and selected:
        outputs = [t for t in selected[-1].get("outputs", []) if t >= 0]

    return inputs, outputs


def compact_subgraph(model_json: dict, selected_ops: list[dict],
                     boundary_inputs: list[int], boundary_outputs: list[int]) -> None:
    sg = model_json["subgraphs"][0]
    used_tensors = set(boundary_inputs) | set(boundary_outputs)
    for op in selected_ops:
        used_tensors.update(t for t in op.get("inputs", []) if t >= 0)
        used_tensors.update(t for t in op.get("outputs", []) if t >= 0)
    tensor_ids = sorted(used_tensors)
    tensor_map = {old: new for new, old in enumerate(tensor_ids)}

    old_tensors = sg["tensors"]
    kept_tensors = [dict(old_tensors[i]) for i in tensor_ids]

    used_buffers = {0}
    for t in kept_tensors:
        bidx = t.get("buffer", 0)
        if isinstance(bidx, int) and bidx >= 0:
            used_buffers.add(bidx)
    buffer_ids = [0] + sorted(b for b in used_buffers if b != 0)
    buffer_map = {old: new for new, old in enumerate(buffer_ids)}
    model_json["buffers"] = [model_json["buffers"][i] for i in buffer_ids]
    for t in kept_tensors:
        t["buffer"] = buffer_map.get(t.get("buffer", 0), 0)

    compact_ops = []
    for op in selected_ops:
        op = dict(op)
        op["inputs"] = [tensor_map[t] if t >= 0 else t for t in op.get("inputs", [])]
        op["outputs"] = [tensor_map[t] if t >= 0 else t for t in op.get("outputs", [])]
        compact_ops.append(op)

    sg["tensors"] = kept_tensors
    sg["operators"] = compact_ops
    sg["inputs"] = [tensor_map[t] for t in boundary_inputs]
    sg["outputs"] = [tensor_map[t] for t in boundary_outputs]


def cut_one(row: dict[str, str], schema: Path, out_dir: Path) -> Path:
    src = Path(row["source_path"])
    start = int(row["start_op"])
    end = int(row["end_op"])
    pattern = row["pattern"]
    stem = Path(row["suggested_slice"]).stem
    pattern_dir = out_dir / pattern
    pattern_dir.mkdir(parents=True, exist_ok=True)
    out_path = pattern_dir / f"{stem}.tflite"

    with tempfile.TemporaryDirectory(prefix="mb_slice_") as td:
        tmp = Path(td)
        run([
            "flatc", "-t", "--strict-json", "--defaults-json", "--raw-binary",
            "-o", str(tmp), str(schema), "--", str(src),
        ])
        json_path = tmp / f"{src.stem}.json"
        data = json.loads(json_path.read_text())
        sg = data["subgraphs"][0]
        all_ops = sg["operators"]
        if start < 0 or end >= len(all_ops) or start > end:
            raise ValueError(f"bad op range {start}:{end} for {src}")
        inputs, outputs = boundary_tensors(data, start, end)
        compact_subgraph(data, all_ops[start:end + 1], inputs, outputs)
        sg["name"] = f"{src.stem}_L{start}_L{end}_{pattern}"
        data["description"] = f"MDLA7 MB pattern slice from {src.name} ops {start}-{end}"
        data["signature_defs"] = []

        cut_json = tmp / f"{stem}.json"
        cut_json.write_text(json.dumps(data, separators=(",", ":")))
        run([
            "flatc", "-b", "--strict-json", "--raw-binary",
            "-o", str(pattern_dir), str(schema), str(cut_json),
        ])
        produced = pattern_dir / f"{stem}.tflite"
        if produced != out_path and produced.exists():
            shutil.move(str(produced), out_path)
    return out_path


def op_count(row: dict[str, str]) -> int:
    return int(row["end_op"]) - int(row["start_op"]) + 1


def select_rows(rows: list[dict[str, str]], patterns: list[str],
                max_per_pattern: int, max_ops: int) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    counts = {p: 0 for p in patterns}
    for row in rows:
        p = row["pattern"]
        if p not in counts:
            continue
        if max_ops and op_count(row) > max_ops:
            continue
        if max_per_pattern and counts[p] >= max_per_pattern:
            continue
        picked.append(row)
        counts[p] += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("model/MB_Path_Slice/microblock_pattern_candidates.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("model/MB_Path_Slice"))
    ap.add_argument("--schema", type=Path, default=Path("third_party/tflite/schema.fbs"))
    ap.add_argument("--pattern", action="append", choices=DEFAULT_PATTERNS,
                    help="pattern to cut; may be repeated. default: all patterns")
    ap.add_argument("--max-per-pattern", type=int, default=3,
                    help="0 means all candidates")
    ap.add_argument("--max-ops", type=int, default=32,
                    help="skip candidate ranges longer than this; 0 keeps all")
    args = ap.parse_args()

    if not args.schema.exists():
        raise SystemExit(f"missing schema: {args.schema}")
    if shutil.which("flatc") is None:
        raise SystemExit("missing flatc")

    patterns = args.pattern or DEFAULT_PATTERNS
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = select_rows(rows, patterns, args.max_per_pattern, args.max_ops)
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            out = cut_one(row, args.schema, args.out_dir)
            status = "ok"
            err = ""
        except Exception as exc:
            out = Path("")
            status = "fail"
            err = str(exc)
        manifest_rows.append({
            **row,
            "slice_path": str(out),
            "slice_status": status,
            "error": err,
        })
        print(f"{status:4s} {row['pattern']:18s} {row['model']} L{row['start_op']}-L{row['end_op']}")

    out_manifest = args.out_dir / "microblock_pattern_slices.csv"
    fields = list(manifest_rows[0].keys()) if manifest_rows else []
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"wrote slice manifest -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
