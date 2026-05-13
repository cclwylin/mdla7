#!/usr/bin/env python3
"""Audit TFLite ops that are unsupported or only materialized by compile_model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import compile_model as cm


def _is_fp_tflite_type(fb, tensor_type: int) -> bool:
    return tensor_type in (fb.TensorType.FLOAT16, fb.TensorType.FLOAT32)


def _models_from_arg(path: Path) -> list[Path]:
    if path.is_file():
        return [] if path.name.startswith("._") else [path]
    return sorted(p for p in path.glob("*.tflite") if not p.name.startswith("._"))


def _op_names(model_path: Path) -> list[str]:
    fb, model, sg = cm._load_flatbuffer(str(model_path))
    return [cm._opcode_name(fb, model, sg.Operators(i))
            for i in range(sg.OperatorsLength())]


def _materialized_fallback_ops(model_path: Path) -> Counter[str]:
    fb, model, sg = cm._load_flatbuffer(str(model_path))
    counts: Counter[str] = Counter()
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        op_name = cm._opcode_name(fb, model, op)
        if op_name in cm.MATERIALIZED_FALLBACK_OPS:
            counts[op_name] += 1
            continue
        if op_name not in cm.DTYPE_MATERIALIZED_FALLBACK_OPS:
            continue
        if op.InputsLength() <= 0 or op.Inputs(0) < 0:
            counts[f"{op_name}:unknown-dtype"] += 1
            continue
        in_t = sg.Tensors(op.Inputs(0))
        if not _is_fp_tflite_type(fb, in_t.Type()):
            counts[f"{op_name}:non-fp-dtype"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit corpus ops missing from or materialized by compile_model")
    parser.add_argument("paths", nargs="+",
                        help="TFLite files or directories containing *.tflite")
    parser.add_argument("--models", action="store_true",
                        help="print per-model unsupported/materialized op counts")
    parser.add_argument("--unsupported-only", action="store_true",
                        help="suppress materialized fallback reporting")
    parser.add_argument("--strict-native", action="store_true",
                        help="exit non-zero if materialized fallback ops are present")
    args = parser.parse_args()

    supported = set(cm.SUPPORTED_OPS)
    any_missing = False
    any_materialized = False

    for raw in args.paths:
        root = Path(raw)
        models = _models_from_arg(root)
        totals: Counter[str] = Counter()
        mat_totals: Counter[str] = Counter()
        model_missing: dict[Path, Counter[str]] = {}
        model_materialized: dict[Path, Counter[str]] = {}
        parse_errors: dict[Path, str] = {}
        by_op: dict[str, list[str]] = defaultdict(list)
        mat_by_op: dict[str, list[str]] = defaultdict(list)

        for model_path in models:
            try:
                op_names = _op_names(model_path)
                mat = _materialized_fallback_ops(model_path)
            except Exception as exc:
                any_missing = True
                parse_errors[model_path] = str(exc)
                continue
            missing = Counter(op for op in op_names if op not in supported)
            if missing:
                any_missing = True
                model_missing[model_path] = missing
                totals.update(missing)
                for op_name in sorted(missing):
                    by_op[op_name].append(model_path.name)
            if mat:
                any_materialized = True
                model_materialized[model_path] = mat
                mat_totals.update(mat)
                for op_name in sorted(mat):
                    mat_by_op[op_name].append(model_path.name)

        label = root.name if root.is_dir() else root.name
        print(f"{label}: {len(model_missing)}/{len(models)} models have unsupported ops")
        if totals:
            print("  unsupported totals: " + ", ".join(
                f"{name}={count}" for name, count in sorted(totals.items())))
        else:
            print("  unsupported totals: none")

        if not args.unsupported_only:
            print(
                f"  materialized fallback: {len(model_materialized)}/{len(models)} "
                "models use supported-but-not-native ops")
            if mat_totals:
                print("  materialized totals: " + ", ".join(
                    f"{name}={count}" for name, count in sorted(mat_totals.items())))
            else:
                print("  materialized totals: none")

        for model_path, err in sorted(parse_errors.items()):
            print(f"  parse-error {model_path.name}: {err}")

        if args.models and model_missing:
            for model_path, missing in sorted(model_missing.items()):
                print(f"  unsupported {model_path.name}: " + ", ".join(
                    f"{name}={count}" for name, count in sorted(missing.items())))
        elif model_missing:
            for op_name, names in sorted(by_op.items()):
                preview = ", ".join(names[:5])
                suffix = "" if len(names) <= 5 else f", ... +{len(names) - 5}"
                print(f"  {op_name}: {len(names)} model(s): {preview}{suffix}")

        if args.models and not args.unsupported_only and model_materialized:
            for model_path, mat in sorted(model_materialized.items()):
                print(f"  materialized {model_path.name}: " + ", ".join(
                    f"{name}={count}" for name, count in sorted(mat.items())))
        elif not args.unsupported_only and model_materialized:
            for op_name, names in sorted(mat_by_op.items()):
                preview = ", ".join(names[:5])
                suffix = "" if len(names) <= 5 else f", ... +{len(names) - 5}"
                print(f"  matrlz {op_name}: {len(names)} model(s): {preview}{suffix}")

    failed = any_missing or (args.strict_native and any_materialized)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
