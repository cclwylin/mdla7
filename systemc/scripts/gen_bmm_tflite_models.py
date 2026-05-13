#!/usr/bin/env python3
"""Generate tiny BMM -> Softmax -> BMM TFLite patterns for L1Mesh tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seeded_values(shape: tuple[int, ...], scale: float = 0.125, bias: float = 0.0) -> np.ndarray:
    count = int(np.prod(shape))
    vals = (np.arange(count, dtype=np.float32) % 17.0) - 8.0
    return (vals.reshape(shape) * scale + bias).astype(np.float32)


def build_bmm_softmax_bmm(
    *,
    heads: int = 2,
    q_len: int = 8,
    kv_len: int = 8,
    depth: int = 8,
    value_dim: int = 8,
) -> tf.keras.Model:
    q = tf.keras.Input(shape=(heads, q_len, depth), batch_size=1, name="q")
    k = tf.keras.Input(shape=(heads, kv_len, depth), batch_size=1, name="k")
    v = tf.keras.Input(shape=(heads, kv_len, value_dim), batch_size=1, name="v")

    score = tf.keras.layers.Lambda(
        lambda xs: tf.matmul(xs[0], xs[1], transpose_b=True),
        name="bmm_qk",
    )([q, k])
    score = tf.keras.layers.Lambda(lambda x: x * np.float32(0.125), name="scale")(score)
    prob = tf.keras.layers.Softmax(axis=-1, name="softmax")(score)
    out = tf.keras.layers.Lambda(
        lambda xs: tf.matmul(xs[0], xs[1]),
        name="bmm_pv",
    )([prob, v])
    return tf.keras.Model([q, k, v], out, name="bmm_softmax_bmm")


def representative_dataset(
    *,
    heads: int,
    q_len: int,
    kv_len: int,
    depth: int,
    value_dim: int,
    count: int = 16,
):
    q_shape = (1, heads, q_len, depth)
    k_shape = (1, heads, kv_len, depth)
    v_shape = (1, heads, kv_len, value_dim)

    def dataset() -> Iterable[list[np.ndarray]]:
        for idx in range(count):
            yield [
                seeded_values(q_shape, 0.03125, idx * 0.003),
                seeded_values(k_shape, 0.0234375, -idx * 0.002),
                seeded_values(v_shape, 0.015625, idx * 0.001),
            ]

    return dataset


def convert_fp32(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def convert_int8(model: tf.keras.Model, dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def tflite_ops(model_path: Path) -> list[str]:
    interp = tf.lite.Interpreter(model_path=str(model_path))
    fn = interp._get_ops_details if hasattr(interp, "_get_ops_details") else interp.get_ops_details
    return [op["op_name"] for op in fn()]


def write_readme(out_dir: Path, ops_by_file: dict[str, list[str]]) -> None:
    lines = [
        "# BMM Unit Patterns",
        "",
        "Tiny attention-style patterns for L1Mesh validation.",
        "",
        "Shape:",
        "",
        "- Q: `[1, 2, 8, 8]`",
        "- K: `[1, 2, 8, 8]`",
        "- V: `[1, 2, 8, 8]`",
        "- Output: `[1, 2, 8, 8]`",
        "",
        "Graph intent:",
        "",
        "1. `BMM(Q, K^T)`",
        "2. scale",
        "3. `Softmax(axis=-1)`",
        "4. `BMM(Softmax, V)`",
        "",
        "Generated files:",
        "",
    ]
    for name, ops in sorted(ops_by_file.items()):
        lines.append(f"- `{name}`: {', '.join(ops)}")
    if out_dir.joinpath("bmm_softmax_bmm_sam_quant_L22_L61.tflite").exists():
        lines += [
            "- `bmm_softmax_bmm_sam_quant_L22_L61.tflite`: real SAM attention slice lowered into supported MDLA7 ops",
        ]
    lines += [
        "",
        "Note: current `compile_model.py` does not list `BATCH_MATMUL` in supported ops, so it ignores the two BMM ops and only emits the middle `MUL + SOFTMAX` today. These files are the target BMM patterns first; the next step is to add BatchMatMul lowering/execution support.",
        "",
        "The SAM candidate is kept as the current runnable L1Mesh stress pattern:",
        "",
        "- QK score path: `trnps -> trnps -> mul`, shape `4,49,49 -> 4,49,49`",
        "- Attention normalization: `softmax`, shape `4,49,49 -> 4,49,49`",
        "- Value/output path: `trnps -> trnps -> matrlz -> fc`, shape `4900,1,128 -> 4900,1,128`",
        "",
    ]
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = repo_root() / "model" / "BMM"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = dict(heads=2, q_len=8, kv_len=8, depth=8, value_dim=8)
    model = build_bmm_softmax_bmm(**params)

    outputs: dict[str, bytes] = {
        "bmm_softmax_bmm_fp32.tflite": convert_fp32(model),
        "bmm_softmax_bmm_int8.tflite": convert_int8(model, representative_dataset(**params)),
    }

    ops_by_file: dict[str, list[str]] = {}
    for name, data in outputs.items():
        path = out_dir / name
        path.write_bytes(data)
        ops_by_file[name] = tflite_ops(path)

    write_readme(out_dir, ops_by_file)
    for name, ops in sorted(ops_by_file.items()):
        print(f"{name}: {', '.join(ops)}")


if __name__ == "__main__":
    main()
