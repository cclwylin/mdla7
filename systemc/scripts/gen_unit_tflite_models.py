#!/usr/bin/env python3
"""Generate tiny INT8 TFLite unit-test models for SystemC/Verilog bring-up."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


ModelBuilder = Callable[[], tf.keras.Model]


@dataclass(frozen=True)
class PatternSpec:
    category: str
    name: str
    builder: ModelBuilder
    input_shape: tuple[int, ...]
    fmt: str = "int8"
    two_inputs: bool = False
    depthwise: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _seeded_values(shape: tuple[int, ...], scale: float = 0.125) -> np.ndarray:
    count = int(np.prod(shape))
    vals = (np.arange(count, dtype=np.float32) % 17.0) - 8.0
    return (vals.reshape(shape) * scale).astype(np.float32)


def _set_conv_weights(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            weights = layer.get_weights()
            if not weights:
                continue
            weights[0] = _seeded_values(tuple(weights[0].shape))
            if len(weights) > 1:
                weights[1] = np.zeros_like(weights[1], dtype=np.float32)
            layer.set_weights(weights)


def _single_input_dataset(shape: tuple[int, ...], count: int = 8):
    def dataset():
        for idx in range(count):
            yield [(_seeded_values((1, *shape), 0.03125) + idx * 0.01).astype(np.float32)]

    return dataset


def _two_input_dataset(shape: tuple[int, ...], count: int = 8):
    def dataset():
        for idx in range(count):
            a = (_seeded_values((1, *shape), 0.03125) + idx * 0.01).astype(np.float32)
            b = (_seeded_values((1, *shape), 0.015625) - idx * 0.005).astype(np.float32)
            yield [a, b]

    return dataset


def _convert_int8(model: tf.keras.Model, representative_dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _convert_int16(model: tf.keras.Model, representative_dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    return converter.convert()


def _convert_fp16(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def _convert_model(spec: PatternSpec, model: tf.keras.Model) -> bytes:
    rep = _two_input_dataset(spec.input_shape) if spec.two_inputs else _single_input_dataset(spec.input_shape)
    if spec.fmt == "int8":
        return _convert_int8(model, rep)
    if spec.fmt == "int16":
        return _convert_int16(model, rep)
    if spec.fmt == "fp16":
        return _convert_fp16(model)
    raise ValueError(f"unknown format: {spec.fmt}")


def build_conv2d() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(8, 8, 3), batch_size=1, name="act")
    out = tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=True, name="conv3x3")(inp)
    model = tf.keras.Model(inp, out, name="g1op_conv2d_int8")
    _set_conv_weights(model)
    return model


def build_depthwise() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.DepthwiseConv2D(3, padding="valid", use_bias=True, name="dw3x3")(inp)
    model = tf.keras.Model(inp, out, name="g1op_dwconv2d_int8")
    _set_conv_weights(model)
    return model


def build_maxpool() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name="maxpool2x2")(inp)
    return tf.keras.Model(inp, out, name="g1op_maxpool_int8")


def build_avgpool() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool2x2")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_avgpool_int8")


def build_add() -> tf.keras.Model:
    a = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="a")
    b = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="b")
    out = tf.keras.layers.Add(name="add")([a, b])
    return tf.keras.Model([a, b], out, name="g1op_add_int8")


def build_mul() -> tf.keras.Model:
    a = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="a")
    b = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="b")
    out = tf.keras.layers.Multiply(name="mul")([a, b])
    return tf.keras.Model([a, b], out, name="g1op_mul_int8")


def build_sub() -> tf.keras.Model:
    a = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="a")
    b = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="b")
    out = tf.keras.layers.Subtract(name="sub")([a, b])
    return tf.keras.Model([a, b], out, name="g1op_ethz_sub_int8")


def build_concat() -> tf.keras.Model:
    a = tf.keras.Input(shape=(2, 2, 2), batch_size=1, name="a")
    b = tf.keras.Input(shape=(2, 2, 2), batch_size=1, name="b")
    out = tf.keras.layers.Concatenate(axis=-1, name="concat")([a, b])
    return tf.keras.Model([a, b], out, name="g1op_ethz_concat_int8")


def build_softmax() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(1, 1, 8), batch_size=1, name="act")
    out = tf.keras.layers.Softmax(axis=-1, name="softmax")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_softmax_int8")


def build_fc() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(8,), batch_size=1, name="act")
    out = tf.keras.layers.Dense(4, use_bias=True, name="fc")(inp)
    model = tf.keras.Model(inp, out, name="g1op_ethz_fc_int8")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            weights[0] = _seeded_values(tuple(weights[0].shape))
            weights[1] = np.zeros_like(weights[1], dtype=np.float32)
            layer.set_weights(weights)
    return model


def build_logistic() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="act")
    out = tf.keras.layers.Activation("sigmoid", name="logistic")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_logistic_fp16")


def build_hard_swish() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="act")
    out = tf.keras.layers.Activation("hard_swish", name="hard_swish")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_hard_swish_fp16")


def build_gelu() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="act")
    out = tf.keras.layers.Activation(tf.keras.activations.gelu, name="gelu")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_gelu_fp16")


def build_reshape() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: tf.nn.space_to_depth(x, block_size=2),
        name="space_to_depth",
    )(inp)
    return tf.keras.Model(inp, out, name="g1op_space_to_depth_tnps_int8")


def build_reshape_move() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="act")
    out = tf.keras.layers.Reshape((4, 4), name="reshape")(inp)
    return tf.keras.Model(inp, out, name="g1op_reshape_tnps")


def build_depth_to_space() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 8), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: tf.nn.depth_to_space(x, block_size=2),
        name="depth_to_space",
    )(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_depth_to_space_int8")


def build_transpose() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 3, 4), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: tf.transpose(x, perm=[0, 2, 1, 3]),
        name="transpose",
    )(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_transpose_int8")


def build_slice() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: tf.slice(x, begin=[0, 1, 0, 0], size=[1, 2, 4, 2]),
        name="slice",
    )(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_slice_int8")


def build_strided_slice() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: x[:, ::2, :, :],
        name="strided_slice",
    )(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_strided_slice_int8")


def build_pad() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 2), batch_size=1, name="act")
    out = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pad")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_pad_int8")


def build_pack() -> tf.keras.Model:
    a = tf.keras.Input(shape=(2, 2, 2), batch_size=1, name="a")
    b = tf.keras.Input(shape=(2, 2, 2), batch_size=1, name="b")
    out = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="pack")([a, b])
    return tf.keras.Model([a, b], out, name="g1op_ethz_pack_int8")


def build_unpack() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 2), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=1)[0], name="unpack")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_unpack_int8")


def build_split() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 4), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis=-1)[0], name="split")(inp)
    return tf.keras.Model(inp, out, name="g1op_ethz_split_int8")


def build_conv_add() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    x = tf.keras.layers.Conv2D(2, 3, padding="same", use_bias=True, name="conv3x3")(inp)
    out = tf.keras.layers.Add(name="conv_add")([x, x])
    model = tf.keras.Model(inp, out, name="g2op_conv_add_int8")
    _set_conv_weights(model)
    return model


def build_conv_pool() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    x = tf.keras.layers.Conv2D(2, 3, padding="same", use_bias=True, name="conv3x3")(inp)
    out = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool2x2")(x)
    model = tf.keras.Model(inp, out, name="g2op_conv_pool_int8")
    _set_conv_weights(model)
    return model


def build_conv_pool_add() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    x = tf.keras.layers.Conv2D(2, 3, padding="same", use_bias=True, name="conv3x3")(inp)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool2x2")(x)
    out = tf.keras.layers.Add(name="pool_add")([x, x])
    model = tf.keras.Model(inp, out, name="g3op_conv_pool_add_int8")
    _set_conv_weights(model)
    return model


def build_mb_space_to_depth() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: tf.nn.space_to_depth(x, block_size=2),
        name="space_to_depth",
    )(inp)
    return tf.keras.Model(inp, out, name="gmb_space_to_depth_int8")


def build_mb_depth_to_space() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(2, 2, 8), batch_size=1, name="act")
    out = tf.keras.layers.Lambda(
        lambda x: tf.nn.depth_to_space(x, block_size=2),
        name="depth_to_space",
    )(inp)
    return tf.keras.Model(inp, out, name="gmb_depth_to_space_int8")


def build_custom_chain() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, name="act")
    x = tf.keras.layers.Conv2D(2, 3, padding="same", use_bias=True, name="conv3x3")(inp)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool2x2")(x)
    x = tf.keras.layers.Conv2D(2, 1, padding="valid", use_bias=True, name="conv1x1")(x)
    x = tf.keras.layers.Add(name="residual_add")([x, x])
    out = tf.keras.layers.Reshape((4, 2), name="reshape_tail")(x)
    model = tf.keras.Model(inp, out, name="gcustom_conv_pool_conv_add_reshape_int8")
    _set_conv_weights(model)
    return model


G1OP_FORMATS = ("int8", "fp16", "int16")


def _g1op_specs() -> tuple[PatternSpec, ...]:
    specs: list[PatternSpec] = []
    for fmt in G1OP_FORMATS:
        specs.extend([
            PatternSpec("g1op", f"g1op_conv2d_{fmt}", build_conv2d, (8, 8, 3), fmt=fmt),
            PatternSpec("g1op", f"g1op_add_{fmt}", build_add, (2, 2, 4), fmt=fmt, two_inputs=True),
            PatternSpec("g1op", f"g1op_mul_{fmt}", build_mul, (2, 2, 4), fmt=fmt, two_inputs=True),
            PatternSpec("g1op", f"g1op_maxpool_{fmt}", build_maxpool, (4, 4, 2), fmt=fmt),
        ])
    specs.append(PatternSpec("g1op", "g1op_space_to_depth_tnps_int8", build_reshape, (4, 4, 2), fmt="int8"))
    specs.append(PatternSpec("g1op", "g1op_reshape_tnps_fp16", build_reshape_move, (2, 2, 4), fmt="fp16"))
    specs.append(PatternSpec("g1op", "g1op_reshape_tnps_int16", build_reshape_move, (2, 2, 4), fmt="int16"))
    return tuple(specs)


PATTERNS: tuple[PatternSpec, ...] = (
    *_g1op_specs(),
    PatternSpec("g1op_ethz", "g1op_ethz_conv2d_int8", build_conv2d, (8, 8, 3), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_dwconv2d_int8", build_depthwise, (4, 4, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_avgpool_int8", build_avgpool, (4, 4, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_maxpool_int8", build_maxpool, (4, 4, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_softmax_int8", build_softmax, (1, 1, 8), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_reshape_int8", build_reshape_move, (2, 2, 4), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_fc_int8", build_fc, (8,), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_add_int8", build_add, (2, 2, 4), fmt="int8", two_inputs=True),
    PatternSpec("g1op_ethz", "g1op_ethz_concat_int8", build_concat, (2, 2, 2), fmt="int8", two_inputs=True),
    PatternSpec("g1op_ethz", "g1op_ethz_mul_int8", build_mul, (2, 2, 4), fmt="int8", two_inputs=True),
    PatternSpec("g1op_ethz", "g1op_ethz_sub_int8", build_sub, (2, 2, 4), fmt="int8", two_inputs=True),
    PatternSpec("g1op_ethz", "g1op_ethz_hard_swish_fp16", build_hard_swish, (2, 2, 4), fmt="fp16"),
    PatternSpec("g1op_ethz", "g1op_ethz_gelu_fp16", build_gelu, (2, 2, 4), fmt="fp16"),
    PatternSpec("g1op_ethz", "g1op_ethz_depth_to_space_int8", build_depth_to_space, (2, 2, 8), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_transpose_int8", build_transpose, (2, 3, 4), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_space_to_depth_int8", build_reshape, (4, 4, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_slice_int8", build_slice, (4, 4, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_strided_slice_int8", build_strided_slice, (4, 4, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_pad_int8", build_pad, (2, 2, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_pack_int8", build_pack, (2, 2, 2), fmt="int8", two_inputs=True),
    PatternSpec("g1op_ethz", "g1op_ethz_unpack_int8", build_unpack, (2, 2, 2), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_split_int8", build_split, (2, 2, 4), fmt="int8"),
    PatternSpec("g1op_ethz", "g1op_ethz_logistic_fp16", build_logistic, (2, 2, 4), fmt="fp16"),
    PatternSpec("g2op", "g2op_conv_add_int8", build_conv_add, (4, 4, 2)),
    PatternSpec("g2op", "g2op_conv_pool_int8", build_conv_pool, (4, 4, 2)),
    PatternSpec("g3op", "g3op_conv_pool_add_int8", build_conv_pool_add, (4, 4, 2)),
    PatternSpec("g_mb_slice", "gmb_space_to_depth_int8", build_mb_space_to_depth, (4, 4, 2)),
    PatternSpec("g_mb_slice", "gmb_depth_to_space_int8", build_mb_depth_to_space, (2, 2, 8)),
    PatternSpec("g_custom", "gcustom_conv_pool_conv_add_reshape_int8", build_custom_chain, (4, 4, 2)),
)

DEPTHWISE_PATTERNS: tuple[PatternSpec, ...] = (
    PatternSpec("g1op", "g1op_dwconv2d_int8", build_depthwise, (4, 4, 2), depthwise=True),
)


def compile_model(model_path: Path, bin_path: Path, max_layers: int) -> None:
    cmd = [
        sys.executable,
        str(repo_root() / "systemc" / "scripts" / "compile_model.py"),
        str(model_path),
        str(bin_path),
    ]
    if max_layers:
        cmd.extend(["--max-layers", str(max_layers)])
    subprocess.run(cmd, check=True)


def sanitize_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "mb_slice"


def import_small_mb_slices(
    out_dir: Path,
    bin_dir: Path,
    compile_bins: bool,
    max_layers: int,
    count: int,
    max_bytes: int,
) -> list[Path]:
    src_root = repo_root() / "model" / "MB_Path_Slice"
    if count <= 0 or not src_root.exists():
        return []
    candidates = [
        p for p in src_root.rglob("*.tflite")
        if p.is_file() and not p.name.startswith("._") and p.stat().st_size <= max_bytes
    ]
    candidates.sort(key=lambda p: (p.stat().st_size, str(p)))
    imported: list[Path] = []
    target_dir = out_dir / "g_mb_slice"
    target_dir.mkdir(parents=True, exist_ok=True)
    if compile_bins:
        (bin_dir / "g_mb_slice").mkdir(parents=True, exist_ok=True)
    for src in candidates[:count]:
        rel = src.relative_to(src_root)
        stem = sanitize_stem("_".join(rel.with_suffix("").parts))
        dst = target_dir / f"gmb_{stem}.tflite"
        shutil.copy2(src, dst)
        imported.append(dst)
        print(f"tflite: {dst}  # copied from {src}")
        if compile_bins:
            compile_model(dst, bin_dir / "g_mb_slice" / f"gmb_{stem}.bin", max_layers)
            print(f"bin:    {bin_dir / 'g_mb_slice' / f'gmb_{stem}.bin'}")
    return imported


def parse_args() -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=root / "model" / "UnitTest",
                    help="root directory for generated .tflite models")
    ap.add_argument("--compile", action="store_true",
                    help="also compile generated .tflite models into MDL7 .bin programs")
    ap.add_argument("--bin-dir", type=Path, default=root / "rtl" / "bin" / "UnitTest",
                    help="root directory for compiled .bin programs when --compile is used")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="pass-through compile_model.py layer cap")
    ap.add_argument("--only", action="append", default=[],
                    help="generate only matching model/category names; may be repeated")
    ap.add_argument("--include-depthwise", action="store_true",
                    help="also generate depthwise-conv; current Verilog closed-loop probe may skip it")
    ap.add_argument("--mb-slice-count", type=int, default=0,
                    help="optional: copy this many smallest existing MB_Path_Slice models into g_mb_slice; 0 disables")
    ap.add_argument("--mb-slice-max-bytes", type=int, default=2048,
                    help="only import MB_Path_Slice models up to this byte size")
    return ap.parse_args()


def _matches(spec: PatternSpec, filters: tuple[str, ...]) -> bool:
    if not filters:
        return True
    haystack = f"{spec.category}/{spec.name}".lower()
    return any(f in haystack for f in filters)


def main() -> int:
    args = parse_args()
    filters = tuple(x.lower() for x in args.only)
    specs = PATTERNS + (DEPTHWISE_PATTERNS if args.include_depthwise else ())

    generated: list[Path] = []
    for spec in specs:
        if not _matches(spec, filters):
            continue
        model = spec.builder()
        model_dir = args.out_dir / spec.category
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{spec.name}.tflite"
        model_path.write_bytes(_convert_model(spec, model))
        generated.append(model_path)
        print(f"tflite: {model_path}")
        if args.compile:
            bin_dir = args.bin_dir / spec.category
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_path = bin_dir / f"{spec.name}.bin"
            compile_model(model_path, bin_path, args.max_layers)
            print(f"bin:    {bin_path}")

    include_mb_slice = bool(args.mb_slice_count) and (
        not filters or any("g_mb_slice".find(f) >= 0 or f in "g_mb_slice" for f in filters)
    )
    if include_mb_slice:
        generated.extend(
            import_small_mb_slices(
                args.out_dir,
                args.bin_dir,
                args.compile,
                args.max_layers,
                args.mb_slice_count,
                args.mb_slice_max_bytes,
            )
        )

    if not generated:
        raise SystemExit("no unit-test models matched --only filters")
    print(f"generated {len(generated)} unit-test tflite model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
