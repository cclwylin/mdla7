#!/usr/bin/env python3
"""
Generate qwen35-style single-step decode attention with KV cache.

Produces:
  model/BMM/qwen35_kvcache_N_fp16.tflite   (FP16)
  model/BMM/qwen35_kvcache_N_int8.tflite   (INT8 / quantized)

Architecture (attention core only, no projection FCs):
  Q:  [1, n_heads, 1,      d_k]   current query token
  K:  [1, n_heads, kv_len, d_k]   KV-cache keys
  V:  [1, n_heads, kv_len, d_v]   KV-cache values

  scores = BMM(Q, K, transpose_b=True)    [1, n_heads, 1, kv_len]
  scores = scores * (1 / sqrt(d_k))
  probs  = Softmax(scores, axis=-1)       [1, n_heads, 1, kv_len]
  out    = BMM(probs, V)                  [1, n_heads, 1, d_v]

Dimensions match qwen35_attention_sXXXX:
  n_heads = 8, d_k = d_v = 256  (2048-dim model / 8 heads)
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    sys.exit("tensorflow required: pip install tensorflow")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seeded_fp32(shape: tuple[int, ...], scale: float = 0.03, offset: float = 0.0) -> np.ndarray:
    n = int(np.prod(shape))
    return ((np.arange(n, dtype=np.float32) % 31 - 15) * scale + offset).reshape(shape)


def build_model(n_heads: int, kv_len: int, d_k: int, d_v: int) -> tf.keras.Model:
    q = tf.keras.Input(shape=(n_heads, 1,      d_k), batch_size=1, name="q")
    k = tf.keras.Input(shape=(n_heads, kv_len, d_k), batch_size=1, name="k")
    v = tf.keras.Input(shape=(n_heads, kv_len, d_v), batch_size=1, name="v")

    scores = tf.keras.layers.Lambda(
        lambda xs: tf.matmul(xs[0], xs[1], transpose_b=True), name="bmm_qk"
    )([q, k])
    scores = tf.keras.layers.Lambda(
        lambda x: x * np.float32(1.0 / np.sqrt(d_k)), name="scale"
    )(scores)
    probs = tf.keras.layers.Softmax(axis=-1, name="softmax")(scores)
    out   = tf.keras.layers.Lambda(
        lambda xs: tf.matmul(xs[0], xs[1]), name="bmm_pv"
    )([probs, v])
    return tf.keras.Model([q, k, v], out, name=f"qwen35_kvcache_{kv_len}")


def rep_dataset(n_heads: int, kv_len: int, d_k: int, d_v: int, count: int = 32):
    q_shp = (1, n_heads, 1,      d_k)
    k_shp = (1, n_heads, kv_len, d_k)
    v_shp = (1, n_heads, kv_len, d_v)
    def _gen() -> Iterable[list[np.ndarray]]:
        for i in range(count):
            yield [
                seeded_fp32(q_shp, 0.02,   i * 0.001),
                seeded_fp32(k_shp, 0.015, -i * 0.001),
                seeded_fp32(v_shp, 0.01,   i * 0.0005),
            ]
    return _gen


def to_fp16(model: tf.keras.Model) -> bytes:
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations          = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
    c.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return c.convert()


def to_int8(model: tf.keras.Model, dataset) -> bytes:
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations              = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset     = dataset
    c.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type       = tf.int8
    c.inference_output_type      = tf.int8
    return c.convert()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-len",   type=int, nargs="+", default=[128, 512, 1024],
                    help="KV cache lengths to generate")
    ap.add_argument("--n-heads",  type=int, default=8)
    ap.add_argument("--d-k",      type=int, default=256)
    ap.add_argument("--d-v",      type=int, default=256)
    ap.add_argument("--dtype",    choices=["fp16","int8","both"], default="both")
    ap.add_argument("--out-dir",  type=Path,
                    default=repo_root() / "model" / "BMM")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nh, dk, dv = args.n_heads, args.d_k, args.d_v

    for kv in args.kv_len:
        print(f"\n=== kv_len={kv}  n_heads={nh}  d_k={dk}  d_v={dv} ===")
        model   = build_model(nh, kv, dk, dv)
        dataset = rep_dataset(nh, kv, dk, dv)

        if args.dtype in ("fp16", "both"):
            data = to_fp16(model)
            path = args.out_dir / f"qwen35_kvcache_{kv}_fp16.tflite"
            path.write_bytes(data)
            print(f"  wrote {path}  ({len(data)//1024} KB)")

        if args.dtype in ("int8", "both"):
            data = to_int8(model, dataset)
            path = args.out_dir / f"qwen35_kvcache_{kv}_int8.tflite"
            path.write_bytes(data)
            print(f"  wrote {path}  ({len(data)//1024} KB)")


if __name__ == "__main__":
    main()
