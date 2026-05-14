#!/usr/bin/env python3
"""
Generate bmm_softmax_bmm_2.5ms tile-sized model for L1Mesh chain validation.

Produces:
  model/BMM/bmm_softmax_bmm_tile_{n_heads}h_{t_q}q_{t_k}k_int8.tflite

Architecture (one tile of the 2.5ms attention workload):
  Q:  [1, n_heads, t_q, depth]   query tile
  K:  [1, n_heads, t_k, depth]   key tile
  V:  [1, n_heads, t_k, depth]   value tile

  scores = BMM(Q, K^T)           [1, n_heads, t_q, t_k]
  scores = scores * scale
  probs  = Softmax(scores, -1)   [1, n_heads, t_q, t_k]
  out    = BMM(probs, V)         [1, n_heads, t_q, depth]

Default: n_heads=32, t_q=64, t_k=64, depth=128
  score tile = 32 × 64 × 64 = 131072 B = 128 KB  → fits in 3 MB L1
  Q/K/V tile = 32 × 64 × 128 = 262144 B = 256 KB → total L1 < 2 MB ✓
"""

from __future__ import annotations
import argparse, os, sys, math
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    sys.exit("tensorflow required: pip install tensorflow")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seeded_fp32(shape, scale=0.02, offset=0.0):
    n = int(np.prod(shape))
    return ((np.arange(n, dtype=np.float32) % 31 - 15) * scale + offset).reshape(shape)


def build_model(n_heads: int, t_q: int, t_k: int, depth: int) -> tf.keras.Model:
    q = tf.keras.Input(shape=(n_heads, t_q, depth), batch_size=1, name="q")
    k = tf.keras.Input(shape=(n_heads, t_k, depth), batch_size=1, name="k")
    v = tf.keras.Input(shape=(n_heads, t_k, depth), batch_size=1, name="v")

    scores = tf.keras.layers.Lambda(
        lambda xs: tf.matmul(xs[0], xs[1], transpose_b=True),
        name="bmm_qk",
    )([q, k])
    scale = np.float32(1.0 / math.sqrt(depth))
    scores = tf.keras.layers.Lambda(
        lambda x: x * scale, name="scale"
    )(scores)

    # Squeeze the t_q=1 inner dim before softmax to avoid TFLite INT8 shape
    # inference bug (accum_dim mismatch on the downstream BATCH_MATMUL).
    if t_q == 1:
        scores_3d = tf.keras.layers.Reshape(
            (n_heads, t_k), name="pre_sm_sq"
        )(scores)
        probs_3d = tf.keras.layers.Softmax(axis=-1, name="softmax")(scores_3d)
        probs = tf.keras.layers.Reshape(
            (n_heads, 1, t_k), name="post_sm_usq"
        )(probs_3d)
    else:
        # General case: softmax over last axis; reshape to 3-D to avoid the
        # 4-D INT8 calibration shape bug.
        scores_3d = tf.keras.layers.Reshape(
            (n_heads * t_q, t_k), name="pre_sm_sq"
        )(scores)
        probs_3d = tf.keras.layers.Softmax(axis=-1, name="softmax")(scores_3d)
        probs = tf.keras.layers.Reshape(
            (n_heads, t_q, t_k), name="post_sm_usq"
        )(probs_3d)

    out = tf.keras.layers.Lambda(
        lambda xs: tf.matmul(xs[0], xs[1]), name="bmm_pv"
    )([probs, v])

    tag = f"{n_heads}h_{t_q}q_{t_k}k"
    return tf.keras.Model([q, k, v], out,
                          name=f"bmm_softmax_bmm_tile_{tag}")


def rep_dataset(n_heads, t_q, t_k, depth, count=32):
    def _gen():
        for i in range(count):
            yield [
                seeded_fp32((1, n_heads, t_q, depth), 0.02,  i * 0.001),
                seeded_fp32((1, n_heads, t_k, depth), 0.015, -i * 0.001),
                seeded_fp32((1, n_heads, t_k, depth), 0.01,   i * 0.0005),
            ]
    return _gen


def to_int8(model, dataset):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = dataset
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--t-q",     type=int, default=64,
                    help="query tile length (t_q × t_k score must fit in 3 MB L1)")
    ap.add_argument("--t-k",     type=int, default=64,
                    help="key tile length")
    ap.add_argument("--depth",   type=int, default=128)
    ap.add_argument("--out-dir", type=Path,
                    default=repo_root() / "model" / "BMM")
    args = ap.parse_args()

    nh, tq, tk, d = args.n_heads, args.t_q, args.t_k, args.depth
    score_kb = nh * tq * tk / 1024
    print(f"tile: heads={nh}  t_q={tq}  t_k={tk}  depth={d}")
    print(f"score tile = {nh}×{tq}×{tk} = {nh*tq*tk:,} B = {score_kb:.0f} KB")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model   = build_model(nh, tq, tk, d)
    dataset = rep_dataset(nh, tq, tk, d)

    data = to_int8(model, dataset)
    tag  = f"{nh}h_{tq}q_{tk}k"
    path = args.out_dir / f"bmm_softmax_bmm_tile_{tag}_int8.tflite"
    path.write_bytes(data)
    print(f"wrote {path}  ({len(data)//1024} KB)")


if __name__ == "__main__":
    main()
