#!/usr/bin/env python3
"""
Generate a Q-tiled attention .tflite for L1Mesh flash-attention-style stress
testing.

Big model `bmm_softmax_bmm_2.5ms_1g_int8` has score = 4 MB/head (32 heads, T_q
= T_k = 2048, INT8) which does NOT fit the 3 MB L1, so the existing
score-tile suppression (cap = L1_BUDGET/4 = 768 KB) cannot fire and 128 MB of
score spills to DRAM.

This script rebuilds the same attention as an UNROLLED per-(head, q-tile)
loop in Keras. Each micro-block is one BMM₁ → SOFTMAX → BMM₂ over a
[1, Tq, T_k] score tile (default Tq=64 → 128 KB INT8 score) that fits L1/4.
compile_model.py then naturally chains each micro-block in L1 via the
existing score-tile suppression — no compile_model.py or runner changes
required.

Big-model topology (defaults):
    H = 32 heads, T_q = T_k = 2048, depth = 128, Tq = 64  → 32 × 32 = 1024
    micro-blocks.  Each micro-block: BMM₁ [1,Tq,D]×[1,D,T_k]ᵀ → softmax →
    BMM₂ [1,Tq,T_k]×[1,T_k,D] = [1,Tq,D].  L1 footprint per block ≈ Q(8KB)
    + K(256KB) + V(256KB) + score(128KB) + out(8KB) = 656 KB.

Validation flow (recommended):
  1. Build SMALL first (H=4, T_q=T_k=256, Tq=64 → 16 micro-blocks):
       python3 systemc/scripts/gen_bmm_attention_tiled_tflite.py \\
         --n-heads 4 --t-q 256 --t-k 256 --t-tile 64 --tag small
     produces model/BMM/bmm_softmax_bmm_tiled_small_int8.tflite
  2. ./batch/run_systemc.py --filter bmm_softmax_bmm_tiled_small \\
         --fast-only --rerun-all --no-html
     Confirm per-layer dram_w is near-zero on fc(bmm) rows that feed softmax
     (score-tile suppression fires).
  3. Build BIG:
       python3 systemc/scripts/gen_bmm_attention_tiled_tflite.py
       (defaults: 32h, 2048q, 2048k, 64-tile, tag=2.5ms)
       produces model/BMM/bmm_softmax_bmm_tiled_2.5ms_int8.tflite
     Compare profile.csv totals vs original bmm_softmax_bmm_2.5ms_1g_int8:
     dram_w on BMM₁ rows should drop from ~4 MB to ~0.
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


def build_tiled_attention(n_heads: int, t_q: int, t_k: int, depth: int,
                          t_tile: int) -> tf.keras.Model:
    """
    Build attention as an explicit per-(head, q-tile) loop.

    Inputs:
        Q: [1, H, T_q, D]
        K: [1, H, T_k, D]
        V: [1, H, T_k, D]
    Output:
        out: [1, H, T_q, D]

    Each micro-block i = (h, q_idx) computes:
        score_i  = matmul(Q[:, h, q_start:q_end, :], K[:, h, :, :], transpose_b=True) * scale
                                                                 # [1, Tq, T_k]
        probs_i  = softmax(score_i, axis=-1)                     # [1, Tq, T_k]
        out_i    = matmul(probs_i, V[:, h, :, :])                # [1, Tq, D]
    Tiles are stacked along T_q then heads back along H.
    """
    if t_q % t_tile != 0:
        raise ValueError(f"t_q ({t_q}) must be divisible by t_tile ({t_tile})")

    q_in = tf.keras.Input(shape=(n_heads, t_q, depth), batch_size=1, name="q")
    k_in = tf.keras.Input(shape=(n_heads, t_k, depth), batch_size=1, name="k")
    v_in = tf.keras.Input(shape=(n_heads, t_k, depth), batch_size=1, name="v")

    scale = np.float32(1.0 / math.sqrt(depth))
    n_q_tiles = t_q // t_tile

    # Pre-extract per-head K and V slices (each used by all tiles of that head).
    # Using a tf.keras Lambda layer per slice keeps the resulting TFLite graph
    # well-formed (each Lambda → 1 STRIDED_SLICE op).
    head_outs = []  # length H, each [1, T_q, D]
    for h in range(n_heads):
        k_h = tf.keras.layers.Lambda(
            lambda x, _h=h: x[:, _h, :, :], name=f"k_h{h}"
        )(k_in)                                                  # [1, T_k, D]
        v_h = tf.keras.layers.Lambda(
            lambda x, _h=h: x[:, _h, :, :], name=f"v_h{h}"
        )(v_in)                                                  # [1, T_k, D]

        tile_outs = []                                           # each [1, Tq, D]
        for qi in range(n_q_tiles):
            q0 = qi * t_tile
            q1 = q0 + t_tile
            q_th = tf.keras.layers.Lambda(
                lambda x, _h=h, _q0=q0, _q1=q1: x[:, _h, _q0:_q1, :],
                name=f"q_h{h}_t{qi}",
            )(q_in)                                              # [1, Tq, D]

            scores = tf.keras.layers.Lambda(
                lambda xs: tf.matmul(xs[0], xs[1], transpose_b=True),
                name=f"bmm_qk_h{h}_t{qi}",
            )([q_th, k_h])                                        # [1, Tq, T_k]
            scores = tf.keras.layers.Lambda(
                lambda x, _s=scale: x * _s, name=f"scale_h{h}_t{qi}"
            )(scores)
            probs = tf.keras.layers.Softmax(
                axis=-1, name=f"softmax_h{h}_t{qi}"
            )(scores)                                            # [1, Tq, T_k]

            out_th = tf.keras.layers.Lambda(
                lambda xs: tf.matmul(xs[0], xs[1]),
                name=f"bmm_pv_h{h}_t{qi}",
            )([probs, v_h])                                       # [1, Tq, D]
            tile_outs.append(out_th)

        head_out = tf.keras.layers.Concatenate(
            axis=1, name=f"cat_h{h}"
        )(tile_outs) if n_q_tiles > 1 else tile_outs[0]          # [1, T_q, D]
        head_outs.append(head_out)

    # Stack heads along a new H axis: list of H × [1, T_q, D] → [1, H, T_q, D]
    out = tf.keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=1), name="stack_heads"
    )(head_outs)

    tag = f"{n_heads}h_{t_q}q_{t_k}k_tq{t_tile}"
    return tf.keras.Model([q_in, k_in, v_in], out,
                          name=f"bmm_softmax_bmm_tiled_{tag}")


def rep_dataset(n_heads, t_q, t_k, depth, count=16):
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
    c.inference_input_type  = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--t-q",     type=int, default=2048,
                    help="total query length (must be multiple of --t-tile)")
    ap.add_argument("--t-k",     type=int, default=2048,
                    help="key/value length (held resident per-head)")
    ap.add_argument("--depth",   type=int, default=128)
    ap.add_argument("--t-tile",  type=int, default=64,
                    help="Q-tile size; score tile = t_tile × t_k × 1 byte must "
                         "fit L1/4 = 768 KB")
    ap.add_argument("--tag",     type=str, default="2.5ms",
                    help="filename tag: bmm_softmax_bmm_tiled_{tag}_int8.tflite")
    ap.add_argument("--out-dir", type=Path,
                    default=repo_root() / "model" / "BMM")
    args = ap.parse_args()

    nh, tq, tk, d, tile = args.n_heads, args.t_q, args.t_k, args.depth, args.t_tile

    if tq % tile != 0:
        sys.exit(f"t_q={tq} must be a multiple of t_tile={tile}")

    n_q_tiles = tq // tile
    n_blocks  = nh * n_q_tiles
    score_tile_kb = tile * tk / 1024
    k_per_head_kb = tk * d / 1024
    print(f"heads={nh}  T_q={tq}  T_k={tk}  depth={d}  Tq_tile={tile}")
    print(f"micro-blocks = {nh} heads × {n_q_tiles} Q-tiles = {n_blocks}")
    print(f"score tile   = {tile} × {tk} = {tile*tk:,} B = {score_tile_kb:.0f} KB (cap L1/4 = 768 KB)")
    print(f"K per head   = {k_per_head_kb:.0f} KB  (cap L1 = 3072 KB)")
    if score_tile_kb > 768:
        print(f"  ⚠ score tile {score_tile_kb:.0f} KB > 768 KB → suppression won't fire; pick smaller --t-tile")
    if k_per_head_kb > 3072:
        print(f"  ⚠ K per head {k_per_head_kb:.0f} KB > 3 MB → K can't stay resident; need K-tiling (flash-attention)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model   = build_tiled_attention(nh, tq, tk, d, tile)
    dataset = rep_dataset(nh, tq, tk, d)

    data = to_int8(model, dataset)
    path = args.out_dir / f"bmm_softmax_bmm_tiled_{args.tag}_int8.tflite"
    path.write_bytes(data)
    print(f"wrote {path}  ({len(data)//1024} KB)")


if __name__ == "__main__":
    main()
