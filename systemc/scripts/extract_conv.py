#!/usr/bin/env python3
"""Extract the first CONV_2D op from a .tflite file and dump a binary blob
the C++ test (test_tflite_conv) can consume.

Blob layout (little-endian):

  uint32_t magic       = 'MDL7' (0x37 4C 44 4D)
  uint32_t version     = 1
  uint16_t in_h, in_w, in_c
  uint16_t out_h, out_w, out_c
  uint8_t  k_h, k_w, s_h, s_w
  uint8_t  pad_t, pad_b, pad_l, pad_r
  int8_t   in[in_h * in_w * in_c]            -- NHWC
  int8_t   wgt[out_c * k_h * k_w * in_c]     -- OHWI
  int32_t  ref_out[out_h * out_w * out_c]    -- NHWC, INT32 partial sums

Reference output is computed in numpy with the same arithmetic the
ConvEngine performs (int8 * int8 -> int32, no quant scale, no bias).

Usage:  python scripts/extract_conv.py <model.tflite> <out.bin>
        (defaults to model/INT8/mobilenet_v1_0.25_128_quant.tflite ->
         build/conv_layer.bin)

Requires: numpy, tflite-runtime  (pip install tflite-runtime)
   or:    tensorflow              (Apple Silicon / macOS).
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---- model parser ---------------------------------------------------------

def _load_interpreter(path: str):
    try:
        import tflite_runtime.interpreter as tflr
        return tflr.Interpreter(model_path=path)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)
    except ImportError:
        import platform, sys as _sys
        py = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
        arch = platform.machine()
        # tflite-runtime has no wheel for Apple Silicon; force tensorflow there.
        cmd = (f"{py} -m pip install --user tensorflow"
               if arch == "arm64"
               else f"{py} -m pip install --user tflite-runtime")
        raise SystemExit(
            "extract_conv: TFLite interpreter unavailable.\n"
            f"  install with:  {cmd}\n"
            f"  (or: pip install -r {Path(__file__).resolve().parent.parent}/requirements.txt)"
        )


def _decode_quant(t):
    qp = t.get("quantization_parameters") or {}
    scales = np.asarray(qp.get("scales", []),       dtype=np.float32)
    zeros  = np.asarray(qp.get("zero_points", []),  dtype=np.int32)
    return scales, zeros


def list_conv_ops(interp):
    """Return all CONV_2D ops in execution order."""
    fn = interp._get_ops_details if hasattr(interp, "_get_ops_details") \
         else interp.get_ops_details
    return [op for op in fn() if op["op_name"] == "CONV_2D"]


def conv_at(interp, idx: int):
    """Return (op_details, in tensor, wgt tensor, out tensor) for the n-th CONV_2D."""
    convs = list_conv_ops(interp)
    if not convs:
        raise SystemExit("no CONV_2D op found in model")
    if idx < 0 or idx >= len(convs):
        raise SystemExit(f"layer index {idx} out of range (model has {len(convs)} CONV_2D ops)")
    op = convs[idx]
    in_idx, w_idx = op["inputs"][0], op["inputs"][1]
    out_idx       = op["outputs"][0]
    td = interp.get_tensor_details()
    return op, td[in_idx], td[w_idx], td[out_idx]


# ---- reference compute ----------------------------------------------------

def conv_int8_ref(act_i8: np.ndarray, wgt_i8: np.ndarray,
                  s_h: int, s_w: int, pad: tuple) -> np.ndarray:
    """Pure int32 reference: int8*int8 -> int32 sum, no quant scale, no bias."""
    H, W, Cin   = act_i8.shape
    OC, K_h, K_w, _ = wgt_i8.shape
    pT, pB, pL, pR = pad
    OH = (H + pT + pB - K_h) // s_h + 1
    OW = (W + pL + pR - K_w) // s_w + 1
    out = np.zeros((OH, OW, OC), dtype=np.int32)
    a = act_i8.astype(np.int32)
    w = wgt_i8.astype(np.int32)
    for oh in range(OH):
        for ow in range(OW):
            for kh in range(K_h):
                ih = oh * s_h + kh - pT
                if ih < 0 or ih >= H:
                    continue
                for kw in range(K_w):
                    iw = ow * s_w + kw - pL
                    if iw < 0 or iw >= W:
                        continue
                    out[oh, ow] += np.einsum("c,oc->o", a[ih, iw],
                                             w[:, kh, kw, :])
    return out


# ---- main -----------------------------------------------------------------

def main():
    import argparse
    systemc_dir = Path(__file__).resolve().parent.parent
    repo_root   = systemc_dir.parent
    DEFAULT_MODEL = str(repo_root / "model/INT8/efficientnet_lite0_int8.tflite")
    DEFAULT_OUT   = str(systemc_dir / "build/conv_layer.bin")

    ap = argparse.ArgumentParser()
    ap.add_argument("model",  nargs="?", default=DEFAULT_MODEL,
                    help="path to .tflite")
    ap.add_argument("output", nargs="?", default=DEFAULT_OUT,
                    help="binary blob output path")
    ap.add_argument("-l", "--layer", type=int, default=0,
                    help="conv layer index (0 = first; default 0)")
    ap.add_argument("--count", action="store_true",
                    help="just print number of CONV_2D ops in the model and exit")
    args = ap.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    interp = _load_interpreter(args.model)
    interp.allocate_tensors()

    if args.count:
        print(len(list_conv_ops(interp)))
        return

    op, in_t, wgt_t, out_t = conv_at(interp, args.layer)

    in_dtype  = in_t["dtype"]
    wgt_dtype = wgt_t["dtype"]
    # The hardware multiplies 8-bit operands; signed/unsigned at TFLite level is
    # just a labelling difference (different zero_point, same bit pattern).
    # We reinterpret uint8 as int8 for the test — the simulator and the numpy
    # reference both see identical bytes, so the comparison is bit-exact.
    if in_dtype not in (np.int8, np.uint8):
        raise SystemExit(f"only INT8/UINT8 input supported, got {in_dtype}")
    if wgt_dtype not in (np.int8, np.uint8):
        raise SystemExit(f"only INT8/UINT8 weights supported, got {wgt_dtype}")
    if in_dtype  == np.uint8: print("  (input is uint8 — reinterpreting as int8)")
    if wgt_dtype == np.uint8: print("  (weights are uint8 — reinterpreting as int8)")

    # Synthetic input (deterministic; we don't depend on bundled test inputs).
    in_shape = in_t["shape"]   # NHWC
    if in_shape[0] != 1:
        print(f"  (batch={in_shape[0]}, taking first sample)")
    H, W, Cin = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])

    rng = np.random.default_rng(0)
    in_i8 = rng.integers(-8, 8, size=(H, W, Cin), dtype=np.int8)

    # Weights from model — reinterpret if uint8.
    wgt = interp.get_tensor(wgt_t["index"])
    if wgt.dtype == np.uint8:
        wgt = wgt.view(np.int8)
    OC, Kh, Kw, _ = wgt.shape

    # Conv params from op options.
    opts = op.get("op_options") or op.get("options") or {}
    s_h = int(opts.get("stride_h", 1))
    s_w = int(opts.get("stride_w", 1))
    pad_kind = opts.get("padding", "SAME")        # 'SAME' | 'VALID'
    if pad_kind in ("SAME", b"SAME", 0):
        # SAME-padding (TF spec): output = ceil(in / s); pad symmetric.
        pT = max((Kh - 1) // 2, 0); pB = (Kh - 1) - pT
        pL = max((Kw - 1) // 2, 0); pR = (Kw - 1) - pL
    else:
        pT = pB = pL = pR = 0

    total_convs = len(list_conv_ops(interp))
    print(f"model       : {args.model}")
    print(f"layer       : {args.layer} / {total_convs - 1}  (CONV_2D)")
    print(f"shape       : in={H}x{W}x{Cin}  wgt={OC}x{Kh}x{Kw}x{Cin}  "
          f"s={s_h}x{s_w}  pad=({pT},{pB},{pL},{pR})")

    OH = (H + pT + pB - Kh) // s_h + 1
    OW = (W + pL + pR - Kw) // s_w + 1
    print(f"output      : {OH}x{OW}x{OC}  ({OH*OW*OC} elements, INT32)")

    ref = conv_int8_ref(in_i8, wgt, s_h, s_w, (pT, pB, pL, pR))

    # ---- write blob ----
    with open(args.output, "wb") as f:
        f.write(struct.pack("<II", 0x374C444D, 1))   # magic, version
        f.write(struct.pack("<HHH", H, W, Cin))
        f.write(struct.pack("<HHH", OH, OW, OC))
        f.write(struct.pack("<8B", Kh, Kw, s_h, s_w, pT, pB, pL, pR))
        f.write(in_i8.astype(np.int8).tobytes(order="C"))
        f.write(wgt   .astype(np.int8).tobytes(order="C"))
        f.write(ref   .astype(np.int32).tobytes(order="C"))
    print(f"wrote       : {args.output}  ({os.path.getsize(args.output)} bytes)")


if __name__ == "__main__":
    main()
