#!/usr/bin/env python3
"""Validate compile_model.py's reference against the real TFLite Interpreter.

Two modes:
  default     — single conv layer (--layer N), bit-exact compare for layer 0.
  --all-layers — sweep every conv-class op and report a per-layer fidelity table.

Important caveats for --all-layers:
  - TFLite is fed the synth that matches compile_model's first rng draw, so
    layer 0 sees the same input in both paths and is bit-exact comparable.
  - Layers N>0 use *chained* TFLite output as input, but compile_model still
    synthesises a fresh per-layer rng draw — the inputs differ, so a non-zero
    diff there is expected and does not indicate a bug. We still report the
    diff because shape/range mismatches are still meaningful smoke checks.

Usage:
  ./validate_tflite.py                       # single layer, default model
  ./validate_tflite.py --layer 3             # single layer 3
  ./validate_tflite.py --all-layers          # sweep all conv layers
  ./validate_tflite.py path/to/model.tflite --all-layers
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compile_model as cm    # reuse helpers


CONV_LIKE_OPS = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")


def make_interp(model_path: str):
    """TFLite Interpreter w/ all intermediate tensors preserved."""
    try:
        import tflite_runtime.interpreter as tflr
        return tflr.Interpreter(model_path=model_path,
                                experimental_preserve_all_tensors=True)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path,
                                   experimental_preserve_all_tensors=True)


def list_conv_ops(interp):
    fn = interp._get_ops_details if hasattr(interp, "_get_ops_details") \
         else interp.get_ops_details
    return [op for op in fn() if op["op_name"] in CONV_LIKE_OPS]


def list_compile_conv_ops(fb, model, sg):
    """Same set as compile_model.py iterates (same ordering)."""
    out = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        name = cm._opcode_name(fb, model, op)
        if name in CONV_LIKE_OPS:
            out.append((name, op))
    return out


def run_tflite_capture(model_path: str, in_data: np.ndarray):
    """Run TFLite once with the given input; return list of conv-op outputs in op order.

    For uint8 outputs we subtract 128 to match compile_model's int8 representation
    (sim stores int8 bytes; tflite_uint8 - 128 == sim_int8)."""
    interp = make_interp(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    if inp["shape"][0] == 1 and in_data.ndim == 3:
        in_data = in_data[None, ...]
    interp.set_tensor(inp["index"], in_data.astype(inp["dtype"]))
    interp.invoke()

    convs = list_conv_ops(interp)
    outs = []
    for op in convs:
        out_idx = op["outputs"][0]
        t = interp.get_tensor(out_idx)
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.dtype == np.uint8:
            t = (t.astype(np.int16) - 128).astype(np.int8)
        outs.append((op["op_name"], t))
    return outs


def parse_layer_meta(buf: bytes, idx: int):
    """Return (op_kind, ref_size, ref_off, out_h, out_w, out_c) for layer idx.

    Layout: see compile_model.py LAYER_FMT.
    Field offsets within the 64-byte meta:
      - out_h @ 6  (uint16)
      - out_w @ 8  (uint16)
      - out_c @ 10 (uint16)
      - ref_size @ 40 (uint32, 5th uint32 after the 8-byte pads)
      - ref_off  @ 52 (uint32, 8th uint32)
      - group    @ 56 (uint16)
      - op_kind  @ 58 (uint16)
      - dtype    @ 60 (uint16)
    """
    meta_off = 16 + 64 * idx
    out_h    = struct.unpack_from("<H", buf, meta_off + 6)[0]
    out_w    = struct.unpack_from("<H", buf, meta_off + 8)[0]
    out_c    = struct.unpack_from("<H", buf, meta_off + 10)[0]
    ref_size = struct.unpack_from("<I", buf, meta_off + 40)[0]
    ref_off  = struct.unpack_from("<I", buf, meta_off + 52)[0]
    op_kind  = struct.unpack_from("<H", buf, meta_off + 58)[0]
    return op_kind, ref_size, ref_off, out_h, out_w, out_c


def compile_model_full(model_path: str, prog_path: str):
    r = subprocess.run(
        [sys.executable, str(HERE / "compile_model.py"), model_path, prog_path],
        capture_output=True, text=True,
    )
    if r.returncode:
        sys.exit("compile_model failed:\n" + r.stderr)


def synth_for_layer0(in_t, fb):
    """Match compile_model.py's first rng draw for the layer-0 input shape."""
    rng = np.random.default_rng(0)
    H = int(in_t.Shape(1)); W = int(in_t.Shape(2)); Cin = int(in_t.Shape(3))
    is_int16 = in_t.Type() == fb.TensorType.INT16
    if is_int16:
        return rng.integers(-128, 128, size=(H, W, Cin), dtype=np.int16), is_int16
    return rng.integers(-8, 8, size=(H, W, Cin), dtype=np.int8), is_int16


def synth_to_tflite_input(synth, model_path):
    """Convert the int8 synth to whatever dtype the TFLite primary input wants."""
    try:
        import tensorflow as tf
        det = tf.lite.Interpreter(model_path=model_path).get_input_details()[0]
    except Exception:
        return synth
    if det["dtype"] == np.uint8:
        return (synth.astype(np.int16) + 128).astype(np.uint8)
    return synth.astype(det["dtype"])


def compare_layer(name, idx, tf_out, ours, is_layer0):
    if tf_out.shape != ours.shape:
        return f"FAIL  shape {tf_out.shape} vs {ours.shape}"
    diff = np.abs(tf_out.astype(np.int32) - ours.astype(np.int32))
    nmis = int((diff != 0).sum())
    if nmis == 0:
        return f"PASS  bit-exact ({tf_out.size} elem)"
    tag = "FAIL" if is_layer0 else "diff"   # N>0 differing is expected (chained input)
    return (f"{tag}  {nmis}/{tf_out.size} differ  "
            f"max={diff.max()} mean={diff.mean():.2f}")


def run_all_layers(model_path: str):
    fb, model, sg = cm._load_flatbuffer(model_path)
    convs = list_compile_conv_ops(fb, model, sg)
    if not convs:
        sys.exit("no CONV/DWCONV/FC ops in model")

    # Synth that matches compile_model layer 0.
    name0, op0 = convs[0]
    in_t0 = sg.Tensors(op0.Inputs(0))
    synth, is_int16 = synth_for_layer0(in_t0, fb)
    tf_input = synth_to_tflite_input(synth, model_path)

    print(f"model:  {model_path}")
    print(f"layers: {len(convs)} conv-class ops "
          f"({'INT16' if is_int16 else 'INT8'} layer-0 input)")

    print("running TFLite (preserve_all_tensors) ...")
    tf_outs = run_tflite_capture(model_path, tf_input)

    if len(tf_outs) != len(convs):
        sys.exit(f"op count mismatch: tflite={len(tf_outs)} compile={len(convs)}")

    # Compile once.
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_:
        prog = tf_.name
    try:
        print("running compile_model ...")
        compile_model_full(model_path, prog)
        with open(prog, "rb") as f:
            buf = f.read()
    finally:
        # We keep `prog` around for parsing then delete at the very end.
        pass

    n_layers = struct.unpack_from("<I", buf, 8)[0]
    if n_layers != len(convs):
        # compile_model also walks non-conv ops (POOL/SOFTMAX/RESHAPE) — find
        # the conv-class layers by op_kind in program.bin.
        pass

    # Build a list of (program.bin layer index) for each conv-class layer.
    OP_CONV, OP_DWCONV, OP_FC = 0, 1, 6
    conv_layer_idx = []
    for i in range(n_layers):
        op_kind, *_ = parse_layer_meta(buf, i)
        if op_kind in (OP_CONV, OP_DWCONV, OP_FC):
            conv_layer_idx.append(i)

    if len(conv_layer_idx) != len(tf_outs):
        sys.exit(f"conv-layer count mismatch: tflite={len(tf_outs)}, "
                 f"compile_model={len(conv_layer_idx)}")

    print("\nlayer  op           shape (tflite)        result")
    print("-" * 72)

    pass_n = fail_n = diff_n = 0
    for k, (idx, (op_name, tf_out)) in enumerate(zip(conv_layer_idx, tf_outs)):
        op_kind, ref_size, ref_off, oh, ow, oc = parse_layer_meta(buf, idx)
        ours_dtype = np.int16 if is_int16 else np.int8   # crude — see compile_model
        ref_bytes  = buf[ref_off : ref_off + ref_size]
        try:
            ours = np.frombuffer(ref_bytes, dtype=ours_dtype).reshape(tf_out.shape)
        except ValueError as e:
            print(f"  L{k:>2}  {op_name:<14}  reshape err: {e}")
            fail_n += 1
            continue
        result = compare_layer(op_name, k, tf_out, ours, is_layer0=(k == 0))
        flag = "*" if k == 0 else " "
        sh = "x".join(str(s) for s in tf_out.shape)
        print(f"  L{k:>2}{flag} {op_name:<14}  {sh:<20}  {result}")

        if "PASS" in result:
            pass_n += 1
        elif result.startswith("FAIL"):
            fail_n += 1
        else:
            diff_n += 1

    os.unlink(prog)

    print("-" * 72)
    print(f"  layer 0 (bit-exact comparable): "
          f"{'PASS' if pass_n >= 1 else 'FAIL'}")
    print(f"  layers >0 (chained-input — expect non-zero diff): "
          f"diff={diff_n}, pass={max(pass_n - 1, 0)}, fail={fail_n}")
    print("  * = layer 0 (only layer with matched inputs in current setup)")


def run_single_layer(model_path: str, layer: int):
    fb, model, sg = cm._load_flatbuffer(model_path)
    convs = list_compile_conv_ops(fb, model, sg)
    if layer >= len(convs):
        sys.exit(f"only {len(convs)} conv ops in model")

    name, op = convs[layer]
    in_t = sg.Tensors(op.Inputs(0))
    H = int(in_t.Shape(1)); W = int(in_t.Shape(2)); Cin = int(in_t.Shape(3))

    rng = np.random.default_rng(0)
    is_int16 = in_t.Type() == fb.TensorType.INT16
    if is_int16:
        synth = rng.integers(-128, 128, size=(H, W, Cin), dtype=np.int16)
    else:
        synth = rng.integers(-8, 8, size=(H, W, Cin), dtype=np.int8)
    tf_input = synth_to_tflite_input(synth, model_path)

    print(f"model:  {model_path}")
    print(f"layer {layer}: {name}  in={H}x{W}x{Cin}")
    print(f"input dtype: {in_t.Type()}  is_int16={is_int16}")

    interp = make_interp(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    if inp["shape"][0] == 1 and tf_input.ndim == 3:
        tf_input = tf_input[None, ...]
    interp.set_tensor(inp["index"], tf_input.astype(inp["dtype"]))
    interp.invoke()
    convs_runtime = list_conv_ops(interp)
    tf_out = interp.get_tensor(convs_runtime[layer]["outputs"][0])
    if tf_out.ndim == 4 and tf_out.shape[0] == 1:
        tf_out = tf_out[0]
    if tf_out.dtype == np.uint8:
        tf_out = (tf_out.astype(np.int16) - 128).astype(np.int8)
    print(f"tflite out: shape={tf_out.shape}  dtype={tf_out.dtype}  "
          f"min/max={tf_out.min()}/{tf_out.max()}")

    print("running compile_model on the same layer ...")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_:
        prog = tf_.name
    try:
        r = subprocess.run(
            [sys.executable, str(HERE / "compile_model.py"),
             model_path, prog,
             "--max-layers", str(layer + 1)],
            capture_output=True, text=True,
        )
        if r.returncode:
            sys.exit("compile_model failed:\n" + r.stderr)
        with open(prog, "rb") as f:
            buf = f.read()
    finally:
        os.unlink(prog) if os.path.exists(prog) else None

    # Find the layer-th conv in program.bin (skipping pools/etc).
    OP_CONV, OP_DWCONV, OP_FC = 0, 1, 6
    n_layers = struct.unpack_from("<I", buf, 8)[0]
    seen = -1
    bin_idx = None
    for i in range(n_layers):
        op_kind, *_ = parse_layer_meta(buf, i)
        if op_kind in (OP_CONV, OP_DWCONV, OP_FC):
            seen += 1
            if seen == layer:
                bin_idx = i
                break
    if bin_idx is None:
        sys.exit("could not locate layer in program.bin")

    op_kind, ref_size, ref_off, oh, ow, oc = parse_layer_meta(buf, bin_idx)
    ours_dtype = np.int16 if is_int16 else np.int8
    ref_bytes = buf[ref_off : ref_off + ref_size]
    ours = np.frombuffer(ref_bytes, dtype=ours_dtype).reshape(tf_out.shape)

    if tf_out.shape != ours.shape:
        sys.exit(f"shape mismatch: tflite={tf_out.shape}  ours={ours.shape}")
    diff = np.abs(tf_out.astype(np.int32) - ours.astype(np.int32))
    nmis = int((diff != 0).sum())
    if nmis == 0:
        print(f"\n  PASS  bit-exact ({tf_out.size} elements)")
        return
    print(f"\n  FAIL  {nmis}/{tf_out.size} elements differ")
    print(f"  max |diff| = {diff.max()}")
    print(f"  mean|diff| = {diff.mean():.2f}")
    sample = np.flatnonzero(diff.ravel())[:5]
    for k in sample:
        print(f"    [{k}] tflite={tf_out.ravel()[k]}  ours={ours.ravel()[k]}  "
              f"|d|={diff.ravel()[k]}")


def main():
    ap = argparse.ArgumentParser()
    repo = HERE.parent.parent
    ap.add_argument("model", nargs="?",
                    default=str(repo / "model/INT8/efficientnet_lite0_int8.tflite"))
    ap.add_argument("--layer", type=int, default=0,
                    help="conv layer index to validate (default 0)")
    ap.add_argument("--all-layers", action="store_true",
                    help="sweep every conv-class op (faster than per-layer loop)")
    args = ap.parse_args()

    if args.all_layers:
        run_all_layers(args.model)
    else:
        run_single_layer(args.model, args.layer)


if __name__ == "__main__":
    main()
