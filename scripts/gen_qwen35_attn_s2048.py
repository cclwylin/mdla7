#!/usr/bin/env python3
"""
Generate Qwen3.5-style attention TFLite for S=2048 prefill.

Architecture (H=8, S=2048, D=256, FP16):
  Q [1,8,2048,256]  K [1,8,2048,256]  V [1,8,2048,256]

  scores = BMM(Q, K, adj_y=True)        [1,8,2048,2048]
  scaled = MUL(scores, 0.0625)          [1,8,2048,2048]   (1/√256)
  masked = SELECT_V2(mask, fill, scaled) [1,8,2048,2048]   (causal mask)
  probs  = SOFTMAX(masked)              [1,8,2048,2048]
  out    = BMM(probs, V)                [1,8,2048,256]

Timing @ 1.9 GHz FP16 (4,096 MACs/cycle):
  MACs = 2 × 8 × 2048² × 256 = 17.18 G
  @100% util : 17.18G / 4096 / 1.9G ≈ 2.21 ms
  @80%  util : ≈ 2.76 ms   ← target 2-5 ms ✓

Output: model/QWEN35/qwen35_attn_s2048_fp16.tflite
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
from flatbuffers import builder as fb

OUT = Path("/Volumes/4T_OFFICE/_Claude/MDLA7_Claude/model/QWEN35/qwen35_attn_s2048_fp16.tflite")
FILE_ID = b"TFL3"

H, S, D = 8, 2048, 256

# ── TFLite schema constants ────────────────────────────────────────────────────
_FP16, _INT32, _BOOL = 1, 2, 6

_OP_MUL, _OP_SOFTMAX, _OP_SELECT_V2, _OP_BMM = 18, 25, 123, 126
_OPT_MUL, _OPT_SOFTMAX, _OPT_SELECT_V2, _OPT_BMM = 21, 9, 98, 101

# ── flatbuffers helpers (all children built BEFORE parent StartObject) ─────────

def _i32v(b, v): return b.CreateNumpyVector(np.array(v, dtype=np.int32))
def _i64v(b, v): return b.CreateNumpyVector(np.array(v, dtype=np.int64))
def _f32v(b, v): return b.CreateNumpyVector(np.array(v, dtype=np.float32))
def _u8v(b, raw): return b.CreateNumpyVector(np.frombuffer(raw, dtype=np.uint8))

def _offv(b, offs):
    b.StartVector(4, len(offs), 4)
    for o in reversed(offs): b.PrependUOffsetTRelative(o)
    return b.EndVector()

def _quant(b, scale=1.0, zp=0):
    sv = _f32v(b, [scale]); zv = _i64v(b, [zp])
    b.StartObject(7)
    b.PrependUOffsetTRelativeSlot(2, sv, 0)
    b.PrependUOffsetTRelativeSlot(3, zv, 0)
    return b.EndObject()

def _tensor(b, name, shape, ttype, buf_idx):
    shpv = _i32v(b, shape); sigv = _i32v(b, shape)
    nm = b.CreateString(name); qp = _quant(b)
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, shpv, 0)
    b.PrependUint8Slot(1, ttype, 0)
    b.PrependUint32Slot(2, buf_idx, 0)
    b.PrependUOffsetTRelativeSlot(3, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, qp, 0)
    b.PrependBoolSlot(8, True, False)
    b.PrependUOffsetTRelativeSlot(7, sigv, 0)
    return b.EndObject()

def _buffer(b, raw: bytes = b''):
    dv = _u8v(b, raw) if raw else None
    b.StartObject(3)
    if dv is not None: b.PrependUOffsetTRelativeSlot(0, dv, 0)
    return b.EndObject()

def _opcode(b, code):
    b.StartObject(4)
    b.PrependInt8Slot(0, min(code, 127), 0)
    b.PrependInt32Slot(2, 1, 0)
    b.PrependInt32Slot(3, code, 0)
    return b.EndObject()

def _opt_bmm(b, adj_y=False):
    b.StartObject(3)
    b.PrependBoolSlot(0, False, False)
    b.PrependBoolSlot(1, adj_y, False)
    b.PrependBoolSlot(2, False, False)
    return b.EndObject()

def _opt_mul(b):
    b.StartObject(1); b.PrependInt8Slot(0, 0, 0); return b.EndObject()

def _opt_softmax(b):
    b.StartObject(1); b.PrependFloat32Slot(0, 1.0, 0.0); return b.EndObject()

def _opt_select_v2(b):
    b.StartObject(0); return b.EndObject()

def _op(b, oci, ins, outs, opt_type, opt_off):
    iv = _i32v(b, ins); ov = _i32v(b, outs)
    b.StartObject(9)
    b.PrependUint32Slot(0, oci, 0)
    b.PrependUOffsetTRelativeSlot(1, iv, 0)
    b.PrependUOffsetTRelativeSlot(2, ov, 0)
    b.PrependUint8Slot(3, opt_type, 0)
    b.PrependUOffsetTRelativeSlot(4, opt_off, 0)
    return b.EndObject()

# ── constant data ──────────────────────────────────────────────────────────────

def make_scale_fp16():
    return np.array([1.0 / np.sqrt(D)], dtype=np.float16).tobytes()   # 0.0625

def make_fill_fp16():
    return np.array([np.finfo(np.float16).min], dtype=np.float16).tobytes()  # -65504

def make_causal_mask():
    """Upper-triangular True = future positions (will be filled with -inf)."""
    m = np.triu(np.ones((S, S), dtype=bool), k=1)
    return m[np.newaxis, np.newaxis].tobytes()   # [1,1,S,S] BOOL

# ── build ──────────────────────────────────────────────────────────────────────

def main():
    scale_raw = make_scale_fp16()
    fill_raw  = make_fill_fp16()

    print(f"Generating causal mask [{1},{1},{S},{S}] = {S*S//1024} KB ...")
    mask_raw  = make_causal_mask()

    macs = 2 * H * S * S * D
    cyc  = macs / 4096
    print(f"Shape  : H={H}  S={S}  D={D}  FP16")
    print(f"MACs   : {macs/1e9:.2f} G")
    print(f"@1.9GHz: {cyc/1.9e9*1000:.2f} ms @100%  /  {cyc/1.9e9/0.8*1000:.2f} ms @80%")

    # Tensor layout:
    #  0=q  1=k  2=v                       inputs
    #  3=scale  4=mask  5=fill              constants
    #  6=scores 7=scaled 8=masked 9=probs  intermediates
    # 10=out                                output
    # Buffer index = tensor index + 1  (buf 0 = sentinel)

    B = fb.Builder(1 << 23)   # start 8 MB, grows as needed

    T = [None] * 11
    T[10] = _tensor(B, "out",    [1,H,S,D], _FP16, 11)
    T[9]  = _tensor(B, "probs",  [1,H,S,S], _FP16, 10)
    T[8]  = _tensor(B, "masked", [1,H,S,S], _FP16,  9)
    T[7]  = _tensor(B, "scaled", [1,H,S,S], _FP16,  8)
    T[6]  = _tensor(B, "scores", [1,H,S,S], _FP16,  7)
    T[5]  = _tensor(B, "fill",   [],        _FP16,  6)
    T[4]  = _tensor(B, "mask",   [1,1,S,S], _BOOL,  5)
    T[3]  = _tensor(B, "scale",  [],        _FP16,  4)
    T[2]  = _tensor(B, "v",      [1,H,S,D], _FP16,  3)
    T[1]  = _tensor(B, "k",      [1,H,S,D], _FP16,  2)
    T[0]  = _tensor(B, "q",      [1,H,S,D], _FP16,  1)
    tvec  = _offv(B, T)

    # opcodes: 0=BMM  1=MUL  2=SELECT_V2  3=SOFTMAX
    OC = _offv(B, [
        _opcode(B, _OP_BMM),
        _opcode(B, _OP_MUL),
        _opcode(B, _OP_SELECT_V2),
        _opcode(B, _OP_SOFTMAX),
    ])

    # ops (build in reverse order for flatbuffers)
    o4 = _op(B, 0, [9,2],    [10], _OPT_BMM,       _opt_bmm(B))
    o3 = _op(B, 3, [8],      [9],  _OPT_SOFTMAX,   _opt_softmax(B))
    o2 = _op(B, 2, [4,5,7],  [8],  _OPT_SELECT_V2, _opt_select_v2(B))
    o1 = _op(B, 1, [6,3],    [7],  _OPT_MUL,       _opt_mul(B))
    o0 = _op(B, 0, [0,1],    [6],  _OPT_BMM,       _opt_bmm(B, adj_y=True))
    opvec = _offv(B, [o0,o1,o2,o3,o4])

    invec  = _i32v(B, [0,1,2])
    outvec = _i32v(B, [10])
    sgname = B.CreateString(f"qwen35_attn_s{S}_fp16")
    B.StartObject(5)
    B.PrependUOffsetTRelativeSlot(0, tvec,   0)
    B.PrependUOffsetTRelativeSlot(1, invec,  0)
    B.PrependUOffsetTRelativeSlot(2, outvec, 0)
    B.PrependUOffsetTRelativeSlot(3, opvec,  0)
    B.PrependUOffsetTRelativeSlot(4, sgname, 0)
    sg_off = B.EndObject()
    sgvec  = _offv(B, [sg_off])

    # buffers: 0=sentinel, 1-3=empty inputs, 4=scale, 5=mask, 6=fill,
    #          7-11=empty intermediates+output
    bufs = _offv(B, [
        _buffer(B),              # 0  sentinel
        _buffer(B),              # 1  q
        _buffer(B),              # 2  k
        _buffer(B),              # 3  v
        _buffer(B, scale_raw),   # 4  scale (FP16 scalar)
        _buffer(B, mask_raw),    # 5  causal mask
        _buffer(B, fill_raw),    # 6  fill=-65504 (FP16 min)
        _buffer(B),              # 7  scores
        _buffer(B),              # 8  scaled
        _buffer(B),              # 9  masked
        _buffer(B),              # 10 probs
        _buffer(B),              # 11 out
    ])

    desc = B.CreateString(
        f"Qwen3.5 prefill attention FP16  H={H} S={S} D={D}  "
        f"~{macs/1e9:.1f}G MACs  ~2.7ms@1.9GHz"
    )
    B.StartObject(8)
    B.PrependUint32Slot(0, 3, 0)
    B.PrependUOffsetTRelativeSlot(1, OC,    0)
    B.PrependUOffsetTRelativeSlot(2, sgvec, 0)
    B.PrependUOffsetTRelativeSlot(3, desc,  0)
    B.PrependUOffsetTRelativeSlot(4, bufs,  0)
    model = B.EndObject()

    B.Finish(model)
    raw = bytes(B.Output())
    root_off = struct.unpack_from('<I', raw, 0)[0]
    result   = struct.pack('<I', root_off + 4) + FILE_ID + raw[4:]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(result)
    sz = len(result)
    print(f"\nWritten: {OUT.name}  ({sz:,} bytes  /  {sz//1024} KB)")
    print("\nGraph:")
    print(f"  Inputs : q,k,v  [1,{H},{S},{D}]  FP16")
    print(f"  op0    : BMM(q, k, adj_y=True)  → scores  [1,{H},{S},{S}]")
    print(f"  op1    : MUL(scores, 0.0625)    → scaled   [1,{H},{S},{S}]")
    print(f"  op2    : SELECT_V2(mask,fill,scaled) → masked [1,{H},{S},{S}]")
    print(f"  op3    : SOFTMAX                → probs   [1,{H},{S},{S}]")
    print(f"  op4    : BMM(probs, v)          → out     [1,{H},{S},{D}]")
    print(f"  Output : out  [1,{H},{S},{D}]  FP16")


if __name__ == "__main__":
    main()
