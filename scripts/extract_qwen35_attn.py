#!/usr/bin/env python3
"""
Extract BMM-SOFTMAX-BMM attention subgraph from qwen35.tflite.

Ops extracted (2nd attention block, FP32):
  8895  BMM(Q[8,128,256], K[8,256,128]) → scores[8,128,128]
  8896  MUL(scores, scale=0.0625)       → scaled[8,128,128]
  8897  RESHAPE(scaled → [1,8,128,128])
  8898  SELECT_V2(mask, fill=-3.4e38, scaled4d) → masked[1,8,128,128]
  8899  SOFTMAX(masked)                 → probs[1,8,128,128]
  8900  RESHAPE(probs → [8,128,128])
  8901  RESHAPE(V[1,8,128,256] → [8,128,256])
  8902  BMM(probs_r, V_r)               → attn_out[8,128,256]

Output: model/QWEN35/qwen35_attn_bmm_softmax_bmm.tflite
"""
from __future__ import annotations
import struct, mmap
from pathlib import Path
import numpy as np
from flatbuffers import builder as fb

SRC = Path("/Volumes/4T_OFFICE/_Claude/MDLA7_Claude/model/QWEN35/qwen35.tflite")
OUT = SRC.parent / "qwen35_attn_bmm_softmax_bmm.tflite"
FILE_ID = b"TFL3"

# TensorType
_FP32, _INT32, _BOOL = 0, 2, 6

# BuiltinOperator
_OP_MUL, _OP_SOFTMAX, _OP_RESHAPE, _OP_SELECT_V2, _OP_BMM = 18, 25, 22, 123, 126

# BuiltinOptions union discriminant
_OPT_MUL, _OPT_SOFTMAX, _OPT_RESHAPE, _OPT_SELECT_V2, _OPT_BMM = 21, 9, 17, 98, 101

# ── source model reader ────────────────────────────────────────────────────────

def _vt(d, o, fi):
    try:
        s = struct.unpack_from('<i',d,o)[0]; v = o-s
        sz = struct.unpack_from('<H',d,v)[0]; sl = 4+fi*2
        if sl >= sz: return None
        r = struct.unpack_from('<H',d,v+sl)[0]
        return None if r==0 else o+r
    except: return None

def _deref(d, f): return f + struct.unpack_from('<I',d,f)[0]

def _vec_offs(d, v):
    n = struct.unpack_from('<I',d,v)[0]
    if n > 200_000: return []
    return [v+4+i*4+struct.unpack_from('<I',d,v+4+i*4)[0] for i in range(n)]

def _buf_data(d, src_bufs, idx) -> bytes:
    if idx >= len(src_bufs): return b''
    buf = src_bufs[idx]
    f = _vt(d, buf, 0)
    if f is None: return b''
    off = _deref(d, f)
    n   = struct.unpack_from('<I',d,off)[0]
    return bytes(d[off+4:off+4+n])

# ── flatbuffers write ──────────────────────────────────────────────────────────
# RULE: all child objects must be built BEFORE StartObject/StartVector of parent.

def _bvec(b, raw: bytes) -> int:
    return b.CreateNumpyVector(np.frombuffer(raw, dtype=np.uint8))

def _i32v(b, vals) -> int:
    return b.CreateNumpyVector(np.array(vals, dtype=np.int32))

def _f32v(b, vals) -> int:
    return b.CreateNumpyVector(np.array(vals, dtype=np.float32))

def _i64v(b, vals) -> int:
    return b.CreateNumpyVector(np.array(vals, dtype=np.int64))

def _offv(b, offs: list[int]) -> int:
    b.StartVector(4, len(offs), 4)
    for o in reversed(offs):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()

def _quant(b, scale=1.0, zp=0) -> int:
    sv = _f32v(b, [scale])
    zv = _i64v(b, [zp])
    b.StartObject(7)
    b.PrependUOffsetTRelativeSlot(2, sv, 0)
    b.PrependUOffsetTRelativeSlot(3, zv, 0)
    return b.EndObject()

def _tensor(b, name, shape, ttype, buf_idx) -> int:
    shpv = _i32v(b, shape)
    sigv = _i32v(b, shape)
    nm   = b.CreateString(name)
    qp   = _quant(b)
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, shpv, 0)
    b.PrependUint8Slot(1, ttype, 0)
    b.PrependUint32Slot(2, buf_idx, 0)
    b.PrependUOffsetTRelativeSlot(3, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, qp, 0)
    b.PrependBoolSlot(8, True, False)
    b.PrependUOffsetTRelativeSlot(7, sigv, 0)
    return b.EndObject()

def _buffer(b, raw: bytes = b'') -> int:
    dv = _bvec(b, raw) if raw else None
    b.StartObject(3)
    if dv is not None:
        b.PrependUOffsetTRelativeSlot(0, dv, 0)
    return b.EndObject()

def _opcode(b, code: int) -> int:
    b.StartObject(4)
    b.PrependInt8Slot(0, min(code, 127), 0)
    b.PrependInt32Slot(2, 1, 0)
    b.PrependInt32Slot(3, code, 0)
    return b.EndObject()

def _opt_bmm(b, adj_y=False) -> int:
    b.StartObject(3)
    b.PrependBoolSlot(0, False, False)
    b.PrependBoolSlot(1, adj_y, False)
    b.PrependBoolSlot(2, False, False)
    return b.EndObject()

def _opt_mul(b) -> int:
    b.StartObject(1); b.PrependInt8Slot(0, 0, 0); return b.EndObject()

def _opt_reshape(b, shape) -> int:
    sv = _i32v(b, shape)
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, sv, 0)
    return b.EndObject()

def _opt_softmax(b, beta=1.0) -> int:
    b.StartObject(1); b.PrependFloat32Slot(0, beta, 0.0); return b.EndObject()

def _opt_select_v2(b) -> int:
    b.StartObject(0); return b.EndObject()

def _op(b, oci, ins, outs, opt_type, opt_off) -> int:
    iv = _i32v(b, ins)
    ov = _i32v(b, outs)
    b.StartObject(9)
    b.PrependUint32Slot(0, oci, 0)
    b.PrependUOffsetTRelativeSlot(1, iv, 0)
    b.PrependUOffsetTRelativeSlot(2, ov, 0)
    b.PrependUint8Slot(3, opt_type, 0)
    b.PrependUOffsetTRelativeSlot(4, opt_off, 0)
    return b.EndObject()

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {SRC.name}  ({SRC.stat().st_size//1024//1024} MB) ...")
    with open(SRC,'rb') as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    root = struct.unpack_from('<I', data, 0)[0]
    f4   = _vt(data, root, 4)
    src_bufs = _vec_offs(data, _deref(data, f4))

    f2   = _vt(data, root, 2)
    sg   = _vec_offs(data, _deref(data, f2))[0]
    src_tensors = _vec_offs(data, _deref(data, _vt(data, sg, 0)))

    def buf_idx(tid):
        t = src_tensors[tid]; f = _vt(data, t, 2)
        return struct.unpack_from('<I', data, f)[0] if f else 0

    # read constant buffers
    scale_raw = _buf_data(data, src_bufs, buf_idx(436))
    rshp1_raw = _buf_data(data, src_bufs, buf_idx(411))
    mask_raw  = _buf_data(data, src_bufs, buf_idx(4878))
    fill_raw  = _buf_data(data, src_bufs, buf_idx(434))
    rshp2_raw = _buf_data(data, src_bufs, buf_idx(408))
    rshp3_raw = _buf_data(data, src_bufs, buf_idx(413))

    print(f"  scale  : {np.frombuffer(scale_raw, np.float32)}")
    print(f"  rshp1  : {np.frombuffer(rshp1_raw, np.int32)}")
    print(f"  fill   : {np.frombuffer(fill_raw,  np.float32)}")
    print(f"  rshp2  : {np.frombuffer(rshp2_raw, np.int32)}")
    print(f"  rshp3  : {np.frombuffer(rshp3_raw, np.int32)}")

    # mask: if empty in source (dynamic), synthesize causal upper-tri mask
    if not mask_raw:
        print(f"  mask   : (not in buffer → synthesise upper-tri causal mask)")
        m = np.triu(np.ones((128,128), dtype=bool), k=1)   # True = future = fill
        m = m[np.newaxis, np.newaxis, :, :]                # [1,1,128,128]
        mask_raw = m.tobytes()
    else:
        print(f"  mask   : {len(mask_raw)} bytes")

    data.close()

    rshp1 = list(np.frombuffer(rshp1_raw, np.int32))
    rshp2 = list(np.frombuffer(rshp2_raw, np.int32))
    rshp3 = list(np.frombuffer(rshp3_raw, np.int32))

    # ── build new TFLite ───────────────────────────────────────────────────────
    # Tensor index map:
    #  0=q  1=k  2=v  (inputs)
    #  3=scale  4=rshp1  5=mask  6=fill  7=rshp2  8=rshp3  (constants)
    #  9=scores 10=scaled 11=reshaped 12=masked 13=probs
    # 14=probs_r 15=V_r 16=attn_out
    # Buffer index = tensor index + 1  (0 = sentinel)

    B = fb.Builder(1 << 23)  # 8 MB

    # tensors (build in reverse order or any order — just must be before subgraph)
    T = [None] * 17
    T[16] = _tensor(B,"attn_out",  [8,128,256],   _FP32, 17)
    T[15] = _tensor(B,"V_r",       [8,128,256],   _FP32, 16)
    T[14] = _tensor(B,"probs_r",   [8,128,128],   _FP32, 15)
    T[13] = _tensor(B,"probs",     [1,8,128,128], _FP32, 14)
    T[12] = _tensor(B,"masked",    [1,8,128,128], _FP32, 13)
    T[11] = _tensor(B,"reshaped",  [1,8,128,128], _FP32, 12)
    T[10] = _tensor(B,"scaled",    [8,128,128],   _FP32, 11)
    T[9]  = _tensor(B,"scores",    [8,128,128],   _FP32, 10)
    T[8]  = _tensor(B,"rshp3",     [3],           _INT32, 9)
    T[7]  = _tensor(B,"rshp2",     [3],           _INT32, 8)
    T[6]  = _tensor(B,"fill",      [],            _FP32,  7)
    T[5]  = _tensor(B,"mask",      [1,1,128,128], _BOOL,  6)
    T[4]  = _tensor(B,"rshp1",     [4],           _INT32, 5)
    T[3]  = _tensor(B,"scale",     [],            _FP32,  4)
    T[2]  = _tensor(B,"v",         [1,8,128,256], _FP32,  3)
    T[1]  = _tensor(B,"k",         [8,256,128],   _FP32,  2)
    T[0]  = _tensor(B,"q",         [8,128,256],   _FP32,  1)
    tvec  = _offv(B, T)

    # opcodes  0=BMM 1=MUL 2=RESHAPE 3=SELECT_V2 4=SOFTMAX
    OC = _offv(B, [
        _opcode(B, _OP_BMM),
        _opcode(B, _OP_MUL),
        _opcode(B, _OP_RESHAPE),
        _opcode(B, _OP_SELECT_V2),
        _opcode(B, _OP_SOFTMAX),
    ])

    # ops (built in reverse execution order so flatbuffers offsets are correct)
    o7 = _op(B,0,[14,15],[16], _OPT_BMM,      _opt_bmm(B))
    o6 = _op(B,2,[2,8],  [15], _OPT_RESHAPE,  _opt_reshape(B,rshp3))
    o5 = _op(B,2,[13,7], [14], _OPT_RESHAPE,  _opt_reshape(B,rshp2))
    o4 = _op(B,4,[12],   [13], _OPT_SOFTMAX,  _opt_softmax(B))
    o3 = _op(B,3,[5,6,11],[12],_OPT_SELECT_V2,_opt_select_v2(B))
    o2 = _op(B,2,[10,4], [11], _OPT_RESHAPE,  _opt_reshape(B,rshp1))
    o1 = _op(B,1,[9,3],  [10], _OPT_MUL,      _opt_mul(B))
    o0 = _op(B,0,[0,1],  [9],  _OPT_BMM,      _opt_bmm(B))
    opvec = _offv(B, [o0,o1,o2,o3,o4,o5,o6,o7])

    invec  = _i32v(B, [0,1,2])
    outvec = _i32v(B, [16])
    sgname = B.CreateString("qwen35_attn_bmm_softmax_bmm")

    B.StartObject(5)
    B.PrependUOffsetTRelativeSlot(0, tvec,   0)
    B.PrependUOffsetTRelativeSlot(1, invec,  0)
    B.PrependUOffsetTRelativeSlot(2, outvec, 0)
    B.PrependUOffsetTRelativeSlot(3, opvec,  0)
    B.PrependUOffsetTRelativeSlot(4, sgname, 0)
    sg_off = B.EndObject()
    sgvec  = _offv(B, [sg_off])

    # buffers (buf 0=sentinel, 1-3=empty inputs, 4-9=constants, 10-17=empty intermediates)
    bufs = _offv(B, [
        _buffer(B),             # 0
        _buffer(B),             # 1 q
        _buffer(B),             # 2 k
        _buffer(B),             # 3 v
        _buffer(B, scale_raw),  # 4
        _buffer(B, rshp1_raw),  # 5
        _buffer(B, mask_raw),   # 6
        _buffer(B, fill_raw),   # 7
        _buffer(B, rshp2_raw),  # 8
        _buffer(B, rshp3_raw),  # 9
        _buffer(B),             # 10-17 intermediates
        _buffer(B), _buffer(B), _buffer(B), _buffer(B),
        _buffer(B), _buffer(B), _buffer(B),
    ])

    desc = B.CreateString(
        "Qwen3.5 attn BMM-SOFTMAX-BMM  Q[8,128,256] K[8,256,128] V[1,8,128,256] → [8,128,256]"
    )
    B.StartObject(8)
    B.PrependUint32Slot(0, 3, 0)
    B.PrependUOffsetTRelativeSlot(1, OC,   0)
    B.PrependUOffsetTRelativeSlot(2, sgvec,0)
    B.PrependUOffsetTRelativeSlot(3, desc, 0)
    B.PrependUOffsetTRelativeSlot(4, bufs, 0)
    model = B.EndObject()

    B.Finish(model)
    raw = bytes(B.Output())
    root_off = struct.unpack_from('<I', raw, 0)[0]
    result = struct.pack('<I', root_off+4) + FILE_ID + raw[4:]

    OUT.write_bytes(result)
    sz = len(result)
    print(f"\nWritten: {OUT.name}  ({sz:,} bytes  /  {sz//1024} KB)")
    print("\nGraph summary:")
    print("  Inputs : q[8,128,256]  k[8,256,128]  v[1,8,128,256]  FP32")
    print("  Ops    : BMM → MUL(×0.0625) → RESHAPE → SELECT_V2(causal_mask)")
    print("           → SOFTMAX → RESHAPE → RESHAPE(V) → BMM")
    print("  Output : attn_out[8,128,256]  FP32")


if __name__ == "__main__":
    main()
