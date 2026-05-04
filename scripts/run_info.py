"""
run_info.py — TFLite 模型結構分析工具
顯示：OP類型、Filter、Stride、Padding、輸入/輸出 Shape

用法：
  python run_info.py alexnet
  python run_info.py mobilenet_v1
  python run_info.py              # 列出所有模型
"""
import sys, os, glob, struct
import numpy as np

# ── 工具 ──────────────────────────────────────────────────────
def fmt(n):
    if n>=1e9: return f"{n/1e9:.2f}G"
    if n>=1e6: return f"{n/1e6:.2f}M"
    if n>=1e3: return f"{n/1e3:.1f}K"
    return str(n)

def sh(shape): return "[" + ",".join(str(d) for d in shape) + "]"

def dtype_bits(d):
    for k,v in {'float32':32,'float16':16,'int8':8,'uint8':8,'int16':16,'int32':32}.items():
        if k in str(d).lower(): return v
    return 32

# ── TFLite FlatBuffer 解析器 ───────────────────────────────────
class TFLiteFB:
    """
    手動解析 TFLite FlatBuffer，取出每個 OP 的 builtin_options
    (stride_h, stride_w, padding, filter_h, filter_w)。
    
    TFLite schema 中 BuiltinOptions union type IDs:
      Conv2DOptions      = 1   (fields: padding,stride_w,stride_h,activation,dil_w,dil_h)
      DepthwiseConv2D    = 2   (fields: padding,stride_w,stride_h,depth_multiplier,activation)
      Pool2DOptions      = 5   (fields: padding,stride_w,stride_h,filter_w,filter_h,activation)
      FullyConnected     = 8
      BatchMatMul        = 93
    
    Padding enum: 0=SAME, 1=VALID
    """
    PAD = {0: 'SAME', 1: 'VALID'}

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.b = bytearray(f.read())

    def i32(self, o): return struct.unpack_from('<i', self.b, o)[0]
    def u32(self, o): return struct.unpack_from('<I', self.b, o)[0]
    def u16(self, o): return struct.unpack_from('<H', self.b, o)[0]
    def i8(self, o):  return struct.unpack_from('<b', self.b, o)[0]

    def field(self, tbl, fid):
        """Absolute offset of field fid in table, or 0 if absent."""
        vt  = tbl - self.i32(tbl)
        vsz = self.u16(vt)
        pos = 4 + fid * 2
        if pos >= vsz: return 0
        off = self.u16(vt + pos)
        return (tbl + off) if off else 0

    def ref(self, off):
        return off + self.u32(off)

    def vec_len(self, vf): return self.u32(self.ref(vf))

    def vec_elem(self, vf, i):
        vs = self.ref(vf)
        eo = vs + 4 + i * 4
        return eo + self.u32(eo)

    def gi32(self, tbl, fid, default=0):
        o = self.field(tbl, fid); return self.i32(o) if o else default

    def gi8(self, tbl, fid, default=0):
        o = self.field(tbl, fid); return self.i8(o) if o else default

    def parse_ops(self):
        root    = self.ref(0)              # Model
        sg_f    = self.field(root, 2)      # Model.subgraphs
        if not sg_f: return []
        sg      = self.vec_elem(sg_f, 0)  # SubGraph[0]
        ops_f   = self.field(sg, 3)        # SubGraph.operators
        if not ops_f: return []

        results = []
        for i in range(self.vec_len(ops_f)):
            op   = self.vec_elem(ops_f, i)
            t    = self.gi8(op, 3)         # builtin_options_type (union discriminant)
            of   = self.field(op, 4)       # builtin_options (union value)
            info = {}
            if of:
                ot = self.ref(of)          # options table
                if t == 1:   # Conv2DOptions
                    info = dict(padding=self.PAD.get(self.gi8(ot,0),'?'),
                                stride_w=self.gi32(ot,1) or 1,
                                stride_h=self.gi32(ot,2) or 1)
                elif t == 2: # DepthwiseConv2DOptions
                    info = dict(padding=self.PAD.get(self.gi8(ot,0),'?'),
                                stride_w=self.gi32(ot,1) or 1,
                                stride_h=self.gi32(ot,2) or 1)
                elif t == 5: # Pool2DOptions
                    info = dict(padding=self.PAD.get(self.gi8(ot,0),'?'),
                                stride_w=self.gi32(ot,1) or 1,
                                stride_h=self.gi32(ot,2) or 1,
                                filter_w=self.gi32(ot,3),
                                filter_h=self.gi32(ot,4))
            results.append(info)
        return results

# ── 模型搜尋 ───────────────────────────────────────────────────
def find_tflite(pattern):
    if os.path.isfile(pattern) and pattern.endswith(".tflite"):
        return pattern, [pattern]
    all_m = glob.glob(os.path.join("model","**","*.tflite"), recursive=True)
    hits  = [m for m in all_m if pattern.lower() in os.path.basename(m).lower()]
    hits.sort(key=lambda x:(len(x),x))
    return (hits[0], hits) if hits else (None, [])

# ── Conv 屬性（從 weight tensor shape 取 filter & channel）──────
def conv_attrs(op_name, inp_idx, tmap):
    a = {}
    if op_name in ('CONV_2D','DEPTHWISE_CONV_2D') and len(inp_idx)>=2:
        wt = tmap.get(inp_idx[1])
        if wt is not None:
            s = list(wt['shape'])
            if len(s)==4:
                a['kH'],a['kW'] = s[1],s[2]
                a['out_ch'] = s[0] if op_name=='CONV_2D' else s[3]
                a['in_ch']  = s[3] if op_name=='CONV_2D' else s[2]
    elif op_name=='FULLY_CONNECTED' and len(inp_idx)>=2:
        wt = tmap.get(inp_idx[1])
        if wt and len(wt['shape'])==2:
            s = wt['shape']
            a['out_u'],a['in_u'] = s[0],s[1]
    return a

# ── 主分析函式 ─────────────────────────────────────────────────
def analyze(model_path):
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ pip install tensorflow"); return

    fsz = os.path.getsize(model_path)
    print(f"\n{'='*80}")
    print(f"  模型：{os.path.basename(model_path)}   ({fsz/1024/1024:.2f} MB)")
    print(f"  路徑：{model_path}")
    print(f"{'='*80}")

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_d  = interp.get_input_details()
    out_d  = interp.get_output_details()
    td     = interp.get_tensor_details()
    tmap   = {t['index']:t for t in td}

    # I/O
    print(f"\n  INPUT  : {inp_d[0]['name']}  {sh(inp_d[0]['shape'])}  {inp_d[0]['dtype'].__name__}")
    print(f"  OUTPUT : {out_d[0]['name']}  {sh(out_d[0]['shape'])}  {out_d[0]['dtype'].__name__}")

    # FlatBuffer stride/padding 解析
    try:
        fb_opts = TFLiteFB(model_path).parse_ops()
    except Exception as e:
        fb_opts = []

    # OPs
    try:    ops = interp._get_ops_details()
    except: ops = []

    op_cnt   = {}
    tot_macs = 0

    if ops:
        print(f"\n  ── OPs ({len(ops)} 個) {'─'*60}")
        hdr = f"  {'#':>3}  {'OP 類型':<24}  {'Filter':<8}  {'Stride':<7}  {'Pad':<5}  {'In Ch→Out Ch / Units':<22}  {'Input Shape':<18}  Output Shape"
        print(hdr)
        print(f"  {'─'*3}  {'─'*24}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*22}  {'─'*18}  {'─'*18}")

        for i, op in enumerate(ops):
            nm      = op.get('op_name','?')
            ii      = [x for x in op.get('inputs',[])  if x>=0]
            oi      = [x for x in op.get('outputs',[]) if x>=0]
            op_cnt[nm] = op_cnt.get(nm,0)+1

            # shapes
            in_sh  = sh(tmap[ii[0]]['shape']) if ii and ii[0] in tmap else '?'
            out_sh = sh(tmap[oi[0]]['shape']) if oi and oi[0] in tmap else '?'

            # attributes from weight shape
            a = conv_attrs(nm, ii, tmap)
            flt_str  = f"{a['kH']}×{a['kW']}" if 'kH' in a else ''
            ch_str   = f"{a.get('in_ch','?')}→{a.get('out_ch','?')}" if 'kH' in a \
                       else (f"{a['in_u']}→{a['out_u']}" if 'out_u' in a else '')

            # stride / padding from flatbuffer
            fb = fb_opts[i] if i < len(fb_opts) else {}
            if 'filter_h' in fb and not flt_str:   # pool filter from fb
                flt_str = f"{fb['filter_h']}×{fb['filter_w']}"
            stride_str = f"{fb.get('stride_h','-')},{fb.get('stride_w','-')}" if fb else '-'
            pad_str    = fb.get('padding','-')

            # MACs
            if 'kH' in a and oi and oi[0] in tmap:
                os4 = tmap[oi[0]]['shape']
                if len(os4)==4:
                    tot_macs += int(os4[1])*int(os4[2])*a['kH']*a['kW']*int(a.get('in_ch',1))
            if 'out_u' in a:
                tot_macs += int(a['in_u'])*int(a['out_u'])

            print(f"  {i:>3}  {nm:<24}  {flt_str:<8}  {stride_str:<7}  {pad_str:<5}  {ch_str:<22}  {in_sh:<18}  → {out_sh}")

    # 統計
    print(f"\n  ── OP 統計 {'─'*68}")
    for nm,cnt in sorted(op_cnt.items(), key=lambda x:-x[1]):
        print(f"  {nm:<35}  {cnt:>5}")

    # 參數量
    tot_p = tot_b = 0
    for t in td:
        is_io = any(t['index']==d['index'] for d in inp_d+out_d)
        if not is_io and len(t['shape'])>0:
            try:
                interp.get_tensor(t['index'])
                n = int(np.prod(t['shape']))
                tot_p += n; tot_b += n*dtype_bits(t['dtype'])//8
            except: pass

    print(f"\n  ── 參數 & 計算量 {'─'*63}")
    print(f"  總參數量 : {fmt(tot_p)} ({tot_p:,})")
    print(f"  權重大小 : {tot_b/1024/1024:.2f} MB")
    if tot_macs: print(f"  MACs     : {fmt(tot_macs)}   FLOPs : {fmt(tot_macs*2)}")
    print(f"\n{'='*80}\n")

# ── PyTorch 分析 ───────────────────────────────────────────────
def analyze_pytorch(model, input_size=(1,3,224,224)):
    try:
        from torchinfo import summary
        summary(model, input_size=input_size,
                col_names=["input_size","output_size","num_params","mult_adds"])
    except ImportError:
        try:
            from torchsummary import summary as ts
            import torch
            ts(model.to("cpu"), input_size=input_size[1:])
        except ImportError:
            print("pip install torchinfo torchsummary")

# ── 主程式 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_info.py <關鍵字>\n所有模型：")
        for m in sorted(glob.glob(os.path.join("model","**","*.tflite"),recursive=True)):
            print(f"  {m:<55}  {os.path.getsize(m)/1024/1024:>7.1f} MB")
        sys.exit(0)

    path, hits = find_tflite(sys.argv[1])
    if not path:
        print(f"❌ 找不到 '{sys.argv[1]}'"); sys.exit(1)
    if len(hits)>1:
        print(f"🔍 多個匹配，選擇：{path}\n   其他：{' | '.join(hits[1:])}")
    analyze(path)
