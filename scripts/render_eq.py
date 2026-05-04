#!/usr/bin/env python3
# render_eq.py — SystemC 教材公式渲染器
#
# 用 matplotlib 的 mathtext 把 LaTeX 渲染成 PNG，**不需要 pdflatex**。
# 對 ~95% 的常見數學符號 (sum, frac, sqrt, sub/sup, Greek letters) 都 OK。
#
# 用法:
#   python render_eq.py [out_dir]    # 預設 ./eq

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (檔名, LaTeX 公式, 對應章節)
EQUATIONS = [
    # ── Ch 2: simcontext / delta cycle ──────────────────────────────
    ('delta_time_axis',
     r't_n = (T_n,\ \delta_n),\ \ t_n < t_m \Leftrightarrow T_n < T_m \vee (T_n = T_m \wedge \delta_n < \delta_m)',
     'Ch 2 §2.x  delta-cycle 雙鍵時間排序'),

    # ── Ch 7: datatypes ─────────────────────────────────────────────
    ('qformat',
     r'x = \sum_{i=-F}^{I-1} b_i\, 2^{i}',
     'Ch 7 §7.x  Q-format 定點數表示'),

    ('saturation',
     r'y = \min(\max(x,\ x_{\min}),\ x_{\max})',
     'Ch 7 §7.x  sc_fixed SC_SAT 飽和'),

    # ── Ch 12: AMS ──────────────────────────────────────────────────
    ('pid',
     r'u(t) = K_p\, e + K_i \int e\, dt + K_d \dfrac{de}{dt}',
     'Ch 12 §12.x  PID controller'),

    ('ams_sampling',
     r'y[n] = x(n T_s),\ \ T_s = \dfrac{1}{f_s}',
     'Ch 12 §12.x  TDF cluster 採樣'),

    # ── Ch 16: scheduler 不變式 ─────────────────────────────────────
    ('scheduler_invariant',
     r'\forall p \in \mathrm{runnable}(t,\delta):\ \ \mathrm{eval}(p) \prec \mathrm{update}(p) \prec \mathrm{notify}(p)',
     'Ch 16 §16.x  delta cycle 三階段順序'),

    # ── Ch 21: fxnum cast ───────────────────────────────────────────
    ('fxnum_cast',
     r"y = \mathrm{sat}\!\left(\mathrm{round}(x \cdot 2^{F'}) \cdot 2^{-F'},\ x_{\min},\ x_{\max}\right)",
     'Ch 21 §21.x  sc_fxnum cast：先 round 再 saturate'),
]


def render_one(name, latex, out_dir):
    fig_w = max(7.0, 0.10 * len(latex))
    fig_h = 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.axis('off')
    ax.text(0.5, 0.5, '$' + latex + '$',
            fontsize=22, ha='center', va='center')
    out_path = os.path.join(out_dir, name + '.png')
    fig.savefig(out_path, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close(fig)
    return out_path


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'eq'
    os.makedirs(out_dir, exist_ok=True)
    print(f'rendering {len(EQUATIONS)} equations → {out_dir}/')
    for name, latex, where in EQUATIONS:
        p = render_one(name, latex, out_dir)
        size = os.path.getsize(p)
        print(f'  {name:24s}  {size:6d} bytes   ({where})')


if __name__ == '__main__':
    main()
