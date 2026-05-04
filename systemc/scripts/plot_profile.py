#!/usr/bin/env python3
"""Render a Gantt chart of an MDLA7 simulation profile.

Reads <build>/<model>.profile.json (written by test_model) and draws:
  - one row per engine (UDMA / CONV / Requant / EWE / POOL)
  - one bar per task interval (start_ns, end_ns)
  - dashed vertical lines at each layer's `cycles_cum` boundary
  - sidebar showing per-engine busy% and per-layer ops

Usage:
  ./plot_profile.py                       # default build/program.profile.json
  ./plot_profile.py path/to/profile.json
  ./plot_profile.py -o out.png            # custom output path
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # headless render — no Tk on CI / Mac
import matplotlib.pyplot as plt

HERE       = Path(__file__).resolve().parent
SYSTEMC    = HERE.parent
DEFAULT_IN = SYSTEMC / "build" / "program.profile.json"

ENGINE_ORDER  = ["udma", "conv", "requant", "ewe", "pool"]
ENGINE_COLORS = {
    "udma":    "#9673a6",   # purple — DMA
    "conv":    "#82b366",   # green  — main compute
    "requant": "#d6b656",   # yellow — chain consumer
    "ewe":     "#b85450",   # red    — element-wise
    "pool":    "#6c8ebf",   # blue   — spatial pooling
}


def render(profile_path: Path, out_path: Path):
    p = json.load(open(profile_path))
    summary = p["summary"]
    engines = p["engines"]
    layers  = p["layers"]
    total   = summary["total_cycles"]

    # Friendly time axis: cycle == ns at 1 GHz, but pick microseconds if long.
    use_us = total >= 50_000
    scale  = 1e-3 if use_us else 1.0
    unit   = "µs" if use_us else "cycles (1 ns = 1 cyc @ 1 GHz)"

    fig, ax = plt.subplots(figsize=(14, 4 + 0.3 * len(layers) / 8))
    bar_h = 8

    # --- engine rows ---
    for i, name in enumerate(ENGINE_ORDER):
        if name not in engines:
            continue
        info = engines[name]
        bars = [(s * scale, max((e - s) * scale, 1 * scale))
                for s, e in info["tasks"]]
        if bars:
            ax.broken_barh(bars, (i * (bar_h + 2), bar_h),
                           facecolors=ENGINE_COLORS[name],
                           edgecolor="black", linewidth=0.3)
        busy_pct = 100.0 * info["busy_cycles"] / total if total else 0.0
        ax.text(-total * scale * 0.015, i * (bar_h + 2) + bar_h / 2,
                f"{name}\n{busy_pct:.0f}%",
                ha="right", va="center", fontsize=9, family="monospace")

    # --- layer boundaries ---
    last_cum = 0
    for L in layers:
        cum = L["cycles_cum"] * scale
        ax.axvline(cum, color="black", linewidth=0.3, alpha=0.25, linestyle="--")
        # tiny op label centred on the layer's interval
        mid = (last_cum + cum) / 2
        ax.text(mid, len(ENGINE_ORDER) * (bar_h + 2) + 1,
                f"{L['id']}\n{L['op'].strip()}",
                ha="center", va="bottom", fontsize=6, alpha=0.8)
        last_cum = cum

    # --- chrome ---
    y_top = len(ENGINE_ORDER) * (bar_h + 2) + 8
    ax.set_xlim(0, total * scale * 1.005)
    ax.set_ylim(-2, y_top)
    ax.set_yticks([])
    ax.set_xlabel(f"sim time [{unit}]")
    title = (f"MDLA7 Gantt — {Path(p['model']).name}   "
             f"{summary['layers']} layers, "
             f"{summary['pass']}/{summary['layers']} PASS, "
             f"{total:,} cycles  "
             f"DRAM r/w={summary['dram_read_bytes']/1024:.1f}/"
             f"{summary['dram_write_bytes']/1024:.1f} KB  "
             f"SRAM r/w={summary['sram_read_bytes']/1024:.1f}/"
             f"{summary['sram_write_bytes']/1024:.1f} KB")
    ax.set_title(title, fontsize=10)

    # remove top/right spines for cleaner look
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  gantt: {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", nargs="?", default=str(DEFAULT_IN),
                    help="profile.json path (default: build/program.profile.json)")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG path (default: profile.json with .png suffix)")
    args = ap.parse_args()

    profile_path = Path(args.profile)
    if not profile_path.exists():
        raise SystemExit(f"profile not found: {profile_path}")
    out_path = Path(args.output) if args.output else \
               profile_path.with_suffix(".png")
    render(profile_path, out_path)


if __name__ == "__main__":
    main()
