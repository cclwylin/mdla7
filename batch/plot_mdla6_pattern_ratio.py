#!/usr/bin/env python3
"""Render MDLA7/MDLA6 pattern ratio chart as SVG."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def nice_label(name: str, max_len: int = 34) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "..."


def svg_text(x: float, y: float, text: str, **attrs: str | int | float) -> str:
    def attr_name(key: str) -> str:
        if key.endswith("_"):
            key = key[:-1]
        return key.replace("_", "-")

    attr = " ".join(f'{attr_name(k)}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attr}>{html.escape(text)}</text>'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("batch/output/mdla6_pattern_regression.csv"))
    ap.add_argument("--out", type=Path, default=Path("batch/mdla6_pattern_ratio_chart.svg"))
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--sort", choices=["ratio", "input"], default="ratio")
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pattern = row.get("pattern", "")
            cx = parse_float(row.get("mdla6_cx", ""))
            ms = parse_float(row.get("mdla7_ms", ""))
            if not pattern or cx is None or ms is None or cx <= 0:
                continue
            rows.append({
                "pattern": pattern,
                "cx": cx,
                "ms": ms,
                "ratio": ms / cx,
                "status": row.get("status", ""),
            })

    if args.sort == "ratio":
        rows.sort(key=lambda r: (float(r["ratio"]), str(r["pattern"])), reverse=True)

    left = 300
    right = 118
    top = 74
    row_h = 24
    bottom = 58
    width = 1280
    height = top + bottom + row_h * len(rows)
    plot_w = width - left - right
    max_ratio = max([float(r["ratio"]) for r in rows] + [args.threshold])
    x_max = max(args.threshold * 1.08, max_ratio * 1.12, 2.2)

    def x_of(value: float) -> float:
        return left + (value / x_max) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#1f2933}",
        ".small{font-size:12px;fill:#52606d}.axis{stroke:#9aa5b1;stroke-width:1}",
        ".grid{stroke:#d9e2ec;stroke-width:1}.threshold{stroke:#d64545;stroke-width:2;stroke-dasharray:7 6}",
        ".label{font-size:12px}.title{font-size:22px;font-weight:700}.bar-ok{fill:#2f80ed}",
        ".bar-fast{fill:#2f9e44}.bar-risk{fill:#d64545}.value{font-size:12px;font-variant-numeric:tabular-nums}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 30, "MDLA7 / MDLA6 Pattern Ratio", class_="title"),
        svg_text(left, 52, "X = MDLA7 / MDLA6 CX （Cycle Ratio）; dashed line marks X = 2", class_="small"),
    ]

    for tick in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
        if tick > x_max:
            continue
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 6}" class="grid"/>')
        parts.append(svg_text(x, height - 28, f"{tick:g}", class_="small", text_anchor="middle"))
    if x_max > 3:
        for tick in range(4, int(x_max) + 1):
            x = x_of(float(tick))
            parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 6}" class="grid"/>')
            parts.append(svg_text(x, height - 28, str(tick), class_="small", text_anchor="middle"))

    th_x = x_of(args.threshold)
    parts.append(f'<line x1="{th_x:.1f}" y1="{top - 18}" x2="{th_x:.1f}" y2="{height - bottom + 6}" class="threshold"/>')
    parts.append(svg_text(th_x + 6, top - 24, f"X={args.threshold:g}", class_="small", fill="#d64545"))
    parts.append(f'<line x1="{left}" y1="{height - bottom + 6}" x2="{width - right}" y2="{height - bottom + 6}" class="axis"/>')

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        ratio = float(row["ratio"])
        ms = float(row["ms"])
        cx = float(row["cx"])
        bar_class = "bar-risk" if ratio >= args.threshold else ("bar-fast" if ratio <= 1.0 else "bar-ok")
        bar_x = left
        bar_y = y + 4
        bar_h = 15
        bar_w = max(1.0, x_of(ratio) - left)
        parts.append(svg_text(18, y + 16, nice_label(str(row["pattern"])), class_="label"))
        parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h}" rx="2" class="{bar_class}"/>')
        value = f'{ratio:.2f}  ({ms:.3f}/{cx:g})'
        tx = min(x_of(ratio) + 7, width - right + 8)
        parts.append(svg_text(tx, y + 16, value, class_="value"))

    parts.append(svg_text(left + plot_w / 2, height - 8, "MDLA7 / MDLA6 CX （Cycle Ratio）", class_="small", text_anchor="middle"))
    parts.append("</svg>\n")

    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
