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
    ap.add_argument("--out", type=Path, default=Path("batch/chart/mdla6_pattern_ratio_chart.svg"))
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--sort", choices=["ratio", "input"], default="ratio")
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pattern = row.get("pattern", "")
            mdla6_cx = parse_float(row.get("mdla6_cx", ""))
            ms = parse_float(row.get("mdla7_ms", ""))
            if not pattern or mdla6_cx is None or ms is None or mdla6_cx <= 0:
                continue
            rows.append({
                "pattern": pattern,
                "mdla6_cx": mdla6_cx,
                "ms": ms,
                "ratio": ms / mdla6_cx,
                "status": row.get("status", ""),
            })

    if args.sort == "ratio":
        rows.sort(key=lambda r: (float(r["ratio"]), str(r["pattern"])))

    left = 82
    right = 74
    top = 74
    bottom = 264
    point_gap = 28
    width = max(1280, left + right + point_gap * max(1, len(rows) - 1))
    height = 760
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_ratio = max([float(r["ratio"]) for r in rows] + [args.threshold])
    y_max = max(args.threshold * 1.08, max_ratio * 1.12, 2.2)
    y_min = 0.0

    def x_of(idx: int) -> float:
        if len(rows) <= 1:
            return left + plot_w / 2
        return left + idx * (plot_w / (len(rows) - 1))

    def y_of(value: float) -> float:
        return top + ((y_max - value) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#1f2933}",
        ".small{font-size:12px;fill:#52606d}.axis{stroke:#9aa5b1;stroke-width:1}",
        ".grid{stroke:#d9e2ec;stroke-width:1}.threshold{stroke:#d64545;stroke-width:2;stroke-dasharray:7 6}",
        ".label{font-size:12px}.title{font-size:22px;font-weight:700}.line{stroke:#2f80ed;stroke-width:2.4;fill:none;stroke-linejoin:round;stroke-linecap:round}",
        ".dot{fill:#2f80ed}.dot-risk{fill:#d64545}.value{font-size:11px;font-variant-numeric:tabular-nums;fill:#334e68}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 30, "MDLA7 / MDLA6 Pattern Ratio", class_="title"),
        svg_text(left, 52, "X = Pattern sorted by MDLA7 / mdla6_cx; Y = MDLA7 / mdla6_cx (Cycle Ratio); dashed line marks Y = 2", class_="small"),
    ]

    ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    if y_max > 3:
        ticks.extend(range(4, int(y_max) + 1))
    for tick in ticks:
        tick = float(tick)
        if tick < y_min or tick > y_max:
            continue
        y = y_of(tick)
        grid_class = "threshold" if abs(tick - args.threshold) < 1e-6 else "grid"
        parts.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right + 8}" y2="{y:.1f}" class="{grid_class}"/>')
        parts.append(svg_text(left - 12, y + 4, f"{tick:g}", class_="small", text_anchor="end"))

    th_y = y_of(args.threshold)
    parts.append(svg_text(left + 6, th_y - 8, f"Y={args.threshold:g}", class_="small", fill="#d64545"))
    axis_y = height - bottom
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{axis_y}" class="axis"/>')

    points = " ".join(f"{x_of(idx):.1f},{y_of(float(row['ratio'])):.1f}" for idx, row in enumerate(rows))
    if points:
        parts.append(f'<polyline points="{points}" class="line"/>')

    for idx, row in enumerate(rows):
        x = x_of(idx)
        ratio = float(row["ratio"])
        ms = float(row["ms"])
        y = y_of(ratio)
        dot_class = "dot-risk" if ratio >= args.threshold else "dot"
        label_y = axis_y + 20
        label = nice_label(str(row["pattern"]), max_len=30)
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}" class="axis"/>')
        parts.append(
            svg_text(
                x + 2,
                label_y,
                label,
                class_="label",
                text_anchor="start",
                transform=f"rotate(58 {x + 2:.1f} {label_y:.1f})",
            )
        )
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" class="{dot_class}"/>')
        if ratio > args.threshold:
            parts.append(svg_text(x + 7, y - 7, f"{ratio:.2f}", class_="value"))

    parts.append(svg_text(left + plot_w / 2, height - 8, "Pattern", class_="small", text_anchor="middle"))
    parts.append(
        svg_text(
            18,
            top + plot_h / 2,
            "MDLA7 / mdla6_cx (Cycle Ratio)",
            class_="small",
            text_anchor="middle",
            transform=f"rotate(-90 18 {top + plot_h / 2:.1f})",
        )
    )
    parts.append("</svg>\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
