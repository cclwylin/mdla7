#!/usr/bin/env python3
"""Render conflict/mesh mode ratios from profile_mdla6_pattern.html as SVG."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


SERIES = [
    ("conflict_ratio", "conflict/fast", "#2f80ed"),
    ("mesh_ratio", "mesh/fast", "#2f9e44"),
    ("mesh_conflict_ratio", "mesh/conflict", "#c2410c"),
]


def parse_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const\s+EMBEDDED_ROWS\s*=\s*(\[.*?\]);", text, re.S)
    if not match:
        raise ValueError(f"cannot find EMBEDDED_ROWS in {path}")
    rows = json.loads(match.group(1))
    if not isinstance(rows, list):
        raise ValueError("EMBEDDED_ROWS is not a list")
    return [row for row in rows if isinstance(row, dict)]


def as_float(value: Any) -> float | None:
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
    ap.add_argument("--html", type=Path, default=Path("batch/profile_mdla6_pattern.html"))
    ap.add_argument("--out", type=Path, default=Path("batch/chart/mdla6_pattern_mode_ratio_chart.svg"))
    ap.add_argument("--sort", choices=["mesh_conflict", "max", "pattern"], default="mesh_conflict")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for row in parse_rows(args.html):
        values = {key: as_float(row.get(key)) for key, _, _ in SERIES}
        if not any(value is not None for value in values.values()):
            continue
        values_present = [value for value in values.values() if value is not None]
        rows.append({
            "pattern": str(row.get("pattern") or row.get("stem") or ""),
            "values": values,
            "max_ratio": max(values_present),
        })

    if args.sort == "mesh_conflict":
        rows.sort(
            key=lambda r: (
                float(r["values"].get("mesh_conflict_ratio") or 0.0),
                str(r["pattern"]),
            ),
        )
    elif args.sort == "max":
        rows.sort(key=lambda r: (float(r["max_ratio"]), str(r["pattern"])), reverse=True)
    else:
        rows.sort(key=lambda r: str(r["pattern"]))

    left = 82
    right = 74
    top = 92
    bottom = 264
    point_gap = 28
    width = max(1280, left + right + point_gap * max(1, len(rows) - 1))
    height = 760
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_ratio = max([float(r["max_ratio"]) for r in rows] + [1.0])
    y_max = max(1.4, max_ratio * 1.08)
    y_min = 0.96
    if y_max - y_min < 0.2:
        y_min = 0.9

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
        ".title{font-size:22px;font-weight:700}.small{font-size:12px;fill:#52606d}",
        ".label{font-size:12px}.value{font-size:11px;font-variant-numeric:tabular-nums;fill:#334e68}",
        ".hotvalue{font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;fill:#c2410c}",
        ".grid{stroke:#d9e2ec;stroke-width:1}.axis{stroke:#9aa5b1;stroke-width:1}",
        ".baseline{stroke:#d64545;stroke-width:2;stroke-dasharray:7 6}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 30, "MDLA7 L1 Mesh Evaluation (v0.1)", class_="title"),
        svg_text(left, 52, "X = Pattern sorted by mesh/conflict; Y = ratio; dashed line marks Y = 1; red labels mark top 10", class_="small"),
    ]

    legend_x = left
    for _, label, color in SERIES:
        parts.append(f'<circle cx="{legend_x:.1f}" cy="73" r="4.5" fill="{color}"/>')
        parts.append(svg_text(legend_x + 10, 77, label, class_="small"))
        legend_x += 132

    tick = 1.0
    ticks: list[float] = []
    while tick <= y_max + 0.001:
        ticks.append(round(tick, 2))
        tick += 0.05
    if y_min < 1.0:
        ticks.insert(0, y_min)
    for tick in ticks:
        if tick < y_min or tick > y_max:
            continue
        y = y_of(tick)
        grid_class = "baseline" if abs(tick - 1.0) < 1e-6 else "grid"
        parts.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right + 8}" y2="{y:.1f}" class="{grid_class}"/>')
        if abs((tick * 100) % 10) < 1e-6 or abs(tick - y_min) < 1e-6:
            parts.append(svg_text(left - 12, y + 4, f"{tick:.2f}", class_="small", text_anchor="end"))

    axis_y = height - bottom
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{axis_y}" class="axis"/>')

    top_red = {
        idx
        for idx, row in sorted(
            enumerate(rows),
            key=lambda item: float(item[1]["values"].get("mesh_conflict_ratio") or 0.0),
            reverse=True,
        )[:10]
    }

    for series_idx, (key, _, color) in enumerate(SERIES):
        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for idx, row in enumerate(rows):
            value = row["values"].get(key)
            if value is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append((x_of(idx), y_of(value)))
        if current:
            segments.append(current)
        for segment in segments:
            if len(segment) < 2:
                continue
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in segment)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" opacity="0.78"/>'
            )

    for idx, row in enumerate(rows):
        x = x_of(idx)
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
        for series_idx, (key, _, color) in enumerate(SERIES):
            value = row["values"].get(key)
            if value is None:
                continue
            cx = x
            cy = y_of(value)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" fill="{color}"/>')
            if key == "mesh_conflict_ratio" and idx in top_red:
                label_x = cx + 7
                label_y = cy - 8
                parts.append(
                    svg_text(
                        label_x,
                        label_y,
                        f"{value:.3f}",
                        class_="hotvalue",
                        text_anchor="start",
                        transform=f"rotate(-42 {label_x:.1f} {label_y:.1f})",
                    )
                )

    parts.append(svg_text(left + plot_w / 2, height - 8, "Pattern", class_="small", text_anchor="middle"))
    parts.append(
        svg_text(
            18,
            top + plot_h / 2,
            "Cycle ratio",
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
