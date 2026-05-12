#!/usr/bin/env python3
"""Shared Hotspot-style sweep runner for model corpora without mdla6_cx baselines."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import time
from pathlib import Path

from run_mdla6_pattern import (  # noqa: E402
    OUT_DIR,
    REPO_ROOT,
    MODEL_RUNNER,
    _artefact_paths,
    _mode_paths,
    _mode_suffix,
    _normalise_l1,
    _normalise_pattern,
    _report_exists_for,
    run_one,
)

HERE = Path(__file__).resolve().parent
MODEL_PROFILE_PY = HERE / "gen_model_profile.py"


def _load_prior_csv(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pattern"):
                    if "cx_ms" in row and "mdla6_cx_ms" not in row:
                        row["mdla6_cx_ms"] = row.get("cx_ms", "")
                    if "rtl_over_cx" in row and "rtl_over_mdla6_cx" not in row:
                        row["rtl_over_mdla6_cx"] = row.get("rtl_over_cx", "")
                    if "synth_over_cx" in row and "cx_over_mdla6_cx" not in row:
                        row["cx_over_mdla6_cx"] = row.get("synth_over_cx", "")
                    out[row["pattern"]] = row
    except Exception:
        return {}
    return out


def _load_prior_results(csv_path: Path, fast_only: bool = False) -> dict[str, dict]:
    rows = {p: r for p, r in _load_prior_csv(csv_path).items()
            if r.get("status") == "ok" and r.get("mdla7_ms")}
    if fast_only:
        return rows
    return {p: r for p, r in rows.items()
            if r.get("conflict_status", "ok") == "ok" and r.get("mdla7_conflict_ms") and
            r.get("mesh_status", "ok") == "ok" and r.get("mdla7_mesh_ms")}


def _discover_models(model_dir: Path, name_filter: str, recursive: bool = False) -> list[str]:
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")
    needle = name_filter.lower()
    patterns = []
    globber = model_dir.rglob if recursive else model_dir.glob
    for path in sorted(globber("*.tflite")):
        if path.name.startswith("._"):
            continue
        pattern = path.relative_to(model_dir).with_suffix("").as_posix()
        if needle and needle not in pattern.lower():
            continue
        patterns.append(pattern)
    return patterns


def _load_pattern_order(csv_path: Path) -> dict[str, tuple[float, int]]:
    if not csv_path.exists():
        raise SystemExit(f"pattern order CSV not found: {csv_path}")
    out: dict[str, tuple[float, int]] = {}
    with csv_path.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            pattern = row.get("Pattern") or row.get("pattern") or ""
            pattern = _normalise_pattern(pattern.strip())
            if not pattern or pattern in out:
                continue
            try:
                mdla6_cx = float(row.get("mdla6_cx") or row.get("CX") or
                                  row.get("cx") or "inf")
            except ValueError:
                mdla6_cx = float("inf")
            out[pattern] = (mdla6_cx, idx)
    return out


def _apply_pattern_order(patterns: list[str], order_csv: Path | None) -> list[str]:
    if not order_csv:
        return patterns
    order = _load_pattern_order(order_csv)

    def key(item: tuple[int, str]) -> tuple[int, float, int, str]:
        original_idx, pattern = item
        mdla6_order = order.get(_normalise_pattern(pattern))
        if mdla6_order:
            mdla6_cx, csv_idx = mdla6_order
            return (0, mdla6_cx, csv_idx, pattern)
        return (1, float("inf"), original_idx, pattern)

    return [pattern for _, pattern in sorted(enumerate(patterns), key=key)]


def _refresh_profile_index(title: str, html_out: str, csv_path: Path) -> None:
    try:
        html_path = Path(html_out)
        if not html_path.is_absolute():
            html_path = HERE / html_path
        if html_path.parent == HERE:
            html_out = f"output/profile/{html_path.name}"
        subprocess.run(
            [sys.executable, str(MODEL_PROFILE_PY),
             "--html-out", html_out,
             "--title", title,
             "--metrics-csv", str(csv_path),
             "--only-metrics-rows",
             "--hide-mdla6-cx"],
            cwd=str(HERE), capture_output=True, text=True,
        )
    except Exception:
        pass


def _ms_cell(value: str) -> str:
    return f"{float(value):>8.2f} ms" if value else f"{'—':>8s}    "


def _ratio_cell(value: str) -> str:
    return f"{float(value):>6.2f}" if value else f"{'':>6s}"


def _ms_value_cell(value: str | float | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def _ratio_from_ms(num: float | str | None, den: float | str | None) -> str:
    if num is None or den in (None, 0):
        return ""
    if num == "" or den == "":
        return ""
    den_f = float(den)
    if den_f <= 0:
        return ""
    return f"{float(num) / den_f:.3f}"


def _format_ms(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def _load_mdla6_cx_ms(csv_path: Path | None) -> dict[str, float]:
    if not csv_path or not csv_path.exists():
        return {}
    out: dict[str, float] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            pattern = row.get("Pattern") or row.get("pattern") or ""
            pattern = _normalise_pattern(pattern.strip())
            if not pattern:
                continue
            value = row.get("mdla6_cx") or row.get("CX") or row.get("cx") or ""
            try:
                out[pattern] = float(value)
            except ValueError:
                continue
    return out


def _fit_cell(value: str, width: int = 30) -> str:
    if len(value) <= width:
        return f"{value:<{width}s}"
    keep = width - 1
    left = keep // 2
    right = keep - left
    return f"{value[:left]}…{value[-right:]}"


def _microblock_metrics_for(model_path: Path, l1_timing: str = "fast",
                            engine_model: str = "fast") -> dict[str, str]:
    suffix = _mode_suffix(l1_timing, engine_model)
    paths = _mode_paths(model_path, suffix) if suffix else _artefact_paths(model_path)
    prof = paths["prof"]
    if not prof.exists():
        return {
            "fuse_hit": "no",
            "fuse_flows": "",
            "streamed_layers": "",
            "mb_hit": "no",
            "mb_count": "0",
            "mb_layers": "",
            "mb_stages": "",
        }
    try:
        data = json.loads(prof.read_text())
    except Exception:
        return {
            "fuse_hit": "profile-error",
            "fuse_flows": "",
            "streamed_layers": "",
            "mb_hit": "profile-error",
            "mb_count": "0",
            "mb_layers": "",
            "mb_stages": "",
        }

    layers_json = data.get("layers") or []
    flow_members: dict[int, list[int]] = {}
    streamed_layers: list[int] = []
    for idx, layer in enumerate(layers_json):
        if not isinstance(layer, dict):
            continue
        lid = int(layer.get("id", idx) or idx)
        flow = int(layer.get("flow", lid) if layer.get("flow") is not None else lid)
        flow_members.setdefault(flow, []).append(lid)
        if layer.get("streamed"):
            streamed_layers.append(lid)
    fused_flows = [
        f"F{flow}:" + "+".join(f"L{x}" for x in members)
        for flow, members in sorted(flow_members.items())
        if len(members) > 1
    ]

    task_meta = data.get("task_meta") or {}
    seen: set[tuple[int, int]] = set()
    layers: set[int] = set()
    stages: set[str] = set()
    for engine, metas in task_meta.items():
        if not isinstance(metas, list):
            continue
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            flags = int(meta.get("flags") or 0)
            stream_flags = int(meta.get("stream_flags") or 0)
            if not (flags & 0x10) or not stream_flags:
                continue
            layer = int(meta.get("layer") or 0)
            mb = int(meta.get("mb") or 0)
            seen.add((layer, mb))
            layers.add(layer)
            if engine in ("udma_r", "udma_w"):
                if stream_flags & 0x8:
                    stages.add("store")
                else:
                    stages.add("load")
            elif engine in ("ewe", "pool", "tnps"):
                stages.add("consumer")
            else:
                stages.add(str(engine))
    return {
        "fuse_hit": "yes" if fused_flows or streamed_layers else "no",
        "fuse_flows": ";".join(fused_flows),
        "streamed_layers": ";".join(f"L{x}" for x in streamed_layers),
        "mb_hit": "yes" if seen else "no",
        "mb_count": str(len(seen)),
        "mb_layers": ";".join(f"L{x}" for x in sorted(layers)),
        "mb_stages": "+".join(s for s in ("load", "conv", "requant", "consumer", "store")
                              if s in stages),
    }


def _compare_paths(model_path: Path) -> dict[str, Path]:
    return _mode_paths(model_path, "rtl_compare")


def _compare_report_exists_for(pattern: str, model_dir: Path) -> bool:
    canonical = _normalise_pattern(pattern)
    return _compare_paths(model_dir / f"{canonical}.tflite")["html"].exists()


def _cx_rtl_compare_paths(model_path: Path) -> dict[str, Path]:
    return _mode_paths(model_path, "cx_rtl_compare")


def _cx_rtl_compare_report_exists_for(pattern: str, model_dir: Path) -> bool:
    canonical = _normalise_pattern(pattern)
    return _cx_rtl_compare_paths(model_dir / f"{canonical}.tflite")["html"].exists()


def _write_two_mode_compare_html(model_path: Path,
                                 compare_suffix: str,
                                 left_label: str,
                                 left_html: Path,
                                 right_label: str,
                                 right_html: Path,
                                 mdla6_cx_ms: float | None,
                                 left_ms: float | None,
                                 right_ms: float | None,
                                 left_status: str,
                                 right_status: str,
                                 ratios: list[tuple[str, float | None, float | None]]) -> Path:
    out = _mode_paths(model_path, compare_suffix)["html"]
    left_doc = left_html.read_text(errors="ignore") if left_html.exists() else ""
    right_doc = right_html.read_text(errors="ignore") if right_html.exists() else ""

    def ms(v: float | None) -> str:
        return f"{v:.3f} ms" if v is not None else ""

    ratio_spans = []
    for label, num, den in ratios:
        ratio = _ratio_from_ms(num, den)
        ratio_spans.append(
            f"<span><b>{html.escape(label)}:</b> "
            f"{html.escape(ratio + 'x' if ratio else '')}</span>"
        )

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDLA7 profile — {html.escape(model_path.name)} — {html.escape(left_label)}/{html.escape(right_label)}</title>
<style>
body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:#222; background:#f6f7f9; }}
header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid #d8dde6;
          padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.04); }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.summary {{ display:flex; flex-wrap:wrap; gap:10px 18px; color:#495464; font-size:12px; }}
.summary b {{ color:#222; }}
.tabs {{ margin-top:10px; display:flex; gap:8px; }}
button {{ border:1px solid #c8d0dc; background:#fff; border-radius:5px; padding:5px 10px; cursor:pointer; }}
button.active {{ background:#1f5fa8; color:#fff; border-color:#1f5fa8; }}
.pane {{ display:none; padding:0; }}
.pane.active {{ display:block; }}
iframe {{ width:100%; height:calc(100vh - 120px); border:0; background:#fff; display:block; }}
</style></head>
<body>
<header>
  <h1>{html.escape(model_path.name)} — {html.escape(left_label)}/{html.escape(right_label)} profile</h1>
  <div class="summary">
    <span><b>mdla6_cx:</b> {html.escape(ms(mdla6_cx_ms))}</span>
    <span><b>{html.escape(left_label)}:</b> {html.escape(ms(left_ms))} {html.escape(left_status)}</span>
    <span><b>{html.escape(right_label)}:</b> {html.escape(ms(right_ms))} {html.escape(right_status)}</span>
    {''.join(ratio_spans)}
  </div>
  <div class="tabs">
    <button class="tab active" data-target="{html.escape(left_label)}">{html.escape(left_label)}</button>
    <button class="tab" data-target="{html.escape(right_label)}">{html.escape(right_label)}</button>
  </div>
</header>
<section id="{html.escape(left_label)}" class="pane active">
  <iframe title="{html.escape(left_label)} profile" srcdoc="{html.escape(left_doc, quote=True)}"></iframe>
</section>
<section id="{html.escape(right_label)}" class="pane">
  <iframe title="{html.escape(right_label)} profile" srcdoc="{html.escape(right_doc, quote=True)}"></iframe>
</section>
<script>
document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.target));
}}));
</script>
</body></html>
"""
    out.write_text(doc)
    return out


def _write_rtl_compare_html(model_path: Path,
                            fast_html: Path,
                            rtl_html: Path,
                            mdla6_cx_ms: float | None,
                            fast_ms: float | None,
                            rtl_ms: float | None,
                            fast_status: str,
                            rtl_status: str) -> Path:
    return _write_two_mode_compare_html(
        model_path, "rtl_compare",
        "fast", fast_html, "rtl", rtl_html,
        mdla6_cx_ms, fast_ms, rtl_ms, fast_status, rtl_status,
        [("rtl/fast", rtl_ms, fast_ms), ("rtl/mdla6_cx", rtl_ms, mdla6_cx_ms)])


def _write_cx_rtl_compare_html(model_path: Path,
                                  rtl_html: Path,
                                  synth_html: Path,
                                  mdla6_cx_ms: float | None,
                                  rtl_ms: float | None,
                                  cx_ms: float | None,
                                  rtl_status: str,
                                  cx_status: str) -> Path:
    return _write_two_mode_compare_html(
        model_path, "cx_rtl_compare",
        "rtl", rtl_html, "cx", synth_html,
        mdla6_cx_ms, rtl_ms, cx_ms, rtl_status, cx_status,
        [("cx/rtl", cx_ms, rtl_ms),
         ("rtl/mdla6_cx", rtl_ms, mdla6_cx_ms),
         ("cx/mdla6_cx", cx_ms, mdla6_cx_ms)])


def _write_rtl_compare_index(title: str, html_out: str,
                             rows: list[dict], csv_path: Path) -> None:
    out_path = HERE / html_out
    link_prefix = "../" if out_path.parent == OUT_DIR / "profile" else "output/"
    body = []
    for row in rows:
        pat = row.get("pattern", "")
        link = f"{link_prefix}{html.escape(pat)}.rtl_compare.html"
        status = row.get("status", "")
        body.append(
            "<tr>"
            f"<td><a href=\"{link}\">{html.escape(pat)}</a></td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('mdla6_cx_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('fast_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('rtl_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('fast_over_mdla6_cx', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('rtl_over_fast', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('rtl_over_mdla6_cx', ''))}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1200px; margin:24px auto; padding:0 16px; color:#222; }}
h1 {{ font-size:20px; margin-bottom:4px; }}
.meta {{ color:#666; font-size:12px; margin-bottom:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ border:1px solid #e4e4e4; padding:5px 8px; font-variant-numeric:tabular-nums; }}
th {{ background:#f4f4f4; text-align:left; }}
a {{ color:#1f5fa8; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">csv: {html.escape(str(csv_path))}</div>
<table>
  <thead><tr><th>pattern</th><th>mdla6_cx ms</th><th>fast ms</th><th>rtl ms</th><th>fast/mdla6_cx</th><th>rtl/fast</th><th>rtl/mdla6_cx</th><th>status</th></tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
</body></html>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)


def _write_cx_rtl_compare_index(title: str, html_out: str,
                                   rows: list[dict], csv_path: Path) -> None:
    out_path = HERE / html_out
    link_prefix = "../" if out_path.parent == OUT_DIR / "profile" else "output/"
    body = []
    for row in rows:
        pat = row.get("pattern", "")
        link = f"{link_prefix}{html.escape(pat)}.cx_rtl_compare.html"
        status = row.get("status", "")
        body.append(
            "<tr>"
            f"<td><a href=\"{link}\">{html.escape(pat)}</a></td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('mdla6_cx_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('rtl_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('cx_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('cx_over_rtl', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('rtl_over_mdla6_cx', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('cx_over_mdla6_cx', ''))}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1200px; margin:24px auto; padding:0 16px; color:#222; }}
h1 {{ font-size:20px; margin-bottom:4px; }}
.meta {{ color:#666; font-size:12px; margin-bottom:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ border:1px solid #e4e4e4; padding:5px 8px; font-variant-numeric:tabular-nums; }}
th {{ background:#f4f4f4; text-align:left; }}
a {{ color:#1f5fa8; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">csv: {html.escape(str(csv_path))}</div>
<table>
  <thead><tr><th>pattern</th><th>mdla6_cx ms</th><th>rtl ms</th><th>cx ms</th><th>cx/rtl</th><th>rtl/mdla6_cx</th><th>cx/mdla6_cx</th><th>status</th></tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)


def _row_print(s: str) -> None:
    sys.stdout.write("\r\033[2K" + s + "\n")
    sys.stdout.flush()


def _row_update(s: str) -> None:
    sys.stdout.write("\r\033[2K" + s)
    sys.stdout.flush()


def run_corpus(*,
               corpus_name: str,
               default_model_dir: Path,
               default_csv_out: Path,
               profile_html: str,
               profile_title: str,
               pattern_order_csv: Path | None = None,
               recursive: bool = False,
               microblock_metrics: bool = False) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(default_model_dir),
                    help=f"directory containing {corpus_name} .tflite models")
    ap.add_argument("--csv-out", "--csv", dest="csv_out",
                    default=str(default_csv_out),
                    help="output regression CSV")
    ap.add_argument("--filter", default="",
                    help="substring filter on model name")
    ap.add_argument("--pattern-order-csv", default=str(pattern_order_csv or ""),
                    help="optional CSV with Pattern,mdla6_cx columns used to order selected models")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N selected models (0 = no limit)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N selected models before applying --limit")
    ap.add_argument("--rerun-all", action="store_true",
                    help="ignore prior --csv-out cache and re-run everything")
    ap.add_argument("--fast-only", action="store_true",
                    help="run fast L1/engine mode only unless --engine is set")
    ap.add_argument("--L1", "--l1", dest="l1_timing",
                    choices=("fast", "rtl", "cx", "synth"), default="rtl",
                    help="L1Mesh timing mode for the primary run")
    ap.add_argument("--l1-timing", dest="l1_timing",
                    choices=("fast", "rtl", "cx", "cx", "conflict", "mesh", "mesh-opt"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--engine", dest="engine_model",
                    choices=("fast", "rtl", "cx", "synth"), default="rtl",
                    help="engine timing model: fast analytical, RTL-style, or cx mode")
    ap.add_argument("--engine-model", dest="engine_model",
                    choices=("fast", "model", "analytical", "rtl", "rtl-style", "cx", "synth"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--rtl-fast", action="store_true",
                    help="legacy alias for --fast-only --engine=rtl")
    ap.add_argument("--compare-rtl-fast", action="store_true",
                    help="run pure fast and rtl-fast, then emit combined CSV/HTML")
    ap.add_argument("--compare-cx-rtl", action="store_true",
                    help="run full RTL and cx modes, then emit combined CSV/HTML")
    ap.add_argument("--keep-bin", action="store_true",
                    help="keep per-model .bin files in output/ after the sweep")
    ap.add_argument("--no-html", action="store_true",
                    help="skip per-model and index HTML generation; keep profile JSON/CSV")
    ap.add_argument("--list", action="store_true",
                    help="list selected models and exit")
    args = ap.parse_args()
    if args.compare_rtl_fast and args.compare_cx_rtl:
        raise SystemExit("--compare-rtl-fast and --compare-cx-rtl are mutually exclusive")
    engine_explicit = any(
        arg == name or arg.startswith(f"{name}=")
        for arg in sys.argv[1:]
        for name in ("--engine", "--engine-model")
    )
    if args.fast_only:
        args.l1_timing = "fast"
        if not engine_explicit:
            args.engine_model = "fast"
    args.l1_timing = _normalise_l1(args.l1_timing)
    if args.engine_model in ("model", "analytical"):
        args.engine_model = "fast"
    elif args.engine_model == "rtl-style":
        args.engine_model = "rtl"
    elif args.engine_model == "synth":
        args.engine_model = "cx"
    if args.compare_rtl_fast:
        args.fast_only = True
        args.l1_timing = "fast"
        args.engine_model = "fast"
    if args.compare_cx_rtl:
        args.fast_only = True
        args.l1_timing = "rtl"
        args.engine_model = "rtl"
    if args.rtl_fast:
        args.fast_only = True
        args.l1_timing = "fast"
        args.engine_model = "rtl"
    if args.l1_timing != "fast":
        args.fast_only = True

    OUT_DIR.mkdir(exist_ok=True)
    default_csv = Path(default_csv_out)
    suffix = "" if (args.compare_rtl_fast or args.compare_cx_rtl) else _mode_suffix(
        args.l1_timing, args.engine_model)
    if suffix and Path(args.csv_out) == default_csv:
        args.csv_out = str(default_csv.with_name(
            f"{default_csv.stem}.{suffix}{default_csv.suffix}"))
    if suffix:
        profile_path = Path(profile_html)
        profile_html = f"{profile_path.stem}.{suffix}{profile_path.suffix}"
    model_dir = Path(args.model_dir)
    patterns = _discover_models(model_dir, args.filter, recursive=recursive)
    order_csv = Path(args.pattern_order_csv) if args.pattern_order_csv else None
    patterns = _apply_pattern_order(patterns, order_csv)
    if args.offset:
        patterns = patterns[args.offset:]
    if args.limit:
        patterns = patterns[:args.limit]

    if args.list:
        for pat in patterns:
            path = model_dir / f"{pat}.tflite"
            print(f"{_fit_cell(pat)} {path.stat().st_size / (1024 * 1024):6.1f} MB")
        return

    if not patterns:
        raise SystemExit(f"no .tflite models found in {model_dir} "
                         f"(filter={args.filter!r})")
    if not MODEL_RUNNER.exists():
        raise SystemExit(f"mdla7_model_runner not built: {MODEL_RUNNER}\n"
                         f"  run `make -C systemc -s` from repo root")

    if args.compare_rtl_fast:
        compare_csv = default_csv.with_name(
            f"{default_csv.stem}.rtl_compare{default_csv.suffix}")
        csv_path = Path(args.csv_out)
        if csv_path == default_csv:
            csv_path = compare_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        prior_full = {} if args.rerun_all else _load_prior_csv(csv_path)
        prior_ok = {
            p: r for p, r in prior_full.items()
            if r.get("status") == "ok" and r.get("fast_ms") and r.get("rtl_ms")
        }
        profile_path = Path(profile_html)
        compare_html = f"{profile_path.stem}.rtl_compare{profile_path.suffix}"
        mdla6_cx_ms_by_pattern = _load_mdla6_cx_ms(order_csv)

        def _mdla6_cx_ms_for(pattern: str) -> float | None:
            return mdla6_cx_ms_by_pattern.get(_normalise_pattern(pattern))

        def _fill_compare_ms(row: dict) -> dict:
            out = dict(row)
            mdla6_cx_ms = out.get("mdla6_cx_ms")
            if not mdla6_cx_ms:
                value = _mdla6_cx_ms_for(out.get("pattern", ""))
                out["mdla6_cx_ms"] = f"{value:.3f}" if value is not None else ""
            if not out.get("rtl_over_fast"):
                out["rtl_over_fast"] = _ratio_from_ms(
                    out.get("rtl_ms", ""), out.get("fast_ms", ""))
            if not out.get("fast_over_mdla6_cx"):
                out["fast_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("fast_ms", ""), out.get("mdla6_cx_ms", ""))
            if not out.get("rtl_over_mdla6_cx"):
                out["rtl_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("rtl_ms", ""), out.get("mdla6_cx_ms", ""))
            return out

        def _checkpoint_compare(rows: list[dict]) -> None:
            seen = {r["pattern"] for r in rows}
            merged = [_fill_compare_ms(r) for r in rows]
            for pat, prow in prior_full.items():
                if pat not in seen:
                    merged.append(_fill_compare_ms(prow))
            fields = [
                "pattern", "mdla6_cx_ms", "fast_ms", "rtl_ms",
                "fast_over_mdla6_cx", "rtl_over_fast", "rtl_over_mdla6_cx",
                "status", "fast_status", "rtl_status",
            ]
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(merged)
            if not args.no_html:
                _write_rtl_compare_index(f"{profile_title} — fast vs rtl",
                                         compare_html, merged, csv_path)

        try:
            rel_model_dir = model_dir.relative_to(REPO_ROOT)
        except ValueError:
            rel_model_dir = model_dir
        print(f"==== MDLA7 {corpus_name} fast vs rtl regression: {len(patterns)} models "
              f"(from {rel_model_dir}) ====", flush=True)
        rows_out = []
        t_total = time.time()
        for i, pat in enumerate(patterns, 1):
            if pat in prior_ok and (args.no_html or _compare_report_exists_for(pat, model_dir)):
                cached = prior_ok[pat]
                cached_filled = _fill_compare_ms(cached)
                display_pat = _fit_cell(pat)
                _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                           f"mdla6_cx={_ms_cell(cached_filled.get('mdla6_cx_ms', ''))} "
                           f"fast={_ms_cell(cached.get('fast_ms', ''))} "
                           f"rtl={_ms_cell(cached.get('rtl_ms', ''))} "
                           f"fast/mdla6_cx={_ratio_cell(cached_filled.get('fast_over_mdla6_cx', ''))} "
                           f"rtl/fast={_ratio_cell(cached_filled.get('rtl_over_fast', ''))} "
                           f"rtl/mdla6_cx={_ratio_cell(cached_filled.get('rtl_over_mdla6_cx', ''))} cached  "
                           f"{cached.get('status', 'ok')}")
                rows_out.append(cached_filled)
                _checkpoint_compare(rows_out)
                continue

            t0 = time.time()

            def _progress(stage: str) -> None:
                elapsed = time.time() - t0
                display_pat = _fit_cell(pat)
                _row_update(f"[{i:>2}/{len(patterns)}] {display_pat} "
                            f"{'—':>10s}      ({elapsed:5.1f}s)  "
                            f"running {stage}...")

            _, fast_ms, _, _, fast_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"fast {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="fast",
                l1_timing="fast")
            _, rtl_ms, _, _, rtl_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"rtl {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="rtl",
                l1_timing="fast")
            mdla6_cx_ms = _mdla6_cx_ms_for(pat)
            fast_mdla6_cx = _ratio_from_ms(fast_ms, mdla6_cx_ms)
            rtl_fast = _ratio_from_ms(rtl_ms, fast_ms)
            rtl_mdla6_cx = _ratio_from_ms(rtl_ms, mdla6_cx_ms)
            status = "ok" if fast_status == "ok" and rtl_status == "ok" else f"{fast_status}/{rtl_status}"
            model_path = model_dir / f"{_normalise_pattern(pat)}.tflite"
            if not args.no_html and fast_ms is not None and rtl_ms is not None:
                try:
                    _write_rtl_compare_html(
                        model_path,
                        OUT_DIR / f"{model_path.stem}.fast.html",
                        _mode_paths(model_path, "rtl")["html"],
                        mdla6_cx_ms, fast_ms, rtl_ms, fast_status, rtl_status)
                except Exception as e:
                    status = f"{status}; html-fail: {str(e)[:80]}"
            elapsed = time.time() - t0
            display_pat = _fit_cell(pat)
            _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                       f"mdla6_cx={f'{mdla6_cx_ms:>8.2f} ms' if mdla6_cx_ms is not None else f'{chr(8212):>8s}    '} "
                       f"fast={f'{fast_ms:>8.2f} ms' if fast_ms is not None else f'{chr(8212):>8s}    '} "
                       f"rtl={f'{rtl_ms:>8.2f} ms' if rtl_ms is not None else f'{chr(8212):>8s}    '} "
                       f"fast/mdla6_cx={_ratio_cell(fast_mdla6_cx)} "
                       f"rtl/fast={_ratio_cell(rtl_fast)} rtl/mdla6_cx={_ratio_cell(rtl_mdla6_cx)}  "
                       f"({elapsed:5.1f}s)  {status}")
            row = {
                "pattern": pat,
                "mdla6_cx_ms": f"{mdla6_cx_ms:.3f}" if mdla6_cx_ms is not None else "",
                "fast_ms": f"{fast_ms:.3f}" if fast_ms is not None else "",
                "rtl_ms": f"{rtl_ms:.3f}" if rtl_ms is not None else "",
                "fast_over_mdla6_cx": fast_mdla6_cx,
                "rtl_over_fast": rtl_fast,
                "rtl_over_mdla6_cx": rtl_mdla6_cx,
                "status": status,
                "fast_status": fast_status,
                "rtl_status": rtl_status,
            }
            rows_out.append(row)
            _checkpoint_compare(rows_out)
            if not args.keep_bin:
                for bin_path in (
                    _artefact_paths(model_path)["prog"],
                    _mode_paths(model_path, "rtl")["prog"],
                    _mode_paths(model_path, "rtl_compare")["prog"],
                ):
                    try:
                        if bin_path.exists():
                            bin_path.unlink()
                    except OSError:
                        pass

        n_both = sum(1 for r in rows_out if r.get("fast_ms") and r.get("rtl_ms"))
        total_fast = sum(float(r["fast_ms"]) for r in rows_out if r.get("fast_ms"))
        total_rtl = sum(float(r["rtl_ms"]) for r in rows_out if r.get("rtl_ms"))
        total_s = time.time() - t_total
        print(f"\n==== summary: compared {n_both}/{len(rows_out)}, "
              f"fast total {total_fast:.1f} ms, rtl total {total_rtl:.1f} ms, "
              f"wall {total_s:.0f}s ====", flush=True)
        print(f"csv: {csv_path}", flush=True)
        if not args.no_html:
            print(f"html: {HERE / compare_html}", flush=True)
        return

    if args.compare_cx_rtl:
        compare_csv = default_csv.with_name(
            f"{default_csv.stem}.cx_rtl_compare{default_csv.suffix}")
        csv_path = Path(args.csv_out)
        if csv_path == default_csv:
            csv_path = compare_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        prior_full = {} if args.rerun_all else _load_prior_csv(csv_path)
        prior_ok = {
            p: r for p, r in prior_full.items()
            if r.get("status") == "ok" and r.get("rtl_ms") and r.get("cx_ms")
        }
        profile_path = Path(profile_html)
        compare_html = f"{profile_path.stem}.cx_rtl_compare{profile_path.suffix}"
        mdla6_cx_ms_by_pattern = _load_mdla6_cx_ms(order_csv)
        rtl_suffix = _mode_suffix("rtl", "rtl")
        cx_suffix = _mode_suffix("cx", "cx")

        def _mdla6_cx_ms_for(pattern: str) -> float | None:
            return mdla6_cx_ms_by_pattern.get(_normalise_pattern(pattern))

        def _fill_synth_compare_ms(row: dict) -> dict:
            out = dict(row)
            mdla6_cx_ms = out.get("mdla6_cx_ms")
            if not mdla6_cx_ms:
                value = _mdla6_cx_ms_for(out.get("pattern", ""))
                out["mdla6_cx_ms"] = f"{value:.3f}" if value is not None else ""
            if not out.get("cx_over_rtl"):
                out["cx_over_rtl"] = _ratio_from_ms(
                    out.get("cx_ms", ""), out.get("rtl_ms", ""))
            if not out.get("rtl_over_mdla6_cx"):
                out["rtl_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("rtl_ms", ""), out.get("mdla6_cx_ms", ""))
            if not out.get("cx_over_mdla6_cx"):
                out["cx_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("cx_ms", ""), out.get("mdla6_cx_ms", ""))
            return out

        def _checkpoint_synth_compare(rows: list[dict]) -> None:
            seen = {r["pattern"] for r in rows}
            merged = [_fill_synth_compare_ms(r) for r in rows]
            for pat, prow in prior_full.items():
                if pat not in seen:
                    merged.append(_fill_synth_compare_ms(prow))
            fields = [
                "pattern", "mdla6_cx_ms", "rtl_ms", "cx_ms",
                "cx_over_rtl", "rtl_over_mdla6_cx", "cx_over_mdla6_cx",
                "status", "rtl_status", "cx_status",
            ]
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(merged)
            if not args.no_html:
                _write_cx_rtl_compare_index(f"{profile_title} — rtl vs cx",
                                               compare_html, merged, csv_path)

        try:
            rel_model_dir = model_dir.relative_to(REPO_ROOT)
        except ValueError:
            rel_model_dir = model_dir
        print(f"==== MDLA7 {corpus_name} rtl vs cx regression: {len(patterns)} models "
              f"(from {rel_model_dir}) ====", flush=True)
        rows_out = []
        t_total = time.time()
        for i, pat in enumerate(patterns, 1):
            if pat in prior_ok and (
                    args.no_html or _cx_rtl_compare_report_exists_for(pat, model_dir)):
                cached = prior_ok[pat]
                cached_filled = _fill_synth_compare_ms(cached)
                display_pat = _fit_cell(pat)
                _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                           f"mdla6_cx={_ms_cell(cached_filled.get('mdla6_cx_ms', ''))} "
                           f"rtl={_ms_cell(cached.get('rtl_ms', ''))} "
                           f"cx={_ms_cell(cached.get('cx_ms', ''))} "
                           f"cx/rtl={_ratio_cell(cached_filled.get('cx_over_rtl', ''))} "
                           f"cx/mdla6_cx={_ratio_cell(cached_filled.get('cx_over_mdla6_cx', ''))} cached  "
                           f"{cached.get('status', 'ok')}")
                rows_out.append(cached_filled)
                _checkpoint_synth_compare(rows_out)
                continue

            t0 = time.time()

            def _progress(stage: str) -> None:
                elapsed = time.time() - t0
                display_pat = _fit_cell(pat)
                _row_update(f"[{i:>2}/{len(patterns)}] {display_pat} "
                            f"{'—':>10s}      ({elapsed:5.1f}s)  "
                            f"running {stage}...")

            _, rtl_ms, _, _, rtl_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"rtl {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="rtl",
                l1_timing="rtl")
            _, cx_ms, _, _, cx_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"cx {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="cx",
                l1_timing="cx")
            mdla6_cx_ms = _mdla6_cx_ms_for(pat)
            cx_rtl = _ratio_from_ms(cx_ms, rtl_ms)
            rtl_mdla6_cx = _ratio_from_ms(rtl_ms, mdla6_cx_ms)
            cx_mdla6_cx = _ratio_from_ms(cx_ms, mdla6_cx_ms)
            status = "ok" if rtl_status == "ok" and cx_status == "ok" else f"{rtl_status}/{cx_status}"
            model_path = model_dir / f"{_normalise_pattern(pat)}.tflite"
            if not args.no_html and rtl_ms is not None and cx_ms is not None:
                try:
                    _write_cx_rtl_compare_html(
                        model_path,
                        _mode_paths(model_path, rtl_suffix)["html"],
                        _mode_paths(model_path, cx_suffix)["html"],
                        mdla6_cx_ms, rtl_ms, cx_ms, rtl_status, cx_status)
                except Exception as e:
                    status = f"{status}; html-fail: {str(e)[:80]}"
            elapsed = time.time() - t0
            display_pat = _fit_cell(pat)
            _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                       f"mdla6_cx={f'{mdla6_cx_ms:>8.2f} ms' if mdla6_cx_ms is not None else f'{chr(8212):>8s}    '} "
                       f"rtl={f'{rtl_ms:>8.2f} ms' if rtl_ms is not None else f'{chr(8212):>8s}    '} "
                       f"cx={f'{cx_ms:>8.2f} ms' if cx_ms is not None else f'{chr(8212):>8s}    '} "
                       f"cx/rtl={_ratio_cell(cx_rtl)} cx/mdla6_cx={_ratio_cell(cx_mdla6_cx)}  "
                       f"({elapsed:5.1f}s)  {status}")
            row = {
                "pattern": pat,
                "mdla6_cx_ms": f"{mdla6_cx_ms:.3f}" if mdla6_cx_ms is not None else "",
                "rtl_ms": f"{rtl_ms:.3f}" if rtl_ms is not None else "",
                "cx_ms": f"{cx_ms:.3f}" if cx_ms is not None else "",
                "cx_over_rtl": cx_rtl,
                "rtl_over_mdla6_cx": rtl_mdla6_cx,
                "cx_over_mdla6_cx": cx_mdla6_cx,
                "status": status,
                "rtl_status": rtl_status,
                "cx_status": cx_status,
            }
            rows_out.append(row)
            _checkpoint_synth_compare(rows_out)
            if not args.keep_bin:
                for bin_path in (
                    _mode_paths(model_path, rtl_suffix)["prog"],
                    _mode_paths(model_path, cx_suffix)["prog"],
                    _mode_paths(model_path, "cx_rtl_compare")["prog"],
                ):
                    try:
                        if bin_path.exists():
                            bin_path.unlink()
                    except OSError:
                        pass

        n_both = sum(1 for r in rows_out if r.get("rtl_ms") and r.get("cx_ms"))
        total_rtl = sum(float(r["rtl_ms"]) for r in rows_out if r.get("rtl_ms"))
        total_cx = sum(float(r["cx_ms"]) for r in rows_out if r.get("cx_ms"))
        total_s = time.time() - t_total
        print(f"\n==== summary: compared {n_both}/{len(rows_out)}, "
              f"rtl total {total_rtl:.1f} ms, cx total {total_cx:.1f} ms, "
              f"wall {total_s:.0f}s ====", flush=True)
        print(f"csv: {csv_path}", flush=True)
        if not args.no_html:
            print(f"html: {HERE / compare_html}", flush=True)
        return

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prior_full = {} if args.rerun_all else _load_prior_csv(csv_path)
    prior_ok = {} if args.rerun_all else _load_prior_results(csv_path, fast_only=args.fast_only)
    if prior_ok:
        print(f"  (cache: {len(prior_ok)} prior ok rows in {csv_path.name}; "
              f"--rerun-all to ignore)", flush=True)
    mdla6_cx_ms_by_pattern = _load_mdla6_cx_ms(order_csv)
    has_mdla6_cx = bool(mdla6_cx_ms_by_pattern)

    def _mdla6_cx_ms_for(pattern: str) -> float | None:
        return mdla6_cx_ms_by_pattern.get(_normalise_pattern(pattern))

    def _attach_mdla6_cx(row: dict) -> dict:
        if not has_mdla6_cx:
            return row
        out = dict(row)
        mdla6_cx_ms = out.get("mdla6_cx")
        if not mdla6_cx_ms:
            value = _mdla6_cx_ms_for(out.get("pattern", ""))
            out["mdla6_cx"] = _format_ms(value)
        if not out.get("fast_over_mdla6_cx"):
            out["fast_over_mdla6_cx"] = _ratio_from_ms(
                out.get("mdla7_ms", ""), out.get("mdla6_cx", ""))
        return out

    def _checkpoint(rows: list[dict]) -> None:
        seen = {r["pattern"] for r in rows}
        merged = [_attach_mdla6_cx(r) for r in rows]
        for pat, prow in prior_full.items():
            if pat not in seen:
                merged.append(_attach_mdla6_cx(prow))
        fields = [
            "pattern",
            *(["mdla6_cx", "fast_over_mdla6_cx"] if has_mdla6_cx else []),
            "mdla7_ms",
            "mdla7_conflict_ms",
            "mdla7_mesh_ms",
            "status",
            "conflict_status",
            "mesh_status",
        ]
        if microblock_metrics:
            fields.extend([
                "fuse_hit", "fuse_flows", "streamed_layers",
                "mb_hit", "mb_count", "mb_layers", "mb_stages",
            ])
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)

    try:
        rel_model_dir = model_dir.relative_to(REPO_ROOT)
    except ValueError:
        rel_model_dir = model_dir
    model_label = ""
    if args.l1_timing != "fast" or args.engine_model != "fast":
        model_label = f", L1={args.l1_timing}, engine={args.engine_model}"
    print(f"==== MDLA7 {corpus_name} regression: {len(patterns)} models "
          f"(from {rel_model_dir}{model_label}) ====", flush=True)

    rows_out = []
    t_total = time.time()
    for i, pat in enumerate(patterns, 1):
        if pat in prior_ok and (args.no_html or _report_exists_for(
                pat, model_dir, engine_model=args.engine_model,
                l1_timing=args.l1_timing)):
            cached = prior_ok[pat]
            cached_conflict_ms = "" if args.fast_only else cached.get("mdla7_conflict_ms", "")
            cached_mesh_ms = "" if args.fast_only else cached.get("mdla7_mesh_ms", "")
            cached_conflict_status = "" if args.fast_only else cached.get("conflict_status", "ok")
            cached_mesh_status = "" if args.fast_only else cached.get("mesh_status", "ok")
            cached = _attach_mdla6_cx(cached)
            mode_suffix = _mode_suffix(args.l1_timing, args.engine_model)
            model_suffix = "" if not mode_suffix else f"/{mode_suffix}"
            suffix = ((cached.get("status", "ok") + model_suffix) if args.fast_only
                      else f"{cached.get('status', 'ok')}/{cached_conflict_status}/{cached_mesh_status}{model_suffix}")
            model_path = model_dir / f"{pat}.tflite"
            mb = (_microblock_metrics_for(model_path, args.l1_timing, args.engine_model)
                  if microblock_metrics else {})
            mb_suffix = (f" fuse={mb.get('fuse_hit', 'no')}"
                         f" mb={mb.get('mb_count', '0')}:{mb.get('mb_stages', '')}"
                         if microblock_metrics else "")
            display_pat = _fit_cell(pat)
            _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                       f"{'mdla6_cx=' + _ms_cell(cached.get('mdla6_cx', '')) + ' ' if has_mdla6_cx else ''}"
                       f"fast={_ms_cell(cached.get('mdla7_ms', ''))} "
                       f"{'fast/mdla6_cx=' + _ratio_cell(cached.get('fast_over_mdla6_cx', '')) + ' ' if has_mdla6_cx else ''}"
                       f"conflict={_ms_cell(cached_conflict_ms)} "
                       f"mesh={_ms_cell(cached_mesh_ms)} cached  "
                       f"{suffix}{mb_suffix}")
            row = {
                "pattern": pat,
                "mdla6_cx": cached.get("mdla6_cx", ""),
                "fast_over_mdla6_cx": cached.get("fast_over_mdla6_cx", ""),
                "mdla7_ms": cached.get("mdla7_ms", ""),
                "mdla7_conflict_ms": cached_conflict_ms,
                "mdla7_mesh_ms": cached_mesh_ms,
                "status": cached.get("status", "ok"),
                "conflict_status": cached_conflict_status,
                "mesh_status": cached_mesh_status,
            }
            row.update(mb)
            rows_out.append(row)
            _checkpoint(rows_out)
            continue

        t0 = time.time()

        def _progress(stage: str) -> None:
            elapsed = time.time() - t0
            display_pat = _fit_cell(pat)
            _row_update(f"[{i:>2}/{len(patterns)}] {display_pat} "
                        f"{'—':>10s}      ({elapsed:5.1f}s)  "
                        f"running {stage}...")

        _, ms, conflict_ms, mesh_ms, status, conflict_status, mesh_status = run_one(
            pat, model_dir, progress=_progress, fast_only=args.fast_only,
            skip_html=args.no_html, engine_model=args.engine_model,
            l1_timing=args.l1_timing)
        elapsed = time.time() - t0
        ms_str = f"{ms:>8.2f} ms" if ms is not None else f"{'—':>8s}    "
        conflict_str = (f"{conflict_ms:>8.2f} ms" if conflict_ms is not None
                        else f"{'—':>8s}    ")
        mesh_str = (f"{mesh_ms:>8.2f} ms" if mesh_ms is not None
                    else f"{'—':>8s}    ")
        mode_suffix = _mode_suffix(args.l1_timing, args.engine_model)
        model_suffix = "" if not mode_suffix else f"/{mode_suffix}"
        suffix = ((status + model_suffix) if args.fast_only
                  else f"{status}/{conflict_status}/{mesh_status}{model_suffix}")
        model_path = model_dir / f"{pat}.tflite"
        mb = (_microblock_metrics_for(model_path, args.l1_timing, args.engine_model)
              if microblock_metrics else {})
        mb_suffix = (f" fuse={mb.get('fuse_hit', 'no')}"
                     f" mb={mb.get('mb_count', '0')}:{mb.get('mb_stages', '')}"
                     if microblock_metrics else "")
        display_pat = _fit_cell(pat)
        mdla6_cx_ms = _mdla6_cx_ms_for(pat)
        fast_mdla6_cx = _ratio_from_ms(ms, mdla6_cx_ms)
        _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                   f"{'mdla6_cx=' + (f'{mdla6_cx_ms:>8.2f} ms' if mdla6_cx_ms is not None else f'{chr(8212):>8s}    ') + ' ' if has_mdla6_cx else ''}"
                   f"fast={ms_str} conflict={conflict_str} mesh={mesh_str}  "
                   f"{'fast/mdla6_cx=' + _ratio_cell(fast_mdla6_cx) + ' ' if has_mdla6_cx else ''}"
                   f"({elapsed:5.1f}s)  "
                   f"{suffix}{mb_suffix}")
        row = {
            "pattern": pat,
            "mdla6_cx": _format_ms(mdla6_cx_ms),
            "fast_over_mdla6_cx": fast_mdla6_cx,
            "mdla7_ms": f"{ms:.3f}" if ms is not None else "",
            "mdla7_conflict_ms": f"{conflict_ms:.3f}" if conflict_ms is not None else "",
            "mdla7_mesh_ms": f"{mesh_ms:.3f}" if mesh_ms is not None else "",
            "status": status,
            "conflict_status": conflict_status,
            "mesh_status": mesh_status,
        }
        row.update(mb)
        rows_out.append(row)
        _checkpoint(rows_out)

        if not args.keep_bin:
            mode_suffix = _mode_suffix(args.l1_timing, args.engine_model)
            if mode_suffix:
                bin_paths = (
                    _mode_paths(model_path, mode_suffix)["prog"],
                    _mode_paths(model_path, f"{mode_suffix}.conflict")["prog"],
                    _mode_paths(model_path, f"{mode_suffix}.mesh")["prog"],
                )
            else:
                bin_paths = (
                    _artefact_paths(model_path)["prog"],
                    OUT_DIR / f"{pat}.conflict.bin",
                    OUT_DIR / f"{pat}.mesh.bin",
                )
            for bin_path in bin_paths:
                try:
                    if bin_path.exists():
                        bin_path.unlink()
                except OSError:
                    pass

    n_fast = sum(1 for r in rows_out if r.get("mdla7_ms"))
    n_conflict = sum(1 for r in rows_out if r.get("mdla7_conflict_ms"))
    n_mesh = sum(1 for r in rows_out if r.get("mdla7_mesh_ms"))
    total_ms = sum(float(r["mdla7_ms"]) for r in rows_out if r.get("mdla7_ms"))
    total_s = time.time() - t_total
    if args.fast_only:
        primary_label = _mode_suffix(args.l1_timing, args.engine_model) or "fast"
        print(f"\n==== summary: {primary_label} {n_fast}/{len(rows_out)} ran, "
              f"sim total {total_ms:.1f} ms, wall {total_s:.0f}s ====",
              flush=True)
    else:
        print(f"\n==== summary: fast {n_fast}/{len(rows_out)} ran, "
              f"conflict {n_conflict}/{len(rows_out)} ran, "
              f"mesh {n_mesh}/{len(rows_out)} ran, "
              f"sim total {total_ms:.1f} ms, wall {total_s:.0f}s ====",
              flush=True)
    print(f"csv: {csv_path}", flush=True)
    if not args.no_html:
        _refresh_profile_index(profile_title, profile_html, csv_path)
        print(f"html: {HERE / profile_html}", flush=True)
