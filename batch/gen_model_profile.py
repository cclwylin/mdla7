#!/usr/bin/env python3
"""Generate an HTML index from current output reports.

The generated page includes an embedded snapshot for direct file:// viewing and
also refreshes itself from output/ when served by a simple HTTP server, where
directory listing is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent              # .../batch/
REPO_ROOT = HERE.parent
SYSTEMC = REPO_ROOT / "systemc"
OUT_DIR = HERE / "output"
DEFAULT_HTML_OUT = HERE / "profile_mdla6_pattern.html"
DEFAULT_REGRESSION_CSV = OUT_DIR / "mdla6_pattern_regression.csv"
DEFAULT_BASELINE_CSV = HERE / "mdla6_ethz_v6_sorted.csv"
TRANSFORMER_PATTERNS = {
    "gpt2_quant",
    "llama2_quant",
    "mobilebert_quant",
    "vit_b16_quant",
    "swin_float",
    "swin_quant",
    "mobilevit_v2_float",
    "mobilevit_v2_quant",
    "sam_float",
    "sam_quant",
}


def normalise_pattern(pat: str) -> str:
    s = pat.strip()
    if s.endswith(".cut"):
        s = s[:-4]
    s = s.replace("__", "_")
    return s


def load_metrics(paths: list[Path]) -> dict[str, tuple[str, str, str, str, str]]:
    out: dict[str, tuple[str, str, str, str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                pat = (row.get("pattern") or row.get("Pattern") or "").strip()
                cx = (row.get("mdla6_cx") or row.get("CX") or "").strip()
                ms = (row.get("mdla7_ms") or "").strip()
                conflict_ms = (row.get("mdla7_conflict_ms") or
                               row.get("conflict_ms") or "").strip()
                mesh_ms = (row.get("mdla7_mesh_ms") or
                           row.get("mesh_ms") or "").strip()
                if not pat:
                    continue
                out[normalise_pattern(pat)] = (pat, cx, ms, conflict_ms, mesh_ms)
    return out


def load_our_ms(stem: str, csv_ms: str = "") -> float | None:
    if csv_ms:
        try:
            return float(csv_ms)
        except ValueError:
            pass
    prof = OUT_DIR / f"{stem}.profile.json"
    if prof.exists():
        try:
            data = json.loads(prof.read_text())
            v = ((data.get("summary") or {}).get("total_cycles"))
            return int(v) / 1.9e6 if v is not None else None
        except Exception:
            pass
    html = OUT_DIR / f"{stem}.html"
    if html.exists():
        text = html.read_text(errors="ignore")
        m = re.search(r"Sim time:</b>\s*([\d.]+)\s*ms", text)
        if m:
            return float(m.group(1))
        m = re.search(r"\(([\d,]+)\s+cycles\)", text)
        if m:
            return int(m.group(1).replace(",", "")) / 1.9e6
    return None


def collect_rows(metrics_csvs: list[Path],
                 only_metric_rows: bool = False) -> list[dict[str, object]]:
    metrics = load_metrics(metrics_csvs)
    allowed = set(metrics) if only_metric_rows else None
    rows: list[dict[str, object]] = []
    if not OUT_DIR.exists():
        return rows
    for html in sorted(OUT_DIR.glob("*.html")):
        stem = html.stem
        if stem.startswith(("model_profile", "profile_")) or stem.startswith("._"):
            continue
        if stem.endswith(".fast") or stem.endswith(".conflict") or stem.endswith(".mesh"):
            continue
        if allowed is not None and stem not in allowed:
            continue
        pat, cx, csv_ms, csv_conflict_ms, csv_mesh_ms = metrics.get(
            stem, (stem, "", "", "", ""))
        our_ms = load_our_ms(stem, csv_ms)
        conflict_ms = None
        if csv_conflict_ms:
            try:
                conflict_ms = float(csv_conflict_ms)
            except ValueError:
                pass
        mesh_ms = None
        if csv_mesh_ms:
            try:
                mesh_ms = float(csv_mesh_ms)
            except ValueError:
                pass
        ratio = None
        conflict_ratio = None
        mesh_ratio = None
        try:
            cx_f = float(cx)
            if cx_f > 0 and our_ms is not None:
                ratio = our_ms / cx_f
        except ValueError:
            pass
        if our_ms and our_ms > 0 and conflict_ms is not None:
            conflict_ratio = conflict_ms / our_ms
        if our_ms and our_ms > 0 and mesh_ms is not None:
            mesh_ratio = mesh_ms / our_ms
        rows.append({
            "pattern": pat,
            "stem": stem,
            "type": "Hotspot" if "_L" in stem else (
                "Transformer" if stem in TRANSFORMER_PATTERNS else ""),
            "link": f"output/{html.name}",
            "cx": cx,
            "our_ms": our_ms,
            "conflict_ms": conflict_ms,
            "mesh_ms": mesh_ms,
            "ratio": ratio,
            "conflict_ratio": conflict_ratio,
            "mesh_ratio": mesh_ratio,
        })
    def key(row: dict[str, object]):
        ratio = row.get("ratio")
        if isinstance(ratio, (int, float)):
            return (0, -ratio, str(row.get("pattern") or ""))
        return (1, str(row.get("pattern") or ""))
    rows.sort(key=key)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-out", default=str(DEFAULT_HTML_OUT))
    ap.add_argument("--title", default="MDLA7 MDLA6 Pattern Profiles")
    ap.add_argument("--metrics-csv", action="append", default=[],
                    help="CSV with pattern/mdla7_ms columns; can be repeated")
    ap.add_argument("--only-metrics-rows", action="store_true",
                    help="only include output HTML whose stem appears in metrics CSV")
    ap.add_argument("--hide-cx", action="store_true",
                    help="hide MDLA6 cx and myms/cx comparison columns")
    args = ap.parse_args()

    metrics_csvs = [Path(p) for p in args.metrics_csv]
    if not metrics_csvs:
        metrics_csvs = [DEFAULT_BASELINE_CSV, DEFAULT_REGRESSION_CSV]

    html_out = Path(args.html_out)
    if not html_out.is_absolute():
        html_out = HERE / html_out

    rows = collect_rows(metrics_csvs, args.only_metrics_rows)
    rows_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    title = args.title
    show_cx = not args.hide_cx
    show_cx_json = "true" if show_cx else "false"
    default_sort_key = "ratio" if show_cx else "mesh_ratio"
    default_sort_key_json = json.dumps(default_sort_key)
    cx_headers = """
      <th class="num"><button class="sort-btn" data-sort-key="cx">cx <span class="sort-mark"></span></button></th>""" if show_cx else ""
    ratio_headers = """
      <th class="num"><button class="sort-btn" data-sort-key="ratio">myms/cx <span class="sort-mark"></span></button></th>""" if show_cx else ""
    live_csv = DEFAULT_REGRESSION_CSV
    for path in reversed(metrics_csvs):
        if path.parent.resolve() == OUT_DIR.resolve():
            live_csv = path
            break
    try:
        live_csv_rel = live_csv.relative_to(HERE).as_posix()
    except ValueError:
        live_csv_rel = live_csv.as_posix()
    live_csv_json = json.dumps(live_csv_rel)
    only_metric_rows_json = "true" if args.only_metrics_rows else "false"
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg:#f7f8fa; --panel:#ffffff; --line:#d8dde6; --text:#17202a;
  --muted:#657080; --head:#eef2f7; --link:#0b5cad;
}}
body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:var(--text); background:var(--bg); }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 12px; font-size:24px; }}
.bar {{ display:flex; align-items:center; gap:10px; margin:0 0 14px; flex-wrap:wrap; }}
button {{ border:1px solid var(--line); background:#fff; padding:6px 10px; border-radius:6px;
         cursor:pointer; }}
input {{ border:1px solid var(--line); border-radius:6px; padding:7px 9px; min-width:240px; }}
.meta {{ color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; background:var(--panel);
        border:1px solid var(--line); }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ background:var(--head); position:sticky; top:0; z-index:1; }}
th.pattern {{ width:32%; min-width:260px; }}
th .sort-btn {{ display:inline-flex; align-items:center; gap:4px; border:0; background:transparent;
                padding:0; color:inherit; font:inherit; font-weight:600; cursor:pointer; }}
th .sort-btn:hover {{ color:var(--link); text-decoration:underline; }}
th .sort-mark {{ display:inline-block; min-width:1.1em; color:var(--muted); font-size:12px; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
a {{ color:var(--link); text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
tr:hover td {{ background:#f4f7fb; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<div class="bar">
  <button id="refresh">Refresh Output</button>
  <input id="filter" placeholder="filter pattern">
  <span class="meta" id="status"></span>
</div>
<table>
  <thead>
    <tr>
      <th class="pattern"><button class="sort-btn" data-sort-key="pattern">pattern <span class="sort-mark"></span></button></th>
      <th><button class="sort-btn" data-sort-key="stem">link <span class="sort-mark"></span></button></th>
      <th><button class="sort-btn" data-sort-key="type">type <span class="sort-mark"></span></button></th>
      {cx_headers}
      <th class="num"><button class="sort-btn" data-sort-key="our_ms">our_ms <span class="sort-mark"></span></button></th>
      <th class="num"><button class="sort-btn" data-sort-key="conflict_ms">conflict_ms <span class="sort-mark"></span></button></th>
      <th class="num"><button class="sort-btn" data-sort-key="mesh_ms">mesh_ms <span class="sort-mark"></span></button></th>
      {ratio_headers}
      <th class="num"><button class="sort-btn" data-sort-key="conflict_ratio">conflict/fast <span class="sort-mark"></span></button></th>
      <th class="num"><button class="sort-btn" data-sort-key="mesh_ratio">mesh/fast <span class="sort-mark"></span></button></th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
</main>
<script>
const EMBEDDED_ROWS = {rows_json};
const LIVE_CSV = {live_csv_json};
const ONLY_METRIC_ROWS = {only_metric_rows_json};
const SHOW_CX = {show_cx_json};
const DEFAULT_SORT_KEY = {default_sort_key_json};
let rows = EMBEDDED_ROWS.slice();
let sortState = {{ key: DEFAULT_SORT_KEY, dir: "desc", default: true }};

function fmtMs(v) {{
  if (v === null || v === undefined || v === "") return "";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : String(v);
}}
function ratioOf(r) {{
  if (r.ratio !== null && r.ratio !== undefined && r.ratio !== "") {{
    const n = Number(r.ratio);
    if (Number.isFinite(n)) return n;
  }}
  const ms = Number(r.our_ms), cx = Number(r.cx);
  return Number.isFinite(ms) && Number.isFinite(cx) && cx > 0 ? ms / cx : null;
}}
function fmtRatio(r) {{
  const n = ratioOf(r);
  return n === null ? "" : n.toFixed(2);
}}
function conflictRatioOf(r) {{
  if (r.conflict_ratio !== null && r.conflict_ratio !== undefined && r.conflict_ratio !== "") {{
    const n = Number(r.conflict_ratio);
    if (Number.isFinite(n)) return n;
  }}
  const conflict = Number(r.conflict_ms), fast = Number(r.our_ms);
  return Number.isFinite(conflict) && Number.isFinite(fast) && fast > 0 ? conflict / fast : null;
}}
function fmtConflictRatio(r) {{
  const n = conflictRatioOf(r);
  return n === null ? "" : n.toFixed(2);
}}
function meshRatioOf(r) {{
  if (r.mesh_ratio !== null && r.mesh_ratio !== undefined && r.mesh_ratio !== "") {{
    const n = Number(r.mesh_ratio);
    if (Number.isFinite(n)) return n;
  }}
  const mesh = Number(r.mesh_ms), fast = Number(r.our_ms);
  return Number.isFinite(mesh) && Number.isFinite(fast) && fast > 0 ? mesh / fast : null;
}}
function fmtMeshRatio(r) {{
  const n = meshRatioOf(r);
  return n === null ? "" : n.toFixed(2);
}}
function sortRows(xs) {{
  if (!sortState.default) {{
    const key = sortState.key;
    const dir = sortState.dir === "asc" ? 1 : -1;
    xs.sort((a,b) => {{
      const av = sortValue(a, key), bv = sortValue(b, key);
      let cmp = 0;
      if (key === "cx" || key === "our_ms" || key === "conflict_ms" ||
          key === "mesh_ms" || key === "ratio" || key === "conflict_ratio" ||
          key === "mesh_ratio") {{
        const an = numOrNull(av), bn = numOrNull(bv);
        if (an !== null && bn !== null) cmp = an - bn;
        else if (an !== null && bn === null) cmp = -1;
        else if (an === null && bn !== null) cmp = 1;
      }} else {{
        cmp = String(av ?? "").localeCompare(String(bv ?? ""), undefined, {{ numeric: true }});
      }}
      if (cmp === 0) cmp = String(a.pattern).localeCompare(String(b.pattern), undefined, {{ numeric: true }});
      return cmp * dir;
    }});
    return xs;
  }}
  xs.sort((a,b) => {{
    const ar = defaultMetric(a), br = defaultMetric(b);
    if (ar !== null && br !== null && ar !== br) return br - ar;
    if (ar !== null && br === null) return -1;
    if (ar === null && br !== null) return 1;
    return String(a.pattern).localeCompare(String(b.pattern));
  }});
  return xs;
}}
function defaultMetric(r) {{
  return DEFAULT_SORT_KEY === "ratio" ? ratioOf(r) : sortValue(r, DEFAULT_SORT_KEY);
}}
function sortValue(r, key) {{
  if (key === "ratio") return ratioOf(r);
  if (key === "conflict_ratio") return conflictRatioOf(r);
  if (key === "mesh_ratio") return meshRatioOf(r);
  if (key === "our_ms") return r.our_ms;
  if (key === "conflict_ms") return r.conflict_ms;
  if (key === "mesh_ms") return r.mesh_ms;
  if (key === "cx") return r.cx;
  if (key === "stem") return r.stem || r.pattern;
  if (key === "type") return r.type || "";
  return r.pattern;
}}
function numOrNull(v) {{
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}}
function updateSortButtons() {{
  document.querySelectorAll(".sort-btn").forEach(btn => {{
    const mark = btn.querySelector(".sort-mark");
    if (!mark) return;
    if (!sortState.default && btn.dataset.sortKey === sortState.key)
      mark.textContent = sortState.dir === "asc" ? "^" : "v";
    else if (sortState.default && btn.dataset.sortKey === DEFAULT_SORT_KEY)
      mark.textContent = "v";
    else
      mark.textContent = "";
  }});
}}
function render() {{
  const q = document.getElementById("filter").value.trim().toLowerCase();
  const body = document.getElementById("rows");
  body.innerHTML = "";
  for (const r of sortRows(rows.slice())) {{
    const parts = [r.pattern, r.stem || "", r.type || "",
                   fmtMs(r.our_ms), fmtMs(r.conflict_ms), fmtMs(r.mesh_ms),
                   fmtConflictRatio(r), fmtMeshRatio(r)];
    if (SHOW_CX) parts.push(r.cx || "", fmtRatio(r));
    const allText = parts.join(" ").toLowerCase();
    if (q && !allText.includes(q)) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${{esc(r.pattern)}}</td>` +
      `<td><a href="${{escAttr(r.link)}}">${{esc(r.stem || r.pattern)}}</a></td>` +
      `<td>${{esc(r.type || "")}}</td>` +
      (SHOW_CX ? `<td class="num">${{esc(r.cx || "")}}</td>` : "") +
      `<td class="num">${{esc(fmtMs(r.our_ms))}}</td>` +
      `<td class="num">${{esc(fmtMs(r.conflict_ms))}}</td>` +
      `<td class="num">${{esc(fmtMs(r.mesh_ms))}}</td>` +
      (SHOW_CX ? `<td class="num">${{esc(fmtRatio(r))}}</td>` : "") +
      `<td class="num">${{esc(fmtConflictRatio(r))}}</td>` +
      `<td class="num">${{esc(fmtMeshRatio(r))}}</td>`;
    body.appendChild(tr);
  }}
  document.getElementById("status").textContent =
    `${{body.children.length}} / ${{rows.length}} profiles`;
  updateSortButtons();
}}
function esc(s) {{
  return String(s ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
}}
function escAttr(s) {{ return esc(s); }}
function csvParse(text) {{
  const lines = text.trim().split(/\\r?\\n/);
  if (!lines.length) return {{}};
  const hdr = lines[0].split(",");
  const out = {{}};
  for (const ln of lines.slice(1)) {{
    const cols = ln.split(",");
    const row = Object.fromEntries(hdr.map((h,i) => [h, cols[i] || ""]));
    const pat = row.pattern || row.Pattern || "";
    const stem = pat.endsWith(".cut") ? pat.slice(0, -4) : pat.replaceAll("__", "_");
    if (stem) out[stem] = {{
      pattern: pat,
      cx: row.mdla6_cx || row.CX || "",
      ms: row.mdla7_ms || "",
      conflict_ms: row.mdla7_conflict_ms || row.conflict_ms || "",
      mesh_ms: row.mdla7_mesh_ms || row.mesh_ms || ""
    }};
  }}
  return out;
}}
async function refreshFromOutput() {{
  const status = document.getElementById("status");
  try {{
    status.textContent = "checking output/ ...";
    const [dirText, csvText] = await Promise.all([
      fetch("output/").then(r => r.text()),
      fetch(LIVE_CSV).then(r => r.ok ? r.text() : "").catch(() => "")
    ]);
    const cx = csvParse(csvText);
    const doc = new DOMParser().parseFromString(dirText, "text/html");
    const names = [...doc.querySelectorAll("a")]
      .map(a => a.getAttribute("href") || "")
      .map(h => decodeURIComponent(h.split("/").pop()))
      .filter(n => n && n.endsWith(".html") && n !== "model_profile.html" &&
                   !n.startsWith("profile_") &&
                   !n.startsWith("._") && !n.endsWith(".fast.html") &&
                   !n.endsWith(".conflict.html") && !n.endsWith(".mesh.html"));
    const next = [];
    for (const name of names) {{
      const stem = name.replace(/\\.html$/, "");
      if (ONLY_METRIC_ROWS && !cx[stem]) continue;
      let ms = cx[stem] && cx[stem].ms ? Number(cx[stem].ms) : null;
      const conflictMs = cx[stem] && cx[stem].conflict_ms ? Number(cx[stem].conflict_ms) : null;
      const meshMs = cx[stem] && cx[stem].mesh_ms ? Number(cx[stem].mesh_ms) : null;
      try {{
        const p = await fetch(`output/${{stem}}.profile.json`);
        if (p.ok) {{
          const j = await p.json();
          if (ms === null && j.summary && j.summary.total_cycles !== undefined)
            ms = Number(j.summary.total_cycles) / 1.9e6;
        }}
      }} catch (_) {{}}
      next.push({{
        pattern: (cx[stem] && cx[stem].pattern) || stem,
        stem, link: `output/${{name}}`,
        type: transformerType(stem),
        cx: (cx[stem] && cx[stem].cx) || "",
        our_ms: ms,
        conflict_ms: conflictMs,
        mesh_ms: meshMs,
        ratio: null,
        conflict_ratio: null,
        mesh_ratio: null
      }});
    }}
    if (next.length) rows = next;
    render();
  }} catch (e) {{
    status.textContent = `using embedded snapshot; serve systemc/ over HTTP for live output scan`;
    render();
  }}
}}
function transformerType(stem) {{
  if (stem.includes("_L")) return "Hotspot";
  const transformer = new Set([
    "gpt2_quant", "llama2_quant", "mobilebert_quant", "vit_b16_quant",
    "swin_float", "swin_quant", "mobilevit_v2_float", "mobilevit_v2_quant",
    "sam_float", "sam_quant"
  ]);
  return transformer.has(stem) ? "Transformer" : "";
}}
document.getElementById("refresh").addEventListener("click", refreshFromOutput);
document.getElementById("filter").addEventListener("input", render);
document.querySelectorAll(".sort-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    const key = btn.dataset.sortKey;
    if (sortState.default || sortState.key !== key) sortState = {{ key, dir: "asc", default: false }};
    else if (sortState.dir === "asc") sortState = {{ key, dir: "desc", default: false }};
    else sortState = {{ key: "ratio", dir: "desc", default: true }};
    render();
  }});
}});
render();
refreshFromOutput();
</script>
</body>
</html>
"""
    html_out.write_text(html)
    print(f"model profile: {html_out}")


if __name__ == "__main__":
    main()
