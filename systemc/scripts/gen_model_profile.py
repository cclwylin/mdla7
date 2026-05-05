#!/usr/bin/env python3
"""Generate systemc/model_profile.html from current output reports.

The generated page includes an embedded snapshot for direct file:// viewing and
also refreshes itself from output/ when served by a simple HTTP server, where
directory listing is available.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEMC = HERE.parent
OUT_DIR = SYSTEMC / "output"
HTML_OUT = SYSTEMC / "model_profile.html"
REGRESSION_CSV = OUT_DIR / "mdla6_pattern_regression.csv"


def normalise_pattern(pat: str) -> str:
    s = pat.strip()
    if s.endswith(".cut"):
        s = s[:-4]
    s = s.replace("__", "_")
    return s


def load_metrics() -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    for path in (SYSTEMC / "mdla6_ethz_v6_sorted.csv", REGRESSION_CSV):
        if not path.exists():
            continue
        with path.open(newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                pat = (row.get("pattern") or row.get("Pattern") or "").strip()
                cx = (row.get("mdla6_cx") or row.get("CX") or "").strip()
                ms = (row.get("mdla7_ms") or "").strip()
                if not pat:
                    continue
                out[normalise_pattern(pat)] = (pat, cx, ms)
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


def collect_rows() -> list[dict[str, object]]:
    metrics = load_metrics()
    rows: list[dict[str, object]] = []
    if not OUT_DIR.exists():
        return rows
    for html in sorted(OUT_DIR.glob("*.html")):
        stem = html.stem
        if stem.startswith("model_profile") or stem.startswith("._"):
            continue
        pat, cx, csv_ms = metrics.get(stem, (stem, "", ""))
        our_ms = load_our_ms(stem, csv_ms)
        ratio = None
        try:
            cx_f = float(cx)
            if cx_f > 0 and our_ms is not None:
                ratio = our_ms / cx_f
        except ValueError:
            pass
        rows.append({
            "pattern": pat,
            "stem": stem,
            "link": f"output/{html.name}",
            "cx": cx,
            "our_ms": our_ms,
            "ratio": ratio,
        })
    def key(row: dict[str, object]):
        ratio = row.get("ratio")
        if isinstance(ratio, (int, float)):
            return (0, -ratio, str(row.get("pattern") or ""))
        return (1, str(row.get("pattern") or ""))
    rows.sort(key=key)
    return rows


def main() -> None:
    rows = collect_rows()
    rows_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDLA7 Model Profiles</title>
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
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
a {{ color:var(--link); text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
tr:hover td {{ background:#f4f7fb; }}
</style>
</head>
<body>
<main>
<h1>MDLA7 Model Profiles</h1>
<div class="bar">
  <button id="refresh">Refresh Output</button>
  <input id="filter" placeholder="filter pattern">
  <span class="meta" id="status"></span>
</div>
<table>
  <thead>
    <tr><th>pattern</th><th>link</th><th class="num">cx</th><th class="num">our_ms</th><th class="num">myms/cx</th></tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
</main>
<script>
const EMBEDDED_ROWS = {rows_json};
let rows = EMBEDDED_ROWS.slice();

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
function sortRows(xs) {{
  xs.sort((a,b) => {{
    const ar = ratioOf(a), br = ratioOf(b);
    if (ar !== null && br !== null && ar !== br) return br - ar;
    if (ar !== null && br === null) return -1;
    if (ar === null && br !== null) return 1;
    return String(a.pattern).localeCompare(String(b.pattern));
  }});
  return xs;
}}
function render() {{
  const q = document.getElementById("filter").value.trim().toLowerCase();
  const body = document.getElementById("rows");
  body.innerHTML = "";
  for (const r of sortRows(rows.slice())) {{
    if (q && !String(r.pattern).toLowerCase().includes(q)) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${{esc(r.pattern)}}</td>` +
      `<td><a href="${{escAttr(r.link)}}">${{esc(r.stem || r.pattern)}}</a></td>` +
      `<td class="num">${{esc(r.cx || "")}}</td>` +
      `<td class="num">${{esc(fmtMs(r.our_ms))}}</td>` +
      `<td class="num">${{esc(fmtRatio(r))}}</td>`;
    body.appendChild(tr);
  }}
  document.getElementById("status").textContent =
    `${{body.children.length}} / ${{rows.length}} profiles`;
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
      ms: row.mdla7_ms || ""
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
      fetch("output/mdla6_pattern_regression.csv").then(r => r.ok ? r.text() : "").catch(() => "")
    ]);
    const cx = csvParse(csvText);
    const doc = new DOMParser().parseFromString(dirText, "text/html");
    const names = [...doc.querySelectorAll("a")]
      .map(a => a.getAttribute("href") || "")
      .map(h => decodeURIComponent(h.split("/").pop()))
      .filter(n => n && n.endsWith(".html") && n !== "model_profile.html" && !n.startsWith("._"));
    const next = [];
    for (const name of names) {{
      const stem = name.replace(/\\.html$/, "");
      let ms = cx[stem] && cx[stem].ms ? Number(cx[stem].ms) : null;
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
        cx: (cx[stem] && cx[stem].cx) || "",
        our_ms: ms,
        ratio: null
      }});
    }}
    if (next.length) rows = next;
    render();
  }} catch (e) {{
    status.textContent = `using embedded snapshot; serve systemc/ over HTTP for live output scan`;
    render();
  }}
}}
document.getElementById("refresh").addEventListener("click", refreshFromOutput);
document.getElementById("filter").addEventListener("input", render);
render();
refreshFromOutput();
</script>
</body>
</html>
"""
    HTML_OUT.write_text(html)
    print(f"model profile: {HTML_OUT}")


if __name__ == "__main__":
    main()
