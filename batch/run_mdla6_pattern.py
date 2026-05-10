#!/usr/bin/env python3
"""Regression sweep restricted to the patterns listed in `mdla6_ethz_v6_sorted.csv`.

For each (pattern, mdla6_cx) row in the input CSV, locates the matching
`.tflite` under `model/ETHZ_v6/`, runs compile_model.py + mdla7_model_runner, and
emits a CSV next to the regular ETHZ sweep that pairs the MDLA7 sim time
with the MDLA6 baseline:

    output/mdla6_pattern_regression.csv
        pattern,mdla6_cx,mdla7_ms,mdla7_conflict_ms,mdla7_mesh_ms,status,conflict_status,mesh_status

    output/<model>.html
        per-model profile report, matching run_model.py's HTML view

Suffix `.cut` in the input CSV (llama2_quant.cut, mobilebert_quant.cut, …)
is stripped before model lookup — those rows in the MDLA6 sheet referred to
truncated/partial graphs; on the MDLA7 side we use the full `.tflite` and let
compile_model skip its unsupported ops via the existing v8.24+ skip paths.

Typos in the input CSV (e.g. `esrgan__int16` with a double underscore) are
normalised to `esrgan_int16`.

By default the script reuses prior `ok` rows from `--csv-out` so a re-run
only re-tests patterns that previously failed (or are new). Pass
`--rerun-all` to ignore the cache and re-compile + re-simulate every row
(useful when a sim or compile-side change might shift cycle counts).

`dped_float` is excluded from the default sweep because its generated program
exceeds the practical 32-bit descriptor/file-offset budget and produces
multi-GB artifacts.

Usage:
    ./batch/run_mdla6_pattern.py                    # default: cache prior ok
    ./batch/run_mdla6_pattern.py --rerun-all        # force re-test all rows
    ./batch/run_mdla6_pattern.py --fast-only        # run fast mode only
    ./batch/run_mdla6_pattern.py --csv-in  <path>   # override input
    ./batch/run_mdla6_pattern.py --csv-out <path>   # override output cache
    ./batch/run_mdla6_pattern.py --filter mobilenet # substring filter
    ./batch/run_mdla6_pattern.py --limit 5          # only run first 5 rows
    ./batch/run_mdla6_pattern.py --offset 10 --limit 5
    ./batch/run_mdla6_pattern.py --include-excluded # debug excluded rows
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE       = Path(__file__).resolve().parent
REPO_ROOT  = HERE.parent
SYSTEMC_DIR = REPO_ROOT / "systemc"
COMPILE_PY = SYSTEMC_DIR / "scripts" / "compile_model.py"
PLOT_PY    = SYSTEMC_DIR / "scripts" / "plot_profile.py"
MODEL_PROFILE_PY = HERE / "gen_model_profile.py"
MODEL_RUNNER = SYSTEMC_DIR / "build" / "mdla7_model_runner"
OUT_DIR    = HERE / "output"

DEFAULT_CSV_IN  = HERE / "mdla6_ethz_v6_sorted.csv"
DEFAULT_CSV_OUT = OUT_DIR / "mdla6_pattern_regression.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "model" / "ETHZ_v6"
EXCLUDED_PATTERNS = {"dped_float"}

# Re-exec into the venv (mirrors run_ethz_v6.py / run_model.py).
VENV_DIR = Path(os.environ.get("MDLA7_VENV") or
                Path.home() / ".venvs/mdla7").expanduser()
VENV_PY  = VENV_DIR / "bin" / "python"

def _reexec_in_venv():
    if not VENV_PY.exists():
        return
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])

_reexec_in_venv()

from run_model import _artefact_paths, _write_html_report  # noqa: E402

# v8.25: mdla7_model_runner.cpp prints "sim time: <cycles> cycles @ 1.9 GHz (= <ms> ms)"
# (pre-v8.25 was "sim time: <ns> ns"). Match both for forward/back compat.
SIM_TIME_RE = re.compile(r"sim time:\s*([\d,]+)\s*(?:cycles|ns)")

# SystemC license banner on stderr (pretty-printed at startup); suppress it
# when picking a meaningful error line out of stderr.
_BANNER_HINTS = ("ALL RIGHTS RESERVED", "Accellera", "Copyright (c)",
                 "ISO/IEC", "SystemC ", "Licensed under")

def _refresh_model_profile_index() -> None:
    try:
        subprocess.run([sys.executable, str(MODEL_PROFILE_PY),
                        "--html-out", "profile_mdla6_pattern.html",
                        "--title", "MDLA7 MDLA6 Pattern Profiles",
                        "--only-metrics-rows"],
                       cwd=str(HERE), capture_output=True, text=True)
    except Exception:
        pass


def _meaningful_stderr_line(stderr: str) -> str:
    """Pick the most informative line from stderr — explicit `layer N: ...`
    errors win; otherwise the last non-banner non-empty line."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"layer\s+\d+:", ln):
            return ln
    for ln in reversed(lines):
        if any(h in ln for h in _BANNER_HINTS):
            continue
        return ln
    return ""


def _normalise_pattern(pat: str) -> str:
    """Strip `.cut` suffix and fix known typos in the MDLA6 sheet."""
    # MDLA6 used truncated graphs for some transformer-class models;
    # MDLA7 runs the full .tflite (compile_model skips unsupported ops).
    if pat.endswith(".cut"):
        pat = pat[: -len(".cut")]
    # Typo: double underscore in esrgan__int16.
    pat = pat.replace("__", "_")
    return pat


def _report_exists_for(pattern: str, model_dir: Path) -> bool:
    canonical = _normalise_pattern(pattern)
    model_path = model_dir / f"{canonical}.tflite"
    return _artefact_paths(model_path)["html"].exists()


def _mode_paths(model_path: Path, mode: str) -> dict[str, Path]:
    paths = _artefact_paths(model_path)
    if mode == "fast":
        return paths
    stem = model_path.stem
    return {
        "prog":  OUT_DIR / f"{stem}.{mode}.bin",
        "prof":  OUT_DIR / f"{stem}.{mode}.profile.json",
        "csv":   OUT_DIR / f"{stem}.{mode}.profile.csv",
        "gantt": OUT_DIR / f"{stem}.{mode}.profile.png",
        "html":  OUT_DIR / f"{stem}.{mode}.html",
    }


def _selected_log_lines(compile_stdout: str, sim_stdout: str) -> list[str]:
    lines: list[str] = []
    for ln in (compile_stdout or "").splitlines():
        if ln.startswith(("compile_model:", "  layer", "  →")):
            lines.append(ln)
    for ln in (sim_stdout or "").splitlines():
        if ln.startswith(("mdla7_model_runner:", "test_model:", "  layer", "  summary",
                          "  sim time", "  DRAM ", "  SRAM ", "  L1Mesh ",
                          "  per-engine", "  utilization", "    ",
                          "  profile", "  csv")):
            lines.append(ln)
    return lines


def _write_mode_html(model_path: Path, paths: dict[str, Path],
                     compile_stdout: str, sim_stdout: str) -> Path:
    if not paths["prof"].exists():
        raise RuntimeError(f"profile missing: {paths['prof'].name}")

    gr = subprocess.run(
        [sys.executable, str(PLOT_PY), str(paths["prof"]), "-o", str(paths["gantt"])],
        capture_output=True, text=True, timeout=120,
    )
    if gr.returncode != 0:
        msg = _meaningful_stderr_line(gr.stderr) or f"exit {gr.returncode}"
        raise RuntimeError(f"gantt-fail: {msg[:80]}")

    return _write_html_report(model_path, paths,
                              _selected_log_lines(compile_stdout, sim_stdout))


def _write_combined_html(model_path: Path,
                         fast_html: Path,
                         conflict_html: Path,
                         mesh_html: Path,
                         fast_ms: float | None,
                         conflict_ms: float | None,
                         mesh_ms: float | None,
                         fast_status: str,
                         conflict_status: str,
                         mesh_status: str) -> Path:
    paths = _artefact_paths(model_path)
    conflict_ratio = ""
    if fast_ms and conflict_ms is not None:
        conflict_ratio = f"{conflict_ms / fast_ms:.3f}x"
    mesh_ratio = ""
    if fast_ms and mesh_ms is not None:
        mesh_ratio = f"{mesh_ms / fast_ms:.3f}x"
    mesh_conflict_ratio = ""
    if conflict_ms and mesh_ms is not None:
        mesh_conflict_ratio = f"{mesh_ms / conflict_ms:.3f}x"
    fast_doc = fast_html.read_text(errors="ignore") if fast_html.exists() else ""
    conflict_doc = conflict_html.read_text(errors="ignore") if conflict_html.exists() else ""
    mesh_doc = mesh_html.read_text(errors="ignore") if mesh_html.exists() else ""
    def ms(v: float | None) -> str:
        return f"{v:.3f} ms" if v is not None else ""
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDLA7 profile — {html.escape(model_path.name)} — fast/conflict/mesh</title>
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
  <h1>{html.escape(model_path.name)} — fast/conflict/mesh profile</h1>
  <div class="summary">
    <span><b>fast:</b> {html.escape(ms(fast_ms))} {html.escape(fast_status)}</span>
    <span><b>conflict:</b> {html.escape(ms(conflict_ms))} {html.escape(conflict_status)}</span>
    <span><b>mesh:</b> {html.escape(ms(mesh_ms))} {html.escape(mesh_status)}</span>
    <span><b>conflict/fast:</b> {html.escape(conflict_ratio)}</span>
    <span><b>mesh/fast:</b> {html.escape(mesh_ratio)}</span>
    <span><b>mesh/conflict:</b> {html.escape(mesh_conflict_ratio)}</span>
  </div>
  <div class="tabs">
    <button class="tab active" data-target="fast">fast</button>
    <button class="tab" data-target="conflict">conflict</button>
    <button class="tab" data-target="mesh">mesh</button>
  </div>
</header>
<section id="fast" class="pane active">
  <iframe title="fast profile" srcdoc="{html.escape(fast_doc, quote=True)}"></iframe>
</section>
<section id="conflict" class="pane">
  <iframe title="conflict profile" srcdoc="{html.escape(conflict_doc, quote=True)}"></iframe>
</section>
<section id="mesh" class="pane">
  <iframe title="mesh profile" srcdoc="{html.escape(mesh_doc, quote=True)}"></iframe>
</section>
<script>
document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.target));
}}));
</script>
</body></html>
"""
    paths["html"].write_text(doc)
    return paths["html"]


def _simulate_one(bin_path: Path, l1_timing: str) -> tuple[float | None, str, str]:
    """Run mdla7_model_runner once. Returns (ms, status, stdout)."""
    try:
        sr = subprocess.run(
            [str(MODEL_RUNNER), str(bin_path), "--quiet", f"--l1-timing={l1_timing}"],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return None, "sim-timeout", ""
    m = SIM_TIME_RE.search(sr.stdout or "")
    if sr.returncode != 0 and not m:
        last = _meaningful_stderr_line(sr.stderr) or f"exit {sr.returncode}"
        return None, f"sim-fail: {last[:80]}", sr.stdout or ""
    if not m:
        return None, "sim-time-missing", sr.stdout or ""
    cycles = int(m.group(1).replace(",", ""))
    ms = cycles / 1.9e6   # spec frequency = 1.9 GHz (post-v8.25)

    sm = re.search(r"summary:\s+(\d+)/(\d+)\s+layers PASS,\s+(\d+)\s+FAIL",
                   sr.stdout or "")
    if sm:
        _, _, n_fail = (int(sm.group(k)) for k in (1, 2, 3))
        status = "ok" if n_fail == 0 else f"{n_fail}-FAIL"
    else:
        status = "ok" if sr.returncode == 0 else f"exit {sr.returncode}"
    return ms, status, sr.stdout or ""


def run_one(pattern: str, model_dir: Path, progress=None,
            fast_only: bool = False,
            skip_html: bool = False) -> tuple[str, float | None, float | None, float | None, str, str, str]:
    """Compile + simulate one model; optionally skip conflict/mesh."""
    canonical  = _normalise_pattern(pattern)
    model_path = model_dir / f"{canonical}.tflite"
    if not model_path.exists():
        return pattern, None, None, None, f"missing-tflite: {model_path.name}", "", ""
    paths = _artefact_paths(model_path)
    bin_path = paths["prog"]
    conflict_paths = _mode_paths(model_path, "conflict")
    mesh_paths = _mode_paths(model_path, "mesh")

    # ---- compile ----
    if progress:
        progress("compile")
    try:
        cr = subprocess.run(
            [sys.executable, str(COMPILE_PY), str(model_path), str(bin_path)],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return pattern, None, None, None, "compile-timeout", "", ""
    if cr.returncode != 0:
        last = _meaningful_stderr_line(cr.stderr) or f"exit {cr.returncode}"
        if "Model provided has model identifier" in last:
            last = "corrupt .tflite"
        return pattern, None, None, None, f"compile-fail: {last[:80]}", "", ""

    # ---- simulate: fast report path ----
    if progress:
        progress("simulate fast")
    ms, status, fast_stdout = _simulate_one(bin_path, "fast")

    fast_html = OUT_DIR / f"{canonical}.fast.html"
    if not skip_html:
        if progress:
            progress("html fast")
        try:
            _write_mode_html(model_path, paths, cr.stdout or "", fast_stdout)
            if paths["html"].exists():
                fast_html.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(paths["html"], fast_html)
        except Exception as e:
            if status == "ok":
                status = f"html-fail: {str(e)[:80]}"
            else:
                status = f"{status}; html-fail"

    if fast_only:
        return pattern, ms, None, None, status, "", ""

    # ---- simulate: conflict timing only ----
    if progress:
        progress("simulate conflict")
    try:
        shutil.copyfile(bin_path, conflict_paths["prog"])
    except OSError:
        pass
    conflict_ms, conflict_status, conflict_stdout = _simulate_one(
        conflict_paths["prog"], "conflict")

    conflict_html = conflict_paths["html"]
    if not skip_html:
        if progress:
            progress("html conflict")
        try:
            _write_mode_html(model_path, conflict_paths, cr.stdout or "", conflict_stdout)
        except Exception as e:
            if conflict_status == "ok":
                conflict_status = f"html-fail: {str(e)[:80]}"
            else:
                conflict_status = f"{conflict_status}; html-fail"

    # ---- simulate: mesh timing + report ----
    if progress:
        progress("simulate mesh")
    try:
        shutil.copyfile(bin_path, mesh_paths["prog"])
    except OSError:
        pass
    mesh_ms, mesh_status, mesh_stdout = _simulate_one(mesh_paths["prog"], "mesh")

    mesh_html = mesh_paths["html"]
    if not skip_html:
        if progress:
            progress("html mesh")
        try:
            _write_mode_html(model_path, mesh_paths, cr.stdout or "", mesh_stdout)
        except Exception as e:
            if mesh_status == "ok":
                mesh_status = f"html-fail: {str(e)[:80]}"
            else:
                mesh_status = f"{mesh_status}; html-fail"

    if not skip_html:
        if progress:
            progress("html combined")
        try:
            if fast_html.exists() and conflict_html.exists() and mesh_html.exists():
                _write_combined_html(model_path, fast_html, conflict_html, mesh_html,
                                     ms, conflict_ms, mesh_ms,
                                     status, conflict_status, mesh_status)
        except Exception as e:
            if status == "ok":
                status = f"html-fail: combined {str(e)[:70]}"
    return pattern, ms, conflict_ms, mesh_ms, status, conflict_status, mesh_status


def _load_prior_csv(csv_path: Path) -> dict[str, dict]:
    """Read EVERY {pattern -> row} from a previous CSV. Used both for
    the cache (only `ok` rows are reusable) and for preserving rows the
    current --filter excluded so they're not dropped on re-write."""
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pattern"):
                    out[row["pattern"]] = row
    except Exception:
        return {}
    return out


def _load_prior_results(csv_path: Path, fast_only: bool = False) -> dict[str, dict]:
    """Subset of _load_prior_csv that returns only reusable cache rows
    (status == 'ok'). `*-FAIL` / `compile-fail` / `sim-fail` etc. should
    re-run because the underlying source likely changed since."""
    rows = {p: r for p, r in _load_prior_csv(csv_path).items()
            if r.get("status") == "ok" and r.get("mdla7_ms")}
    if fast_only:
        return rows
    return {p: r for p, r in rows.items()
            if r.get("conflict_status", "ok") == "ok" and r.get("mdla7_conflict_ms") and
            r.get("mesh_status", "ok") == "ok" and r.get("mdla7_mesh_ms")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in",  default=str(DEFAULT_CSV_IN))
    ap.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--filter", default="",
                    help="substring filter on pattern name")
    ap.add_argument("--keep-bin", action="store_true",
                    help="keep per-model .bin in output/ after the sweep")
    ap.add_argument("--rerun-all", action="store_true",
                    help="ignore prior --csv-out cache and re-run everything")
    ap.add_argument("--fast-only", action="store_true",
                    help="run only fast mode; leave conflict/mesh CSV fields empty")
    ap.add_argument("--include-excluded", action="store_true",
                    help="include patterns excluded from the default sweep")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N selected patterns (0 = no limit)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N selected patterns before applying --limit")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    csv_in = Path(args.csv_in)
    if not csv_in.exists():
        sys.exit(f"input CSV not found: {csv_in}")
    if not MODEL_RUNNER.exists():
        sys.exit(f"mdla7_model_runner not built: {MODEL_RUNNER}\n"
                 f"  run `make -C ../systemc -s` from {HERE}\n"
                 f"  or `make -C systemc -s` from {REPO_ROOT}")

    # Read MDLA6 patterns + their CX baselines.
    rows_in: list[tuple[str, str]] = []
    with csv_in.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)              # skip "Pattern,CX"
        for row in reader:
            if not row or not row[0].strip():
                continue
            pat = row[0].strip()
            cx  = row[1].strip() if len(row) > 1 else ""
            if args.filter and args.filter not in pat:
                continue
            if (not args.include_excluded and
                    _normalise_pattern(pat) in EXCLUDED_PATTERNS):
                continue
            rows_in.append((pat, cx))
    if args.offset:
        rows_in = rows_in[args.offset:]
    if args.limit:
        rows_in = rows_in[:args.limit]

    if not rows_in:
        excluded = ", ".join(sorted(EXCLUDED_PATTERNS))
        sys.exit(f"no patterns from {csv_in} (filter={args.filter!r}; "
                 f"excluded={excluded})")

    csv_out = Path(args.csv_out)
    # v8.29: cache prior `ok` rows from the same csv-out so a re-run of the
    # sweep skips already-passing models. Pass --rerun-all to force re-test
    # (e.g. after a sim or compile-side change that might affect cycles).
    prior_ok = {} if args.rerun_all else _load_prior_results(csv_out, fast_only=args.fast_only)
    if prior_ok:
        print(f"  (cache: {len(prior_ok)} prior ok rows in {csv_out.name}; "
              f"--rerun-all to ignore)", flush=True)

    # v8.30: each row print is one full line that begins with `\r\033[2K`
    # (carriage return + erase-line) so any partial subprocess output that
    # leaked to the parent tty gets overwritten before our row text. Belt-
    # and-braces: subprocesses already use capture_output=True so this
    # shouldn't happen, but seen in the wild when a Ctrl-C'd prior run left
    # a zombie mdla7_model_runner still streaming to the terminal.
    def _row_print(s: str):
        sys.stdout.write("\r\033[2K" + s + "\n")
        sys.stdout.flush()

    def _row_update(s: str):
        sys.stdout.write("\r\033[2K" + s)
        sys.stdout.flush()

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # v8.30: checkpoint after every row so a Ctrl-C / kill mid-sweep doesn't
    # lose work. The previous "write at end of loop" form lost everything that
    # ran but didn't get to the post-loop write block. Also preserves rows
    # the current --filter excluded — without this, a filtered re-run would
    # overwrite the cache to only contain the filtered subset.
    prior_full = _load_prior_csv(csv_path)
    def _checkpoint(rows: list[dict]):
        # Merge: current run's rows override, prior rows fill in for any
        # pattern not in this run.
        seen = {r["pattern"] for r in rows}
        merged = list(rows)
        # Preserve prior rows in their original order for any pattern this
        # run didn't touch (e.g. filter excluded).
        for pat, prow in prior_full.items():
            if pat not in seen:
                merged.append(prow)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pattern", "mdla6_cx",
                                              "mdla7_ms", "mdla7_conflict_ms",
                                              "mdla7_mesh_ms", "status",
                                              "conflict_status", "mesh_status"])
            w.writeheader()
            w.writerows(merged)

    print(f"==== MDLA6 pattern regression: {len(rows_in)} patterns "
          f"(from {csv_in.name}) ====", flush=True)
    rows_out = []
    t_total = time.time()
    for i, (pat, cx) in enumerate(rows_in, 1):
        # Reuse prior ok result if available and its per-model HTML already
        # exists. Older cache rows predate the report, so they re-run once.
        if pat in prior_ok and _report_exists_for(pat, Path(args.model_dir)):
            cached = prior_ok[pat]
            cached_ms = cached.get("mdla7_ms", "")
            cached_conflict_ms = "" if args.fast_only else cached.get("mdla7_conflict_ms", "")
            cached_mesh_ms = "" if args.fast_only else cached.get("mdla7_mesh_ms", "")
            cached_status = cached.get("status", "ok")
            cached_conflict_status = "" if args.fast_only else cached.get("conflict_status", "ok")
            cached_mesh_status = "" if args.fast_only else cached.get("mesh_status", "ok")
            ms_str = f"{float(cached_ms):>10.3f} ms" if cached_ms else f"{'—':>10s}    "
            cms_str = f"{float(cached_conflict_ms):>10.3f} ms" if cached_conflict_ms else f"{'—':>10s}    "
            mesh_str = f"{float(cached_mesh_ms):>10.3f} ms" if cached_mesh_ms else f"{'—':>10s}    "
            suffix = (f"{cached_status}" if args.fast_only
                      else f"{cached_status}/{cached_conflict_status}/{cached_mesh_status}")
            _row_print(f"[{i:>2}/{len(rows_in)}] {pat:<28s} cx={cx:<6s} "
                       f"fast={ms_str} conflict={cms_str} mesh={mesh_str} cached  "
                       f"{suffix}")
            rows_out.append({
                "pattern":           pat,
                "mdla6_cx":          cx,
                "mdla7_ms":          cached_ms,
                "mdla7_conflict_ms": cached_conflict_ms,
                "mdla7_mesh_ms":     cached_mesh_ms,
                "status":            cached_status,
                "conflict_status":   cached_conflict_status,
                "mesh_status":       cached_mesh_status,
            })
            _checkpoint(rows_out)
            continue
        t0 = time.time()
        def _progress(stage: str):
            elapsed = time.time() - t0
            _row_update(f"[{i:>2}/{len(rows_in)}] {pat:<28s} cx={cx:<6s} "
                        f"{'—':>10s}      ({elapsed:5.1f}s)  running {stage}...")

        pattern, ms, conflict_ms, mesh_ms, status, conflict_status, mesh_status = run_one(
            pat, Path(args.model_dir), progress=_progress, fast_only=args.fast_only)
        elapsed = time.time() - t0
        ms_str  = f"{ms:>10.3f} ms" if ms is not None else f"{'—':>10s}    "
        cms_str = (f"{conflict_ms:>10.3f} ms" if conflict_ms is not None
                   else f"{'—':>10s}    ")
        mesh_str = (f"{mesh_ms:>10.3f} ms" if mesh_ms is not None
                    else f"{'—':>10s}    ")
        suffix = status if args.fast_only else f"{status}/{conflict_status}/{mesh_status}"
        _row_print(f"[{i:>2}/{len(rows_in)}] {pat:<28s} cx={cx:<6s} "
                   f"fast={ms_str} conflict={cms_str} mesh={mesh_str}  ({elapsed:5.1f}s)  "
                   f"{suffix}")
        rows_out.append({
            "pattern":           pat,
            "mdla6_cx":          cx,
            "mdla7_ms":          f"{ms:.3f}" if ms is not None else "",
            "mdla7_conflict_ms": f"{conflict_ms:.3f}" if conflict_ms is not None else "",
            "mdla7_mesh_ms":     f"{mesh_ms:.3f}" if mesh_ms is not None else "",
            "status":            status,
            "conflict_status":   conflict_status,
            "mesh_status":       mesh_status,
        })
        _checkpoint(rows_out)            # durable after each row
        if not args.keep_bin:
            canonical = _normalise_pattern(pat)
            for bin_path in (OUT_DIR / f"{canonical}.bin",
                             OUT_DIR / f"{canonical}.conflict.bin",
                             OUT_DIR / f"{canonical}.mesh.bin"):
                try:
                    if bin_path.exists():
                        bin_path.unlink()
                except OSError:
                    pass

    n_ok    = sum(1 for r in rows_out if r["mdla7_ms"])
    n_conflict_ok = sum(1 for r in rows_out if r.get("mdla7_conflict_ms"))
    n_mesh_ok = sum(1 for r in rows_out if r.get("mdla7_mesh_ms"))
    n_fail  = len(rows_out) - n_ok
    total_s = time.time() - t_total
    total_ms = sum(float(r["mdla7_ms"]) for r in rows_out if r["mdla7_ms"])
    if args.fast_only:
        print(f"\n==== summary: fast {n_ok}/{len(rows_out)} ran "
              f"({n_fail} skipped/failed),"
              f"  sim total {total_ms:.1f} ms,  wall {total_s:.0f}s  ====", flush=True)
    else:
        print(f"\n==== summary: fast {n_ok}/{len(rows_out)} ran, "
              f"conflict {n_conflict_ok}/{len(rows_out)} ran, "
              f"mesh {n_mesh_ok}/{len(rows_out)} ran  ({n_fail} skipped/failed),"
              f"  sim total {total_ms:.1f} ms,  wall {total_s:.0f}s  ====", flush=True)
    print(f"csv: {csv_path}", flush=True)
    _refresh_model_profile_index()
    print(f"html: {HERE / 'profile_mdla6_pattern.html'}", flush=True)


if __name__ == "__main__":
    main()
