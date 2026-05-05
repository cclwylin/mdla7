#!/usr/bin/env python3
"""Regression sweep restricted to the patterns listed in `mdla6_ethz_v6_sorted.csv`.

For each (pattern, mdla6_cx) row in the input CSV, locates the matching
`.tflite` under `model/ETHZ_v6/`, runs compile_model.py + test_model, and
emits a CSV next to the regular ETHZ sweep that pairs the MDLA7 sim time
with the MDLA6 baseline:

    output/mdla6_pattern_regression.csv
        pattern,mdla6_cx,mdla7_ms,status

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
    python3 systemc/run_mdla6_pattern.py                    # default: cache prior ok
    python3 systemc/run_mdla6_pattern.py --rerun-all        # force re-test all rows
    python3 systemc/run_mdla6_pattern.py --csv-in  <path>   # override input
    python3 systemc/run_mdla6_pattern.py --csv-out <path>   # override output cache
    python3 systemc/run_mdla6_pattern.py --filter mobilenet # substring filter
    python3 systemc/run_mdla6_pattern.py --include-excluded # debug excluded rows
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE       = Path(__file__).resolve().parent
REPO_ROOT  = HERE.parent
COMPILE_PY = HERE / "scripts" / "compile_model.py"
PLOT_PY    = HERE / "scripts" / "plot_profile.py"
MODEL_PROFILE_PY = HERE / "scripts" / "gen_model_profile.py"
TEST_BIN   = HERE / "build" / "test_model"
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

# v8.25: test_model.cpp prints "sim time: <cycles> cycles @ 1.9 GHz (= <ms> ms)"
# (pre-v8.25 was "sim time: <ns> ns"). Match both for forward/back compat.
SIM_TIME_RE = re.compile(r"sim time:\s*([\d,]+)\s*(?:cycles|ns)")

# SystemC license banner on stderr (pretty-printed at startup); suppress it
# when picking a meaningful error line out of stderr.
_BANNER_HINTS = ("ALL RIGHTS RESERVED", "Accellera", "Copyright (c)",
                 "ISO/IEC", "SystemC ", "Licensed under")

def _refresh_model_profile_index() -> None:
    try:
        subprocess.run([sys.executable, str(MODEL_PROFILE_PY)],
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


def _selected_log_lines(compile_stdout: str, sim_stdout: str) -> list[str]:
    lines: list[str] = []
    for ln in (compile_stdout or "").splitlines():
        if ln.startswith(("compile_model:", "  layer", "  →")):
            lines.append(ln)
    for ln in (sim_stdout or "").splitlines():
        if ln.startswith(("test_model:", "  layer", "  summary",
                          "  sim time", "  DRAM ", "  SRAM ",
                          "  per-engine", "  utilization", "    ",
                          "  profile", "  csv")):
            lines.append(ln)
    return lines


def _write_pattern_html(model_path: Path, compile_stdout: str, sim_stdout: str) -> Path:
    paths = _artefact_paths(model_path)
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


def run_one(pattern: str, model_dir: Path, progress=None) -> tuple[str, float | None, str]:
    """Compile + simulate one model. Returns (pattern, ms, status)."""
    canonical  = _normalise_pattern(pattern)
    model_path = model_dir / f"{canonical}.tflite"
    if not model_path.exists():
        return pattern, None, f"missing-tflite: {model_path.name}"
    paths = _artefact_paths(model_path)
    bin_path = paths["prog"]

    # ---- compile ----
    if progress:
        progress("compile")
    try:
        cr = subprocess.run(
            [sys.executable, str(COMPILE_PY), str(model_path), str(bin_path)],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return pattern, None, "compile-timeout"
    if cr.returncode != 0:
        last = _meaningful_stderr_line(cr.stderr) or f"exit {cr.returncode}"
        if "Model provided has model identifier" in last:
            last = "corrupt .tflite"
        return pattern, None, f"compile-fail: {last[:80]}"

    # ---- simulate ----
    if progress:
        progress("simulate")
    try:
        sr = subprocess.run(
            [str(TEST_BIN), str(bin_path), "--quiet"],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return pattern, None, "sim-timeout"
    m = SIM_TIME_RE.search(sr.stdout or "")
    if sr.returncode != 0 and not m:
        last = _meaningful_stderr_line(sr.stderr) or f"exit {sr.returncode}"
        return pattern, None, f"sim-fail: {last[:80]}"
    if not m:
        return pattern, None, "sim-time-missing"
    cycles = int(m.group(1).replace(",", ""))
    ms = cycles / 1.9e6   # spec frequency = 1.9 GHz (post-v8.25)

    sm = re.search(r"summary:\s+(\d+)/(\d+)\s+layers PASS,\s+(\d+)\s+FAIL",
                   sr.stdout or "")
    if sm:
        n_pass, n_total, n_fail = (int(sm.group(k)) for k in (1, 2, 3))
        status = "ok" if n_fail == 0 else f"{n_fail}-FAIL"
    else:
        status = "ok" if sr.returncode == 0 else f"exit {sr.returncode}"

    if progress:
        progress("html")
    try:
        _write_pattern_html(model_path, cr.stdout or "", sr.stdout or "")
    except Exception as e:
        if status == "ok":
            status = f"html-fail: {str(e)[:80]}"
        else:
            status = f"{status}; html-fail"
    return pattern, ms, status


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


def _load_prior_results(csv_path: Path) -> dict[str, dict]:
    """Subset of _load_prior_csv that returns only reusable cache rows
    (status == 'ok'). `*-FAIL` / `compile-fail` / `sim-fail` etc. should
    re-run because the underlying source likely changed since."""
    return {p: r for p, r in _load_prior_csv(csv_path).items()
            if r.get("status") == "ok" and r.get("mdla7_ms")}


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
    ap.add_argument("--include-excluded", action="store_true",
                    help="include patterns excluded from the default sweep")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    csv_in = Path(args.csv_in)
    if not csv_in.exists():
        sys.exit(f"input CSV not found: {csv_in}")
    if not TEST_BIN.exists():
        sys.exit(f"test_model not built: {TEST_BIN}\n  run `make` in {HERE}")

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

    if not rows_in:
        excluded = ", ".join(sorted(EXCLUDED_PATTERNS))
        sys.exit(f"no patterns from {csv_in} (filter={args.filter!r}; "
                 f"excluded={excluded})")

    csv_out = Path(args.csv_out)
    # v8.29: cache prior `ok` rows from the same csv-out so a re-run of the
    # sweep skips already-passing models. Pass --rerun-all to force re-test
    # (e.g. after a sim or compile-side change that might affect cycles).
    prior_ok = {} if args.rerun_all else _load_prior_results(csv_out)
    if prior_ok:
        print(f"  (cache: {len(prior_ok)} prior ok rows in {csv_out.name}; "
              f"--rerun-all to ignore)", flush=True)

    # v8.30: each row print is one full line that begins with `\r\033[2K`
    # (carriage return + erase-line) so any partial subprocess output that
    # leaked to the parent tty gets overwritten before our row text. Belt-
    # and-braces: subprocesses already use capture_output=True so this
    # shouldn't happen, but seen in the wild when a Ctrl-C'd prior run left
    # a zombie test_model still streaming to the terminal.
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
                                              "mdla7_ms", "status"])
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
            cached_status = cached.get("status", "ok")
            ms_str = f"{float(cached_ms):>10.3f} ms" if cached_ms else f"{'—':>10s}    "
            _row_print(f"[{i:>2}/{len(rows_in)}] {pat:<28s} cx={cx:<6s} "
                       f"{ms_str}      cached  {cached_status}")
            rows_out.append({
                "pattern":   pat,
                "mdla6_cx":  cx,
                "mdla7_ms":  cached_ms,
                "status":    cached_status,
            })
            _checkpoint(rows_out)
            continue
        t0 = time.time()
        def _progress(stage: str):
            elapsed = time.time() - t0
            _row_update(f"[{i:>2}/{len(rows_in)}] {pat:<28s} cx={cx:<6s} "
                        f"{'—':>10s}      ({elapsed:5.1f}s)  running {stage}...")

        pattern, ms, status = run_one(pat, Path(args.model_dir), progress=_progress)
        elapsed = time.time() - t0
        ms_str  = f"{ms:>10.3f} ms" if ms is not None else f"{'—':>10s}    "
        _row_print(f"[{i:>2}/{len(rows_in)}] {pat:<28s} cx={cx:<6s} "
                   f"{ms_str}  ({elapsed:5.1f}s)  {status}")
        rows_out.append({
            "pattern":   pat,
            "mdla6_cx":  cx,
            "mdla7_ms":  f"{ms:.3f}" if ms is not None else "",
            "status":    status,
        })
        _checkpoint(rows_out)            # durable after each row
        if not args.keep_bin:
            canonical = _normalise_pattern(pat)
            bin_path = OUT_DIR / f"{canonical}.bin"
            try:
                if bin_path.exists():
                    bin_path.unlink()
            except OSError:
                pass

    n_ok    = sum(1 for r in rows_out if r["mdla7_ms"])
    n_fail  = len(rows_out) - n_ok
    total_s = time.time() - t_total
    total_ms = sum(float(r["mdla7_ms"]) for r in rows_out if r["mdla7_ms"])
    print(f"\n==== summary: {n_ok}/{len(rows_out)} ran  ({n_fail} skipped/failed),"
          f"  sim total {total_ms:.1f} ms,  wall {total_s:.0f}s  ====", flush=True)
    print(f"csv: {csv_path}", flush=True)
    _refresh_model_profile_index()
    print(f"html: {HERE / 'model_profile.html'}", flush=True)


if __name__ == "__main__":
    main()
