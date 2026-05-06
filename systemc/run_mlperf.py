#!/usr/bin/env python3
"""Regression sweep over model/MLPerf_Tiny/*.tflite - emit CSV (pattern, ms).

For each .tflite in the bundle, runs the standard compile_model.py + test_model
flow, parses the simulator's "sim time: NNNN ns" line, and writes:

    output/mlperf_regression.csv
        pattern,ms,status

where `pattern` is the .tflite stem and `ms` is sim cycles / 1e6 (1 cycle = 1 ns
@ 1 GHz). Failed compiles (unsupported op, missing buffer, …) emit a blank ms
and a short status note so you can see what's still gating coverage.

Usage:
    python3 systemc/run_mlperf.py
    python3 systemc/run_mlperf.py --model-dir <path>     # override input dir
    python3 systemc/run_mlperf.py --csv <path>           # override output csv
    python3 systemc/run_mlperf.py --filter resnet        # substring filter
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
TEST_BIN   = HERE / "build" / "test_model"
OUT_DIR    = HERE / "output"

# Re-exec into the venv (same convention as run_model.py — see §5 of handoff.md
# for why ~/.venvs/mdla7 lives outside the repo on the 4T_OFFICE volume).
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

# v8.25: test_model.cpp prints "sim time: <cycles> cycles @ 1.9 GHz (= <ms> ms)";
# pre-v8.25 it was "sim time: <ns> ns  (= <cycles> cycles @ 1 GHz)". Match both.
SIM_TIME_RE = re.compile(r"sim time:\s*([\d,]+)\s*(?:cycles|ns)")

# v8.21: SystemC writes a multi-line license banner to stderr at startup
# (Accellera / Copyright / "ALL RIGHTS RESERVED"); the previous "last line of
# stderr" heuristic picked that up instead of the real error. Filter it.
_BANNER_HINTS = ("ALL RIGHTS RESERVED", "Accellera", "Copyright (c)",
                 "ISO/IEC", "SystemC ", "Licensed under")

def _meaningful_stderr_line(stderr: str) -> str:
    """Pick the most informative line from test_model's stderr.

    Prefers explicit `layer N: ...` errors emitted by test_model.cpp; falls
    back to the last non-banner, non-empty line. Returns "" if nothing useful.
    """
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    # Strongest signal: explicit layer-level error from test_model.cpp.
    for ln in reversed(lines):
        if re.match(r"layer\s+\d+:", ln):
            return ln
    # Otherwise: scan back skipping banner / license boilerplate.
    for ln in reversed(lines):
        if any(h in ln for h in _BANNER_HINTS):
            continue
        return ln
    return ""


def run_one(model: Path, l1_timing: str = "fast") -> tuple[str, float | None, str]:
    """Compile + simulate one model. Returns (pattern, ms, status)."""
    pattern  = model.stem
    bin_path = OUT_DIR / f"{pattern}.bin"

    # ---- compile ----
    try:
        cr = subprocess.run(
            [sys.executable, str(COMPILE_PY), str(model), str(bin_path)],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return pattern, None, "compile-timeout"
    if cr.returncode != 0:
        last = _meaningful_stderr_line(cr.stderr) or f"exit {cr.returncode}"
        if "Model provided has model identifier" in last:
            last = "corrupt .tflite"
        return pattern, None, f"compile-fail: {last[:80]}"

    # ---- simulate ----
    try:
        sr = subprocess.run(
            [str(TEST_BIN), str(bin_path), "--quiet", f"--l1-timing={l1_timing}"],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return pattern, None, "sim-timeout"
    # test_model returns 1 if any layer FAILed bit-exact verify, but the sim
    # still completed and we have timing — surface that as "N-FAIL" in status
    # rather than discarding the timing as a hard failure. Reserve sim-fail
    # for crashes (exit < 0) or early-aborts (no "sim time:" line).
    m = SIM_TIME_RE.search(sr.stdout or "")
    if sr.returncode != 0 and not m:
        last = _meaningful_stderr_line(sr.stderr) or f"exit {sr.returncode}"
        return pattern, None, f"sim-fail: {last[:80]}"
    if not m:
        return pattern, None, "sim-time-missing"
    cycles = int(m.group(1).replace(",", ""))
    ms = cycles / 1.9e6   # v8.25: spec frequency = 1.9 GHz

    # Pass/fail count from the summary line so we can flag silent regressions.
    sm = re.search(r"summary:\s+(\d+)/(\d+)\s+layers PASS,\s+(\d+)\s+FAIL", sr.stdout or "")
    if sm:
        _, _, n_fail = (int(sm.group(k)) for k in (1, 2, 3))
        status = "ok" if n_fail == 0 else f"{n_fail}-FAIL"
    else:
        status = "ok" if sr.returncode == 0 else f"exit {sr.returncode}"
    return pattern, ms, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO_ROOT / "model" / "MLPerf_Tiny"))
    ap.add_argument("--csv",       default=str(OUT_DIR / "mlperf_regression.csv"))
    ap.add_argument("--filter",    default="",
                    help="substring filter on model basename (e.g. 'resnet')")
    ap.add_argument("--keep-bin",  action="store_true",
                    help="keep per-model .bin in output/ after the sweep")
    ap.add_argument("--l1-timing", choices=("fast", "conflict"), default="fast",
                    help="L1Mesh timing mode: fast aggregate estimate (default) "
                         "or per-bank SRAM port conflict model")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"model dir not found: {model_dir}")
    if not TEST_BIN.exists():
        sys.exit(f"test_model not built: {TEST_BIN}\n  run `make` in {HERE}")

    models = sorted(p for p in model_dir.glob("*.tflite")
                    if (not args.filter) or args.filter in p.name)
    if not models:
        sys.exit(f"no .tflite matched in {model_dir} (filter={args.filter!r})")

    print(f"==== MLPerf_Tiny regression: {len(models)} models ====")
    rows = []
    t_total = time.time()
    for i, m in enumerate(models, 1):
        t0 = time.time()
        pattern, ms, status = run_one(m, args.l1_timing)
        elapsed = time.time() - t0
        ms_str  = f"{ms:>10.3f} ms" if ms is not None else f"{'—':>10s}    "
        print(f"[{i:>2}/{len(models)}] {pattern:<40s} {ms_str}  ({elapsed:5.1f}s)  {status}")
        rows.append({
            "pattern": pattern,
            "ms":      f"{ms:.3f}" if ms is not None else "",
            "status":  status,
        })
        if not args.keep_bin:
            bin_path = OUT_DIR / f"{pattern}.bin"
            try:
                if bin_path.exists(): bin_path.unlink()
            except OSError:
                pass

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "ms", "status"])
        w.writeheader()
        w.writerows(rows)

    n_ok   = sum(1 for r in rows if r["ms"])
    n_fail = len(rows) - n_ok
    total_s = time.time() - t_total
    total_ms = sum(float(r["ms"]) for r in rows if r["ms"])
    print(f"\n==== summary: {n_ok}/{len(rows)} ran  ({n_fail} skipped/failed),  "
          f"sim total {total_ms:.1f} ms,  wall {total_s:.0f}s  ====")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
