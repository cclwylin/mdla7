#!/usr/bin/env python3
"""Regression sweep for extracted Hotspot `.tflite` slices.

The Hotspot directory contains small repeated bottleneck blocks sliced out of
larger ETHZ_v6 transformer-like models. This runner mirrors
`run_mdla6_pattern.py`: each slice is compiled once, simulated in fast,
conflict, and mesh L1 timing modes, and written to a combined per-slice HTML
profile. Pass `--fast-only` to run only fast mode and leave conflict/mesh CSV
fields empty.

Outputs:
    output/hotspot_regression.csv
        pattern,mdla7_ms,mdla7_conflict_ms,mdla7_mesh_ms,status,conflict_status,mesh_status

    output/<slice>.html
        combined fast/conflict/mesh profile report

Usage:
    ./batch/run_hotspot.py
    ./batch/run_hotspot.py --filter vit
    ./batch/run_hotspot.py --limit 3
    ./batch/run_hotspot.py --offset 3 --limit 3
    ./batch/run_hotspot.py --rerun-all
    ./batch/run_hotspot.py --fast-only
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SYSTEMC_DIR = REPO_ROOT / "systemc"
OUT_DIR = HERE / "output"
DEFAULT_MODEL_DIR = REPO_ROOT / "model" / "Hotspot"
DEFAULT_CSV_OUT = OUT_DIR / "hotspot_regression.csv"
MODEL_RUNNER = SYSTEMC_DIR / "build" / "mdla7_model_runner"
MODEL_PROFILE_PY = HERE / "gen_model_profile.py"

# Re-exec into the same venv policy used by run_model.py / run_mdla6_pattern.py.
VENV_DIR = Path(os.environ.get("MDLA7_VENV") or
                Path.home() / ".venvs/mdla7").expanduser()
VENV_PY = VENV_DIR / "bin" / "python"


def _reexec_in_venv() -> None:
    if not VENV_PY.exists():
        return
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])


_reexec_in_venv()

from run_mdla6_pattern import (  # noqa: E402
    _artefact_paths,
    _report_exists_for,
    run_one,
)


def _refresh_hotspot_profile_index(csv_path: Path) -> None:
    try:
        subprocess.run(
            [sys.executable, str(MODEL_PROFILE_PY),
             "--html-out", "profile_hotspot.html",
             "--title", "MDLA7 Hotspot Profiles",
             "--metrics-csv", str(csv_path),
             "--only-metrics-rows",
             "--hide-mdla6-cx"],
            cwd=str(HERE), capture_output=True, text=True,
        )
    except Exception:
        pass


def _load_prior_csv(csv_path: Path) -> dict[str, dict]:
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
    rows = {p: r for p, r in _load_prior_csv(csv_path).items()
            if r.get("status") == "ok" and r.get("mdla7_ms")}
    if fast_only:
        return rows
    return {p: r for p, r in rows.items()
            if r.get("conflict_status", "ok") == "ok" and r.get("mdla7_conflict_ms") and
            r.get("mesh_status", "ok") == "ok" and r.get("mdla7_mesh_ms")}


def _discover_hotspots(model_dir: Path, name_filter: str) -> list[str]:
    if not model_dir.exists():
        raise SystemExit(f"Hotspot model dir not found: {model_dir}")
    patterns = []
    for path in sorted(model_dir.glob("*.tflite")):
        if path.name.startswith("._"):
            continue
        if name_filter and name_filter.lower() not in path.stem.lower():
            continue
        patterns.append(path.stem)
    return patterns


def _ms_cell(value: str) -> str:
    return f"{float(value):>10.3f} ms" if value else f"{'—':>10s}    "


def _row_print(s: str) -> None:
    sys.stdout.write("\r\033[2K" + s + "\n")
    sys.stdout.flush()


def _row_update(s: str) -> None:
    sys.stdout.write("\r\033[2K" + s)
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                    help="directory containing Hotspot .tflite slices")
    ap.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
    ap.add_argument("--filter", default="",
                    help="substring filter on Hotspot slice name")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N selected slices (0 = no limit)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N selected slices before applying --limit")
    ap.add_argument("--rerun-all", action="store_true",
                    help="ignore prior --csv-out cache and re-run everything")
    ap.add_argument("--fast-only", action="store_true",
                    help="run only fast mode; leave conflict/mesh CSV fields empty")
    ap.add_argument("--keep-bin", action="store_true",
                    help="keep per-slice .bin files in output/ after the sweep")
    ap.add_argument("--list", action="store_true",
                    help="list selected Hotspot slices and exit")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    model_dir = Path(args.model_dir)
    patterns = _discover_hotspots(model_dir, args.filter)
    if args.offset:
        patterns = patterns[args.offset:]
    if args.limit:
        patterns = patterns[:args.limit]

    if args.list:
        for pat in patterns:
            path = model_dir / f"{pat}.tflite"
            print(f"{pat:<42s} {path.stat().st_size / (1024 * 1024):6.1f} MB")
        return

    if not patterns:
        raise SystemExit(f"no Hotspot slices found in {model_dir} "
                         f"(filter={args.filter!r})")
    if not MODEL_RUNNER.exists():
        raise SystemExit(f"mdla7_model_runner not built: {MODEL_RUNNER}\n"
                         f"  run `make -C ../systemc -s` from {HERE}\n"
                         f"  or `make -C systemc -s` from {REPO_ROOT}")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prior_full = _load_prior_csv(csv_path)
    prior_ok = {} if args.rerun_all else _load_prior_results(csv_path, fast_only=args.fast_only)
    if prior_ok:
        print(f"  (cache: {len(prior_ok)} prior ok rows in {csv_path.name}; "
              f"--rerun-all to ignore)", flush=True)

    def _checkpoint(rows: list[dict]) -> None:
        seen = {r["pattern"] for r in rows}
        merged = list(rows)
        for pat, prow in prior_full.items():
            if pat not in seen:
                merged.append(prow)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "pattern",
                    "mdla7_ms",
                    "mdla7_conflict_ms",
                    "mdla7_mesh_ms",
                    "status",
                    "conflict_status",
                    "mesh_status",
                ],
            )
            w.writeheader()
            w.writerows(merged)

    print(f"==== MDLA7 Hotspot regression: {len(patterns)} slices "
          f"(from {model_dir.relative_to(REPO_ROOT)}) ====", flush=True)
    rows_out = []
    t_total = time.time()
    for i, pat in enumerate(patterns, 1):
        if pat in prior_ok and _report_exists_for(pat, model_dir):
            cached = prior_ok[pat]
            cached_conflict_ms = "" if args.fast_only else cached.get("mdla7_conflict_ms", "")
            cached_mesh_ms = "" if args.fast_only else cached.get("mdla7_mesh_ms", "")
            cached_conflict_status = "" if args.fast_only else cached.get("conflict_status", "ok")
            cached_mesh_status = "" if args.fast_only else cached.get("mesh_status", "ok")
            suffix = (cached.get("status", "ok") if args.fast_only
                      else f"{cached.get('status', 'ok')}/{cached_conflict_status}/{cached_mesh_status}")
            _row_print(f"[{i:>2}/{len(patterns)}] {pat:<42s} "
                       f"fast={_ms_cell(cached.get('mdla7_ms', ''))} "
                       f"conflict={_ms_cell(cached_conflict_ms)} "
                       f"mesh={_ms_cell(cached_mesh_ms)} cached  "
                       f"{suffix}")
            rows_out.append({
                "pattern": pat,
                "mdla7_ms": cached.get("mdla7_ms", ""),
                "mdla7_conflict_ms": cached_conflict_ms,
                "mdla7_mesh_ms": cached_mesh_ms,
                "status": cached.get("status", "ok"),
                "conflict_status": cached_conflict_status,
                "mesh_status": cached_mesh_status,
            })
            _checkpoint(rows_out)
            continue

        t0 = time.time()

        def _progress(stage: str) -> None:
            elapsed = time.time() - t0
            _row_update(f"[{i:>2}/{len(patterns)}] {pat:<42s} "
                        f"{'—':>10s}      ({elapsed:5.1f}s)  "
                        f"running {stage}...")

        _, ms, conflict_ms, mesh_ms, status, conflict_status, mesh_status = run_one(
            pat, model_dir, progress=_progress, fast_only=args.fast_only)
        elapsed = time.time() - t0
        ms_str = f"{ms:>10.3f} ms" if ms is not None else f"{'—':>10s}    "
        conflict_str = (f"{conflict_ms:>10.3f} ms" if conflict_ms is not None
                        else f"{'—':>10s}    ")
        mesh_str = (f"{mesh_ms:>10.3f} ms" if mesh_ms is not None
                    else f"{'—':>10s}    ")
        suffix = status if args.fast_only else f"{status}/{conflict_status}/{mesh_status}"
        _row_print(f"[{i:>2}/{len(patterns)}] {pat:<42s} "
                   f"fast={ms_str} conflict={conflict_str} mesh={mesh_str}  "
                   f"({elapsed:5.1f}s)  "
                   f"{suffix}")
        rows_out.append({
            "pattern": pat,
            "mdla7_ms": f"{ms:.3f}" if ms is not None else "",
            "mdla7_conflict_ms": f"{conflict_ms:.3f}" if conflict_ms is not None else "",
            "mdla7_mesh_ms": f"{mesh_ms:.3f}" if mesh_ms is not None else "",
            "status": status,
            "conflict_status": conflict_status,
            "mesh_status": mesh_status,
        })
        _checkpoint(rows_out)

        if not args.keep_bin:
            model_path = model_dir / f"{pat}.tflite"
            for bin_path in (_artefact_paths(model_path)["prog"],
                             OUT_DIR / f"{pat}.conflict.bin",
                             OUT_DIR / f"{pat}.mesh.bin"):
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
        print(f"\n==== summary: fast {n_fast}/{len(rows_out)} ran, "
              f"sim total {total_ms:.1f} ms, wall {total_s:.0f}s ====",
              flush=True)
    else:
        print(f"\n==== summary: fast {n_fast}/{len(rows_out)} ran, "
              f"conflict {n_conflict}/{len(rows_out)} ran, "
              f"mesh {n_mesh}/{len(rows_out)} ran, "
              f"sim total {total_ms:.1f} ms, wall {total_s:.0f}s ====",
              flush=True)
    print(f"csv: {csv_path}", flush=True)
    _refresh_hotspot_profile_index(csv_path)
    print(f"html: {HERE / 'profile_hotspot.html'}", flush=True)


if __name__ == "__main__":
    main()
