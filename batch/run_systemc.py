#!/usr/bin/env python3
"""Unified SystemC regression runner.

Examples:
    ./batch/run_systemc.py --filter ethz
    ./batch/run_systemc.py --filter ethz_v6 --model-filter mobilenet --limit 3
    ./batch/run_systemc.py --filter hotspot --L1 cx --engine cx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUT_DIR = HERE / "output"

VENV_DIR = Path(os.environ.get("MDLA7_VENV") or
                Path.home() / ".venvs/mdla7").expanduser()
VENV_PY = VENV_DIR / "bin" / "python"

if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])

from corpus_runner import run_corpus  # noqa: E402


CORPORA = {
    "ethz": {
        "name": "ETHZ_v6",
        "model_dir": REPO_ROOT / "model" / "ETHZ_v6",
        "csv": OUT_DIR / "ethz_v6_regression.csv",
        "profile": "profile_ethz_v6.html",
        "title": "MDLA7 ETHZ_v6 Profiles",
        "order": REPO_ROOT / "batch" / "mdla6_ethz_v6_sorted.csv",
    },
    "ethz_v6": {
        "name": "ETHZ_v6",
        "model_dir": REPO_ROOT / "model" / "ETHZ_v6",
        "csv": OUT_DIR / "ethz_v6_regression.csv",
        "profile": "profile_ethz_v6.html",
        "title": "MDLA7 ETHZ_v6 Profiles",
        "order": REPO_ROOT / "batch" / "mdla6_ethz_v6_sorted.csv",
    },
    "ethz_v5": {
        "name": "ETHZ_v5",
        "model_dir": REPO_ROOT / "model" / "ETHZ_v5",
        "csv": OUT_DIR / "ethz_v5_regression.csv",
        "profile": "profile_ethz_v5.html",
        "title": "MDLA7 ETHZ_v5 Profiles",
        "order": None,
    },
    "hotspot": {
        "name": "Hotspot",
        "model_dir": REPO_ROOT / "model" / "Hotspot",
        "csv": OUT_DIR / "hotspot_regression.csv",
        "profile": "profile_hotspot.html",
        "title": "MDLA7 Hotspot Profiles",
        "order": None,
    },
    "slice": {
        "name": "MB_Path_Slice",
        "model_dir": REPO_ROOT / "model" / "MB_Path_Slice",
        "csv": OUT_DIR / "mb_path_regression.csv",
        "profile": "profile_mb_path.html",
        "title": "MDLA7 MB Path Slice Profiles",
        "order": None,
        "recursive": True,
        "microblock_metrics": True,
    },
    "mlperf": {
        "name": "MLPerf_Tiny",
        "model_dir": REPO_ROOT / "model" / "MLPerf_Tiny",
        "csv": OUT_DIR / "mlperf_regression.csv",
        "profile": "profile_mlperf.html",
        "title": "MDLA7 MLPerf_Tiny Profiles",
        "order": None,
    },
}


def _has_option(args: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=")
               for arg in args for name in names)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--filter", default="ethz",
                    help="corpus selector: ethz, ethz_v6, ethz_v5, hotspot, slice, mlperf")
    ap.add_argument("--model-filter", default="",
                    help="substring filter inside the selected corpus")
    ap.add_argument("-h", "--help", action="store_true")
    ns, rest = ap.parse_known_args()

    corpus_key = ns.filter.lower()
    if corpus_key == "ethz":
        corpus_key = "ethz_v6"
    if ns.help:
        print(__doc__.strip())
        print("\nCommon options: --limit N --offset N --rerun-all --L1 fast|rtl|cx --engine fast|rtl|cx")
        print("Corpus keys:", ", ".join(sorted(k for k in CORPORA if k != "ethz")))
        return
    if corpus_key not in CORPORA:
        raise SystemExit(f"unknown --filter corpus {ns.filter!r}; use ethz_v6/ethz_v5/hotspot/slice/mlperf")

    runner_args = list(rest)
    if ns.model_filter:
        runner_args.extend(["--filter", ns.model_filter])

    compare_mode = _has_option(runner_args, "--compare-rtl-fast", "--compare-cx-rtl")
    if not compare_mode and not _has_option(runner_args, "--fast-only"):
        runner_args.append("--fast-only")
    if not _has_option(runner_args, "--L1", "--l1", "--l1-timing"):
        runner_args.extend(["--L1", "fast"])
    if not _has_option(runner_args, "--engine", "--engine-model"):
        runner_args.extend(["--engine", "fast"])

    sys.argv = [sys.argv[0], *runner_args]
    cfg = CORPORA[corpus_key]
    print(f"[run_systemc] corpus={cfg['name']} mode_args={' '.join(runner_args)}", flush=True)
    run_corpus(
        corpus_name=str(cfg["name"]),
        default_model_dir=Path(cfg["model_dir"]),
        default_csv_out=Path(cfg["csv"]),
        profile_html=f"profile/{cfg['profile']}",
        profile_title=str(cfg["title"]),
        pattern_order_csv=cfg.get("order"),
        recursive=bool(cfg.get("recursive", False)),
        microblock_metrics=bool(cfg.get("microblock_metrics", False)),
    )


if __name__ == "__main__":
    main()
