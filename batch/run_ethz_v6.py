#!/usr/bin/env python3
"""Regression sweep over model/ETHZ_v6/*.tflite.

Models run in the same small-mdla6_cx-first order as run_mdla6_pattern.py's
mdla6_ethz_v6_sorted.csv by default.

Usage:
    ./batch/run_ethz_v6.py
    ./batch/run_ethz_v6.py --filter mobilenet
    ./batch/run_ethz_v6.py --limit 3
    ./batch/run_ethz_v6.py --offset 3 --limit 3
    ./batch/run_ethz_v6.py --rerun-all
    ./batch/run_ethz_v6.py --fast-only
    ./batch/run_ethz_v6.py --rtl-fast
    ./batch/run_ethz_v6.py --compare-rtl-fast
    ./batch/run_ethz_v6.py --fast-only --engine-model rtl
    ./batch/run_ethz_v6.py --pattern-order-csv other_order.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VENV_DIR = Path(os.environ.get("MDLA7_VENV") or
                Path.home() / ".venvs/mdla7").expanduser()
VENV_PY = VENV_DIR / "bin" / "python"

if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])

from corpus_runner import REPO_ROOT, OUT_DIR, run_corpus  # noqa: E402


if __name__ == "__main__":
    run_corpus(
        corpus_name="ETHZ_v6",
        default_model_dir=REPO_ROOT / "model" / "ETHZ_v6",
        default_csv_out=OUT_DIR / "ethz_v6_regression.csv",
        profile_html="profile_ethz_v6.html",
        profile_title="MDLA7 ETHZ_v6 Profiles",
        pattern_order_csv=REPO_ROOT / "batch" / "mdla6_ethz_v6_sorted.csv",
    )
