#!/usr/bin/env python3
"""Regression sweep over model/MLPerf_Tiny/*.tflite.

Usage:
    ./batch/run_mlperf.py
    ./batch/run_mlperf.py --filter vww
    ./batch/run_mlperf.py --limit 3
    ./batch/run_mlperf.py --offset 3 --limit 3
    ./batch/run_mlperf.py --rerun-all
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
        corpus_name="MLPerf_Tiny",
        default_model_dir=REPO_ROOT / "model" / "MLPerf_Tiny",
        default_csv_out=OUT_DIR / "mlperf_regression.csv",
        profile_html="profile_mlperf.html",
        profile_title="MDLA7 MLPerf_Tiny Profiles",
    )
