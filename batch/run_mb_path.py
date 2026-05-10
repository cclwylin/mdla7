#!/usr/bin/env python3
"""Regression sweep over model/MB_Path_Slice/**/*.tflite.

Usage:
    ./batch/run_mb_path.py
    ./batch/run_mb_path.py --filter fanout_live_range
    ./batch/run_mb_path.py --limit 3
    ./batch/run_mb_path.py --offset 3 --limit 3
    ./batch/run_mb_path.py --rerun-all
    ./batch/run_mb_path.py --fast-only
    ./batch/run_mb_path.py --list
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
        corpus_name="MB_Path_Slice",
        default_model_dir=REPO_ROOT / "model" / "MB_Path_Slice",
        default_csv_out=OUT_DIR / "mb_path_regression.csv",
        profile_html="profile_mb_path.html",
        profile_title="MDLA7 MB Path Slice Profiles",
        recursive=True,
    )
