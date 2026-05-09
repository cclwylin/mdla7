#!/usr/bin/env python3
"""Regression sweep over model/ETHZ_v5/*.tflite.

Usage:
    ./batch/run_ethz_v5.py
    ./batch/run_ethz_v5.py --filter mobilenet
    ./batch/run_ethz_v5.py --limit 3
    ./batch/run_ethz_v5.py --offset 3 --limit 3
    ./batch/run_ethz_v5.py --rerun-all
    ./batch/run_ethz_v5.py --fast-only
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
        corpus_name="ETHZ_v5",
        default_model_dir=REPO_ROOT / "model" / "ETHZ_v5",
        default_csv_out=OUT_DIR / "ethz_v5_regression.csv",
        profile_html="profile_ethz_v5.html",
        profile_title="MDLA7 ETHZ_v5 Profiles",
    )
