#!/usr/bin/env bash
# One-shot environment setup for MDLA7 SystemC simulator.
#   - creates ~/.venvs/mdla7 (outside repo so volumes without xattrs don't
#     poison Python's site loader with AppleDouble "._*" sidecar files)
#   - installs requirements.txt
#   - prints how to run
#
# Override venv location:   MDLA7_VENV=/path/to/venv ./setup.sh

set -euo pipefail

VENV="${MDLA7_VENV:-$HOME/.venvs/mdla7}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[setup] venv      = $VENV"
echo "[setup] requirements = $HERE/requirements.txt"
echo

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[setup] creating venv ..."
    python3 -m venv "$VENV"
fi

echo "[setup] upgrading pip ..."
"$VENV/bin/python" -m pip install --upgrade --quiet pip

echo "[setup] installing requirements ..."
"$VENV/bin/python" -m pip install --quiet -r "$HERE/requirements.txt"

echo
echo "[setup] done. quick test:"
"$VENV/bin/python" - <<'PY'
import platform, sys
print(f"  python   : {sys.version.split()[0]}  ({platform.machine()})")
import numpy as n
print(f"  numpy    : {n.__version__}")
try:
    import tflite_runtime as t
    print(f"  tflite_runtime: {t.__version__}")
except ImportError:
    import tensorflow as tf
    print(f"  tensorflow: {tf.__version__}")
PY

cat <<EOF

next: cd .. && ./batch/run_systemc.py --filter ethz --model-filter mobilenet --limit 1
      (run_systemc.py auto-re-execs into $VENV/bin/python; no need to activate.)
EOF
