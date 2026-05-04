#!/usr/bin/env bash
# Build SystemC textbook PDF.
#
# Steps:
#   1. Render every drawio/*.drawio -> drawio/*.drawio.png
#   2. Concatenate md/0*.md (with \newpage between) -> combined.md
#   3. md2html.py: combined.md -> textbook.html
#   4. Chrome headless: textbook.html -> textbook.pdf
#
# Usage: scripts/build_pdf.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRAWIO="/Applications/draw.io.app/Contents/MacOS/draw.io"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PY="/tmp/mdpdf_venv/bin/python"
MAIN="SystemC_textbook"

cd "$ROOT"
echo "[0/4] rendering equations (LaTeX -> PNG)"
"$PY" scripts/render_eq.py eq | tail -1

cd "$ROOT/drawio"
echo "[1/4] rendering drawio -> png"
for f in *.drawio; do
  "$DRAWIO" -x -f png -b 10 -s 2 -o "${f}.png" "$f" 2>/dev/null \
    | grep -v "SharedImageManager" || true
  printf '       %s\n' "${f}.png"
done

cd "$ROOT/md"
echo "[2/4] concatenating chapters"
{
  for f in [0-9][0-9]_*.md; do
    cat "$f"; printf '\n\n\\newpage\n\n'
  done
} > "${MAIN}_combined.md"
printf '       %s lines\n' "$(wc -l < ${MAIN}_combined.md)"

echo "[3/4] markdown -> html"
"$PY" "$ROOT/scripts/md2html.py" "${MAIN}_combined.md" "${MAIN}.html"

echo "[4/4] html -> pdf"
mkdir -p "$ROOT/pdf"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$ROOT/pdf/${MAIN}.pdf" \
  --virtual-time-budget=30000 --run-all-compositor-stages-before-draw \
  "file://$(pwd)/${MAIN}.html" 2>&1 | grep "bytes written" || true

ls -lh "$ROOT/pdf/${MAIN}.pdf"
echo "done -> $ROOT/pdf/${MAIN}.pdf"
