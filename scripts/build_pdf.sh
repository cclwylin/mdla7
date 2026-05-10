#!/usr/bin/env bash
# Build MDLA7 textbook PDF.
#
# Steps:
#   1. Render every drawio/*.drawio -> drawio/*.drawio.png
#   2. Concatenate md/0*.md (with \newpage between) -> mdla7_textbook.md
#   3. md2html.py: mdla7_textbook.md -> mdla7_textbook.html
#   4. Chrome headless: mdla7_textbook.html -> mdla7_textbook.pdf
#
# Usage: scripts/build_pdf.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRAWIO="/Applications/draw.io.app/Contents/MacOS/draw.io"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PY="/tmp/mdpdf_venv/bin/python"
MAIN="mdla7_textbook"
TEXTBOOK_MD="${MAIN}.md"
TEXTBOOK_HTML="${MAIN}.html"
TEXTBOOK_PDF="${MAIN}.pdf"

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
  chapters=([0-9][0-9]_*.md)
  for idx in "${!chapters[@]}"; do
    cat "${chapters[$idx]}"
    if (( idx + 1 < ${#chapters[@]} )); then
      printf '\n\n\\newpage\n\n'
    else
      printf '\n\n\\newpage\n'
    fi
  done
} > "${TEXTBOOK_MD}"
printf '       %s lines\n' "$(wc -l < "${TEXTBOOK_MD}")"

echo "[3/4] markdown -> html"
"$PY" "$ROOT/scripts/md2html.py" "${TEXTBOOK_MD}" "${TEXTBOOK_HTML}"

echo "[4/4] html -> pdf"
mkdir -p "$ROOT/pdf"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$ROOT/pdf/${TEXTBOOK_PDF}" \
  --virtual-time-budget=30000 --run-all-compositor-stages-before-draw \
  "file://$(pwd)/${TEXTBOOK_HTML}" 2>&1 | grep "bytes written" || true

ls -lh "$ROOT/pdf/${TEXTBOOK_PDF}"
echo "done -> $ROOT/pdf/${TEXTBOOK_PDF}"
