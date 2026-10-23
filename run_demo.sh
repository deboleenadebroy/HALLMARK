#!/usr/bin/env bash
# Hallmark demo launcher for macOS / Linux.  Usage:  bash run_demo.sh
set -e
cd "$(dirname "$0")"

if ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract OCR is not installed."
  echo "  macOS:          brew install tesseract"
  echo "  Ubuntu/Debian:  sudo apt-get install -y tesseract-ocr"
  exit 1
fi

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then echo "Python 3.10+ is required."; exit 1; fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Python 3.10+ is required."; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo
echo "== Self-check: analysing every demo file live =="
python scripts/check_demo.py || true

export HALLMARK_PORT="${HALLMARK_PORT:-8000}"
echo
echo "== Hallmark is starting. Open  http://127.0.0.1:${HALLMARK_PORT}  in your browser. Ctrl+C to stop. =="
python app.py
