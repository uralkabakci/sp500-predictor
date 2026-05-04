#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MeTrade Live ==="

# Create data dir
mkdir -p data logs

# Install deps if venv does not exist
if [ ! -d ".venv" ]; then
  echo "[*] Creating venv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --quiet -r requirements.txt
else
  source .venv/bin/activate
fi

export PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH

echo "[*] Starting server on http://0.0.0.0:8000"
python app.py
