#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN was not found. Set PYTHON_BIN to your Python executable." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import huggingface_hub, tqdm" >/dev/null 2>&1; then
  echo "Installing downloader dependencies..."
  "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/download.py" "$@"

