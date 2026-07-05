#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE_SRC="$(cd "${SCRIPT_DIR}/../refine_stamp_src" && pwd)"

cd "${REFINE_SRC}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m unittest discover -s tests -p "test_*.py"
