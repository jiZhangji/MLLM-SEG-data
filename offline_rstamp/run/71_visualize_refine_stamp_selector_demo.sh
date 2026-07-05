#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE_SRC="$(cd "${SCRIPT_DIR}/../refine_stamp_src" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REFINE_SRC}/outputs/selector_demo}"
TOP_K="${TOP_K:-64}"
SELECTOR_MODE="${SELECTOR_MODE:-hybrid}"

cd "${REFINE_SRC}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.visualize_selector \
  --demo \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --selector-mode "${SELECTOR_MODE}"

echo "Selector visualization written to: ${OUTPUT_DIR}"
