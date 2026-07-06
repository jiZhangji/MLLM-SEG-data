#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash offline_rstamp/run/72_visualize_refine_stamp_selector_from_dump.sh /path/to/stamp_refinement_dump.pt"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE_SRC="$(cd "${SCRIPT_DIR}/../refine_stamp_src" && pwd)"
INPUT_PT="$1"
OUTPUT_DIR="${OUTPUT_DIR:-${REFINE_SRC}/outputs/selector_from_dump}"
TOP_K="${TOP_K:-64}"
SELECTOR_MODE="${SELECTOR_MODE:-hybrid}"

cd "${REFINE_SRC}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.visualize_selector \
  --input "${INPUT_PT}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --selector-mode "${SELECTOR_MODE}"

echo "Selector visualization written to: ${OUTPUT_DIR}"
