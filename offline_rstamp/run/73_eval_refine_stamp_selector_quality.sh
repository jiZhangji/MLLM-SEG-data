#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash offline_rstamp/run/73_eval_refine_stamp_selector_quality.sh /path/to/refinement_dump_dir"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE_SRC="$(cd "${SCRIPT_DIR}/../refine_stamp_src" && pwd)"
INPUT_DIR="$1"
OUTPUT_DIR="${OUTPUT_DIR:-${REFINE_SRC}/outputs/selector_quality}"
LIMIT="${LIMIT:-0}"
TOP_K="${TOP_K:-64}"
VISUALIZE_LIMIT="${VISUALIZE_LIMIT:-8}"

cd "${REFINE_SRC}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.eval_selector_quality \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --top-k "${TOP_K}" \
  --visualize-limit "${VISUALIZE_LIMIT}"

echo "Selector quality report:"
echo "${OUTPUT_DIR}/selector_quality_summary.md"
