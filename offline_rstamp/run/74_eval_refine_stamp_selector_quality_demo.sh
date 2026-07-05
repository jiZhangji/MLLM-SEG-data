#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE_SRC="$(cd "${SCRIPT_DIR}/../refine_stamp_src" && pwd)"
DUMP_DIR="${DUMP_DIR:-${REFINE_SRC}/outputs/demo_dumps}"
OUTPUT_DIR="${OUTPUT_DIR:-${REFINE_SRC}/outputs/demo_selector_quality}"
COUNT="${COUNT:-8}"
TOP_K="${TOP_K:-64}"

cd "${REFINE_SRC}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.make_demo_refinement_dumps \
  --output-dir "${DUMP_DIR}" \
  --count "${COUNT}"

PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.eval_selector_quality \
  --input-dir "${DUMP_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --visualize-limit "${COUNT}"

echo "Demo selector quality report:"
echo "${OUTPUT_DIR}/selector_quality_summary.md"
