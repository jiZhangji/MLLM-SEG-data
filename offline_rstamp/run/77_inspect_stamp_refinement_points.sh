#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
REFINE_SRC="${TOOL_REPO}/offline_rstamp/refine_stamp_src"
OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_phase2_inspection}"

cd "${STAMP_CODE_DIR}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.inspect_stamp_refinement_points \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "Inspection output:"
echo "${OUTPUT_DIR}"
