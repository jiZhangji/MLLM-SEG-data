#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"

python "${TOOL_REPO}/offline_rstamp/scripts/patch_stamp_refinement_export.py" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --target segment_predictor

echo "Checking refinement export method:"
OUTPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_phase2_inspection_after_patch" \
bash "${TOOL_REPO}/offline_rstamp/run/77_inspect_stamp_refinement_points.sh"
