#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"

python "${TOOL_REPO}/offline_rstamp/scripts/fix_stamp_missing_segment_predictor.py" \
  --stamp-code-dir "${STAMP_CODE_DIR}"

echo "Import fix complete. Checking:"
ls -l "${STAMP_CODE_DIR}/segment_predictor.py" || true

