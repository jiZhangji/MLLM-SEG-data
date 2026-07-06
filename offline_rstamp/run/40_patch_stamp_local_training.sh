#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"

python "${TOOL_REPO}/offline_rstamp/scripts/patch_stamp_local_training.py" \
  --stamp-code-dir "${STAMP_CODE_DIR}"

echo "Patch complete. You can inspect:"
echo "grep -n \"STAMP_JSON_DIR\\|STAMP_USE_STRUCTURED_PRIOR\\|STAMP_ATTN_IMPL\" ${STAMP_CODE_DIR}/train/main_uni.py"

