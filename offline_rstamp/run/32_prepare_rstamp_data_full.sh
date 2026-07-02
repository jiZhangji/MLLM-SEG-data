#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"

python "${TOOL_REPO}/offline_rstamp/scripts/prepare_stamp_training_data.py" \
  --root "${MLLM_SEG_ROOT}" \
  --output-json-dir "${MLLM_SEG_ROOT}/code/STAMP/playground/data/json_files_rstamp" \
  --mask-root "${MLLM_SEG_ROOT}/code/STAMP/playground/data/masks_rstamp" \
  --duplicate 1 \
  --include-prior
