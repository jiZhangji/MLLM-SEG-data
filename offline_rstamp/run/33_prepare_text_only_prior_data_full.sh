#!/usr/bin/env bash
set -euo pipefail

# Prepare a fair text-only prior version of the STAMP data.
# Unlike json_files_rstamp, this does NOT use GT bbox or GT center in
# structured_prior_text. It only wraps the original referring expression.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"

python "${TOOL_REPO}/offline_rstamp/scripts/prepare_stamp_training_data.py" \
  --root "${MLLM_SEG_ROOT}" \
  --output-json-dir "${MLLM_SEG_ROOT}/code/STAMP/playground/data/json_files_text_prior" \
  --mask-root "${MLLM_SEG_ROOT}/code/STAMP/playground/data/masks_text_prior" \
  --duplicate 1 \
  --include-prior \
  --prior-mode text_only
