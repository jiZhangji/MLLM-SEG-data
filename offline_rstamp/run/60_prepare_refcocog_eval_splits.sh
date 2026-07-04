#!/usr/bin/env bash
set -euo pipefail

# Prepare official-like RefCOCOg val/test evaluation JSONs.
# This produces:
#   json_eval_baseline/refcocog_val.json, refcocog_test.json
#   json_eval_text_prior/refcocog_val.json, refcocog_test.json
#   json_eval_oracle_prior/refcocog_val.json, refcocog_test.json

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
STAMP_DATA="${MLLM_SEG_ROOT}/code/STAMP/playground/data"

python "${TOOL_REPO}/offline_rstamp/scripts/prepare_refcoco_eval_data.py" \
  --root "${MLLM_SEG_ROOT}" \
  --splits refcocog_val refcocog_test \
  --prior-mode none \
  --output-json-dir "${STAMP_DATA}/json_eval_baseline" \
  --mask-root "${STAMP_DATA}/masks_eval_baseline"

python "${TOOL_REPO}/offline_rstamp/scripts/prepare_refcoco_eval_data.py" \
  --root "${MLLM_SEG_ROOT}" \
  --splits refcocog_val refcocog_test \
  --prior-mode text_only \
  --output-json-dir "${STAMP_DATA}/json_eval_text_prior" \
  --mask-root "${STAMP_DATA}/masks_eval_text_prior"

python "${TOOL_REPO}/offline_rstamp/scripts/prepare_refcoco_eval_data.py" \
  --root "${MLLM_SEG_ROOT}" \
  --splits refcocog_val refcocog_test \
  --prior-mode oracle \
  --output-json-dir "${STAMP_DATA}/json_eval_oracle_prior" \
  --mask-root "${STAMP_DATA}/masks_eval_oracle_prior"

echo "Prepared RefCOCOg evaluation splits under ${STAMP_DATA}"
