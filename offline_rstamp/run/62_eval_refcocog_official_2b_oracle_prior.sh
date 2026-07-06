#!/usr/bin/env bash
set -euo pipefail

# Upper-bound diagnostic: official STAMP-2B original query vs oracle
# target/GT-box/GT-center prior on RefCOCOg val/test.
# Do NOT report this as a fair deployable result.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
MODEL_NAME="${MODEL_NAME:-${MLLM_SEG_ROOT}/models/STAMP-2B-uni}"
STAMP_DATA="${MLLM_SEG_ROOT}/code/STAMP/playground/data"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export STAMP_ATTN_IMPL="${STAMP_ATTN_IMPL:-sdpa}"
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"

cd "${STAMP_CODE_DIR}"

for SPLIT in refcocog_val refcocog_test; do
  echo "=== Evaluating ${SPLIT}: official STAMP-2B original vs oracle prior ==="
  python "${TOOL_REPO}/offline_rstamp/scripts/eval_smoke_iou.py" \
    --root "${MLLM_SEG_ROOT}" \
    --stamp-code-dir "${STAMP_CODE_DIR}" \
    --baseline-model "${MODEL_NAME}" \
    --rstamp-model "${MODEL_NAME}" \
    --baseline-json "${STAMP_DATA}/json_eval_baseline/${SPLIT}.json" \
    --rstamp-json "${STAMP_DATA}/json_eval_oracle_prior/${SPLIT}.json" \
    --output-dir "${MLLM_SEG_ROOT}/outputs/eval_refcocog_official_2b_oracle_prior/${SPLIT}" \
    --limit "${EVAL_LIMIT}"
done

echo "Reports:"
find "${MLLM_SEG_ROOT}/outputs/eval_refcocog_official_2b_oracle_prior" -name "smoke_iou_comparison.md" -print
