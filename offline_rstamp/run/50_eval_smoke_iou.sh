#!/usr/bin/env bash
set -euo pipefail

# Quick validation-IoU comparison for smoke-trained baseline vs R-STAMP.
# This reads the prepared JSON/mask data directly and does not require the
# original REFER pickle data layout.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
TOOL_REPO_DIR="${TOOL_REPO_DIR:-${MLLM_SEG_ROOT}/MLLM-SEG-data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/smoke_eval_iou}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export STAMP_ATTN_IMPL="${STAMP_ATTN_IMPL:-sdpa}"
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"

cd "${STAMP_CODE_DIR}"

python "${TOOL_REPO_DIR}/offline_rstamp/scripts/eval_smoke_iou.py" \
  --root "${MLLM_SEG_ROOT}" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --baseline-model "${MLLM_SEG_ROOT}/outputs/smoke_baseline_1x48g/final_model" \
  --rstamp-model "${MLLM_SEG_ROOT}/outputs/smoke_rstamp_1x48g/final_model" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${EVAL_LIMIT:-50}" \
  --min-pixels "${STAMP_MIN_PIXELS:-50176}" \
  --max-pixels "${STAMP_MAX_PIXELS:-200704}"
