#!/usr/bin/env bash
set -euo pipefail

# Official-aligned single-split evaluation.
# Differences from earlier smoke eval:
#   - official STAMP prompt template by default;
#   - official-like min/max pixel settings from run_seg_ref.py;
#   - one split at a time, so two GPUs can run val/test independently.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
MODEL_NAME="${MODEL_NAME:-${MLLM_SEG_ROOT}/models/STAMP-2B-uni}"
STAMP_DATA="${MLLM_SEG_ROOT}/code/STAMP/playground/data"
SPLIT="${SPLIT:-refcocog_val}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export STAMP_ATTN_IMPL="${STAMP_ATTN_IMPL:-sdpa}"
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
export STAMP_MIN_PIXELS="${STAMP_MIN_PIXELS:-802816}"
export STAMP_MAX_PIXELS="${STAMP_MAX_PIXELS:-1003520}"
export STAMP_PROMPT_MODE="${STAMP_PROMPT_MODE:-official}"

cd "${STAMP_CODE_DIR}"

echo "=== Official-aligned evaluation: ${SPLIT} ==="
echo "MODEL_NAME=${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "STAMP_MIN_PIXELS=${STAMP_MIN_PIXELS}"
echo "STAMP_MAX_PIXELS=${STAMP_MAX_PIXELS}"
echo "STAMP_PROMPT_MODE=${STAMP_PROMPT_MODE}"

python "${TOOL_REPO}/offline_rstamp/scripts/eval_smoke_iou.py" \
  --root "${MLLM_SEG_ROOT}" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --baseline-model "${MODEL_NAME}" \
  --rstamp-model "${MODEL_NAME}" \
  --baseline-json "${STAMP_DATA}/json_eval_baseline/${SPLIT}.json" \
  --rstamp-json "${STAMP_DATA}/json_eval_text_prior/${SPLIT}.json" \
  --output-dir "${MLLM_SEG_ROOT}/outputs/eval_refcocog_official_aligned_2b_text_prior/${SPLIT}" \
  --limit "${EVAL_LIMIT}" \
  --min-pixels "${STAMP_MIN_PIXELS}" \
  --max-pixels "${STAMP_MAX_PIXELS}" \
  --prompt-mode "${STAMP_PROMPT_MODE}"

echo "Report:"
echo "${MLLM_SEG_ROOT}/outputs/eval_refcocog_official_aligned_2b_text_prior/${SPLIT}/smoke_iou_comparison.md"
