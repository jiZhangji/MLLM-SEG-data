#!/usr/bin/env bash
set -euo pipefail

# Export STAMP Phase-2 tensors for Refine-STAMP selector/refiner diagnosis.
# Requires the server STAMP GenerativeSegmenter to expose one refinement export
# method documented by export_stamp_refinement_dumps.py.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
REFINE_SRC="${TOOL_REPO}/offline_rstamp/refine_stamp_src"
MODEL_NAME="${MODEL_NAME:-${MLLM_SEG_ROOT}/models/STAMP-2B-uni}"
STAMP_DATA="${MLLM_SEG_ROOT}/code/STAMP/playground/data"
SPLIT="${SPLIT:-refcocog_val}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
OFFSET="${OFFSET:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}}"
JSON_PATH="${JSON_PATH:-${STAMP_DATA}/json_eval_baseline/${SPLIT}.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export STAMP_ATTN_IMPL="${STAMP_ATTN_IMPL:-sdpa}"
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
export STAMP_MIN_PIXELS="${STAMP_MIN_PIXELS:-802816}"
export STAMP_MAX_PIXELS="${STAMP_MAX_PIXELS:-1003520}"
export STAMP_PROMPT_MODE="${STAMP_PROMPT_MODE:-official}"

cd "${STAMP_CODE_DIR}"

PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.export_stamp_refinement_dumps \
  --root "${MLLM_SEG_ROOT}" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --model "${MODEL_NAME}" \
  --json "${JSON_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split-name "${SPLIT}" \
  --limit "${EVAL_LIMIT}" \
  --offset "${OFFSET}" \
  --min-pixels "${STAMP_MIN_PIXELS}" \
  --max-pixels "${STAMP_MAX_PIXELS}" \
  --prompt-mode "${STAMP_PROMPT_MODE}"

echo "Refinement dumps written to:"
echo "${OUTPUT_DIR}"
