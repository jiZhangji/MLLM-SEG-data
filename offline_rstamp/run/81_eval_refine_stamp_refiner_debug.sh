#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
REFINE_SRC="${TOOL_REPO}/offline_rstamp/refine_stamp_src"
SPLIT="${SPLIT:-refcocog_val}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
SELECTOR="${SELECTOR:-hybrid}"
INPUT_DIR="${INPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}}"
CHECKPOINT="${CHECKPOINT:-${MLLM_SEG_ROOT}/outputs/refine_stamp_refiner/${SPLIT}_${EVAL_LIMIT}_${SELECTOR}/local_refiner.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_refiner_eval/${SPLIT}_${EVAL_LIMIT}_${SELECTOR}}"

cd "${TOOL_REPO}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.eval_refiner_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --selector "${SELECTOR}" \
  --use-checkpoint-val \
  --top-k "${TOP_K:-64}" \
  --image-size "${IMAGE_SIZE:-896}" \
  --blend-weight "${BLEND_WEIGHT:-0.8}" \
  --visualize-limit "${VISUALIZE_LIMIT:-8}"

echo "Report:"
echo "${OUTPUT_DIR}/refiner_eval_summary.md"
