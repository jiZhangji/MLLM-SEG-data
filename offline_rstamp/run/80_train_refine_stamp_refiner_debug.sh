#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
REFINE_SRC="${TOOL_REPO}/offline_rstamp/refine_stamp_src"
SPLIT="${SPLIT:-refcocog_val}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
SELECTOR="${SELECTOR:-hybrid}"
INPUT_DIR="${INPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_refiner/${SPLIT}_${EVAL_LIMIT}_${SELECTOR}}"

cd "${TOOL_REPO}"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.train_refiner_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --selector "${SELECTOR}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --top-k "${TOP_K:-64}" \
  --image-size "${IMAGE_SIZE:-896}" \
  --learning-rate "${LR:-1e-4}" \
  --refiner-chunk-size "${REFINER_CHUNK_SIZE:-16}"

echo "Checkpoint:"
echo "${OUTPUT_DIR}/local_refiner.pt"
