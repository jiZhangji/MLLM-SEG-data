#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-../outputs/refine_stamp_dumps/refcocog_train_0}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/token_head_train_full_h128_e30_lr3e4_b256}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
DEVICE="${DEVICE:-cuda}"

python -m token_refine.train_head_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --device "${DEVICE}"
