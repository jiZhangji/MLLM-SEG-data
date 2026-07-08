#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-../outputs/refine_stamp_dumps/refcocog_val_full}"
CHECKPOINT="${CHECKPOINT:-../outputs/token_head_train_full_h128_e30_lr3e4_b256/head.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/token_head_eval_full_h128_e30_lr3e4_b256_on_val_full}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

python -m token_refine.eval_head_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"
