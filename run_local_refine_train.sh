#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-../outputs/refine_stamp_dumps/refcocog_val_0}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/local_refine_train}"
TOP_K="${TOP_K:-64}"
EPOCHS="${EPOCHS:-5}"
LIMIT="${LIMIT:-0}"

python -m local_refine.train_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --selector "${SELECTOR:-hybrid}" \
  --top-k "${TOP_K}" \
  --epochs "${EPOCHS}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --device "${DEVICE:-cuda}"

