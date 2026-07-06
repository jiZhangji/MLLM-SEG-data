#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-../outputs/refine_stamp_dumps/refcocog_val_0}"
CHECKPOINT="${CHECKPOINT:-../outputs/local_refine_train/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/local_refine_eval}"
TOP_K="${TOP_K:-64}"
LIMIT="${LIMIT:-0}"

python -m local_refine.eval_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --device "${DEVICE:-cuda}"

