#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-../outputs/refine_stamp_dumps/refcocog_train_1000}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/token_refine_train}"

python -m token_refine.train_adapter_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT:-0}" \
  --epochs "${EPOCHS:-10}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-cuda}" \
  --image-size "${IMAGE_SIZE:-896}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --learning-rate "${LR:-1e-3}" \
  --uncertainty-loss-weight "${UNCERTAINTY_LOSS_WEIGHT:-2.0}" \
  --delta-reg-weight "${DELTA_REG_WEIGHT:-0.01}" \
  --grid-cache "${GRID_CACHE:-${INPUT_DIR}/token_refine_grid_cache.json}" \
  "${GATE_FLAG:---use-uncertainty-gate}"
