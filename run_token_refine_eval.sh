#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-../outputs/refine_stamp_dumps/refcocog_val_200}"
CHECKPOINT="${CHECKPOINT:-../outputs/token_refine_train/adapter.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/token_refine_eval}"

python -m token_refine.eval_adapter_from_dumps \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT:-0}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-cuda}" \
  --image-size "${IMAGE_SIZE:-896}" \
  --grid-cache "${GRID_CACHE:-${INPUT_DIR}/token_refine_grid_cache.json}"
