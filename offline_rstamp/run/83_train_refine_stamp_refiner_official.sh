#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
SELECTOR="${SELECTOR:-boundary}"

cd "${MLLM_SEG_ROOT}/MLLM-SEG-data"

INPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/refcocog_train_${TRAIN_LIMIT}" \
OUTPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_refiner/refcocog_train_${TRAIN_LIMIT}_${SELECTOR}" \
SPLIT="refcocog_train" \
EVAL_LIMIT="${TRAIN_LIMIT}" \
SELECTOR="${SELECTOR}" \
EPOCHS="${EPOCHS:-5}" \
REFINER_CHUNK_SIZE="${REFINER_CHUNK_SIZE:-8}" \
bash offline_rstamp/run/80_train_refine_stamp_refiner_debug.sh
