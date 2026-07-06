#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
SELECTOR="${SELECTOR:-boundary}"
SPLIT="${SPLIT:-refcocog_val}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

cd "${MLLM_SEG_ROOT}/MLLM-SEG-data"

if [[ ! -d "${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}" ]]; then
  SPLIT="${SPLIT}" \
  EVAL_LIMIT="${EVAL_LIMIT}" \
  OUTPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}" \
  bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh
fi

INPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}" \
CHECKPOINT="${MLLM_SEG_ROOT}/outputs/refine_stamp_refiner/refcocog_train_${TRAIN_LIMIT}_${SELECTOR}/local_refiner.pt" \
OUTPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_refiner_eval/official_${SPLIT}_train${TRAIN_LIMIT}_${SELECTOR}" \
SPLIT="${SPLIT}" \
EVAL_LIMIT="${EVAL_LIMIT}" \
SELECTOR="${SELECTOR}" \
EVAL_ALL_FLAG="--eval-all" \
bash offline_rstamp/run/81_eval_refine_stamp_refiner_debug.sh
