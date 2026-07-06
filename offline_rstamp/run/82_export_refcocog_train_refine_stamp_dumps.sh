#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
STAMP_DATA="${MLLM_SEG_ROOT}/code/STAMP/playground/data"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
OFFSET="${OFFSET:-0}"

cd "${MLLM_SEG_ROOT}/MLLM-SEG-data"

SPLIT="refcocog_train" \
EVAL_LIMIT="${TRAIN_LIMIT}" \
OFFSET="${OFFSET}" \
JSON_PATH="${STAMP_DATA}/json_files_baseline/refcocog_formatted_all_sentences_doubled_mp.json" \
OUTPUT_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/refcocog_train_${TRAIN_LIMIT}" \
bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh
