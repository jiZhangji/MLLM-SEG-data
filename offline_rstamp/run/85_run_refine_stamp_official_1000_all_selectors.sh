#!/usr/bin/env bash
set -euo pipefail

# One-command official-comparable Refine-STAMP run.
# It trains local refiners on RefCOCOg train dumps and evaluates val/test.

MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
REFINE_SRC="${TOOL_REPO}/offline_rstamp/refine_stamp_src"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
EPOCHS="${EPOCHS:-5}"
REFINER_CHUNK_SIZE="${REFINER_CHUNK_SIZE:-8}"
TOP_K="${TOP_K:-64}"
SELECTORS="${SELECTORS:-random uncertainty boundary hybrid}"
SPLITS="${SPLITS:-refcocog_val refcocog_test}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES

cd "${TOOL_REPO}"

echo "=== Step 1/4: export RefCOCOg train dumps (${TRAIN_LIMIT}) ==="
TRAIN_LIMIT="${TRAIN_LIMIT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash offline_rstamp/run/82_export_refcocog_train_refine_stamp_dumps.sh

echo "=== Step 2/4: train selectors ==="
for selector in ${SELECTORS}; do
  echo "=== Training selector: ${selector} ==="
  TRAIN_LIMIT="${TRAIN_LIMIT}" \
  SELECTOR="${selector}" \
  EPOCHS="${EPOCHS}" \
  TOP_K="${TOP_K}" \
  REFINER_CHUNK_SIZE="${REFINER_CHUNK_SIZE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  bash offline_rstamp/run/83_train_refine_stamp_refiner_official.sh
done

echo "=== Step 3/4: evaluate selectors on official splits ==="
for split in ${SPLITS}; do
  for selector in ${SELECTORS}; do
    echo "=== Evaluating split=${split}, selector=${selector} ==="
    TRAIN_LIMIT="${TRAIN_LIMIT}" \
    SELECTOR="${selector}" \
    SPLIT="${split}" \
    EVAL_LIMIT=0 \
    TOP_K="${TOP_K}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash offline_rstamp/run/84_eval_refine_stamp_refiner_official_split.sh
  done
done

echo "=== Step 4/4: summarize ==="
SUMMARY_DIR="${MLLM_SEG_ROOT}/outputs/refine_stamp_official_summary"
SUMMARY_PATH="${SUMMARY_DIR}/official_refine_stamp_train${TRAIN_LIMIT}_summary.md"
PYTHONPATH="${REFINE_SRC}:${PYTHONPATH:-}" python -m refine_stamp.scripts.summarize_official_refiner_results \
  --root "${MLLM_SEG_ROOT}" \
  --train-limit "${TRAIN_LIMIT}" \
  --selectors ${SELECTORS} \
  --splits ${SPLITS} \
  --output "${SUMMARY_PATH}"

echo "Summary:"
echo "${SUMMARY_PATH}"
