#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO="${ROOT}/MLLM-SEG-data"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
MODEL_NAME="${STAMP7B_MODEL_NAME:-${ROOT}/models/STAMP-7B-lora}"
GPU="${CUDA_DEVICE:-0}"

VAL_DUMPS="${STAMP7B_VAL_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_val_full_stamp7b}"
TEST_DUMPS="${STAMP7B_TEST_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full_stamp7b}"
VAL_OUTPUT="${STAMP7B_VAL_OUTPUT:-${ROOT}/outputs/training_free_refine_stamp7b_refcocog_val_full}"
TEST_OUTPUT="${STAMP7B_TEST_OUTPUT:-${ROOT}/outputs/training_free_refine_stamp7b_refcocog_test_full}"
COMBINED_OUTPUT="${STAMP7B_COMBINED_OUTPUT:-${ROOT}/outputs/training_free_refine_stamp7b_refcocog_full_comparison}"

if [[ ! -x "${STAMP_ENV}/bin/python" ]]; then
  echo "ERROR: STAMP Python environment not found at ${STAMP_ENV}." >&2
  exit 1
fi
if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "ERROR: STAMP-7B model not found at ${MODEL_NAME}." >&2
  exit 1
fi

export PATH="${STAMP_ENV}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
mkdir -p "${ROOT}/outputs" "${VAL_DUMPS}" "${TEST_DUMPS}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.training_free_stamp7b_full.lock"
  if ! flock -n 9; then
    echo "Another STAMP-7B training-free full job already holds the lock." >&2
    exit 0
  fi
fi

count_dumps() {
  find "$1" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l
}

cd "${REPO}"

echo "[1/4] Exporting/resuming STAMP-7B RefCOCOg val dumps"
echo "Existing val dumps: $(count_dumps "${VAL_DUMPS}") / 4896"
MODEL_NAME="${MODEL_NAME}" SPLIT=refcocog_val EVAL_LIMIT=0 OUTPUT_DIR="${VAL_DUMPS}" \
  bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh

echo "[2/4] Exporting/resuming STAMP-7B RefCOCOg test dumps"
echo "Existing test dumps: $(count_dumps "${TEST_DUMPS}") / 9602"
MODEL_NAME="${MODEL_NAME}" SPLIT=refcocog_test EVAL_LIMIT=0 OUTPUT_DIR="${TEST_DUMPS}" \
  bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh

echo "[3/4] Running the frozen training-free refiner on val/test"
VAL_DUMPS="${VAL_DUMPS}" TEST_DUMPS="${TEST_DUMPS}" \
VAL_OUTPUT="${VAL_OUTPUT}" TEST_OUTPUT="${TEST_OUTPUT}" COMBINED_OUTPUT="${COMBINED_OUTPUT}" \
  bash run_training_free_refine_full_eval.sh

echo "[4/4] STAMP-7B training-free full evaluation completed"
echo "Summary: ${COMBINED_OUTPUT}/combined_summary.md"
