#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
MODEL_NAME="${STAMP_BATCH_MODEL_NAME:-${ROOT}/models/STAMP-2B-uni}"
GPU="${CUDA_DEVICE:-1}"
BATCH_SIZE="${STAMP_BATCH_SIZE:-8}"
LIMIT="${STAMP_BATCH_CHECK_SAMPLES:-16}"
SPLIT="${STAMP_BATCH_CHECK_SPLIT:-refcoco_val}"
JSON_PATH="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json"
OUTPUT_ROOT="${STAMP_BATCH_CHECK_OUTPUT:-${ROOT}/outputs/stamp_batch_equivalence/stamp2b_bs${BATCH_SIZE}}"
SINGLE_DIR="${OUTPUT_ROOT}/single"
BATCH_DIR="${OUTPUT_ROOT}/batch"
REPORT="${OUTPUT_ROOT}/equivalence.json"

if [[ ! -x "${STAMP_ENV}/bin/python" ]]; then
  echo "ERROR: STAMP environment not found: ${STAMP_ENV}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_NAME}" || ! -f "${JSON_PATH}" ]]; then
  echo "ERROR: model or evaluation JSON is missing." >&2
  exit 1
fi
if (( BATCH_SIZE < 2 )); then
  echo "ERROR: STAMP_BATCH_SIZE must be at least 2 for equivalence testing." >&2
  exit 1
fi

export PATH="${STAMP_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
mkdir -p "${SINGLE_DIR}" "${BATCH_DIR}"
cd "${SCRIPT_DIR}"

python offline_rstamp/scripts/patch_stamp_refinement_export.py \
  --stamp-code-dir "${ROOT}/code/STAMP" \
  --target both

echo "[1/3] Exporting ${LIMIT} reference samples with batch size 1"
MODEL_NAME="${MODEL_NAME}" SPLIT="${SPLIT}" JSON_PATH="${JSON_PATH}" \
EVAL_LIMIT="${LIMIT}" OUTPUT_DIR="${SINGLE_DIR}" BATCH_SIZE=1 OVERWRITE=1 \
  bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh

echo "[2/3] Exporting the same samples with requested batch size ${BATCH_SIZE}"
MODEL_NAME="${MODEL_NAME}" SPLIT="${SPLIT}" JSON_PATH="${JSON_PATH}" \
EVAL_LIMIT="${LIMIT}" OUTPUT_DIR="${BATCH_DIR}" BATCH_SIZE="${BATCH_SIZE}" OVERWRITE=1 \
  bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh

echo "[3/3] Comparing text, grids, token labels and logits"
python -m training_free_refine.compare_stamp_batch_dumps \
  --single-dir "${SINGLE_DIR}" \
  --batch-dir "${BATCH_DIR}" \
  --expected "${LIMIT}" \
  --output "${REPORT}"
touch "${OUTPUT_ROOT}/PASSED"
echo "STAMP batch equivalence passed: ${REPORT}"
