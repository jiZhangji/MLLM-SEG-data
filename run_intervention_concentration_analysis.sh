#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO="${INTERVENTION_REPO:-${ROOT}/MLLM-SEG-data}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON="${INTERVENTION_PYTHON:-${STAMP_ENV}/bin/python}"
OUTPUT="${INTERVENTION_OUTPUT:-${ROOT}/outputs/freeref_intervention_concentration}"
LIMIT="${INTERVENTION_LIMIT:-0}"
RUN_TEXT4SEG="${INTERVENTION_RUN_TEXT4SEG:-1}"

STAMP_ROWS="${INTERVENTION_STAMP_ROWS:-${ROOT}/outputs/training_free_refine_stamp7b_refcocog_val_full/eval_rows.csv}"
TEXT4SEG_ROWS="${INTERVENTION_TEXT4SEG_ROWS:-${ROOT}/outputs/text4seg_training_free_refcocog_val/eval_rows.csv}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python environment is unavailable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${STAMP_ROWS}" ]]; then
  echo "ERROR: STAMP rows are unavailable: ${STAMP_ROWS}" >&2
  exit 1
fi
if [[ "${RUN_TEXT4SEG}" == "1" && ! -f "${TEXT4SEG_ROWS}" ]]; then
  echo "ERROR: Text4Seg rows are unavailable: ${TEXT4SEG_ROWS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

run_one() {
  local kind="$1"
  local rows="$2"
  local label="$3"
  local output="$4"
  "${PYTHON}" -m training_free_refine.analyze_intervention_concentration \
    --kind "${kind}" \
    --rows "${rows}" \
    --output-dir "${output}" \
    --label "${label}" \
    --limit "${LIMIT}" \
    --n-segments 1024 \
    --graph-lambda 1.0 \
    --confidence-power 2.0 \
    --fusion-power 1.0 \
    --foreground-seed 0.9 \
    --background-seed 0.1 \
    --seed-strength 50 \
    --threshold 0.5
}

echo "[1/2] STAMP-7B soft-probability concentration analysis"
run_one stamp "${STAMP_ROWS}" "STAMP-7B" "${OUTPUT}/stamp7b_refcocog_val"

if [[ "${RUN_TEXT4SEG}" == "1" ]]; then
  echo "[2/2] Text4Seg-p24 hard-mask boundary concentration analysis"
  run_one text4seg "${TEXT4SEG_ROWS}" "Text4Seg-p24" "${OUTPUT}/text4seg_refcocog_val"
else
  echo "[2/2] Text4Seg analysis skipped"
fi

touch "${OUTPUT}/COMPLETE"
echo "Complete: ${OUTPUT}"
