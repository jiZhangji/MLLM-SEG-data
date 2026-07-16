#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TRAINING_FREE_REPO:-${SCRIPT_DIR}}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${TRAINING_FREE_PYTHON:-${STAMP_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

STAMP_ROWS="${STAMP_VIS_ROWS:-${ROOT}/outputs/training_free_refine_stamp7b_refcocog_val_full/eval_rows.csv}"
TEXT4SEG_ROWS="${TEXT4SEG_VIS_ROWS:-${ROOT}/outputs/text4seg_training_free_refcocog_val/eval_rows.csv}"
OUTPUT_ROOT="${TRAINING_FREE_VIS_OUTPUT:-${ROOT}/outputs/training_free_visualizations}"
LIMIT="${TRAINING_FREE_VIS_LIMIT:-0}"
PANELS="${TRAINING_FREE_VIS_PANELS_PER_GROUP:-3}"

for path in "${STAMP_ROWS}" "${TEXT4SEG_ROWS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required evaluation rows not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO}"

echo "[1/3] Analyzing original STAMP versus STAMP + Training-Free"
PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m training_free_refine.visualize_comparison run \
  --kind stamp \
  --rows "${STAMP_ROWS}" \
  --output-dir "${OUTPUT_ROOT}/stamp7b_val" \
  --label "STAMP-7B" \
  --limit "${LIMIT}" \
  --panels-per-group "${PANELS}"

echo "[2/3] Analyzing original Text4Seg versus Text4Seg + Training-Free"
PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m training_free_refine.visualize_comparison run \
  --kind text4seg \
  --rows "${TEXT4SEG_ROWS}" \
  --output-dir "${OUTPUT_ROOT}/text4seg_val" \
  --label "Text4Seg-7B-p24" \
  --limit "${LIMIT}" \
  --panels-per-group "${PANELS}" \
  --boundary-sigma 8.0

echo "[3/3] Building the cross-model comparison"
PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m training_free_refine.visualize_comparison compare \
  --run "STAMP-7B=${OUTPUT_ROOT}/stamp7b_val/visual_analysis_rows.csv" \
  --run "Text4Seg-7B-p24=${OUTPUT_ROOT}/text4seg_val/visual_analysis_rows.csv" \
  --output-dir "${OUTPUT_ROOT}/combined"

echo "Training-Free visualizations completed."
echo "STAMP: ${OUTPUT_ROOT}/stamp7b_val"
echo "Text4Seg: ${OUTPUT_ROOT}/text4seg_val"
echo "Combined: ${OUTPUT_ROOT}/combined/cross_model_comparison.png"
