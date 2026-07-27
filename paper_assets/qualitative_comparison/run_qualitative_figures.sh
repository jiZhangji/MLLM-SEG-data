#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO="${QUALITATIVE_REPO:-${ROOT}/MLLM-SEG-data}"
FINAL_ROOT="${FINAL_OUTPUT_ROOT:-${ROOT}/outputs/freeref_final_h100_overnight_v2}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${QUALITATIVE_PYTHON:-${STAMP_ENV}/bin/python}"
OUTPUT_DIR="${QUALITATIVE_OUTPUT_DIR:-${ROOT}/outputs/freeref_qualitative_figures}"

STAMP_ROWS="${QUALITATIVE_STAMP_ROWS:-${FINAL_ROOT}/studies/stamp7b/graph_lambda/lambda_1/eval_rows.csv}"
TEXT4SEG_ROWS="${QUALITATIVE_TEXT4SEG_ROWS:-${ROOT}/outputs/text4seg_training_free_refcoco_val/eval_rows.csv}"
PIXELLM_ROWS="${QUALITATIVE_PIXELLM_ROWS:-${ROOT}/outputs/pixellm_public_freeref_full/refcoco_val/freeref/eval_rows.csv}"
PIXELLM_MANIFEST="${QUALITATIVE_PIXELLM_MANIFEST:-${ROOT}/outputs/pixellm_public_freeref_full/refcoco_val/official/manifest.jsonl}"
POSTPROCESS_ROWS="${QUALITATIVE_POSTPROCESS_ROWS:-${FINAL_ROOT}/postprocess/stamp7b/refcoco_val/eval_rows.csv}"

for path in \
  "${STAMP_ROWS}" \
  "${TEXT4SEG_ROWS}" \
  "${PIXELLM_ROWS}" \
  "${PIXELLM_MANIFEST}" \
  "${POSTPROCESS_ROWS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required qualitative input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

command=(
  "${PYTHON_BIN}"
  -m paper_assets.qualitative_comparison.generate_qualitative_figures
  --stamp-rows "${STAMP_ROWS}"
  --text4seg-rows "${TEXT4SEG_ROWS}"
  --pixellm-rows "${PIXELLM_ROWS}"
  --pixellm-manifest "${PIXELLM_MANIFEST}"
  --postprocess-rows "${POSTPROCESS_ROWS}"
  --output-dir "${OUTPUT_DIR}"
  --sample-count "${QUALITATIVE_SAMPLE_COUNT:-4}"
  --rows-per-page "${QUALITATIVE_ROWS_PER_PAGE:-4}"
  --zoom-rows-per-page "${QUALITATIVE_ZOOM_ROWS_PER_PAGE:-2}"
  --candidate-pool "${QUALITATIVE_CANDIDATE_POOL:-96}"
  --render-style "${QUALITATIVE_RENDER_STYLE:-overlay}"
  --dpi "${QUALITATIVE_DPI:-260}"
  --main-selection-mode "${QUALITATIVE_MAIN_SELECTION_MODE:-balanced}"
  --post-selection-mode "${QUALITATIVE_POST_SELECTION_MODE:-balanced}"
  --hard-max-base-iou "${QUALITATIVE_HARD_MAX_BASE_IOU:-0.78}"
  --hard-min-final-iou "${QUALITATIVE_HARD_MIN_FINAL_IOU:-0.72}"
  --hard-min-iou-gain "${QUALITATIVE_HARD_MIN_IOU_GAIN:-0.04}"
  --hard-min-improved-models "${QUALITATIVE_HARD_MIN_IMPROVED_MODELS:-2}"
)

for sample_id in ${QUALITATIVE_MAIN_SAMPLE_IDS:-}; do
  command+=(--main-sample-id "${sample_id}")
done
for sample_id in ${QUALITATIVE_POST_SAMPLE_IDS:-}; do
  command+=(--post-sample-id "${sample_id}")
done

"${command[@]}"

export QUALITATIVE_OUTPUT_DIR="${OUTPUT_DIR}"
"${PYTHON_BIN}" - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["QUALITATIVE_OUTPUT_DIR"])
archive = root / "freeref_qualitative_figures.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
    for path in root.rglob("*"):
        if path.is_file() and path != archive:
            handle.write(path, path.relative_to(root).as_posix())
print(archive)
PY

echo "Main-table overlay figures: ${OUTPUT_DIR}/main_table_qualitative*.pdf"
echo "Post-process overlay figures: ${OUTPUT_DIR}/postprocess_qualitative*.pdf"
echo "Main-table binary zoom figures: ${OUTPUT_DIR}/main_table_binary_zoom*.pdf"
echo "Post-process binary zoom figures: ${OUTPUT_DIR}/postprocess_binary_zoom*.pdf"
echo "Bundle: ${OUTPUT_DIR}/freeref_qualitative_figures.zip"
