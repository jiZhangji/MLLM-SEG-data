#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${INTRO_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${INTRO_PYTHON:-${STAMP_ENV}/bin/python}"
FINAL_ROOT="${FINAL_OUTPUT_ROOT:-${ROOT}/outputs/freeref_final_h100_overnight_v2}"
PIXELLM_ROOT="${PIXELLM_OUTPUT_ROOT:-${ROOT}/outputs/pixellm_public_freeref}"
SPLIT="${INTRO_SPLIT:-refcoco_val}"
SLUG="${SPLIT//+/plus}"
OUTPUT_DIR="${INTRO_OUTPUT_DIR:-${ROOT}/outputs/freeref_intro_motivation}"

first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: STAMP Python is unavailable: ${PYTHON_BIN}" >&2
  exit 1
fi

STAMP_ROWS="${INTRO_STAMP_ROWS:-}"
if [[ -z "${STAMP_ROWS}" ]]; then
  STAMP_ROWS="$(first_existing \
    "${FINAL_ROOT}/studies/stamp7b/graph_lambda/lambda_1/eval_rows.csv" \
    "${FINAL_ROOT}/studies/stamp7b/n_segments/k_1024/eval_rows.csv" \
    "${ROOT}/outputs/training_free_refine_stamp7b_${SPLIT}_full/eval_rows.csv" \
  )" || {
    echo "ERROR: could not locate a complete STAMP-7B ${SPLIT} eval_rows.csv." >&2
    exit 1
  }
fi

TEXT4SEG_ROWS="${INTRO_TEXT4SEG_ROWS:-}"
if [[ -z "${TEXT4SEG_ROWS}" ]]; then
  TEXT4SEG_ROWS="$(first_existing \
    "${ROOT}/outputs/text4seg_training_free_${SPLIT}/eval_rows.csv" \
    "${ROOT}/outputs/text4seg_training_free_${SLUG}/eval_rows.csv" \
    "${FINAL_ROOT}/postprocess/text4seg_p24/${SLUG}/freeref/eval_rows.csv" \
  )" || {
    echo "ERROR: could not locate complete Text4Seg-p24 ${SPLIT} eval_rows.csv." >&2
    exit 1
  }
fi

PIXELLM_ROWS="${INTRO_PIXELLM_ROWS:-}"
PIXELLM_MANIFEST="${INTRO_PIXELLM_MANIFEST:-}"
if [[ -z "${PIXELLM_ROWS}" || -z "${PIXELLM_MANIFEST}" ]]; then
  for root in \
    "${PIXELLM_ROOT}" \
    "${ROOT}/outputs/pixellm_public_freeref_full" \
    "${ROOT}/outputs/pixellm_public_freeref"; do
    rows="${root}/${SLUG}/freeref/eval_rows.csv"
    manifest="${root}/${SLUG}/official/manifest.jsonl"
    if [[ -f "${rows}" && -f "${manifest}" ]]; then
      PIXELLM_ROWS="${PIXELLM_ROWS:-${rows}}"
      PIXELLM_MANIFEST="${PIXELLM_MANIFEST:-${manifest}}"
      break
    fi
  done
fi

for required in \
  "${STAMP_ROWS}" \
  "${TEXT4SEG_ROWS}" \
  "${PIXELLM_ROWS}" \
  "${PIXELLM_MANIFEST}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    echo "ERROR: required paired evaluation input is missing: ${required:-<empty>}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "STAMP rows:      ${STAMP_ROWS}"
echo "Text4Seg rows:   ${TEXT4SEG_ROWS}"
echo "PixelLM rows:    ${PIXELLM_ROWS}"
echo "PixelLM manifest:${PIXELLM_MANIFEST}"
echo "Output:          ${OUTPUT_DIR}"

command=(
  "${PYTHON_BIN}"
  -m paper_assets.intro_figure.generate_intro_motivation_figure
  --stamp-rows "${STAMP_ROWS}"
  --text4seg-rows "${TEXT4SEG_ROWS}"
  --pixellm-rows "${PIXELLM_ROWS}"
  --pixellm-manifest "${PIXELLM_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --sample-rank "${INTRO_SAMPLE_RANK:-1}"
  --candidate-pool "${INTRO_CANDIDATE_POOL:-48}"
  --contact-sheet-count "${INTRO_CONTACT_SHEET_COUNT:-8}"
  --gallery-count "${INTRO_GALLERY_COUNT:-6}"
  --package-output
  --minimum-box-iou "${INTRO_MINIMUM_BOX_IOU:-0.50}"
  --n-segments "${INTRO_N_SEGMENTS:-1024}"
  --dpi "${INTRO_DPI:-300}"
)
if [[ -n "${INTRO_SAMPLE_ID:-}" ]]; then
  command+=(--sample-id "${INTRO_SAMPLE_ID}")
fi

"${command[@]}"

echo
echo "Introduction figure completed."
echo "Preview:  ${OUTPUT_DIR}/freeref_intro_motivation.png"
echo "Paper PDF:${OUTPUT_DIR}/freeref_intro_motivation.pdf"
echo "SVG:      ${OUTPUT_DIR}/freeref_intro_motivation.svg"
echo "Candidates:${OUTPUT_DIR}/intro_candidate_contact_sheet.png"
echo "Record:   ${OUTPUT_DIR}/intro_figure_manifest.json"
echo "Complete figures: ${OUTPUT_DIR}/intro_complete_figures.zip"
echo "Individual panels:${OUTPUT_DIR}/intro_individual_panels.zip"
echo "All materials:    ${OUTPUT_DIR}/freeref_intro_figure_bundle.zip"
