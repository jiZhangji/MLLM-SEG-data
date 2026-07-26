#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FRAMEWORK_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${FRAMEWORK_PYTHON:-${STAMP_ENV}/bin/python}"
KIND="${FRAMEWORK_KIND:-stamp}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="$(conda run -n STAMP python -c 'import sys; print(sys.executable)')"
  else
    echo "ERROR: STAMP Python was not found at ${PYTHON_BIN}, and conda is unavailable." >&2
    exit 1
  fi
fi

case "${KIND}" in
  stamp)
    DEFAULT_ROWS="${ROOT}/outputs/training_free_refine_stamp7b_refcocog_val_full/eval_rows.csv"
    DEFAULT_LABEL="STAMP-7B"
    ;;
  text4seg)
    DEFAULT_ROWS="${ROOT}/outputs/text4seg_training_free_refcocog_val/eval_rows.csv"
    DEFAULT_LABEL="Text4Seg-7B-p24"
    ;;
  *)
    echo "ERROR: FRAMEWORK_KIND must be stamp or text4seg, got ${KIND}." >&2
    exit 1
    ;;
esac

ROWS="${FRAMEWORK_ROWS:-${DEFAULT_ROWS}}"
LABEL="${FRAMEWORK_LABEL:-${DEFAULT_LABEL}}"
OUTPUT_DIR="${FRAMEWORK_OUTPUT_DIR:-${ROOT}/outputs/framework_figure_real/${KIND}}"
SELECTION="${FRAMEWORK_SELECTION:-representative_success}"
RANK="${FRAMEWORK_SAMPLE_RANK:-1}"
CANDIDATE_POOL="${FRAMEWORK_CANDIDATE_POOL:-16}"
SAMPLE_NAME="${FRAMEWORK_SAMPLE_NAME:-}"
UPLOAD_HF="${FRAMEWORK_UPLOAD_HF:-0}"
HF_REPO_ID="${FRAMEWORK_HF_REPO_ID:-shimiandeshu/MLLM-SEG}"
HF_PATH="${FRAMEWORK_HF_PATH:-paper_assets/framework_runs/${KIND}}"

if [[ ! -f "${ROWS}" ]]; then
  echo "ERROR: evaluation CSV not found: ${ROWS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -c \
  'import matplotlib, numpy, PIL, scipy, skimage, torch; print("Runtime imports: OK")'
if [[ "${UPLOAD_HF}" == "1" ]]; then
  "${PYTHON_BIN}" -c \
    'from huggingface_hub import HfApi; print("Hugging Face account:", HfApi().whoami()["name"])'
fi

selector=(
  "${PYTHON_BIN}"
  paper_assets/framework/select_real_framework_sample.py
  --kind "${KIND}"
  --rows "${ROWS}"
  --output-dir "${OUTPUT_DIR}"
  --label "${LABEL}"
  --selection "${SELECTION}"
  --rank "${RANK}"
  --candidate-pool "${CANDIDATE_POOL}"
)
if [[ -n "${SAMPLE_NAME}" ]]; then
  selector+=(--sample-name "${SAMPLE_NAME}")
fi

echo "[1/2] Selecting and exporting a real ${LABEL} evaluation sample"
"${selector[@]}"

echo "[2/2] Rendering the paper framework figure from real experiment arrays"
"${PYTHON_BIN}" paper_assets/framework/generate_framework_figure.py \
  --sample-bundle "${OUTPUT_DIR}/selected_real_sample.npz" \
  --output-dir "${OUTPUT_DIR}" \
  --stem "freeref_framework_real"

echo
echo "Completed. No AI-generated or synthetic image is used in the paper output."
echo "Candidate sheet: ${OUTPUT_DIR}/framework_candidate_contact_sheet.png"
echo "Selection record: ${OUTPUT_DIR}/selected_real_sample.json"
echo "Paper PDF:       ${OUTPUT_DIR}/freeref_framework_real.pdf"
echo "Paper PNG:       ${OUTPUT_DIR}/freeref_framework_real.png"
echo "Editable SVG:    ${OUTPUT_DIR}/freeref_framework_real.svg"

if [[ "${UPLOAD_HF}" == "1" ]]; then
  echo
  echo "[HF] Uploading viewable outputs to ${HF_REPO_ID}/${HF_PATH}"
  "${PYTHON_BIN}" paper_assets/framework/upload_framework_outputs.py \
    --output-dir "${OUTPUT_DIR}" \
    --repo-id "${HF_REPO_ID}" \
    --path-in-repo "${HF_PATH}"
else
  echo
  echo "HF upload skipped. Set FRAMEWORK_UPLOAD_HF=1 to upload viewable outputs."
fi
