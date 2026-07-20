#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FREEREF_REPO:-${SCRIPT_DIR}}"
LISA_DIR="${LISA_DIR:-${ROOT}/code/third_party/lisa}"
CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"
WEIGHTS_ROOT="${FREEREF_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
MODEL_PATH="${LISA_MODEL_PATH:-${WEIGHTS_ROOT}/lisa/LISA-7B-v1}"
VISION_TOWER="${LISA_VISION_TOWER:-${WEIGHTS_ROOT}/shared/clip-vit-large-patch14}"
SAM_PATH="${LISA_SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
OUTPUT_ROOT="${LISA_PAPER_OUTPUT_ROOT:-${ROOT}/outputs/lisa_paper_reproduction}"
SPLITS="${LISA_PAPER_SPLITS:-refcoco|unc|testA}"
PAPER_ROW="${LISA_PAPER_ROW:-finetuned_referseg}"
PRECISION="${LISA_PRECISION:-bf16}"
WORKERS="${LISA_WORKERS:-4}"
LIMIT_IMAGES="${LISA_PAPER_LIMIT_IMAGES:-0}"
OFFSET_IMAGES="${LISA_PAPER_OFFSET_IMAGES:-0}"
TOLERANCE="${LISA_PAPER_TOLERANCE_POINTS:-0.5}"
REQUIRE_MATCH="${LISA_REQUIRE_PAPER_MATCH:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ -z "${LISA_PAPER_DATA_ROOT:-}" ]]; then
  refs_path="$(find "${ROOT}" -maxdepth 9 -type f -path '*/refcoco/refs(unc).p' -print -quit 2>/dev/null || true)"
  if [[ -n "${refs_path}" ]]; then
    LISA_PAPER_DATA_ROOT="$(dirname "$(dirname "${refs_path}")")"
  else
    echo "ERROR: official refs(unc).p was not found below ${ROOT}." >&2
    echo "Set LISA_PAPER_DATA_ROOT to the directory containing refcoco/, refcoco+/, refcocog/, and images/." >&2
    exit 2
  fi
fi
DATA_ROOT="$(cd "${LISA_PAPER_DATA_ROOT}" && pwd)"

for required in \
  "${LISA_DIR}/model/LISA.py" \
  "${MODEL_PATH}/config.json" \
  "${VISION_TOWER}/config.json" \
  "${SAM_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required paper-reproduction input is missing: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
echo "LISA paper reproduction only; FreeRef is intentionally disabled."
echo "checkpoint=${MODEL_PATH}"
echo "official_data=${DATA_ROOT}"
echo "paper_row=${PAPER_ROW}"
echo "splits=${SPLITS}"

for split_spec in ${SPLITS}; do
  slug="${split_spec//|/_}"
  output_dir="${OUTPUT_ROOT}/${slug}"
  require_args=()
  if [[ "${REQUIRE_MATCH}" == "1" ]]; then
    require_args+=(--require-paper-match)
  fi
  echo "===== Reproduce ${split_spec} ====="
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.eval_lisa_paper_protocol \
      --lisa-code-dir "${LISA_DIR}" \
      --model-path "${MODEL_PATH}" \
      --vision-tower "${VISION_TOWER}" \
      --sam-path "${SAM_PATH}" \
      --dataset-dir "${DATA_ROOT}" \
      --val-dataset "${split_spec}" \
      --output-dir "${output_dir}" \
      --paper-row "${PAPER_ROW}" \
      --precision "${PRECISION}" \
      --workers "${WORKERS}" \
      --limit-images "${LIMIT_IMAGES}" \
      --offset-images "${OFFSET_IMAGES}" \
      --tolerance-points "${TOLERANCE}" \
      "${require_args[@]}"
done

echo "LISA paper-protocol reproduction completed."
echo "Results: ${OUTPUT_ROOT}"
