#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${TEXT4SEG_P24_MODEL_PATH:-lmc22/text4seg-llava-7b-p24}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SPLITS="${TEXT4SEG_P24_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
COMBINED_OUTPUT="${TEXT4SEG_P24_COMBINED_OUTPUT:-${ROOT}/outputs/text4seg_public_p24_freeref_full_comparison}"
VISION_TOWER="${TEXT4SEG_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
SAM_PATH="${TEXT4SEG_SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"

if [[ "${MODEL_PATH,,}" != *p24* ]]; then
  echo "ERROR: TEXT4SEG_P24_MODEL_PATH must identify the public p24 checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ROOT}/code/Text4Seg/llava" ]]; then
  echo "ERROR: Text4Seg code is missing: ${ROOT}/code/Text4Seg" >&2
  exit 1
fi
if [[ ! -d "${VISION_TOWER}" ]]; then
  echo "ERROR: local CLIP vision tower is missing: ${VISION_TOWER}" >&2
  exit 1
fi
if [[ ! -f "${SAM_PATH}" ]]; then
  echo "ERROR: local SAM-H checkpoint is missing: ${SAM_PATH}" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${COMBINED_OUTPUT}"
SUMMARY_ARGS=()

for SPLIT in ${SPLITS}; do
  EVAL_JSON="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json"
  BASE_OUTPUT="${ROOT}/outputs/text4seg_official_${SPLIT}"
  REFINE_OUTPUT="${ROOT}/outputs/text4seg_training_free_${SPLIT}"
  SUMMARY="${REFINE_OUTPUT}/eval_summary.json"

  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "ERROR: paired evaluation JSON is missing: ${EVAL_JSON}" >&2
    exit 1
  fi

  EXPECTED="$(conda run -n "${CONDA_ENV}" python -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "${EVAL_JSON}")"
  COMPLETE=0
  if [[ -f "${SUMMARY}" ]]; then
    COMPLETE="$(conda run -n "${CONDA_ENV}" python -c \
      'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("samples", 0)))' \
      "${SUMMARY}" 2>/dev/null || echo 0)"
  fi

  if [[ "${COMPLETE}" == "${EXPECTED}" ]]; then
    echo "SKIP completed Text4Seg public-p24 ${SPLIT}: ${COMPLETE}/${EXPECTED}"
  else
    echo "===== Text4Seg public-p24 ${SPLIT}: ${COMPLETE}/${EXPECTED} ====="
    TEXT4SEG_MODEL_PATH="${MODEL_PATH}" \
    TEXT4SEG_SETUP_MODE=offline \
    TEXT4SEG_DESCRIPTOR_GRID_SIZE=24 \
    TEXT4SEG_VISION_TOWER="${VISION_TOWER}" \
    TEXT4SEG_SAM_PATH="${SAM_PATH}" \
    TEXT4SEG_EVAL_JSON="${EVAL_JSON}" \
    TEXT4SEG_RESULTS_ROOT="${BASE_OUTPUT}" \
    TEXT4SEG_REFINE_OUTPUT="${REFINE_OUTPUT}" \
    CUDA_DEVICE="${CUDA_DEVICE}" \
      bash "${SCRIPT_DIR}/run_text4seg_training_free_eval.sh"
  fi

  SUMMARY_ARGS+=(--summary "${SPLIT}=${SUMMARY}")
done

conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m training_free_refine.summarize_splits \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${COMBINED_OUTPUT}" \
  --title "Text4Seg public LLaVA-1.5-7B-p24 + FreeRef paired evaluation"

echo "Text4Seg public-p24 full evaluation completed."
echo "Summary: ${COMBINED_OUTPUT}/combined_summary.md"
echo "Protocol: public mixed-data p24 checkpoint with paired flat-JSON evaluation."

