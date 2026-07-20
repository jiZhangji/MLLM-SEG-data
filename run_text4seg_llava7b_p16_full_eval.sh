#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${TEXT4SEG_P16_MODEL_PATH:-${ROOT}/models/Text4Seg/llava-v1.5-7b-p16}"
TEXT4SEG_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
VISION_TOWER="${TEXT4SEG_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
SAM_PATH="${TEXT4SEG_SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SPLITS="${TEXT4SEG_P16_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
BASE_OUTPUT="${TEXT4SEG_P16_BASE_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_paired}"
REFINE_OUTPUT="${TEXT4SEG_P16_REFINE_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_freeref}"
COMBINED_OUTPUT="${TEXT4SEG_P16_COMBINED_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_freeref_comparison}"

if [[ ! -d "${TEXT4SEG_DIR}/llava" ]]; then
  echo "ERROR: official Text4Seg code is missing: ${TEXT4SEG_DIR}" >&2
  echo "Run the separate download/setup step first." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: the official LLaVA-1.5-7B-p16 checkpoint is missing: ${MODEL_PATH}" >&2
  echo "The public p24 demo checkpoint is not a substitute for this paper configuration." >&2
  exit 1
fi
if [[ "${MODEL_PATH,,}" != *p16* ]]; then
  echo "ERROR: TEXT4SEG_P16_MODEL_PATH must identify a p16 checkpoint: ${MODEL_PATH}" >&2
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

mkdir -p "${BASE_OUTPUT}" "${REFINE_OUTPUT}" "${COMBINED_OUTPUT}"
SUMMARY_ARGS=()

for SPLIT in ${SPLITS}; do
  EVAL_JSON="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json"
  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "ERROR: paired evaluation JSON is missing: ${EVAL_JSON}" >&2
    exit 1
  fi
  echo "===== Text4Seg LLaVA-1.5-7B p16: ${SPLIT} ====="
  TEXT4SEG_MODEL_PATH="${MODEL_PATH}" \
  TEXT4SEG_SETUP_MODE=offline \
  TEXT4SEG_DESCRIPTOR_GRID_SIZE=16 \
  TEXT4SEG_VISION_TOWER="${VISION_TOWER}" \
  TEXT4SEG_SAM_PATH="${SAM_PATH}" \
  TEXT4SEG_EVAL_JSON="${EVAL_JSON}" \
  TEXT4SEG_RESULTS_ROOT="${BASE_OUTPUT}/${SPLIT}" \
  TEXT4SEG_REFINE_OUTPUT="${REFINE_OUTPUT}/${SPLIT}" \
  CUDA_DEVICE="${CUDA_DEVICE}" \
    bash "${SCRIPT_DIR}/run_text4seg_training_free_eval.sh"
  SUMMARY_ARGS+=(--summary "${SPLIT}=${REFINE_OUTPUT}/${SPLIT}/eval_summary.json")
done

conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV:-text4seg-tf}" \
  python -m training_free_refine.summarize_splits \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${COMBINED_OUTPUT}" \
  --title "Text4Seg LLaVA-1.5-7B p16 + FreeRef paired evaluation"

echo "Text4Seg p16 paired evaluation completed."
echo "Summary: ${COMBINED_OUTPUT}/combined_summary.md"
echo "NOTE: this runner matches the paper checkpoint/grid family but uses paired flat JSON inputs."
echo "Paper-row claims still require the official REFER-loader baseline reproduction gate."
