#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READ_DIR="${READ_DIR:-${ROOT}/code/third_party/read}"
CONDA_ENV="${READ_CONDA_ENV:-read-freeref}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
WEIGHTS_ROOT="${READ_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
MODEL_PATH="${READ_MODEL_PATH:-${WEIGHTS_ROOT}/read/READ-LLaVA-v1.5-7B-for-fprefcoco}"
VISION_TOWER="${READ_VISION_TOWER:-${WEIGHTS_ROOT}/shared/clip-vit-large-patch14-336}"
DATA_ROOT="${READ_DATA_ROOT:-${ROOT}/data/read_eval}"
DATASET="${READ_DATASET:-refcoco}"
SPLIT="${READ_SPLIT:-val}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SLUG="${DATASET//+/plus}_${SPLIT}"
OUTPUT_ROOT="${READ_OUTPUT_ROOT:-${ROOT}/outputs/read_freeref_full/${SLUG}}"
EXPORT_DIR="${OUTPUT_ROOT}/official_export"
FREEREF_DIR="${OUTPUT_ROOT}/freeref"

for required in \
  "${READ_DIR}/model/READ.py" \
  "${MODEL_PATH}/config.json" \
  "${VISION_TOWER}/config.json" \
  "${DATA_ROOT}/refer_seg/${DATASET}/instances.json"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing READ prerequisite: ${required}" >&2; exit 2; }
done
if ! find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) \
  -size +100M -print -quit | grep -q .; then
  echo "ERROR: no full READ model shards found below ${MODEL_PATH}." >&2
  exit 2
fi
mkdir -p "${EXPORT_DIR}" "${FREEREF_DIR}"

echo "READ official teacher-forced validation: ${DATASET}/${SPLIT} GPU=${CUDA_DEVICE}"
cd "${SCRIPT_DIR}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${READ_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  python -m universal_freeref.export_read_masks \
    --read-code-dir "${READ_DIR}" \
    --model-path "${MODEL_PATH}" \
    --vision-tower "${VISION_TOWER}" \
    --dataset-dir "${DATA_ROOT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output-dir "${EXPORT_DIR}" \
    --method "${READ_METHOD_NAME:-READ-LLaVA-v1.5-7B-official}" \
    --precision "${READ_PRECISION:-bf16}" \
    --workers "${READ_WORKERS:-4}"

conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.evaluate \
    --manifest "${EXPORT_DIR}/manifest.jsonl" \
    --output-dir "${FREEREF_DIR}" \
    --save-visualizations "${READ_SAVE_VISUALIZATIONS:-8}"
conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.summarize \
    --summary "READ_${DATASET}_${SPLIT}=${FREEREF_DIR}/eval_summary.json" \
    --output-dir "${OUTPUT_ROOT}/comparison" \
    --title "READ Official SasP/SAM Mask vs. FreeRef"
echo "READ paired evaluation complete: ${OUTPUT_ROOT}/comparison/comparison.md"
