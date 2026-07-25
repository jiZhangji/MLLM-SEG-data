#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSVA_DIR="${GSVA_DIR:-${ROOT}/code/third_party/gsva}"
CONDA_ENV="${GSVA_CONDA_ENV:-gsva-freeref}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
WEIGHTS_ROOT="${GSVA_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
MODEL_PATH="${GSVA_MLLM_MODEL_PATH:-${WEIGHTS_ROOT}/gsva/LLaVA-Lightning-7B-v1-1-merged}"
CHECKPOINT="${GSVA_CHECKPOINT:-${WEIGHTS_ROOT}/gsva/official_checkpoints/gsva-7b-ft-res.bin}"
VISION_TOWER="${GSVA_VISION_TOWER:-${WEIGHTS_ROOT}/shared/clip-vit-large-patch14}"
SAM_CHECKPOINT="${GSVA_SAM_CHECKPOINT:-${WEIGHTS_ROOT}/shared/sam_vit_h_4b8939.pth}"
DATA_ROOT="${GSVA_DATA_ROOT:-${ROOT}/data/gsva_eval}"
DATASET="${GSVA_DATASET:-refcoco}"
SPLIT="${GSVA_SPLIT:-val}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PRECISION="${GSVA_PRECISION:-fp32}"
METHOD="${GSVA_METHOD_NAME:-GSVA-7B-ft-res-official}"

case "${DATASET}" in
  refcoco|refcoco+) SPLIT_BY=unc ;;
  refcocog) SPLIT_BY=umd ;;
  *) echo "ERROR: unsupported GSVA dataset: ${DATASET}" >&2; exit 2 ;;
esac
SLUG="${DATASET//+/plus}_${SPLIT}"
OUTPUT_ROOT="${GSVA_OUTPUT_ROOT:-${ROOT}/outputs/gsva_freeref_full/${SLUG}}"
EXPORT_DIR="${OUTPUT_ROOT}/official_export"
FREEREF_DIR="${OUTPUT_ROOT}/freeref"
LOG_DIR="${OUTPUT_ROOT}/official_logs"
for required in \
  "${GSVA_DIR}/main.py" \
  "${MODEL_PATH}/.freeref_merge_complete" \
  "${MODEL_PATH}/config.json" \
  "${CHECKPOINT}" \
  "${VISION_TOWER}/config.json" \
  "${SAM_CHECKPOINT}" \
  "${DATA_ROOT}/refer_seg/${DATASET}/instances.json"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing GSVA prerequisite: ${required}" >&2; exit 2; }
done
if ! find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) \
  -size +100M -print -quit | grep -q .; then
  echo "ERROR: merged GSVA LLaVA model weights are missing below ${MODEL_PATH}." >&2
  exit 2
fi
mkdir -p "${EXPORT_DIR}" "${FREEREF_DIR}" "${LOG_DIR}"

echo "GSVA official inference: ${DATASET}/${SPLIT} GPU=${CUDA_DEVICE}"
cd "${GSVA_DIR}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    FREEREF_GSVA_EXPORT=1 \
    FREEREF_GSVA_EXPORT_DIR="${EXPORT_DIR}" \
    FREEREF_GSVA_METHOD="${METHOD}" \
    FREEREF_GSVA_SPLIT="${DATASET}_${SPLIT}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    DS_SKIP_CUDA_CHECK=1 \
    PYTHONPATH="${SCRIPT_DIR}/universal_freeref/gsva_hook:${GSVA_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  deepspeed --num_gpus 1 main.py \
    --val_dataset="${DATASET}|${SPLIT_BY}|${SPLIT}" \
    --segmentation_model_path="${SAM_CHECKPOINT}" \
    --mllm_model_path="${MODEL_PATH}" \
    --vision-tower="${VISION_TOWER}" \
    --dataset_dir="${DATA_ROOT}" \
    --weight="${CHECKPOINT}" \
    --precision="${PRECISION}" \
    --lora_r=8 \
    --workers="${GSVA_WORKERS:-6}" \
    --log_base_dir="${LOG_DIR}" \
    --exp_name="${SLUG}" \
    --eval_only

[[ -s "${EXPORT_DIR}/manifest.jsonl" ]] || {
  echo "ERROR: GSVA export hook did not produce ${EXPORT_DIR}/manifest.jsonl" >&2
  exit 1
}
cd "${SCRIPT_DIR}"
conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.evaluate \
    --manifest "${EXPORT_DIR}/manifest.jsonl" \
    --output-dir "${FREEREF_DIR}" \
    --save-visualizations "${GSVA_SAVE_VISUALIZATIONS:-8}"
conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.summarize \
    --summary "GSVA_${DATASET}_${SPLIT}=${FREEREF_DIR}/eval_summary.json" \
    --output-dir "${OUTPUT_ROOT}/comparison" \
    --title "GSVA Official SAM Mask vs. FreeRef"
echo "GSVA paired evaluation complete: ${OUTPUT_ROOT}/comparison/comparison.md"
