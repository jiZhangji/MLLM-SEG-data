#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELA_DIR="${RELA_DIR:-${ROOT}/code/third_party/rela}"
CONDA_ENV="${RELA_CONDA_ENV:-rela-freeref}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
DATA_ROOT="${RELA_DATA_ROOT:-${ROOT}/data/rela_refer}"
IMAGE_ROOT="${RELA_IMAGE_ROOT:-${DATA_ROOT}/images/train2014}"
WEIGHTS_ROOT="${RELA_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
BERT_DIR="${RELA_BERT_DIR:-${WEIGHTS_ROOT}/shared/bert-base-uncased}"
MODEL_ROOT="${RELA_CLASSIC_MODEL_ROOT:-${WEIGHTS_ROOT}/rela/classic}"
DATASET="${RELA_DATASET:-refcoco}"
SPLIT="${RELA_SPLIT:-val}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
METHOD="${RELA_METHOD_NAME:-ReLA-Swin-B-local-classic-retrain}"

case "${DATASET}" in
  refcoco|refcoco+) SPLIT_BY=unc ;;
  refcocog) SPLIT_BY=umd ;;
  *) echo "ERROR: unsupported ReLA dataset: ${DATASET}" >&2; exit 2 ;;
esac
case "${DATASET}" in
  refcoco) CHECKPOINT="${RELA_REFCOCO_CHECKPOINT:-${MODEL_ROOT}/refcoco/model_final.pth}" ;;
  refcoco+) CHECKPOINT="${RELA_REFCOCOPLUS_CHECKPOINT:-${MODEL_ROOT}/refcocoplus/model_final.pth}" ;;
  refcocog) CHECKPOINT="${RELA_REFCOCOG_CHECKPOINT:-${MODEL_ROOT}/refcocog/model_final.pth}" ;;
esac
CHECKPOINT="${RELA_CHECKPOINT:-${CHECKPOINT}}"
SLUG="${DATASET//+/plus}_${SPLIT}"
OUTPUT_ROOT="${RELA_OUTPUT_ROOT:-${ROOT}/outputs/rela_freeref_full/${SLUG}}"
OFFICIAL_DIR="${OUTPUT_ROOT}/official"
IMPORT_DIR="${OUTPUT_ROOT}/imported"
FREEREF_DIR="${OUTPUT_ROOT}/freeref"
PREDICTIONS="${OFFICIAL_DIR}/inference/ref_seg_predictions.pth"

for required in \
  "${RELA_DIR}/train_net.py" \
  "${CHECKPOINT}" \
  "${DATA_ROOT}/${DATASET}/instances.json" \
  "${DATA_ROOT}/${DATASET}/refs(${SPLIT_BY}).p" \
  "${BERT_DIR}/config.json"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing ReLA prerequisite: ${required}" >&2; exit 2; }
done
[[ -d "${IMAGE_ROOT}" ]] || { echo "ERROR: missing ReLA images: ${IMAGE_ROOT}" >&2; exit 2; }
mkdir -p "${OFFICIAL_DIR}" "${IMPORT_DIR}" "${FREEREF_DIR}"

echo "ReLA official inference: ${DATASET}/${SPLIT} GPU=${CUDA_DEVICE}"
cd "${RELA_DIR}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    DETECTRON2_DATASETS="${DATA_ROOT}" \
    FREEREF_RELA_SAVE_PREDICTIONS=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${SCRIPT_DIR}/universal_freeref/rela_hook:${RELA_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 1 --dist-url auto --eval-only \
    MODEL.WEIGHTS "${CHECKPOINT}" \
    DATASETS.TEST "(\"${DATASET}_${SPLIT_BY}_${SPLIT}\",)" \
    DATASETS.DATASET_NAME "${DATASET}" \
    DATASETS.SPLIT_BY "${SPLIT_BY}" \
    REFERRING.BERT_TYPE "${BERT_DIR}" \
    DATALOADER.NUM_WORKERS "${RELA_NUM_WORKERS:-8}" \
    OUTPUT_DIR "${OFFICIAL_DIR}"

[[ -s "${PREDICTIONS}" ]] || {
  echo "ERROR: ReLA save hook did not produce ${PREDICTIONS}" >&2
  exit 1
}
cd "${SCRIPT_DIR}"
conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.import_rela_outputs \
    --input-pth "${PREDICTIONS}" \
    --output-dir "${IMPORT_DIR}" \
    --image-root "${IMAGE_ROOT}" \
    --split "${DATASET}_${SPLIT}" \
    --method "${METHOD}"
conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.evaluate \
    --manifest "${IMPORT_DIR}/manifest.jsonl" \
    --output-dir "${FREEREF_DIR}" \
    --save-visualizations "${RELA_SAVE_VISUALIZATIONS:-8}"
conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.summarize \
    --summary "ReLA_${DATASET}_${SPLIT}=${FREEREF_DIR}/eval_summary.json" \
    --output-dir "${OUTPUT_ROOT}/comparison" \
    --title "ReLA Original Mask vs. FreeRef"
echo "ReLA paired evaluation complete: ${OUTPUT_ROOT}/comparison/comparison.md"
