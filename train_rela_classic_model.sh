#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELA_DIR="${RELA_DIR:-${ROOT}/code/third_party/rela}"
CONDA_ENV="${RELA_CONDA_ENV:-rela-freeref}"
DATA_ROOT="${RELA_DATA_ROOT:-${ROOT}/data/rela_refer}"
WEIGHTS_ROOT="${RELA_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
BERT_DIR="${RELA_BERT_DIR:-${WEIGHTS_ROOT}/shared/bert-base-uncased}"
SWIN_D2="${RELA_SWIN_D2:-${WEIGHTS_ROOT}/rela/swin_base_patch4_window12_384_22k.pkl}"
MODEL_ROOT="${RELA_CLASSIC_MODEL_ROOT:-${WEIGHTS_ROOT}/rela/classic}"
DATASET="${RELA_TRAIN_DATASET:-refcoco}"
CUDA_DEVICES="${RELA_TRAIN_CUDA_DEVICES:-0}"
GLOBAL_BATCH="${RELA_GLOBAL_BATCH:-24}"
MAX_ITER="${RELA_MAX_ITER:-150000}"
STEPS="${RELA_LR_STEPS:-(110000, 140000)}"
NUM_WORKERS="${RELA_NUM_WORKERS:-8}"
RESUME="${RELA_RESUME:-1}"

case "${DATASET}" in
  refcoco|refcoco+) SPLIT_BY=unc ;;
  refcocog) SPLIT_BY=umd ;;
  *) echo "ERROR: unsupported ReLA training dataset: ${DATASET}" >&2; exit 2 ;;
esac
OUTPUT_DIR="${MODEL_ROOT}/${DATASET//+/plus}"
FINAL_CHECKPOINT="${OUTPUT_DIR}/model_final.pth"
if [[ -s "${FINAL_CHECKPOINT}" ]]; then
  echo "SKIP completed ReLA ${DATASET}: ${FINAL_CHECKPOINT}"
  exit 0
fi
for required in \
  "${RELA_DIR}/train_net.py" \
  "${RELA_DIR}/configs/referring_swin_base.yaml" \
  "${DATA_ROOT}/${DATASET}/instances.json" \
  "${DATA_ROOT}/${DATASET}/refs(${SPLIT_BY}).p" \
  "${BERT_DIR}/config.json" \
  "${SWIN_D2}"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing ReLA prerequisite: ${required}" >&2; exit 2; }
done

IFS=',' read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
(( NUM_GPUS > 0 )) || { echo "ERROR: RELA_TRAIN_CUDA_DEVICES is empty." >&2; exit 2; }
if (( GLOBAL_BATCH % NUM_GPUS != 0 )); then
  echo "ERROR: RELA_GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by ${NUM_GPUS} GPUs." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
RESUME_ARGS=()
if [[ "${RESUME}" == 1 && -f "${OUTPUT_DIR}/last_checkpoint" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "Training ReLA on ${DATASET}/${SPLIT_BY}"
echo "GPUs=${CUDA_DEVICES} global_batch=${GLOBAL_BATCH} max_iter=${MAX_ITER}"
echo "This is local retraining because the authors did not publish classic-RES checkpoints."
cd "${RELA_DIR}"
env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  DETECTRON2_DATASETS="${DATA_ROOT}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${RELA_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus "${NUM_GPUS}" --dist-url auto "${RESUME_ARGS[@]}" \
    MODEL.WEIGHTS "${SWIN_D2}" \
    DATASETS.TRAIN "(\"${DATASET}_${SPLIT_BY}_train\",)" \
    DATASETS.TEST "(\"${DATASET}_${SPLIT_BY}_val\",)" \
    DATASETS.DATASET_NAME "${DATASET}" \
    DATASETS.SPLIT_BY "${SPLIT_BY}" \
    REFERRING.BERT_TYPE "${BERT_DIR}" \
    SOLVER.IMS_PER_BATCH "${GLOBAL_BATCH}" \
    SOLVER.BASE_LR 0.00001 \
    SOLVER.WEIGHT_DECAY 0.01 \
    SOLVER.STEPS "${STEPS}" \
    SOLVER.MAX_ITER "${MAX_ITER}" \
    TEST.EVAL_PERIOD 0 \
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
    OUTPUT_DIR "${OUTPUT_DIR}"

[[ -s "${FINAL_CHECKPOINT}" ]] || {
  echo "ERROR: ReLA training ended without ${FINAL_CHECKPOINT}" >&2
  exit 1
}
echo "ReLA ${DATASET} training complete: ${FINAL_CHECKPOINT}"
