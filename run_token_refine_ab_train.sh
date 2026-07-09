#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/MLLM-SEG-data}"
TRAIN_DUMPS="${TRAIN_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_train_0}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
BATCH_SIZE_A="${BATCH_SIZE_A:-512}"
BATCH_SIZE_B="${BATCH_SIZE_B:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LR="${LR:-3e-4}"
DELTA_REG_WEIGHT="${DELTA_REG_WEIGHT:-0.01}"

cd "${DATA_ROOT}"

mkdir -p "${OUT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_A}" nohup bash -lc "
python -m token_refine.train_adapter_from_dumps \
  --input-dir '${TRAIN_DUMPS}' \
  --output-dir '${OUT_ROOT}/token_refine_A_frozen_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${BATCH_SIZE_A}' \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE_A} \
  --num-workers ${NUM_WORKERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --learning-rate ${LR} \
  --delta-reg-weight ${DELTA_REG_WEIGHT} \
  --device cuda \
  --no-use-uncertainty-gate \
  --no-trainable-logit-calibration
" > "${OUT_ROOT}/token_refine_A_frozen_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${BATCH_SIZE_A}.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES="${GPU_B}" nohup bash -lc "
python -m token_refine.train_adapter_from_dumps \
  --input-dir '${TRAIN_DUMPS}' \
  --output-dir '${OUT_ROOT}/token_refine_B_calibrated_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${BATCH_SIZE_B}' \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE_B} \
  --num-workers ${NUM_WORKERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --learning-rate ${LR} \
  --delta-reg-weight ${DELTA_REG_WEIGHT} \
  --device cuda \
  --no-use-uncertainty-gate \
  --trainable-logit-calibration
" > "${OUT_ROOT}/token_refine_B_calibrated_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${BATCH_SIZE_B}.log" 2>&1 &
PID_B=$!

echo "Started scheme A on GPU ${GPU_A}: PID ${PID_A}"
echo "  log: ${OUT_ROOT}/token_refine_A_frozen_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${BATCH_SIZE_A}.log"
echo "Started scheme B on GPU ${GPU_B}: PID ${PID_B}"
echo "  log: ${OUT_ROOT}/token_refine_B_calibrated_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${BATCH_SIZE_B}.log"
