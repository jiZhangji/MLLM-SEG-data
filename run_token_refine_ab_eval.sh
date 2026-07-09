#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/MLLM-SEG-data}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs}"
VAL_DUMPS="${VAL_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_val_full}"
TEST_DUMPS="${TEST_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
TRAIN_BATCH_SIZE_A="${TRAIN_BATCH_SIZE_A:-512}"
TRAIN_BATCH_SIZE_B="${TRAIN_BATCH_SIZE_B:-512}"

cd "${DATA_ROOT}"

A_DIR="${OUT_ROOT}/token_refine_A_frozen_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${TRAIN_BATCH_SIZE_A}"
B_DIR="${OUT_ROOT}/token_refine_B_calibrated_head_residual_h${HIDDEN_SIZE}_e${EPOCHS}_b${TRAIN_BATCH_SIZE_B}"

CUDA_VISIBLE_DEVICES="${GPU_A}" nohup bash -lc "
python -m token_refine.eval_adapter_from_dumps \
  --input-dir '${VAL_DUMPS}' \
  --checkpoint '${A_DIR}/adapter.pt' \
  --output-dir '${A_DIR}_on_val_full' \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --device cuda

python -m token_refine.eval_adapter_from_dumps \
  --input-dir '${TEST_DUMPS}' \
  --checkpoint '${A_DIR}/adapter.pt' \
  --output-dir '${A_DIR}_on_test_full' \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --device cuda
" > "${A_DIR}_eval.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES="${GPU_B}" nohup bash -lc "
python -m token_refine.eval_adapter_from_dumps \
  --input-dir '${VAL_DUMPS}' \
  --checkpoint '${B_DIR}/adapter.pt' \
  --output-dir '${B_DIR}_on_val_full' \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --device cuda

python -m token_refine.eval_adapter_from_dumps \
  --input-dir '${TEST_DUMPS}' \
  --checkpoint '${B_DIR}/adapter.pt' \
  --output-dir '${B_DIR}_on_test_full' \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --device cuda
" > "${B_DIR}_eval.log" 2>&1 &
PID_B=$!

echo "Started scheme A eval on GPU ${GPU_A}: PID ${PID_A}"
echo "  log: ${A_DIR}_eval.log"
echo "Started scheme B eval on GPU ${GPU_B}: PID ${PID_B}"
echo "  log: ${B_DIR}_eval.log"
