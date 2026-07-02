#!/usr/bin/env bash
set -euo pipefail

# Single-GPU smoke training for official STAMP baseline on 48GB GPU.
# It intentionally avoids upstream launch_all_*.sh because those scripts are
# multi-node templates and reinstall requirements.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
MODEL_ROOT="${MODEL_ROOT:-${MLLM_SEG_ROOT}/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MLLM_SEG_ROOT}/outputs}"

cd "${STAMP_CODE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

export STAMP_ATTN_IMPL="${STAMP_ATTN_IMPL:-sdpa}"
export STAMP_JSON_DIR="${STAMP_JSON_DIR:-${STAMP_CODE_DIR}/playground/data/json_files_baseline}"
export STAMP_USE_STRUCTURED_PRIOR=0
export STAMP_MAX_SAMPLES="${STAMP_MAX_SAMPLES:-1000}"
export STAMP_REPEAT_NON_LLAVA="${STAMP_REPEAT_NON_LLAVA:-1}"

export STAMP_BATCH_SIZE="${STAMP_BATCH_SIZE:-1}"
export STAMP_GRAD_ACCUM="${STAMP_GRAD_ACCUM:-8}"
export STAMP_NUM_EPOCHS="${STAMP_NUM_EPOCHS:-1}"
export STAMP_LORA_R="${STAMP_LORA_R:-16}"
export STAMP_LORA_ALPHA="${STAMP_LORA_ALPHA:-32}"
export STAMP_LR="${STAMP_LR:-3e-5}"
export STAMP_MAX_LENGTH="${STAMP_MAX_LENGTH:-2048}"
export STAMP_SAVE_STEPS="${STAMP_SAVE_STEPS:-200}"
export STAMP_REPORT_TO="${STAMP_REPORT_TO:-none}"

MODEL_NAME="${MODEL_NAME:-${MODEL_ROOT}/STAMP-2B-uni}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/smoke_baseline_1x48g}"

mkdir -p "${OUT_DIR}"

echo "Running STAMP baseline smoke training"
echo "MODEL_NAME=${MODEL_NAME}"
echo "OUT_DIR=${OUT_DIR}"
echo "STAMP_JSON_DIR=${STAMP_JSON_DIR}"
echo "STAMP_MAX_SAMPLES=${STAMP_MAX_SAMPLES}"

python -m train.main_uni \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_DIR}"

