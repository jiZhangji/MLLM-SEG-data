#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${REPO_DIR}/.." && pwd)"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-${PROJECT_ROOT}/outputs/onepass7b_stamp_lora_warmstart_e2/onepass_qwen7b.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/onepass7b_stamp_lora_seg_grounding_e2}"
MIN_FREE_GPU_GB="${MIN_FREE_GPU_GB:-100}"

if [[ ! -f "${PARENT_CHECKPOINT}" ]]; then
  echo "Parent OnePass checkpoint not found: ${PARENT_CHECKPOINT}" >&2
  exit 1
fi

echo "=== Two-GPU preflight ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m onepass_qwen7b.gpu_preflight \
  --min-free-gb "${MIN_FREE_GPU_GB}" \
  --probe-mb 256

echo "=== SEG-grounding fine-tuning ==="
echo "parent=${PARENT_CHECKPOINT}"
echo "output=${OUTPUT_DIR}"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m onepass_qwen7b.train \
  --stamp-code-dir "${PROJECT_ROOT}/code/STAMP" \
  --base-model "${PROJECT_ROOT}/models/Qwen2-VL-7B-Instruct" \
  --init-onepass-checkpoint "${PARENT_CHECKPOINT}" \
  --train-json "${PROJECT_ROOT}/code/STAMP/playground/data/json_files_baseline/refcocog_formatted_all_sentences_doubled_mp.json" \
  --data-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 2 \
  --batch-size 6 \
  --gradient-accumulation 4 \
  --num-workers 12 \
  --query-learning-rate 1e-5 \
  --head-learning-rate 1e-5 \
  --lora-learning-rate 5e-6 \
  --seg-learning-rate 3e-5 \
  --use-seg-grounding \
  --seg-grounding-size 256 \
  --seg-grounding-temperature 0.1 \
  --seg-grounding-loss-weight 0.1 \
  --use-seg-fusion \
  --bce-weight 0.3 \
  --dice-weight 0.7 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --min-pixels 802816 \
  --max-pixels 1003520 \
  --attn-implementation sdpa \
  --no-gradient-checkpointing \
  --logging-steps 1 \
  --save-steps 500
