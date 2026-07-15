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
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"

if [[ ! -f "${PARENT_CHECKPOINT}" ]]; then
  echo "Parent OnePass checkpoint not found: ${PARENT_CHECKPOINT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
  echo "Exactly two GPU ids are required; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

wait_for_two_gpus() {
  while true; do
    local all_ready=1
    local status=()
    local gpu line free_mb total_mb utilization free_gb total_gb ready
    for gpu in "${GPU_IDS[@]}"; do
      gpu="$(echo "${gpu}" | xargs)"
      line="$(
        nvidia-smi --id="${gpu}" \
          --query-gpu=memory.free,memory.total,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null || true
      )"
      if [[ -z "${line}" ]]; then
        all_ready=0
        status+=("GPU${gpu}=unavailable")
        continue
      fi
      IFS=',' read -r free_mb total_mb utilization <<< "${line}"
      free_mb="$(echo "${free_mb}" | xargs)"
      total_mb="$(echo "${total_mb}" | xargs)"
      utilization="$(echo "${utilization}" | xargs)"
      free_gb="$(awk -v value="${free_mb}" 'BEGIN {printf "%.1f", value / 1024}')"
      total_gb="$(awk -v value="${total_mb}" 'BEGIN {printf "%.1f", value / 1024}')"
      ready="$(awk -v value="${free_mb}" -v minimum="${MIN_FREE_GPU_GB}" \
        'BEGIN {print (value / 1024 >= minimum) ? 1 : 0}')"
      if [[ "${ready}" -ne 1 ]]; then
        all_ready=0
      fi
      status+=("GPU${gpu}:free=${free_gb}/${total_gb}GB,util=${utilization}%")
    done
    echo "[$(date '+%F %T')] waiting for both GPUs: ${status[*]} required_free=${MIN_FREE_GPU_GB}GB"
    if [[ "${all_ready}" -eq 1 ]]; then
      return 0
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

echo "=== Waiting for two GPUs ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} poll=${GPU_POLL_SECONDS}s"
while true; do
  wait_for_two_gpus
  echo "Both GPUs meet the memory threshold; validating CUDA allocation and NCCL..."
  if python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m onepass_qwen7b.gpu_preflight \
    --min-free-gb "${MIN_FREE_GPU_GB}" \
    --probe-mb 256; then
    echo "Two-GPU preflight passed. Starting training."
    break
  fi
  echo "GPU/NCCL preflight failed or resources changed; retrying in ${GPU_POLL_SECONDS}s."
  sleep "${GPU_POLL_SECONDS}"
done

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
