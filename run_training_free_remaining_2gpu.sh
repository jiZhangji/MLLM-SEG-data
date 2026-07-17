#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_7B="${STAMP7B_GPU:-0}"
GPU_2B="${STAMP2B_GPU:-1}"
MIN_FREE_2B_MB="${STAMP2B_MIN_FREE_GPU_MB:-24000}"
POLL_SECONDS="${STAMP_GPU_POLL_SECONDS:-10}"
STAMP2B_BATCH_SIZE="${STAMP2B_BATCH_SIZE:-8}"
STAMP7B_BATCH_SIZE="${STAMP7B_BATCH_SIZE:-1}"
LOG_DIR="${ROOT}/outputs"
LOG_7B="${LOG_DIR}/training_free_stamp7b_refcoco_family_remaining.log"
LOG_2B="${LOG_DIR}/training_free_stamp2b_refcoco_family_full.log"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required." >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( gpu_count < 2 )); then
  echo "ERROR: two visible physical GPUs are required; detected ${gpu_count}." >&2
  exit 1
fi
if [[ "${GPU_7B}" == "${GPU_2B}" ]]; then
  echo "ERROR: STAMP7B_GPU and STAMP2B_GPU must be different." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9'
}

wait_for_gpu() {
  local gpu="$1"
  local required="$2"
  local label="$3"
  while true; do
    local free_mb
    free_mb="$(gpu_free_mb "${gpu}")"
    echo "[$(date '+%F %T')] ${label}: GPU ${gpu} free ${free_mb} MiB; required ${required} MiB."
    if [[ -n "${free_mb}" ]] && (( free_mb >= required )); then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

echo "Inference policy: batched autoregressive generation with per-sample mask export and automatic OOM splitting."
echo "Parallel policy: STAMP-7B on physical GPU ${GPU_7B}; STAMP-2B on physical GPU ${GPU_2B}."

if pgrep -af 'bash run_training_free_stamp7b_refcoco_family_eval.sh' >/dev/null; then
  echo "STAMP-7B RefCOCO-family runner is already active; leaving it untouched."
  echo "Existing 7B log remains the log selected when that job was launched."
else
  echo "Starting/resuming only the missing STAMP-7B RefCOCO+ splits on GPU ${GPU_7B}."
  CUDA_DEVICE="${GPU_7B}" \
  STAMP7B_EVAL_BATCH_SIZE="${STAMP7B_BATCH_SIZE}" \
  STAMP7B_OTHER_SPLITS="refcoco+_val refcoco+_testA refcoco+_testB" \
    nohup bash run_training_free_stamp7b_refcoco_family_eval.sh \
      > "${LOG_7B}" 2>&1 < /dev/null &
  echo "STAMP-7B PID: $!"
fi

if pgrep -af 'bash run_training_free_stamp2b_refcoco_family_eval.sh' >/dev/null; then
  echo "STAMP-2B RefCOCO-family runner is already active; not starting a duplicate."
else
  wait_for_gpu "${GPU_2B}" "${MIN_FREE_2B_MB}" "STAMP-2B launch"
  batch_marker="${ROOT}/outputs/stamp_batch_equivalence/stamp2b_bs${STAMP2B_BATCH_SIZE}/PASSED"
  if (( STAMP2B_BATCH_SIZE > 1 )) && [[ ! -f "${batch_marker}" ]]; then
    echo "Validating STAMP-2B batch size ${STAMP2B_BATCH_SIZE} against batch size 1."
    CUDA_DEVICE="${GPU_2B}" STAMP_BATCH_SIZE="${STAMP2B_BATCH_SIZE}" \
      bash run_stamp_batch_equivalence.sh
  fi
  echo "Starting/resuming STAMP-2B RefCOCO/RefCOCO+ on GPU ${GPU_2B}."
  CUDA_DEVICE="${GPU_2B}" \
  STAMP2B_EVAL_BATCH_SIZE="${STAMP2B_BATCH_SIZE}" \
    nohup bash run_training_free_stamp2b_refcoco_family_eval.sh \
      > "${LOG_2B}" 2>&1 < /dev/null &
  echo "STAMP-2B PID: $!"
fi

echo "Both model scales now run independently. This launcher can exit safely."
echo "Status: bash check_training_free_remaining_2gpu_status.sh"
