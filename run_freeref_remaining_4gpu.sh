#!/usr/bin/env bash
set -uo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${ROOT}/outputs"
SAM_PATH="${SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
TEXT4SEG_JOBS="${TEXT4SEG_H200_PARALLEL_JOBS:-5}"
STAMP_SAMH_JOBS="${STAMP_H200_PARALLEL_JOBS:-6}"
LISA_JOBS_PER_H100="${LISA_H100_PARALLEL_JOBS:-2}"
LISA_H100_START_FREE_MB="${LISA_H100_START_FREE_MB:-70000}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${SAM_PATH}" ]]; then
  echo "ERROR: SAM-H checkpoint is missing: ${SAM_PATH}" >&2
  exit 2
fi

declare -a H100_GPUS=()
declare -a H200_GPUS=()
while IFS=',' read -r index name; do
  index="${index//[[:space:]]/}"
  name="${name#${name%%[![:space:]]*}}"
  case "${name}" in
    *H200*) H200_GPUS+=("${index}") ;;
    *H100*) H100_GPUS+=("${index}") ;;
  esac
done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)

if (( ${#H100_GPUS[@]} < 2 || ${#H200_GPUS[@]} < 2 )); then
  echo "ERROR: expected at least two H100 and two H200 GPUs." >&2
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2
  exit 2
fi

H200_TEXT4SEG="${H200_GPUS[0]}"
H200_STAMP="${H200_GPUS[1]}"
H100_LISA_A="${H100_GPUS[0]}"
H100_LISA_B="${H100_GPUS[1]}"

echo "4-GPU assignment:"
echo "  GPU ${H200_TEXT4SEG} (H200): Text4Seg-p24, ${TEXT4SEG_JOBS} splits in parallel"
echo "  GPU ${H200_STAMP} (H200): STAMP official SAM-H, ${STAMP_SAMH_JOBS} splits in parallel"
echo "  GPU ${H100_LISA_A} (H100): LISA queue A, ${LISA_JOBS_PER_H100} splits in parallel"
echo "  GPU ${H100_LISA_B} (H100): LISA queue B, ${LISA_JOBS_PER_H100} splits in parallel"

LISA_DATA_ROOT="${LISA_PAPER_DATA_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
LISA_PAPER_DATA_ROOT="${LISA_DATA_ROOT}" bash "${SCRIPT_DIR}/prepare_lisa_paper_data.sh"

run_text4seg() {
  CUDA_DEVICE="${H200_TEXT4SEG}" \
  TEXT4SEG_P24_PARALLEL_JOBS="${TEXT4SEG_JOBS}" \
  TEXT4SEG_P24_MIN_FREE_MB="${TEXT4SEG_H200_MIN_FREE_MB:-100000}" \
  SAM_PATH="${SAM_PATH}" \
    bash "${SCRIPT_DIR}/run_text4seg_public_p24_full_eval.sh"
}

run_stamp_samh() {
  CUDA_DEVICE="${H200_STAMP}" \
  SAMH_PARALLEL_JOBS="${STAMP_SAMH_JOBS}" \
  SAMH_MIN_FREE_MB="${STAMP_H200_MIN_FREE_MB:-48000}" \
  SAM_PATH="${SAM_PATH}" \
    bash "${SCRIPT_DIR}/run_frozen_samh_full_eval.sh"
}

wait_for_gpu() {
  local gpu="$1"
  local required="$2"
  local free_mb
  while true; do
    free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${free_mb}" ]] && (( free_mb >= required )); then
      return
    fi
    echo "GPU ${gpu} free: ${free_mb:-unknown} MiB; waiting 10 seconds for ${required} MiB..."
    sleep 10
  done
}

run_lisa_split() {
  local gpu="$1"
  local split="$2"
  local slug="${split//|/_}"
  echo "RUN LISA ${split} on GPU ${gpu}"
  CUDA_DEVICE="${gpu}" \
  MIN_FREE_MB=0 \
  LISA_PAPER_DATA_ROOT="${LISA_DATA_ROOT}" \
  LISA_PAPER_SPLITS="${split}" \
    bash "${SCRIPT_DIR}/run_lisa_paper_freeref_eval.sh" \
      >"${OUTPUT_ROOT}/lisa_paper_freeref_${slug}.log" 2>&1
}

run_lisa_queue() {
  local gpu="$1"
  shift
  local split pid index failed
  local -a pids=()
  local -a names=()
  failed=0
  for split in "$@"; do
    run_lisa_split "${gpu}" "${split}" &
    pids+=("$!")
    names+=("${split}")
    if (( ${#pids[@]} >= LISA_JOBS_PER_H100 )); then
      for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        if wait "${pid}"; then
          echo "DONE LISA ${names[$index]} on GPU ${gpu}"
        else
          echo "LISA ${names[$index]} stopped at the paper-match gate or failed; inspect its split log." >&2
          failed=1
        fi
      done
      pids=()
      names=()
      wait_for_gpu "${gpu}" "${LISA_H100_START_FREE_MB}"
    fi
  done
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    if wait "${pid}"; then
      echo "DONE LISA ${names[$index]} on GPU ${gpu}"
    else
      echo "LISA ${names[$index]} stopped at the paper-match gate or failed; inspect its split log." >&2
      failed=1
    fi
  done
  return "${failed}"
}

run_text4seg >"${OUTPUT_ROOT}/freeref_4gpu_text4seg.log" 2>&1 &
TEXT4SEG_PID="$!"
run_stamp_samh >"${OUTPUT_ROOT}/freeref_4gpu_stamp_samh.log" 2>&1 &
STAMP_PID="$!"

(
  wait_for_gpu "${H100_LISA_A}" "${LISA_H100_START_FREE_MB}"
  run_lisa_queue "${H100_LISA_A}" \
    'refcoco|unc|val' 'refcoco|unc|testA' \
    'refcoco|unc|testB' 'refcoco+|unc|val'
) >"${OUTPUT_ROOT}/freeref_4gpu_lisa_h100_0.log" 2>&1 &
LISA_A_PID="$!"

(
  wait_for_gpu "${H100_LISA_B}" "${LISA_H100_START_FREE_MB}"
  run_lisa_queue "${H100_LISA_B}" \
    'refcoco+|unc|testA' 'refcoco+|unc|testB' \
    'refcocog|umd|val' 'refcocog|umd|test'
) >"${OUTPUT_ROOT}/freeref_4gpu_lisa_h100_1.log" 2>&1 &
LISA_B_PID="$!"

echo "Launchers: Text4Seg=${TEXT4SEG_PID}, STAMP-SAMH=${STAMP_PID}, LISA-A=${LISA_A_PID}, LISA-B=${LISA_B_PID}"
FAILED=0
for pid in "${TEXT4SEG_PID}" "${STAMP_PID}" "${LISA_A_PID}" "${LISA_B_PID}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done

echo "All four GPU queues have stopped. Status follows:"
bash "${SCRIPT_DIR}/check_freeref_all_status.sh"
exit "${FAILED}"
