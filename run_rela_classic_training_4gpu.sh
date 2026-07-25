#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${RELA_TRAIN_LOG_ROOT:-${ROOT}/outputs/rela_classic_training_logs}"
RUN_EVAL_AFTER_TRAIN="${RUN_RELA_EVAL_AFTER_TRAIN:-1}"
mkdir -p "${LOG_ROOT}"

mapfile -t H100_GPUS < <(
  nvidia-smi --query-gpu=index,name --format=csv,noheader |
    awk -F, 'tolower($2) ~ /h100/ {gsub(/ /, "", $1); print $1}'
)
mapfile -t H200_GPUS < <(
  nvidia-smi --query-gpu=index,name --format=csv,noheader |
    awk -F, 'tolower($2) ~ /h200/ {gsub(/ /, "", $1); print $1}'
)
if (( ${#H100_GPUS[@]} < 2 || ${#H200_GPUS[@]} < 2 )); then
  echo "ERROR: expected at least two H100 and two H200 GPUs." >&2
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2
  exit 2
fi

REFCOCO_DEVICES="${RELA_REFCOCO_TRAIN_DEVICES:-${H200_GPUS[0]}}"
REFCOCOPLUS_DEVICES="${RELA_REFCOCOPLUS_TRAIN_DEVICES:-${H100_GPUS[0]},${H100_GPUS[1]}}"
REFCOCOG_DEVICES="${RELA_REFCOCOG_TRAIN_DEVICES:-${H200_GPUS[1]}}"

echo "ReLA parallel classic-RES training allocation:"
echo "  refcoco  -> ${REFCOCO_DEVICES}"
echo "  refcoco+ -> ${REFCOCOPLUS_DEVICES}"
echo "  refcocog -> ${REFCOCOG_DEVICES}"
echo "The two H200 jobs use one large-memory GPU each; RefCOCO+ uses two H100s."

declare -a PIDS=() NAMES=() LOGS=()
launch() {
  local dataset="$1" devices="$2" slug="${1//+/plus}" log
  log="${LOG_ROOT}/${slug}.log"
  env \
    MLLM_SEG_ROOT="${ROOT}" \
    RELA_TRAIN_DATASET="${dataset}" \
    RELA_TRAIN_CUDA_DEVICES="${devices}" \
    bash "${SCRIPT_DIR}/train_rela_classic_model.sh" >"${log}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${dataset}")
  LOGS+=("${log}")
  echo "START ${dataset}: PID=$! log=${log}"
}
launch refcoco "${REFCOCO_DEVICES}"
launch refcoco+ "${REFCOCOPLUS_DEVICES}"
launch refcocog "${REFCOCOG_DEVICES}"

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "DONE ReLA ${NAMES[$index]}"
  else
    echo "ERROR ReLA ${NAMES[$index]}: ${LOGS[$index]}" >&2
    tail -n 80 "${LOGS[$index]}" >&2 || true
    failed=1
  fi
done
(( failed == 0 )) || exit 1

if [[ "${RUN_EVAL_AFTER_TRAIN}" == 1 ]]; then
  MLLM_SEG_ROOT="${ROOT}" \
  RELA_EVAL_CUDA_DEVICES="${H100_GPUS[0]} ${H100_GPUS[1]} ${H200_GPUS[0]} ${H200_GPUS[1]}" \
    bash "${SCRIPT_DIR}/run_rela_freeref_full_eval.sh"
fi
