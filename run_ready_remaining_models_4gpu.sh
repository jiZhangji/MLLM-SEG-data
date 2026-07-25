#!/usr/bin/env bash
set -uo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${READY_REMAINING_OUTPUT_ROOT:-${ROOT}/outputs/freeref_ready_remaining_4gpu}"
RUN_LISA="${RUN_LISA_FULL:-1}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}"

declare -a H100_GPUS=() H200_GPUS=()
while IFS=',' read -r index name; do
  index="${index//[[:space:]]/}"
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

polyformer_gpus="${H200_GPUS[0]} ${H200_GPUS[1]} ${H100_GPUS[0]}"
lisa_gpu="${H100_GPUS[1]}"
echo "GPU assignment:"
echo "  PolyFormer full eight splits: ${polyformer_gpus}"
if [[ "${RUN_LISA}" == 1 ]]; then
  echo "  LISA paired eight splits: ${lisa_gpu}"
else
  echo "  LISA skipped by RUN_LISA_FULL=${RUN_LISA}"
fi
echo "  ReLA/GSVA/READ: blocked until per-sample official-output adapters are available"
echo "  UNINEXT: blocked until the official checkpoint is available"

POLYFORMER_CUDA_DEVICES="${polyformer_gpus}" \
POLYFORMER_FULL_OUTPUT_ROOT="${ROOT}/outputs/polyformer_freeref_full" \
POLYFORMER_FULL_LOG_ROOT="${ROOT}/outputs/polyformer_freeref_full_logs" \
  bash "${SCRIPT_DIR}/run_polyformer_freeref_full_eval.sh" \
  >"${OUTPUT_ROOT}/polyformer.log" 2>&1 &
polyformer_pid="$!"

lisa_pid=""
if [[ "${RUN_LISA}" == 1 ]]; then
  CUDA_DEVICE="${lisa_gpu}" \
  LISA_SETUP_ENV=0 \
  LISA_SPLITS="refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test" \
  LISA_RESULTS_ROOT="${ROOT}/outputs/lisa_official_all8" \
  LISA_FREEREF_ROOT="${ROOT}/outputs/universal_freeref_lisa_all8" \
    bash "${SCRIPT_DIR}/run_lisa_freeref_eval.sh" \
    >"${OUTPUT_ROOT}/lisa.log" 2>&1 &
  lisa_pid="$!"
fi

printf 'polyformer_pid=%s\nlisa_pid=%s\n' "${polyformer_pid}" "${lisa_pid:-not_started}" |
  tee "${OUTPUT_ROOT}/pids.txt"

failed=0
if ! wait "${polyformer_pid}"; then
  echo "PolyFormer failed; inspect ${OUTPUT_ROOT}/polyformer.log" >&2
  failed=1
fi
if [[ -n "${lisa_pid}" ]] && ! wait "${lisa_pid}"; then
  echo "LISA failed; inspect ${OUTPUT_ROOT}/lisa.log" >&2
  failed=1
fi

echo "Ready-method queues stopped."
[[ -f "${ROOT}/outputs/polyformer_freeref_full/combined/comparison.md" ]] && \
  cat "${ROOT}/outputs/polyformer_freeref_full/combined/comparison.md"
[[ -f "${ROOT}/outputs/universal_freeref_lisa_all8/combined/comparison.md" ]] && \
  cat "${ROOT}/outputs/universal_freeref_lisa_all8/combined/comparison.md"
exit "${failed}"
