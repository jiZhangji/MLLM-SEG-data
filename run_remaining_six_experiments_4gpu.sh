#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${REMAINING_SIX_PHASE:-all}"
PREPARE="${PREPARE_REMAINING_RUNTIMES:-0}"
OUTPUT_ROOT="${REMAINING_SIX_RUN_ROOT:-${ROOT}/outputs/remaining_six_4gpu}"
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
ALL_GPUS="${H100_GPUS[0]} ${H100_GPUS[1]} ${H200_GPUS[0]} ${H200_GPUS[1]}"
if [[ "${PREPARE}" == 1 ]]; then
  MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_remaining_six_runtimes.sh" \
    |& tee "${OUTPUT_ROOT}/prepare_runtimes.log"
fi

summarize_lisa_diagnostic() {
  local root="${ROOT}/outputs/universal_freeref_lisa_all8"
  local -a args=()
  local split slug
  for split in \
    refcoco_val refcoco_testA refcoco_testB \
    refcoco+_val refcoco+_testA refcoco+_testB \
    refcocog_val refcocog_test; do
    slug="${split//+/plus}"
    [[ -f "${root}/${slug}/eval_summary.json" ]] || return 0
    args+=(--summary "${split}=${root}/${slug}/eval_summary.json")
  done
  cd "${SCRIPT_DIR}"
  conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
    python -m universal_freeref.summarize_table1b \
      --method LISA-7B-v1-public-checkpoint \
      "${args[@]}" \
      --output-dir "${root}/combined" \
      --eligibility diagnostic_only \
      --expected-baseline 74.9 79.1 72.3 65.1 70.8 58.1 67.9 70.6 \
      --note "Flat-JSON paired diagnostic; the public checkpoint failed the selected paper-row baseline gate (70.40 vs 79.1)."
}

run_ready_phase() {
  echo "Parallel ready-method allocation:"
  echo "  READ full eight splits       -> H200 ${H200_GPUS[0]}"
  echo "  PolyFormer-L full eight      -> H100 ${H100_GPUS[0]}"
  echo "  LISA public paired diagnostic-> H100 ${H100_GPUS[1]}"
  local gsva_model="${GSVA_MLLM_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/gsva/LLaVA-Lightning-7B-v1-1-merged}"
  local gsva_ready=0
  if [[ -f "${gsva_model}/.freeref_merge_complete" && -f "${gsva_model}/config.json" ]] &&
     find "${gsva_model}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) \
       -size +100M -print -quit | grep -q .; then
    gsva_ready=1
    echo "  GSVA full eight splits       -> H200 ${H200_GPUS[1]}"
  else
    echo "  GSVA deferred (authorized merged LLaVA base missing); H200 ${H200_GPUS[1]} remains free"
  fi

  declare -a PIDS=() NAMES=() LOGS=()
  READ_EVAL_CUDA_DEVICES="${H200_GPUS[0]}" \
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_read_freeref_full_eval.sh" \
    >"${OUTPUT_ROOT}/read.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(READ); LOGS+=("${OUTPUT_ROOT}/read.log")
  POLYFORMER_CUDA_DEVICES="${H100_GPUS[0]}" \
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_polyformer_freeref_full_eval.sh" \
    >"${OUTPUT_ROOT}/polyformer.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(PolyFormer); LOGS+=("${OUTPUT_ROOT}/polyformer.log")
  CUDA_DEVICE="${H100_GPUS[1]}" \
    LISA_SETUP_ENV=0 \
    LISA_SPLITS="refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test" \
    LISA_RESULTS_ROOT="${ROOT}/outputs/lisa_official_all8" \
    LISA_FREEREF_ROOT="${ROOT}/outputs/universal_freeref_lisa_all8" \
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_lisa_freeref_eval.sh" \
    >"${OUTPUT_ROOT}/lisa_diagnostic.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(LISA-diagnostic); LOGS+=("${OUTPUT_ROOT}/lisa_diagnostic.log")
  if [[ "${gsva_ready}" == 1 ]]; then
    GSVA_EVAL_CUDA_DEVICES="${H200_GPUS[1]}" \
      MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_gsva_freeref_full_eval.sh" \
      >"${OUTPUT_ROOT}/gsva.log" 2>&1 &
    PIDS+=("$!"); NAMES+=(GSVA); LOGS+=("${OUTPUT_ROOT}/gsva.log")
  fi
  printf '%s\n' "${PIDS[@]}" >"${OUTPUT_ROOT}/ready_phase.pids"

  local failed=0 index
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      echo "DONE ${NAMES[$index]}"
    else
      echo "ERROR ${NAMES[$index]}: ${LOGS[$index]}" >&2
      tail -n 80 "${LOGS[$index]}" >&2 || true
      failed=1
    fi
  done
  summarize_lisa_diagnostic
  return "${failed}"
}

run_rela_train_phase() {
  echo "ReLA classic checkpoints are not public; starting the documented three-model retraining."
  MLLM_SEG_ROOT="${ROOT}" RUN_RELA_EVAL_AFTER_TRAIN=0 \
    bash "${SCRIPT_DIR}/run_rela_classic_training_4gpu.sh" \
    |& tee "${OUTPUT_ROOT}/rela_training.log"
}

run_rela_eval_phase() {
  MLLM_SEG_ROOT="${ROOT}" RELA_EVAL_CUDA_DEVICES="${ALL_GPUS}" \
    bash "${SCRIPT_DIR}/run_rela_freeref_full_eval.sh" \
    |& tee "${OUTPUT_ROOT}/rela_eval.log"
}

case "${PHASE}" in
  ready) run_ready_phase ;;
  rela-train) run_rela_train_phase ;;
  rela-eval) run_rela_eval_phase ;;
  all)
    run_ready_phase
    run_rela_train_phase
    run_rela_eval_phase
    ;;
  *)
    echo "ERROR: REMAINING_SIX_PHASE must be ready, rela-train, rela-eval, or all." >&2
    exit 2
    ;;
esac
echo "UNINEXT-L remains gated by the unavailable official Stage-2 ConvNeXt-L checkpoint."
echo "Remaining-six phase '${PHASE}' completed."
