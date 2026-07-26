#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${REMAINING_SIX_PHASE:-ready}"
ROLE="${REMAINING_SIX_INSTANCE_ROLE:-auto}"
PREPARE="${PREPARE_REMAINING_RUNTIMES:-0}"
mapfile -t H100_GPUS < <(
  nvidia-smi --query-gpu=index,name --format=csv,noheader |
    awk -F, 'tolower($2) ~ /h100/ {gsub(/ /, "", $1); print $1}'
)
mapfile -t H200_GPUS < <(
  nvidia-smi --query-gpu=index,name --format=csv,noheader |
    awk -F, 'tolower($2) ~ /h200/ {gsub(/ /, "", $1); print $1}'
)
if [[ "${ROLE}" == auto ]]; then
  if (( ${#H100_GPUS[@]} >= 2 && ${#H200_GPUS[@]} == 0 )); then
    ROLE=h100
  elif (( ${#H200_GPUS[@]} >= 2 && ${#H100_GPUS[@]} == 0 )); then
    ROLE=h200
  else
    echo "ERROR: cannot infer instance role; expected a two-H100 or two-H200 instance." >&2
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2
    exit 2
  fi
fi
case "${ROLE}" in
  h100)
    GPU0="${H100_GPUS[0]:-}"
    GPU1="${H100_GPUS[1]:-}"
    ;;
  h200)
    GPU0="${H200_GPUS[0]:-}"
    GPU1="${H200_GPUS[1]:-}"
    ;;
  *)
    echo "ERROR: REMAINING_SIX_INSTANCE_ROLE must be auto, h100, or h200." >&2
    exit 2
    ;;
esac
[[ -n "${GPU0}" && -n "${GPU1}" ]] || {
  echo "ERROR: role ${ROLE} requires two matching local GPUs." >&2
  exit 2
}

RUN_BASE="${REMAINING_SIX_RUN_ROOT:-${ROOT}/outputs/remaining_six_instances}"
OUTPUT_ROOT="${RUN_BASE}/${ROLE}"
mkdir -p "${OUTPUT_ROOT}"
echo "Instance role=${ROLE}, host=$(hostname), local GPUs=${GPU0},${GPU1}, phase=${PHASE}"
if [[ "${PREPARE}" == 1 ]]; then
  MLLM_SEG_ROOT="${ROOT}" REMAINING_SIX_INSTANCE_ROLE="${ROLE}" \
    bash "${SCRIPT_DIR}/prepare_remaining_six_runtimes.sh" \
    |& tee "${OUTPUT_ROOT}/prepare_runtimes.log"
fi

wait_jobs() {
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
  return "${failed}"
}

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

run_ready_h100() {
  echo "H100 allocation: PolyFormer-L -> ${GPU0}; LISA diagnostic -> ${GPU1}"
  declare -a PIDS=() NAMES=() LOGS=()
  POLYFORMER_CUDA_DEVICES="${GPU0}" \
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_polyformer_freeref_full_eval.sh" \
    >"${OUTPUT_ROOT}/polyformer.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(PolyFormer); LOGS+=("${OUTPUT_ROOT}/polyformer.log")
  CUDA_DEVICE="${GPU1}" \
    LISA_SETUP_ENV=0 \
    LISA_SPLITS="refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test" \
    LISA_RESULTS_ROOT="${ROOT}/outputs/lisa_official_all8" \
    LISA_FREEREF_ROOT="${ROOT}/outputs/universal_freeref_lisa_all8" \
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_lisa_freeref_eval.sh" \
    >"${OUTPUT_ROOT}/lisa_diagnostic.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(LISA-diagnostic); LOGS+=("${OUTPUT_ROOT}/lisa_diagnostic.log")
  wait_jobs
  summarize_lisa_diagnostic
}

run_ready_h200() {
  local gsva_model="${GSVA_MLLM_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/gsva/LLaVA-Lightning-7B-v1-1-merged}"
  local gsva_ready=0 read_devices="${GPU0} ${GPU1}"
  if [[ -f "${gsva_model}/.freeref_merge_complete" && -f "${gsva_model}/config.json" ]] &&
     find "${gsva_model}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) \
       -size +100M -print -quit | grep -q .; then
    gsva_ready=1
    read_devices="${GPU0}"
    echo "H200 allocation: READ -> ${GPU0}; GSVA -> ${GPU1}"
  else
    echo "H200 allocation: READ eight splits -> ${GPU0},${GPU1}; GSVA deferred (licensed base missing)"
  fi
  declare -a PIDS=() NAMES=() LOGS=()
  READ_EVAL_CUDA_DEVICES="${read_devices}" \
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_read_freeref_full_eval.sh" \
    >"${OUTPUT_ROOT}/read.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(READ); LOGS+=("${OUTPUT_ROOT}/read.log")
  if [[ "${gsva_ready}" == 1 ]]; then
    GSVA_EVAL_CUDA_DEVICES="${GPU1}" \
      MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/run_gsva_freeref_full_eval.sh" \
      >"${OUTPUT_ROOT}/gsva.log" 2>&1 &
    PIDS+=("$!"); NAMES+=(GSVA); LOGS+=("${OUTPUT_ROOT}/gsva.log")
  fi
  wait_jobs
}

run_rela_train_h100() {
  echo "H100 ReLA allocation: RefCOCO+ -> ${GPU0},${GPU1}"
  MLLM_SEG_ROOT="${ROOT}" \
  RELA_TRAIN_DATASET=refcoco+ \
  RELA_TRAIN_CUDA_DEVICES="${GPU0},${GPU1}" \
    bash "${SCRIPT_DIR}/train_rela_classic_model.sh" \
    |& tee "${OUTPUT_ROOT}/rela_refcocoplus_train.log"
}

run_rela_train_h200() {
  echo "H200 ReLA allocation: RefCOCO -> ${GPU0}; RefCOCOg -> ${GPU1}"
  declare -a PIDS=() NAMES=() LOGS=()
  MLLM_SEG_ROOT="${ROOT}" RELA_TRAIN_DATASET=refcoco \
    RELA_TRAIN_CUDA_DEVICES="${GPU0}" \
    bash "${SCRIPT_DIR}/train_rela_classic_model.sh" \
    >"${OUTPUT_ROOT}/rela_refcoco_train.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(ReLA-RefCOCO); LOGS+=("${OUTPUT_ROOT}/rela_refcoco_train.log")
  MLLM_SEG_ROOT="${ROOT}" RELA_TRAIN_DATASET=refcocog \
    RELA_TRAIN_CUDA_DEVICES="${GPU1}" \
    bash "${SCRIPT_DIR}/train_rela_classic_model.sh" \
    >"${OUTPUT_ROOT}/rela_refcocog_train.log" 2>&1 &
  PIDS+=("$!"); NAMES+=(ReLA-RefCOCOg); LOGS+=("${OUTPUT_ROOT}/rela_refcocog_train.log")
  wait_jobs
}

run_rela_eval_assigned() {
  local specs
  if [[ "${ROLE}" == h100 ]]; then
    specs="refcoco+|val refcoco+|testA refcoco+|testB"
  else
    specs="refcoco|val refcoco|testA refcoco|testB refcocog|val refcocog|test"
  fi
  MLLM_SEG_ROOT="${ROOT}" \
  RELA_EVAL_CUDA_DEVICES="${GPU0} ${GPU1}" \
  RELA_SPLIT_SPECS="${specs}" \
  RELA_FINALIZE_FULL=0 \
    bash "${SCRIPT_DIR}/run_rela_freeref_full_eval.sh" \
    |& tee "${OUTPUT_ROOT}/rela_assigned_eval.log"
}

finalize_rela() {
  echo "Finalizing ReLA only; all eight shared split summaries must already exist."
  MLLM_SEG_ROOT="${ROOT}" \
  RELA_EVAL_CUDA_DEVICES="${GPU0} ${GPU1}" \
  RELA_FINALIZE_ONLY=1 \
    bash "${SCRIPT_DIR}/run_rela_freeref_full_eval.sh" \
    |& tee "${OUTPUT_ROOT}/rela_finalize.log"
}

run_phase() {
  case "${PHASE}" in
    ready)
      if [[ "${ROLE}" == h100 ]]; then run_ready_h100; else run_ready_h200; fi
      ;;
    rela-train)
      if [[ "${ROLE}" == h100 ]]; then run_rela_train_h100; else run_rela_train_h200; fi
      ;;
    rela-eval) run_rela_eval_assigned ;;
    rela-finalize) finalize_rela ;;
    all)
      if [[ "${ROLE}" == h100 ]]; then run_ready_h100; run_rela_train_h100
      else run_ready_h200; run_rela_train_h200
      fi
      run_rela_eval_assigned
      echo "Both instances must finish before running REMAINING_SIX_PHASE=rela-finalize once."
      ;;
    *)
      echo "ERROR: phase must be ready, rela-train, rela-eval, rela-finalize, or all." >&2
      exit 2
      ;;
  esac
}

run_phase
echo "Instance role=${ROLE} phase=${PHASE} completed."
echo "UNINEXT-L remains gated by the unavailable official Stage-2 ConvNeXt-L checkpoint."
