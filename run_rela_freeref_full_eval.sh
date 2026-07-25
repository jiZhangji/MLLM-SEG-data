#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${RELA_FULL_OUTPUT_ROOT:-${ROOT}/outputs/rela_freeref_full}"
LOG_ROOT="${RELA_FULL_LOG_ROOT:-${ROOT}/outputs/rela_freeref_full_logs}"
CUDA_DEVICES="${RELA_EVAL_CUDA_DEVICES:-0 1 2 3}"
SPLIT_SPECS="${RELA_SPLIT_SPECS:-refcoco|val refcoco|testA refcoco|testB refcoco+|val refcoco+|testA refcoco+|testB refcocog|val refcocog|test}"
read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "ERROR: RELA_EVAL_CUDA_DEVICES is empty." >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_split() {
  local spec="$1" gpu="$2" dataset split slug
  IFS='|' read -r dataset split <<<"${spec}"
  slug="${dataset//+/plus}_${split}"
  if [[ -f "${OUTPUT_ROOT}/${slug}/freeref/eval_summary.json" ]]; then
    echo "SKIP completed ReLA ${spec}"
    return 0
  fi
  MLLM_SEG_ROOT="${ROOT}" \
  CUDA_DEVICE="${gpu}" \
  RELA_DATASET="${dataset}" \
  RELA_SPLIT="${split}" \
  RELA_OUTPUT_ROOT="${OUTPUT_ROOT}/${slug}" \
    bash "${SCRIPT_DIR}/run_rela_freeref_split.sh"
}

declare -a ACTIVE_PIDS=() ACTIVE_NAMES=() ACTIVE_LOGS=()
failed=0
wait_batch() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "DONE ReLA ${ACTIVE_NAMES[$index]}"
    else
      echo "ERROR ReLA ${ACTIVE_NAMES[$index]}: ${ACTIVE_LOGS[$index]}" >&2
      tail -n 80 "${ACTIVE_LOGS[$index]}" >&2 || true
      failed=1
    fi
  done
  ACTIVE_PIDS=(); ACTIVE_NAMES=(); ACTIVE_LOGS=()
}

index=0
for spec in ${SPLIT_SPECS}; do
  gpu="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  slug="${spec//+/plus}"
  slug="${slug//|/_}"
  log="${LOG_ROOT}/${slug}.log"
  run_split "${spec}" "${gpu}" >"${log}" 2>&1 &
  ACTIVE_PIDS+=("$!"); ACTIVE_NAMES+=("${spec}"); ACTIVE_LOGS+=("${log}")
  index=$((index + 1))
  if (( ${#ACTIVE_PIDS[@]} >= ${#GPU_ARRAY[@]} )); then wait_batch; fi
done
if (( ${#ACTIVE_PIDS[@]} > 0 )); then wait_batch; fi
(( failed == 0 )) || exit 1

summary_args=()
table_args=()
for spec in ${SPLIT_SPECS}; do
  IFS='|' read -r dataset split <<<"${spec}"
  slug="${dataset//+/plus}_${split}"
  summary="${OUTPUT_ROOT}/${slug}/freeref/eval_summary.json"
  [[ -f "${summary}" ]] || { echo "ERROR: missing ReLA summary ${summary}" >&2; exit 1; }
  summary_args+=(--summary "ReLA_${dataset}_${split}=${summary}")
  table_args+=(--summary "${dataset}_${split}=${summary}")
done
cd "${SCRIPT_DIR}"
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize "${summary_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/combined" \
    --title "ReLA Original Mask vs. FreeRef — Eight Standard Splits"
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize_table1b \
    --method ReLA-Swin-B-local-classic-retrain \
    "${table_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/combined" \
    --eligibility paper_candidate \
    --expected-baseline 73.8 76.5 70.2 66.0 71.0 57.7 65.0 66.0 \
    --note "Locally retrained because the authors publish only gRefCOCO checkpoints; the automatic split-level baseline gate controls promotion."
echo "ReLA full paired suite complete: ${OUTPUT_ROOT}/combined/comparison.md"
