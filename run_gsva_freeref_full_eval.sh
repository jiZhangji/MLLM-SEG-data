#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${GSVA_FULL_OUTPUT_ROOT:-${ROOT}/outputs/gsva_freeref_full}"
LOG_ROOT="${GSVA_FULL_LOG_ROOT:-${ROOT}/outputs/gsva_freeref_full_logs}"
CUDA_DEVICES="${GSVA_EVAL_CUDA_DEVICES:-0 1 2 3}"
SPLIT_SPECS="${GSVA_SPLIT_SPECS:-refcoco|val refcoco|testA refcoco|testB refcoco+|val refcoco+|testA refcoco+|testB refcocog|val refcocog|test}"
read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "ERROR: GSVA_EVAL_CUDA_DEVICES is empty." >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_split() {
  local spec="$1" gpu="$2" dataset split slug
  IFS='|' read -r dataset split <<<"${spec}"
  slug="${dataset//+/plus}_${split}"
  if [[ -f "${OUTPUT_ROOT}/${slug}/freeref/eval_summary.json" ]]; then
    echo "SKIP completed GSVA ${spec}"
    return 0
  fi
  MLLM_SEG_ROOT="${ROOT}" \
  CUDA_DEVICE="${gpu}" \
  GSVA_DATASET="${dataset}" \
  GSVA_SPLIT="${split}" \
  GSVA_OUTPUT_ROOT="${OUTPUT_ROOT}/${slug}" \
    bash "${SCRIPT_DIR}/run_gsva_freeref_split.sh"
}

declare -a ACTIVE_PIDS=() ACTIVE_NAMES=() ACTIVE_LOGS=()
failed=0
wait_batch() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "DONE GSVA ${ACTIVE_NAMES[$index]}"
    else
      echo "ERROR GSVA ${ACTIVE_NAMES[$index]}: ${ACTIVE_LOGS[$index]}" >&2
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
  [[ -f "${summary}" ]] || { echo "ERROR: missing GSVA summary ${summary}" >&2; exit 1; }
  summary_args+=(--summary "GSVA_${dataset}_${split}=${summary}")
  table_args+=(--summary "${dataset}_${split}=${summary}")
done
cd "${SCRIPT_DIR}"
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize "${summary_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/combined" \
    --title "GSVA Official SAM Mask vs. FreeRef — Eight Standard Splits"
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize_table1b \
    --method GSVA-7B-ft-res-official \
    "${table_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/combined" \
    --eligibility paper_candidate \
    --expected-baseline 77.2 78.9 73.5 65.9 69.6 59.8 72.7 73.3 \
    --note "Official ft-res checkpoint and official validation path; requires the authorized legacy LLaVA base."
echo "GSVA full paired suite complete: ${OUTPUT_ROOT}/combined/comparison.md"
