#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${READ_FULL_OUTPUT_ROOT:-${ROOT}/outputs/read_freeref_full}"
LOG_ROOT="${READ_FULL_LOG_ROOT:-${ROOT}/outputs/read_freeref_full_logs}"
CUDA_DEVICES="${READ_EVAL_CUDA_DEVICES:-0 1 2 3}"
SPLIT_SPECS="${READ_SPLIT_SPECS:-refcoco|val refcoco|testA refcoco|testB refcoco+|val refcoco+|testA refcoco+|testB refcocog|val refcocog|test}"
read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "ERROR: READ_EVAL_CUDA_DEVICES is empty." >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_split() {
  local spec="$1" gpu="$2" dataset split slug
  IFS='|' read -r dataset split <<<"${spec}"
  slug="${dataset//+/plus}_${split}"
  if [[ -f "${OUTPUT_ROOT}/${slug}/freeref/eval_summary.json" ]]; then
    echo "SKIP completed READ ${spec}"
    return 0
  fi
  MLLM_SEG_ROOT="${ROOT}" \
  CUDA_DEVICE="${gpu}" \
  READ_DATASET="${dataset}" \
  READ_SPLIT="${split}" \
  READ_OUTPUT_ROOT="${OUTPUT_ROOT}/${slug}" \
    bash "${SCRIPT_DIR}/run_read_freeref_split.sh"
}

declare -a ACTIVE_PIDS=() ACTIVE_NAMES=() ACTIVE_LOGS=()
failed=0
wait_batch() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "DONE READ ${ACTIVE_NAMES[$index]}"
    else
      echo "ERROR READ ${ACTIVE_NAMES[$index]}: ${ACTIVE_LOGS[$index]}" >&2
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
  [[ -f "${summary}" ]] || { echo "ERROR: missing READ summary ${summary}" >&2; exit 1; }
  summary_args+=(--summary "READ_${dataset}_${split}=${summary}")
  table_args+=(--summary "${dataset}_${split}=${summary}")
done
cd "${SCRIPT_DIR}"
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize "${summary_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/combined" \
    --title "READ Official SasP/SAM Mask vs. FreeRef — Eight Standard Splits"
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize_table1b \
    --method READ-LLaVA-v1.5-7B-official \
    "${table_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/combined" \
    --eligibility paper_candidate \
    --expected-baseline 78.1 80.2 73.2 68.4 73.7 60.4 70.1 71.4 \
    --note "Official full checkpoint with the standard teacher-forced [SEG] validation protocol."
echo "READ full paired suite complete: ${OUTPUT_ROOT}/combined/comparison.md"
