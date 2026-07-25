#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
OUTPUT_ROOT="${POLYFORMER_FULL_OUTPUT_ROOT:-${ROOT}/outputs/polyformer_freeref_full}"
LOG_ROOT="${POLYFORMER_FULL_LOG_ROOT:-${ROOT}/outputs/polyformer_freeref_full_logs}"
CUDA_DEVICES="${POLYFORMER_CUDA_DEVICES:-0}"
SPLIT_SPECS="${POLYFORMER_SPLIT_SPECS:-refcoco|unc|val refcoco|unc|testA refcoco|unc|testB refcoco+|unc|val refcoco+|unc|testA refcoco+|unc|testB refcocog|umd|val refcocog|umd|test}"

read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "ERROR: POLYFORMER_CUDA_DEVICES is empty." >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_split() {
  local spec="$1" gpu="$2" dataset split_by split slug
  IFS='|' read -r dataset split_by split <<<"${spec}"
  slug="${dataset//+/plus}_${split}"
  if [[ -f "${OUTPUT_ROOT}/${slug}/comparison.md" ]]; then
    echo "SKIP completed PolyFormer ${spec}"
    return 0
  fi
  CUDA_DEVICE="${gpu}" \
  POLYFORMER_DATASET="${dataset}" \
  POLYFORMER_SPLIT_BY="${split_by}" \
  POLYFORMER_SPLIT="${split}" \
  POLYFORMER_LIMIT=0 \
  POLYFORMER_OUTPUT_ROOT="${OUTPUT_ROOT}/${slug}" \
    bash "${SCRIPT_DIR}/run_polyformer_freeref_smoke.sh"
}

declare -a ACTIVE_PIDS=() ACTIVE_NAMES=() ACTIVE_LOGS=()
failed=0
wait_batch() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "DONE PolyFormer ${ACTIVE_NAMES[$index]}"
    else
      echo "ERROR PolyFormer ${ACTIVE_NAMES[$index]}: ${ACTIVE_LOGS[$index]}" >&2
      tail -n 60 "${ACTIVE_LOGS[$index]}" >&2 || true
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
for spec in ${SPLIT_SPECS}; do
  IFS='|' read -r dataset split_by split <<<"${spec}"
  slug="${dataset//+/plus}_${split}"
  summary="${OUTPUT_ROOT}/${slug}/freeref/eval_summary.json"
  [[ -f "${summary}" ]] || { echo "ERROR: missing summary ${summary}" >&2; exit 1; }
  summary_args+=(--summary "PolyFormer-L_${dataset}_${split}=${summary}")
done
conda run --no-capture-output -n "${FREEREF_CONDA_ENV:-STAMP}" \
  python -m universal_freeref.summarize "${summary_args[@]}" \
  --output-dir "${OUTPUT_ROOT}/combined" \
  --title "PolyFormer-L Original Rasterized Polygon vs. FreeRef"
echo "PolyFormer full paired suite complete: ${OUTPUT_ROOT}/combined/comparison.md"
