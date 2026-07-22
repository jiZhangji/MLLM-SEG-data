#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
LIMIT="${PAPER_STUDY_SAMPLES:-500}"
JOBS="${PAPER_STUDY_PARALLEL_JOBS:-4}"
OUTPUT_ROOT="${PAPER_STUDY_OUTPUT_ROOT:-${ROOT}/outputs/freeref_paper_studies_n${LIMIT}}"
STAMP_INPUT="${PAPER_STUDY_STAMP_INPUT:-${ROOT}/outputs/refine_stamp_dumps/refcoco_testA_full_stamp7b}"
TEXT4SEG_MANIFEST="${PAPER_STUDY_TEXT4SEG_MANIFEST:-${ROOT}/outputs/text4seg_official_refcoco_testA/manifest.jsonl}"

if ! [[ "${LIMIT}" =~ ^[1-9][0-9]*$ && "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_STUDY_SAMPLES and PAPER_STUDY_PARALLEL_JOBS must be positive integers." >&2
  exit 1
fi
if [[ ! -d "${STAMP_INPUT}" ]]; then
  echo "ERROR: STAMP study dumps are missing: ${STAMP_INPUT}" >&2
  exit 1
fi
if [[ ! -f "${TEXT4SEG_MANIFEST}" ]]; then
  echo "ERROR: Text4Seg study manifest is missing: ${TEXT4SEG_MANIFEST}" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/combined"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.freeref_paper_studies.lock"
  if ! flock -n 9; then
    echo "Another FreeRef paper-study suite is already running." >&2
    exit 0
  fi
fi

complete() {
  local summary="$1"
  [[ -f "${summary}" ]] && \
    python -c 'import json,sys; raise SystemExit(0 if int(json.load(open(sys.argv[1])).get("samples",0))==int(sys.argv[2]) else 1)' \
      "${summary}" "${LIMIT}"
}

run_study() {
  local model="$1" study="$2" setting="$3"
  shift 3
  local output="${OUTPUT_ROOT}/${model}/${study}/${setting}"
  local log="${OUTPUT_ROOT}/logs/${model}_${study}_${setting}.log"
  if complete "${output}/eval_summary.json"; then
    echo "SKIP ${model}/${study}/${setting}"
    return
  fi
  mkdir -p "${output}"
  echo "RUN ${model}/${study}/${setting}; log=${log}"
  if [[ "${model}" == "stamp7b" ]]; then
    "${STAMP_ENV_PATH}/bin/python" -m training_free_refine.eval_stamp_dumps \
      --input-dir "${STAMP_INPUT}" --output-dir "${output}" --limit "${LIMIT}" \
      --save-visualizations 0 "$@" >"${log}" 2>&1
  elif [[ "${model}" == "text4seg_p24" ]]; then
    conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
      python -m training_free_refine.eval_text4seg_outputs \
      --manifest "${TEXT4SEG_MANIFEST}" --output-dir "${output}" --limit "${LIMIT}" \
      --save-visualizations 0 "$@" >"${log}" 2>&1
  else
    echo "ERROR: unsupported study model ${model}" >&2
    return 1
  fi
  echo "DONE ${model}/${study}/${setting}"
}

declare -a ACTIVE_PIDS=()
schedule() {
  if (( ${#ACTIVE_PIDS[@]} >= JOBS )); then
    wait "${ACTIVE_PIDS[0]}"
    ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  fi
  run_study "$@" &
  ACTIVE_PIDS+=("$!")
}

# Component ablations: remove exactly one component from the complete method.
schedule stamp7b ablation full
schedule stamp7b ablation uniform_anchoring --no-uncertainty-aware-anchoring
schedule stamp7b ablation unweighted_graph --no-appearance-weighted-graph
schedule stamp7b ablation global_replacement --no-selective-fusion

# Soft-probability sensitivity on the same STAMP-7B subset.
for value in 0.125 0.25 0.5 1 2 4 8; do
  slug="${value//./p}"
  schedule stamp7b graph_lambda "lambda_${slug}" --graph-lambda "${value}"
done
for value in 0.5 1 2 4 8; do
  slug="${value//./p}"
  schedule stamp7b confidence_power "gamma_${slug}" --confidence-power "${value}"
done
for value in 250 500 1000 1500 2000; do
  schedule stamp7b n_segments "k_${value}" --n-segments "${value}"
done

# Hard-mask uncertainty sensitivity for Text4Seg-p24.
for value in 2 4 8 16 32; do
  schedule text4seg_p24 boundary_sigma "sigma_${value}" --boundary-sigma "${value}"
done

for pid in "${ACTIVE_PIDS[@]}"; do
  wait "${pid}"
done
"${STAMP_ENV_PATH}/bin/python" -m training_free_refine.summarize_paper_studies \
  --input-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/combined"
echo "Paper studies complete: ${OUTPUT_ROOT}/combined/paper_studies.md"
