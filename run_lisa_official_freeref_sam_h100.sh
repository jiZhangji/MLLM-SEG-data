#!/usr/bin/env bash
set -uo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISA_DIR="${LISA_DIR:-${ROOT}/code/third_party/lisa}"
CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"
WEIGHTS_ROOT="${FREEREF_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
MODEL_PATH="${LISA_MODEL_PATH:-${WEIGHTS_ROOT}/lisa/LISA-7B-v1}"
VISION_TOWER="${LISA_VISION_TOWER:-${WEIGHTS_ROOT}/shared/clip-vit-large-patch14}"
SAM_PATH="${LISA_SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
DATA_ROOT="${LISA_PAPER_DATA_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
OUTPUT_ROOT="${LISA_OFFICIAL_FREEREF_SAM_OUTPUT_ROOT:-${ROOT}/outputs/lisa_official_freeref_before_sam}"
PARALLEL_PER_GPU="${LISA_H100_PARALLEL_PER_GPU:-2}"
MIN_FREE_MB="${LISA_H100_MIN_FREE_MB:-70000}"
LIMIT_IMAGES="${LISA_OFFICIAL_LIMIT_IMAGES:-0}"
OFFSET_IMAGES="${LISA_OFFICIAL_OFFSET_IMAGES:-0}"
SPLITS="${LISA_OFFICIAL_SPLITS:-refcoco|unc|val refcoco|unc|testA refcoco|unc|testB refcoco+|unc|val refcoco+|unc|testA refcoco+|unc|testB refcocog|umd|val refcocog|umd|test}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}" "${ROOT}/outputs/lisa_official_freeref_sam_logs"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.lisa_official_freeref_sam_h100.lock"
  if ! flock -n 9; then
    echo "Another LISA official FreeRef-before-SAM suite is running."
    exit 0
  fi
fi

LISA_PAPER_DATA_ROOT="${DATA_ROOT}" bash "${SCRIPT_DIR}/prepare_lisa_paper_data.sh"
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F, 'tolower($2) ~ /h100/ {gsub(/ /,"",$1); print $1}')
if (( ${#GPUS[@]} < 2 )); then
  echo "ERROR: this runner requires an instance exposing two H100 GPUs." >&2
  nvidia-smi -L >&2
  exit 2
fi

declare -a JOBS_A=()
declare -a JOBS_B=()
job_index=0
for split in ${SPLITS}; do
  if (( job_index % 2 == 0 )); then JOBS_A+=("${split}"); else JOBS_B+=("${split}"); fi
  job_index=$((job_index + 1))
done

wait_for_gpu() {
  local gpu="$1" free_mb
  while true; do
    free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${free_mb}" ]] && (( free_mb >= MIN_FREE_MB )); then return; fi
    echo "GPU ${gpu} free ${free_mb:-unknown} MiB; waiting for ${MIN_FREE_MB} MiB..."
    sleep 10
  done
}

run_split() {
  local gpu="$1" split="$2" slug output summary complete
  slug="${split//|/_}"
  output="${OUTPUT_ROOT}/${slug}"
  summary="${output}/eval_summary.json"
  complete=0
  if [[ -f "${summary}" && "${LIMIT_IMAGES}" == "0" && "${OFFSET_IMAGES}" == "0" ]]; then
    complete="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("sample_count_matches") is True))' "${summary}" 2>/dev/null || echo 0)"
  fi
  if [[ "${complete}" == "1" ]]; then
    echo "SKIP complete ${split}"
    return 0
  fi
  echo "RUN ${split} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.eval_lisa_official_freeref_sam \
      --lisa-code-dir "${LISA_DIR}" \
      --model-path "${MODEL_PATH}" \
      --vision-tower "${VISION_TOWER}" \
      --sam-path "${SAM_PATH}" \
      --dataset-dir "${DATA_ROOT}" \
      --val-dataset "${split}" \
      --output-dir "${output}" \
      --paper-row finetuned_referseg \
      --precision bf16 \
      --workers 4 \
      --limit-images "${LIMIT_IMAGES}" \
      --offset-images "${OFFSET_IMAGES}" \
      --seed 0
}

run_queue() {
  local gpu="$1"
  shift
  local split slug pid index failed
  local -a pids=() names=()
  failed=0
  for split in "$@"; do
    if (( ${#pids[@]} == 0 )); then
      wait_for_gpu "${gpu}"
    fi
    slug="${split//|/_}"
    run_split "${gpu}" "${split}" >"${ROOT}/outputs/lisa_official_freeref_sam_logs/${slug}.log" 2>&1 &
    pids+=("$!")
    names+=("${split}")
    if (( ${#pids[@]} >= PARALLEL_PER_GPU )); then
      for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        if wait "${pid}"; then echo "DONE ${names[$index]}"; else echo "ERROR ${names[$index]}" >&2; failed=1; fi
      done
      pids=()
      names=()
    fi
  done
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    if wait "${pid}"; then echo "DONE ${names[$index]}"; else echo "ERROR ${names[$index]}" >&2; failed=1; fi
  done
  return "${failed}"
}

echo "LISA official four-branch scheduler: GPU ${GPUS[0]} and GPU ${GPUS[1]}, ${PARALLEL_PER_GPU} jobs/GPU"
run_queue "${GPUS[0]}" "${JOBS_A[@]}" >"${ROOT}/outputs/lisa_official_freeref_sam_gpu0.log" 2>&1 &
PID_A="$!"
run_queue "${GPUS[1]}" "${JOBS_B[@]}" >"${ROOT}/outputs/lisa_official_freeref_sam_gpu1.log" 2>&1 &
PID_B="$!"
FAILED=0
if ! wait "${PID_A}"; then FAILED=1; fi
if ! wait "${PID_B}"; then FAILED=1; fi

if (( FAILED == 0 && LIMIT_IMAGES == 0 && OFFSET_IMAGES == 0 )); then
  SUMMARY_ARGS=()
  for split in ${SPLITS}; do
    slug="${split//|/_}"
    SUMMARY_ARGS+=(--summary "${split}=${OUTPUT_ROOT}/${slug}/eval_summary.json")
  done
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.summarize_lisa_official_freeref_sam \
      "${SUMMARY_ARGS[@]}" \
      --output-dir "${OUTPUT_ROOT}/combined"
fi

echo "LISA official FreeRef-before-second-SAM suite stopped; failed=${FAILED}."
exit "${FAILED}"
