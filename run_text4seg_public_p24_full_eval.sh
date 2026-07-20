#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${TEXT4SEG_P24_MODEL_PATH:-lmc22/text4seg-llava-7b-p24}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PARALLEL_JOBS="${TEXT4SEG_P24_PARALLEL_JOBS:-4}"
MIN_FREE_MB="${TEXT4SEG_P24_MIN_FREE_MB:-$((18000 * PARALLEL_JOBS))}"
SPLITS="${TEXT4SEG_P24_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
COMBINED_OUTPUT="${TEXT4SEG_P24_COMBINED_OUTPUT:-${ROOT}/outputs/text4seg_public_p24_freeref_samh_full_comparison}"
VISION_TOWER="${TEXT4SEG_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
SAM_PATH="${TEXT4SEG_SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
TEXT4SEG_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
WORKER_LOG_DIR="${ROOT}/outputs/text4seg_public_p24_worker_logs"

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TEXT4SEG_P24_PARALLEL_JOBS must be a positive integer." >&2
  exit 1
fi
if ! [[ "${MIN_FREE_MB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TEXT4SEG_P24_MIN_FREE_MB must be a non-negative integer." >&2
  exit 1
fi
if [[ "${MODEL_PATH,,}" != *p24* ]]; then
  echo "ERROR: TEXT4SEG_P24_MODEL_PATH must identify the public p24 checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${TEXT4SEG_DIR}/llava" ]]; then
  echo "ERROR: Text4Seg code is missing: ${TEXT4SEG_DIR}" >&2
  exit 1
fi
if [[ ! -d "${VISION_TOWER}" ]]; then
  echo "ERROR: local CLIP vision tower is missing: ${VISION_TOWER}" >&2
  exit 1
fi
if [[ ! -f "${SAM_PATH}" ]] || (( $(stat -c '%s' "${SAM_PATH}" 2>/dev/null || echo 0) < 2000000000 )); then
  echo "ERROR: complete SAM-H checkpoint is missing: ${SAM_PATH}" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${COMBINED_OUTPUT}" "${WORKER_LOG_DIR}"
if command -v flock >/dev/null 2>&1; then
  exec 8>"${ROOT}/outputs/.text4seg_public_p24_full.lock"
  if ! flock -n 8; then
    echo "Another Text4Seg public-p24 full evaluation is already running." >&2
    exit 0
  fi
fi

while true; do
  FREE_MB="$(nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
  if [[ -n "${FREE_MB}" ]] && (( FREE_MB >= MIN_FREE_MB )); then
    break
  fi
  echo "GPU ${CUDA_DEVICE} free: ${FREE_MB:-unknown} MiB; waiting 10 seconds for ${MIN_FREE_MB} MiB..."
  sleep 10
done

run_split() {
  local SPLIT="$1"
  local EVAL_JSON BASE_OUTPUT REFINE_OUTPUT MANIFEST SAMPLES EXPECTED SAMH_OUTPUT SAMH_SUMMARY PAIRED_READY
  EVAL_JSON="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json"
  BASE_OUTPUT="${ROOT}/outputs/text4seg_official_${SPLIT}"
  REFINE_OUTPUT="${ROOT}/outputs/text4seg_training_free_${SPLIT}"
  MANIFEST="${BASE_OUTPUT}/manifest.jsonl"
  SAMH_OUTPUT="${ROOT}/outputs/text4seg_public_p24_samh_${SPLIT}"
  SAMH_SUMMARY="${SAMH_OUTPUT}/eval_summary.json"

  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "ERROR: paired evaluation JSON is missing: ${EVAL_JSON}" >&2
    return 1
  fi
  EXPECTED="$(conda run -n "${CONDA_ENV}" python -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "${EVAL_JSON}")"
  SAMPLES=0
  if [[ -f "${SAMH_SUMMARY}" ]]; then
    SAMPLES="$(conda run -n "${CONDA_ENV}" python -c \
      'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("samples", 0)))' \
      "${SAMH_SUMMARY}" 2>/dev/null || echo 0)"
  fi
  if [[ "${SAMPLES}" == "${EXPECTED}" ]]; then
    echo "SKIP complete four-branch result ${SPLIT}: ${SAMPLES}/${EXPECTED}"
    return 0
  fi

  echo "RUN Text4Seg public-p24 ${SPLIT}: ${SAMPLES}/${EXPECTED}"
  PAIRED_READY=0
  if [[ -f "${BASE_OUTPUT}/export_summary.json" && -f "${REFINE_OUTPUT}/eval_summary.json" && -f "${MANIFEST}" ]]; then
    PAIRED_READY="$(conda run -n "${CONDA_ENV}" python -c \
      'import json,sys; expected=int(sys.argv[3]); a=json.load(open(sys.argv[1], encoding="utf-8")); b=json.load(open(sys.argv[2], encoding="utf-8")); print(int(a.get("samples", 0)==expected and b.get("samples", 0)==expected))' \
      "${BASE_OUTPUT}/export_summary.json" "${REFINE_OUTPUT}/eval_summary.json" "${EXPECTED}" \
      2>/dev/null || echo 0)"
  fi
  if [[ "${PAIRED_READY}" == "1" ]]; then
    echo "REUSE completed Text4Seg + FreeRef artifacts for ${SPLIT}"
  else
    TEXT4SEG_MODEL_PATH="${MODEL_PATH}" \
    TEXT4SEG_SETUP_MODE=offline \
    TEXT4SEG_DESCRIPTOR_GRID_SIZE=24 \
    TEXT4SEG_VISION_TOWER="${VISION_TOWER}" \
    TEXT4SEG_SAM_PATH="${SAM_PATH}" \
    TEXT4SEG_EVAL_JSON="${EVAL_JSON}" \
    TEXT4SEG_RESULTS_ROOT="${BASE_OUTPUT}" \
    TEXT4SEG_REFINE_OUTPUT="${REFINE_OUTPUT}" \
    CUDA_DEVICE="${CUDA_DEVICE}" \
      bash "${SCRIPT_DIR}/run_text4seg_training_free_eval.sh"
  fi

  if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Text4Seg manifest was not generated: ${MANIFEST}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m training_free_refine.eval_text4seg_sam_h \
      --manifest "${MANIFEST}" \
      --output-dir "${SAMH_OUTPUT}" \
      --text4seg-code-dir "${TEXT4SEG_DIR}" \
      --sam-path "${SAM_PATH}" \
      --model-label "Text4Seg-7B-p24" \
      --split-name "${SPLIT}" \
      --point-count 10 \
      --cascade-steps 2 \
      --seed 0 \
      --save-visualizations 8
}

wait_batch() {
  local index pid split log_path
  for index in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$index]}"
    split="${ACTIVE_SPLITS[$index]}"
    log_path="${ACTIVE_LOGS[$index]}"
    if wait "${pid}"; then
      echo "DONE ${split}; worker log: ${log_path}"
    else
      echo "ERROR ${split}; worker log: ${log_path}" >&2
      tail -n 40 "${log_path}" >&2 || true
      FAILED=1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_SPLITS=()
  ACTIVE_LOGS=()
}

cd "${SCRIPT_DIR}"
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_SPLITS=()
declare -a ACTIVE_LOGS=()
FAILED=0

echo "Text4Seg public-p24 scheduler: parallel_jobs=${PARALLEL_JOBS}, gpu=${CUDA_DEVICE}, min_free_mb=${MIN_FREE_MB}"
for SPLIT in ${SPLITS}; do
  SAFE_SPLIT="${SPLIT//+/plus}"
  WORKER_LOG="${WORKER_LOG_DIR}/${SAFE_SPLIT}.log"
  run_split "${SPLIT}" >"${WORKER_LOG}" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_SPLITS+=("${SPLIT}")
  ACTIVE_LOGS+=("${WORKER_LOG}")
  echo "START ${SPLIT}; worker log: ${WORKER_LOG}"
  if (( ${#ACTIVE_PIDS[@]} >= PARALLEL_JOBS )); then
    wait_batch
  fi
done
if (( ${#ACTIVE_PIDS[@]} > 0 )); then
  wait_batch
fi
if (( FAILED != 0 )); then
  echo "ERROR: at least one Text4Seg split failed; combined summary was not generated." >&2
  exit 1
fi

SUMMARY_ARGS=()
for SPLIT in ${SPLITS}; do
  SUMMARY_ARGS+=(--summary "${SPLIT}=${ROOT}/outputs/text4seg_public_p24_samh_${SPLIT}/eval_summary.json")
done
conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m training_free_refine.summarize_text4seg_sam_h \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${COMBINED_OUTPUT}"

echo "Text4Seg public-p24 four-branch evaluation completed."
echo "Summary: ${COMBINED_OUTPUT}/combined_summary.md"
echo "Branches: Text4Seg / +FreeRef / +SAM-H / +FreeRef+SAM-H"
