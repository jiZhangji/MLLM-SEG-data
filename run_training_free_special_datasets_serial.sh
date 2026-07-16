#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TRAINING_FREE_REPO:-${SCRIPT_DIR}}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
STAMP7B_MODEL="${STAMP7B_MODEL_NAME:-${ROOT}/models/STAMP-7B-lora}"
REASONSEG_MODEL="${REASONSEG_MODEL_NAME:-}"
GPU="${CUDA_DEVICE:-0}"
MIN_FREE_MB="${SPECIAL_MIN_FREE_GPU_MB:-24000}"
POLL_SECONDS="${SPECIAL_GPU_POLL_SECONDS:-10}"
EVAL_LIMIT="${SPECIAL_EVAL_LIMIT:-0}"
JSON_DIR="${ROOT}/code/STAMP/playground/data/json_eval_special"
MASK_ROOT="${ROOT}/code/STAMP/playground/data/masks_eval_special"
COMBINED_OUTPUT="${SPECIAL_COMBINED_OUTPUT:-${ROOT}/outputs/training_free_special_datasets_comparison}"
STATUS_FILE="${COMBINED_OUTPUT}/serial_status.tsv"

if [[ ! -x "${STAMP_ENV}/bin/python" ]]; then
  echo "ERROR: STAMP environment not found: ${STAMP_ENV}" >&2
  exit 1
fi
if [[ ! -d "${STAMP7B_MODEL}" ]]; then
  echo "ERROR: STAMP-7B model not found: ${STAMP7B_MODEL}" >&2
  exit 1
fi
if [[ -z "${REASONSEG_MODEL}" ]]; then
  if [[ -d "${ROOT}/models/STAMP-2B-reasonseg" ]]; then
    REASONSEG_MODEL="${ROOT}/models/STAMP-2B-reasonseg"
    REASONSEG_MODE="reasoning_finetuned"
  else
    REASONSEG_MODEL="${ROOT}/models/STAMP-2B-uni"
    REASONSEG_MODE="base_zero_shot"
  fi
else
  REASONSEG_MODE="user_checkpoint"
fi
if [[ ! -d "${REASONSEG_MODEL}" ]]; then
  echo "ERROR: ReasonSeg model not found: ${REASONSEG_MODEL}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for the 10-second GPU memory gate." >&2
  exit 1
fi

export PATH="${STAMP_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
mkdir -p "${JSON_DIR}" "${MASK_ROOT}" "${COMBINED_OUTPUT}" "${ROOT}/outputs/refine_stamp_dumps"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.training_free_special_datasets_serial.lock"
  if ! flock -n 9; then
    echo "Another special-dataset serial evaluation already holds the lock." >&2
    exit 0
  fi
fi

printf "dataset\tsplit\tstatus\tmodel_mode\tsummary\n" > "${STATUS_FILE}"
SUMMARY_ARGS=()

gpu_free_mb() {
  nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9'
}

wait_for_gpu() {
  local label="$1"
  while true; do
    local free_mb
    free_mb="$(gpu_free_mb)"
    if [[ -z "${free_mb}" ]]; then
      echo "ERROR: could not read free memory for GPU ${GPU}." >&2
      exit 1
    fi
    echo "[$(date '+%F %T')] ${label}: GPU ${GPU} free ${free_mb} MiB; required ${MIN_FREE_MB} MiB."
    if (( free_mb >= MIN_FREE_MB )); then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

prepare_dataset() {
  local dataset="$1"
  shift
  python offline_rstamp/scripts/prepare_special_eval_data.py \
    --root "${ROOT}" \
    --dataset "${dataset}" \
    --splits "$@" \
    --output-json-dir "${JSON_DIR}" \
    --mask-root "${MASK_ROOT}"
}

evaluate_split() {
  local dataset="$1"
  local split="$2"
  local model="$3"
  local model_mode="$4"
  local name="${dataset}_${split}"
  local json_path="${JSON_DIR}/${name}.json"
  local suffix="full"
  if [[ "${EVAL_LIMIT}" != "0" ]]; then
    suffix="limit${EVAL_LIMIT}"
  fi
  local dump_dir="${ROOT}/outputs/refine_stamp_dumps/${name}_${suffix}"
  local result_dir="${ROOT}/outputs/training_free_refine_${name}_${suffix}"
  mkdir -p "${dump_dir}" "${result_dir}"
  local expected
  expected="$(python -c 'import json,sys; n=len(json.load(open(sys.argv[1], encoding="utf-8"))); limit=int(sys.argv[2]); print(min(n, limit) if limit else n)' "${json_path}" "${EVAL_LIMIT}")"
  local existing
  existing="$(find "${dump_dir}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  echo "===== ${name}: dumps ${existing}/${expected}; model mode ${model_mode} ====="
  wait_for_gpu "${name}"
  MODEL_NAME="${model}" \
  SPLIT="${name}" \
  JSON_PATH="${json_path}" \
  EVAL_LIMIT="${EVAL_LIMIT}" \
  OUTPUT_DIR="${dump_dir}" \
  EMPTY_ON_FAILURE=1 \
    bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh
  python -m training_free_refine.eval_stamp_dumps \
    --input-dir "${dump_dir}" \
    --output-dir "${result_dir}" \
    --n-segments 1024 \
    --graph-lambda 1.0 \
    --save-visualizations 8
  SUMMARY_ARGS+=(--summary "${name}=${result_dir}/eval_summary.json")
  printf "%s\t%s\tcomplete\t%s\t%s\n" "${dataset}" "${split}" "${model_mode}" "${result_dir}/eval_summary.json" >> "${STATUS_FILE}"
}

cd "${REPO}"
echo "ReasonSeg checkpoint mode: ${REASONSEG_MODE} (${REASONSEG_MODEL})"

echo "[1/3] gRefCOCO val/testA/testB"
if prepare_dataset grefcoco val testA testB; then
  for split in val testA testB; do
    evaluate_split grefcoco "${split}" "${STAMP7B_MODEL}" "stamp7b_grefcoco"
  done
else
  printf "grefcoco\tall\tskipped_missing_data\tstamp7b_grefcoco\t\n" >> "${STATUS_FILE}"
  echo "WARNING: gRefCOCO preparation failed; continuing serial evaluation." >&2
fi

echo "[2/3] ReasonSeg val/test"
if prepare_dataset reasonseg val test; then
  for split in val test; do
    evaluate_split reasonseg "${split}" "${REASONSEG_MODEL}" "${REASONSEG_MODE}"
  done
else
  printf "reasonseg\tall\tskipped_missing_data\t%s\t\n" "${REASONSEG_MODE}" >> "${STATUS_FILE}"
  echo "WARNING: ReasonSeg preparation failed; continuing serial evaluation." >&2
fi

echo "[3/3] RefCLEF/ReferIt val/test"
if prepare_dataset refclef val test; then
  for split in val test; do
    evaluate_split refclef "${split}" "${STAMP7B_MODEL}" "stamp7b_refclef"
  done
else
  printf "refclef\tall\tskipped_missing_authorized_data\tstamp7b_refclef\t\n" >> "${STATUS_FILE}"
  echo "WARNING: RefCLEF official annotations/images are incomplete; marked skipped." >&2
fi

if (( ${#SUMMARY_ARGS[@]} > 0 )); then
  python -m training_free_refine.summarize_splits \
    "${SUMMARY_ARGS[@]}" \
    --output-dir "${COMBINED_OUTPUT}" \
    --title "Training-Free gRefCOCO, ReasonSeg and RefCLEF Evaluation"
fi

echo "Special-dataset serial evaluation completed."
echo "Status: ${STATUS_FILE}"
echo "Summary: ${COMBINED_OUTPUT}/combined_summary.md"
