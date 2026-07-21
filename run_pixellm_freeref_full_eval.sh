#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXELLM_DIR="${PIXELLM_DIR:-${ROOT}/code/third_party/pixellm}"
MODEL_PATH="${PIXELLM_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/pixellm/PixelLM-7B}"
VISION_TOWER="${PIXELLM_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
PREPROCESSOR_CONFIG="${PIXELLM_PREPROCESSOR_CONFIG:-${PIXELLM_DIR}/configs/preprocessor_448.json}"
CONDA_ENV="${PIXELLM_CONDA_ENV:-lisa-freeref}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
SPLITS="${PIXELLM_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
CUDA_DEVICES="${PIXELLM_CUDA_DEVICES:-0 1}"
OUTPUT_ROOT="${PIXELLM_OUTPUT_ROOT:-${ROOT}/outputs/pixellm_public_freeref}"
LOG_ROOT="${PIXELLM_WORKER_LOG_ROOT:-${ROOT}/outputs/pixellm_public_freeref_worker_logs}"
LIMIT="${PIXELLM_LIMIT:-0}"
OFFSET="${PIXELLM_OFFSET:-0}"
MIN_FREE_MB="${PIXELLM_MIN_FREE_MB_PER_JOB:-22000}"
LAUNCH_STAGGER="${PIXELLM_LAUNCH_STAGGER_SECONDS:-5}"

if [[ ! -f "${MODEL_PATH}/config.json" && -f "${MODEL_PATH}/hf_model/config.json" ]]; then
  MODEL_PATH="${MODEL_PATH}/hf_model"
fi
for required in \
  "${PIXELLM_DIR}/model/PixelLM.py" \
  "${MODEL_PATH}/config.json" \
  "${VISION_TOWER}/config.json" \
  "${PREPROCESSOR_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: PixelLM prerequisite is missing: ${required}" >&2
    exit 1
  fi
done
read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "ERROR: PIXELLM_CUDA_DEVICES is empty." >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_split() {
  local split="$1" gpu="$2" eval_json base_dir refine_dir
  eval_json="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${split}.json"
  base_dir="${OUTPUT_ROOT}/${split//+/plus}/official"
  refine_dir="${OUTPUT_ROOT}/${split//+/plus}/freeref"
  if [[ ! -f "${eval_json}" ]]; then
    echo "ERROR: evaluation JSON is missing: ${eval_json}" >&2
    return 1
  fi
  if [[ -f "${refine_dir}/eval_summary.json" ]]; then
    echo "SKIP completed PixelLM + FreeRef ${split}"
    return 0
  fi
  while true; do
    free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${free_mb}" ]] && (( free_mb >= MIN_FREE_MB )); then break; fi
    echo "GPU ${gpu} free: ${free_mb:-unknown} MiB; waiting 10 seconds for ${MIN_FREE_MB} MiB..."
    sleep 10
  done
  echo "RUN PixelLM public checkpoint ${split} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.export_pixellm_masks \
      --pixellm-code-dir "${PIXELLM_DIR}" \
      --model-path "${MODEL_PATH}" \
      --vision-tower "${VISION_TOWER}" \
      --preprocessor-config "${PREPROCESSOR_CONFIG}" \
      --eval-json "${eval_json}" \
      --data-root "${ROOT}" \
      --output-dir "${base_dir}" \
      --split "${split}" \
      --precision bf16 \
      --seg-token-num 3 \
      --image-feature-scale-num 2 \
      --limit "${LIMIT}" \
      --offset "${OFFSET}" \
      --seed 0
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
    python -m universal_freeref.evaluate \
      --manifest "${base_dir}/manifest.jsonl" \
      --output-dir "${refine_dir}" \
      --n-segments 1024 \
      --graph-lambda 1.0 \
      --boundary-sigma 8.0 \
      --save-visualizations 12
}

declare -a ACTIVE_PIDS=() ACTIVE_SPLITS=() ACTIVE_LOGS=()
FAILED=0
wait_batch() {
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "DONE ${ACTIVE_SPLITS[$index]}"
    else
      echo "ERROR ${ACTIVE_SPLITS[$index]}; ${ACTIVE_LOGS[$index]}" >&2
      tail -n 50 "${ACTIVE_LOGS[$index]}" >&2 || true
      FAILED=1
    fi
  done
  ACTIVE_PIDS=(); ACTIVE_SPLITS=(); ACTIVE_LOGS=()
}

split_index=0
for split in ${SPLITS}; do
  gpu="${GPU_ARRAY[$((split_index % ${#GPU_ARRAY[@]}))]}"
  log_path="${LOG_ROOT}/${split//+/plus}.log"
  run_split "${split}" "${gpu}" >"${log_path}" 2>&1 &
  ACTIVE_PIDS+=("$!"); ACTIVE_SPLITS+=("${split}"); ACTIVE_LOGS+=("${log_path}")
  split_index=$((split_index + 1))
  sleep "${LAUNCH_STAGGER}"
  if (( ${#ACTIVE_PIDS[@]} >= ${#GPU_ARRAY[@]} )); then wait_batch; fi
done
if (( ${#ACTIVE_PIDS[@]} > 0 )); then wait_batch; fi
if (( FAILED != 0 )); then exit 1; fi

SUMMARY_ARGS=()
for split in ${SPLITS}; do
  summary="${OUTPUT_ROOT}/${split//+/plus}/freeref/eval_summary.json"
  [[ -f "${summary}" ]] && SUMMARY_ARGS+=(--summary "PixelLM_${split}=${summary}")
done
if (( ${#SUMMARY_ARGS[@]} > 0 )); then
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
    python -m universal_freeref.summarize "${SUMMARY_ARGS[@]}" \
      --output-dir "${OUTPUT_ROOT}/combined" \
      --title "PixelLM-7B Public Checkpoint Original Mask vs. FreeRef"
fi
echo "PixelLM paired FreeRef suite complete: ${OUTPUT_ROOT}/combined/comparison.md"
