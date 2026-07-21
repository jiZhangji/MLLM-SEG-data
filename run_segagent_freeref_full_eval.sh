#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEGAGENT_DIR="${SEGAGENT_DIR:-${ROOT}/code/third_party/segagent}"
MODEL_PATH="${SEGAGENT_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/segagent/SegAgent-Model}"
DATASET_ROOT="${SEGAGENT_DATASET_ROOT:-${ROOT}/models/freeref_missing_methods/segagent/SegAgent-Dataset}"
IMAGE_ROOT="${SEGAGENT_IMAGE_ROOT:-${ROOT}/data/lisa_paper_refer_seg/images}"
SIMPLECLICK_ROOT="${SEGAGENT_SIMPLECLICK_ROOT:-${ROOT}/models/freeref_missing_methods/segagent/simpleclick_models}"
SIMPLECLICK_CHECKPOINT="${SEGAGENT_SIMPLECLICK_CHECKPOINT:-}"
SEGAGENT_ENV="${SEGAGENT_CONDA_ENV:-segagent-freeref}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
SPLITS="${SEGAGENT_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
CUDA_DEVICES="${SEGAGENT_CUDA_DEVICES:-0 1}"
OUTPUT_ROOT="${SEGAGENT_OUTPUT_ROOT:-${ROOT}/outputs/segagent_freeref}"
LOG_ROOT="${SEGAGENT_WORKER_LOG_ROOT:-${ROOT}/outputs/segagent_freeref_worker_logs}"
LIMIT_ITEMS="${SEGAGENT_LIMIT_ITEMS:-0}"
OFFSET_ITEMS="${SEGAGENT_OFFSET_ITEMS:-0}"
N_CLICKS="${SEGAGENT_N_CLICKS:-7}"
MIN_FREE_MB="${SEGAGENT_MIN_FREE_MB_PER_JOB:-22000}"
LAUNCH_STAGGER="${SEGAGENT_LAUNCH_STAGGER_SECONDS:-5}"
CLICK_GUIDANCE="${SEGAGENT_FREEREF_CLICK_GUIDANCE:-0}"
CLICK_GUIDANCE_BOUNDARY_SIGMA="${SEGAGENT_FREEREF_BOUNDARY_SIGMA:-8.0}"

if [[ ! -d "${SEGAGENT_DIR}/evaltools" ]]; then
  echo "ERROR: SegAgent code is missing: ${SEGAGENT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  nested_model="$(find "${MODEL_PATH}" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit 2>/dev/null || true)"
  [[ -n "${nested_model}" ]] && MODEL_PATH="${nested_model}"
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: SegAgent model is incomplete: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: SegAgent-Dataset is missing: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "ERROR: SegAgent image root is missing: ${IMAGE_ROOT}" >&2
  exit 1
fi
if [[ -z "${SIMPLECLICK_CHECKPOINT}" ]]; then
  SIMPLECLICK_CHECKPOINT="$(find "${SIMPLECLICK_ROOT}" -type f -name '*cocolvis*vit*large*.pth' -print -quit 2>/dev/null || true)"
fi
if [[ -z "${SIMPLECLICK_CHECKPOINT}" ]]; then
  SIMPLECLICK_CHECKPOINT="$(find "${SIMPLECLICK_ROOT}" -type f -name '*large*.pth' -print -quit 2>/dev/null || true)"
fi
if [[ -z "${SIMPLECLICK_CHECKPOINT}" || ! -f "${SIMPLECLICK_CHECKPOINT}" ]]; then
  echo "ERROR: SimpleClick cocolvis ViT-L checkpoint is missing below ${SIMPLECLICK_ROOT}." >&2
  exit 1
fi

read -r -a GPU_ARRAY <<<"${CUDA_DEVICES}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "ERROR: SEGAGENT_CUDA_DEVICES is empty." >&2
  exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd -P)"
LOG_ROOT="$(cd "${LOG_ROOT}" && pwd -P)"

find_dataset_json() {
  local split="$1" candidate
  for candidate in \
    "${DATASET_ROOT}/${split}.json" \
    "${DATASET_ROOT}/data/${split}.json" \
    "${DATASET_ROOT}/segagent-data/${split}.json"; do
    [[ -f "${candidate}" ]] && { printf '%s\n' "${candidate}"; return 0; }
  done
  find "${DATASET_ROOT}" -type f -name "${split}.json" -print -quit 2>/dev/null
}

find_official_output_json() {
  local official_dir="$1"
  find "${official_dir}" -maxdepth 1 -type f \
    -name "*_newresults_${N_CLICKS}_simple_click_qwen-full_radius0use_gt_box0.json" \
    -print -quit 2>/dev/null
}

run_split() {
  local split="$1" gpu="$2" master_port="$3" data_json official_dir import_dir refine_dir output_json free_mb method_label
  local -a guidance_args=()
  data_json="$(find_dataset_json "${split}")"
  if [[ -z "${data_json}" || ! -f "${data_json}" ]]; then
    echo "ERROR: SegAgent dataset JSON not found for ${split}." >&2
    return 1
  fi
  method_label="SegAgent-Qwen7B-SimpleClick"
  if [[ "${CLICK_GUIDANCE}" == "1" ]]; then
    method_label="SegAgent-Qwen7B-SimpleClick-FreeRefGuidedClicks"
    guidance_args+=(
      --freeref-click-guidance
      --freeref-boundary-sigma "${CLICK_GUIDANCE_BOUNDARY_SIGMA}"
    )
    official_dir="${OUTPUT_ROOT}/${split//+/plus}/click_guided_official"
    import_dir="${OUTPUT_ROOT}/${split//+/plus}/click_guided_import"
    refine_dir="${OUTPUT_ROOT}/${split//+/plus}/click_guided_evaluation"
  else
    official_dir="${OUTPUT_ROOT}/${split//+/plus}/official"
    import_dir="${OUTPUT_ROOT}/${split//+/plus}/import"
    refine_dir="${OUTPUT_ROOT}/${split//+/plus}/freeref"
  fi
  mkdir -p "${official_dir}"
  output_json="$(find_official_output_json "${official_dir}")"

  if [[ ! -f "${refine_dir}/eval_summary.json" ]]; then
    if [[ -z "${output_json}" || ! -f "${output_json}" ]]; then
      echo "RUN SegAgent ${split} on physical GPU ${gpu}"
      while true; do
        free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
        if [[ -n "${free_mb}" ]] && (( free_mb >= MIN_FREE_MB )); then break; fi
        echo "GPU ${gpu} free: ${free_mb:-unknown} MiB; waiting 10 seconds for ${MIN_FREE_MB} MiB..."
        sleep 10
      done
      VIS_DIR="${official_dir}" CUDA_VISIBLE_DEVICES="${gpu}" MASTER_PORT="${master_port}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
        conda run --no-capture-output -n "${SEGAGENT_ENV}" \
        python -m universal_freeref.run_segagent_official \
          --segagent-code-dir "${SEGAGENT_DIR}" \
          --limit-items "${LIMIT_ITEMS}" \
          --offset-items "${OFFSET_ITEMS}" \
          "${guidance_args[@]}" \
          NoBRS \
          --model "${MODEL_PATH}" \
          --img "${IMAGE_ROOT}" \
          --json "${data_json}" \
          --checkpoint "${SIMPLECLICK_CHECKPOINT}" \
          --config-path "${SEGAGENT_DIR}/third_party/SimpleClick/config.yml" \
          --no_use_mask_module \
          --use_previous_mask 1 \
          --n-clicks "${N_CLICKS}" \
          --seg_model simple_click \
          --grounding_model qwen-full \
          --gpus 0
      output_json="$(find_official_output_json "${official_dir}")"
      if [[ -z "${output_json}" || ! -f "${output_json}" ]]; then
        echo "ERROR: SegAgent completed but its official result JSON was not found in ${official_dir}." >&2
        return 1
      fi
    else
      echo "REUSE SegAgent official output ${split}: ${output_json}"
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
      python -m universal_freeref.import_segagent_outputs \
        --input-json "${output_json}" \
        --output-dir "${import_dir}" \
        --split "${split}" \
        --method "${method_label}" \
        --selection final \
        --image-root "${IMAGE_ROOT}"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
      python -m universal_freeref.evaluate \
        --manifest "${import_dir}/manifest.jsonl" \
        --output-dir "${refine_dir}" \
        --n-segments 1024 \
        --graph-lambda 1.0 \
        --boundary-sigma 8.0 \
        --save-visualizations 12
  else
    echo "SKIP completed SegAgent + FreeRef ${split}"
  fi
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
  master_port="$((12500 + split_index))"
  log_path="${LOG_ROOT}/${split//+/plus}.log"
  run_split "${split}" "${gpu}" "${master_port}" >"${log_path}" 2>&1 &
  ACTIVE_PIDS+=("$!"); ACTIVE_SPLITS+=("${split}"); ACTIVE_LOGS+=("${log_path}")
  split_index=$((split_index + 1))
  sleep "${LAUNCH_STAGGER}"
  if (( ${#ACTIVE_PIDS[@]} >= ${#GPU_ARRAY[@]} )); then wait_batch; fi
done
if (( ${#ACTIVE_PIDS[@]} > 0 )); then wait_batch; fi
if (( FAILED != 0 )); then exit 1; fi

SUMMARY_ARGS=()
for split in ${SPLITS}; do
  if [[ "${CLICK_GUIDANCE}" == "1" ]]; then
    summary="${OUTPUT_ROOT}/${split//+/plus}/click_guided_evaluation/eval_summary.json"
  else
    summary="${OUTPUT_ROOT}/${split//+/plus}/freeref/eval_summary.json"
  fi
  [[ -f "${summary}" ]] && SUMMARY_ARGS+=(--summary "SegAgent_${split}=${summary}")
done
if (( ${#SUMMARY_ARGS[@]} > 0 )); then
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
    python -m universal_freeref.summarize "${SUMMARY_ARGS[@]}" \
      --output-dir "${OUTPUT_ROOT}/combined" \
      --title "SegAgent-SimpleClick Original Mask vs. FreeRef"
fi
echo "SegAgent paired FreeRef suite complete: ${OUTPUT_ROOT}/combined/comparison.md"
