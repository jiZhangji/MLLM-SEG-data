#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
GPU_LIST="${FREEREF_BASELINE_GPUS:-0 1}"
LIMIT="${FREEREF_BASELINE_SAMPLES:-500}"
MIN_FREE_MB="${FREEREF_BASELINE_MIN_FREE_MB:-60000}"
FORCE="${FREEREF_BASELINE_FORCE:-0}"
OUTPUT_ROOT="${FREEREF_BASELINE_OUTPUT_ROOT:-${ROOT}/outputs/freeref_postprocess_baselines_n${LIMIT}}"
STAMP_INPUT="${FREEREF_BASELINE_STAMP_INPUT:-${ROOT}/outputs/refine_stamp_dumps/refcoco_testA_full_stamp7b}"
TEXT4SEG_MANIFEST="${FREEREF_BASELINE_TEXT4SEG_MANIFEST:-${ROOT}/outputs/text4seg_official_refcoco_testA/manifest.jsonl}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${ROOT}/code/STAMP}"
TEXT4SEG_CODE_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
SAM_PATH="${SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"

read -r -a GPUS <<<"${GPU_LIST}"
if (( ${#GPUS[@]} != 2 )); then
  echo "ERROR: FREEREF_BASELINE_GPUS must contain exactly two physical GPU indices." >&2
  exit 1
fi
GPU0="${GPUS[0]}"
GPU1="${GPUS[1]}"

if ! [[ "${LIMIT}" =~ ^[1-9][0-9]*$ && "${MIN_FREE_MB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: sample count and minimum free memory must be positive integers." >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  name="$(nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader | head -n1)"
  if [[ "${name,,}" != *h100* ]]; then
    echo "ERROR: physical GPU ${gpu} is ${name}, not H100." >&2
    exit 1
  fi
done
for path in \
  "${STAMP_ENV_PATH}/bin/python" "${STAMP_INPUT}" "${TEXT4SEG_MANIFEST}" \
  "${STAMP_CODE_DIR}" "${TEXT4SEG_CODE_DIR}" "${SAM_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required input is missing: ${path}" >&2
    exit 1
  fi
done

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/combined"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.freeref_postprocess_baselines_n${LIMIT}.lock"
  if ! flock -n 9; then
    echo "Another FreeRef post-processing baseline suite is already running." >&2
    exit 0
  fi
fi

if ! "${STAMP_ENV_PATH}/bin/python" -c \
  'import pydensecrf.densecrf, scipy, skimage, cupy, cucim' >/dev/null 2>&1; then
  echo "ERROR: STAMP baseline dependencies are missing." >&2
  echo "Run: bash prepare_freeref_baseline_env.sh" >&2
  exit 1
fi
if ! conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" python -c \
  'import pydensecrf.densecrf, scipy, skimage, cupy, cucim' >/dev/null 2>&1; then
  echo "ERROR: Text4Seg baseline dependencies are missing." >&2
  echo "Run: bash prepare_freeref_baseline_env.sh" >&2
  exit 1
fi

wait_for_gpu() {
  local gpu="$1" free
  while true; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${free}" ]] && (( free >= MIN_FREE_MB )); then
      return
    fi
    echo "GPU ${gpu}: ${free:-unknown} MiB free; waiting for ${MIN_FREE_MB} MiB."
    sleep 30
  done
}

complete() {
  local summary="$1" protocol="$2"
  [[ "${FORCE}" != "1" && -f "${summary}" ]] && \
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d.get("samples",0))==int(sys.argv[2]) and d.get("protocol")==sys.argv[3] else 1)' \
      "${summary}" "${LIMIT}" "${protocol}"
}

run_logged() {
  local name="$1" protocol="$2"
  shift 2
  local output="${OUTPUT_ROOT}/${name}" log="${OUTPUT_ROOT}/logs/${name//\//_}.log"
  if complete "${output}/eval_summary.json" "${protocol}"; then
    echo "SKIP complete ${name}"
    return
  fi
  mkdir -p "${output}"
  echo "RUN ${name}; log=${log}"
  if "$@" >"${log}" 2>&1; then
    echo "DONE ${name}"
  else
    local code="$?"
    echo "ERROR ${name} exited with ${code}; log=${log}" >&2
    tail -n 80 "${log}" >&2 || true
    return "${code}"
  fi
}

run_stamp_baselines() {
  local stage="$1" gpu="$2" timing_args=() backend=cpu
  [[ "${stage}" == timing ]] && timing_args+=(--record-timing) && backend=gpu
  run_logged "${stage}/stamp7b/baselines" paired_saved_output_postprocess_baselines_v1 \
    env CUDA_VISIBLE_DEVICES="${gpu}" "${STAMP_ENV_PATH}/bin/python" \
    -m training_free_refine.eval_postprocess_baselines \
    --source stamp --input-dir "${STAMP_INPUT}" \
    --output-dir "${OUTPUT_ROOT}/${stage}/stamp7b/baselines" \
    --model-label STAMP-7B --split-name refcoco_testA --limit "${LIMIT}" \
    --freeref-backend "${backend}" --methods base densecrf guided_filter slic_average freeref \
    "${timing_args[@]}"
}

run_text4seg_baselines() {
  local stage="$1" gpu="$2" timing_args=() backend=cpu
  [[ "${stage}" == timing ]] && timing_args+=(--record-timing) && backend=gpu
  run_logged "${stage}/text4seg_p24/baselines" paired_saved_output_postprocess_baselines_v1 \
    env CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
    python -m training_free_refine.eval_postprocess_baselines \
    --source text4seg --manifest "${TEXT4SEG_MANIFEST}" \
    --output-dir "${OUTPUT_ROOT}/${stage}/text4seg_p24/baselines" \
    --model-label Text4Seg-p24 --split-name refcoco_testA --limit "${LIMIT}" \
    --freeref-backend "${backend}" --methods base densecrf guided_filter slic_average freeref \
    "${timing_args[@]}"
}

run_stamp_sam() {
  local stage="$1" gpu="$2"
  wait_for_gpu "${gpu}"
  run_logged "${stage}/stamp7b/sam_h" stamp_official_frozen_sam_h_v1 \
    env CUDA_VISIBLE_DEVICES="${gpu}" "${STAMP_ENV_PATH}/bin/python" \
    -m training_free_refine.eval_stamp_sam_h \
    --input-dir "${STAMP_INPUT}" --stamp-code-dir "${STAMP_CODE_DIR}" \
    --sam-path "${SAM_PATH}" --output-dir "${OUTPUT_ROOT}/${stage}/stamp7b/sam_h" \
    --model-label STAMP-7B --split-name refcoco_testA --limit "${LIMIT}" \
    --save-visualizations 0
}

run_text4seg_sam() {
  local stage="$1" gpu="$2"
  wait_for_gpu "${gpu}"
  run_logged "${stage}/text4seg_p24/sam_h" text4seg_public_p24_paired_frozen_sam_h_v1 \
    env CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
    python -m training_free_refine.eval_text4seg_sam_h \
    --manifest "${TEXT4SEG_MANIFEST}" --text4seg-code-dir "${TEXT4SEG_CODE_DIR}" \
    --sam-path "${SAM_PATH}" --output-dir "${OUTPUT_ROOT}/${stage}/text4seg_p24/sam_h" \
    --model-label Text4Seg-p24 --split-name refcoco_testA --limit "${LIMIT}" \
    --save-visualizations 0
}

accuracy_stamp() {
  run_stamp_baselines accuracy "${GPU0}"
  run_stamp_sam accuracy "${GPU0}"
}
accuracy_text4seg() {
  run_text4seg_baselines accuracy "${GPU1}"
  run_text4seg_sam accuracy "${GPU1}"
}

echo "Stage 1/3: paired accuracy in parallel on H100 ${GPU0} and ${GPU1}"
accuracy_stamp & pid0=$!
accuracy_text4seg & pid1=$!
failed=0
wait "${pid0}" || failed=1
wait "${pid1}" || failed=1
(( failed == 0 )) || exit 1

echo "Stage 2/3: isolated post-processing timing, strictly serial on H100 ${GPU0}"
wait_for_gpu "${GPU0}"
run_stamp_baselines timing "${GPU0}"
wait_for_gpu "${GPU0}"
run_text4seg_baselines timing "${GPU0}"
run_stamp_sam timing "${GPU0}"
run_text4seg_sam timing "${GPU0}"

echo "Stage 3/3: combined comparison table"
"${STAMP_ENV_PATH}/bin/python" -m training_free_refine.summarize_postprocess_baselines \
  --input-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/combined"
touch "${OUTPUT_ROOT}/COMPLETE"
echo "Complete: ${OUTPUT_ROOT}/combined/postprocess_comparison.md"
