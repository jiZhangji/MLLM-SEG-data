#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LIST="${FREEREF_H100_GPUS:-0 1}"
SAMPLES="${EFFICIENCY_SAMPLES:-500}"
WARMUP="${EFFICIENCY_WARMUP:-20}"
SEED="${EFFICIENCY_SEED:-0}"
MIN_FREE_MB="${EFFICIENCY_MIN_FREE_MB:-30000}"
FORCE="${EFFICIENCY_FORCE:-0}"
OUTPUT_ROOT="${EFFICIENCY_OUTPUT_ROOT:-${ROOT}/outputs/freeref_efficiency_h100}"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
LISA_CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"

STAMP_CODE_DIR="${STAMP_CODE_DIR:-${ROOT}/code/STAMP}"
STAMP2B_MODEL="${STAMP2B_MODEL_PATH:-${ROOT}/models/STAMP-2B-uni}"
STAMP7B_MODEL="${STAMP7B_MODEL_PATH:-${ROOT}/models/STAMP-7B-lora}"
TEXT4SEG_CODE_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
TEXT4SEG_MODEL="${TEXT4SEG_P24_MODEL_PATH:-lmc22/text4seg-llava-7b-p24}"
VISION_TOWER_336="${TEXT4SEG_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
LISA_CODE_DIR="${LISA_CODE_DIR:-${ROOT}/code/third_party/lisa}"
LISA_MODEL="${LISA_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/lisa/LISA-7B-v1}"
LISA_VISION_TOWER="${LISA_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14}"
LISA_DATASET_DIR="${LISA_DATASET_DIR:-${ROOT}/data/lisa_paper_refer_seg}"
SAM_PATH="${SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
EVAL_JSON="${EFFICIENCY_EVAL_JSON:-${ROOT}/code/STAMP/playground/data/json_eval_baseline/refcoco_testA.json}"

read -r -a GPUS <<<"${GPU_LIST}"
if (( ${#GPUS[@]} != 2 )); then
  echo "ERROR: FREEREF_H100_GPUS must contain exactly two physical GPU indices." >&2
  exit 1
fi
GPU0="${GPUS[0]}"
GPU1="${GPUS[1]}"

for gpu in "${GPUS[@]}"; do
  name="$(nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader | head -n1)"
  if [[ "${name,,}" != *h100* ]]; then
    echo "ERROR: physical GPU ${gpu} is ${name}, not H100." >&2
    exit 1
  fi
done
for path in \
  "${STAMP_ENV_PATH}/bin/python" "${STAMP_CODE_DIR}" "${STAMP2B_MODEL}" "${STAMP7B_MODEL}" \
  "${TEXT4SEG_CODE_DIR}" "${VISION_TOWER_336}" "${LISA_CODE_DIR}" "${LISA_MODEL}" \
  "${LISA_VISION_TOWER}" "${LISA_DATASET_DIR}" "${SAM_PATH}" "${EVAL_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required benchmark input is missing: ${path}" >&2
    exit 1
  fi
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.freeref_efficiency_h100.lock"
  if ! flock -n 9; then
    echo "Another H100 efficiency suite is already running." >&2
    exit 0
  fi
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
  local summary="$1"
  [[ "${FORCE}" != "1" && -f "${summary}" ]] && \
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=int(d.get("samples",0))==int(sys.argv[2]) and int(d.get("warmup",-1))==int(sys.argv[3]) and "H100" in str(d.get("device","")); raise SystemExit(0 if ok else 1)' \
      "${summary}" "${SAMPLES}" "${WARMUP}"
}

run_logged() {
  local name="$1" gpu="$2"
  shift 2
  local output_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${OUTPUT_ROOT}/logs/${name}.log"
  if complete "${output_dir}/summary.json"; then
    echo "SKIP complete ${name}"
    return
  fi
  wait_for_gpu "${gpu}"
  mkdir -p "${output_dir}"
  echo "RUN ${name} on physical GPU ${gpu}; log=${log_path}"
  if env CUDA_VISIBLE_DEVICES="${gpu}" "$@" >"${log_path}" 2>&1; then
    echo "DONE ${name}"
  else
    local code="$?"
    echo "ERROR ${name} exited with ${code}; log=${log_path}" >&2
    tail -n 80 "${log_path}" >&2 || true
    return "${code}"
  fi
}

run_stamp_group() {
  local scale="$1" model="$2" gpu="$3" variant name
  for variant in base freeref_gpu sam_h freeref_sam_h; do
    name="stamp${scale}b_${variant}"
    run_logged "${name}" "${gpu}" \
      "${STAMP_ENV_PATH}/bin/python" -m efficiency_benchmark.run_stamp \
        --root "${ROOT}" --stamp-code-dir "${STAMP_CODE_DIR}" --model "${model}" \
        --method-label "STAMP-${scale}B" --eval-json "${EVAL_JSON}" --sam-path "${SAM_PATH}" \
        --output-dir "${OUTPUT_ROOT}/${name}" --variant "${variant}" \
        --warmup "${WARMUP}" --samples "${SAMPLES}" --seed "${SEED}" --allow-other-gpu
  done
}

run_text4seg_group() {
  local gpu="$1" variant name
  for variant in base freeref_gpu sam_h freeref_sam_h; do
    name="text4seg_${variant}"
    run_logged "${name}" "${gpu}" \
      conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
      python -m efficiency_benchmark.run_text4seg \
        --root "${ROOT}" --text4seg-code-dir "${TEXT4SEG_CODE_DIR}" \
        --model-path "${TEXT4SEG_MODEL}" --vision-tower "${VISION_TOWER_336}" \
        --eval-json "${EVAL_JSON}" --sam-path "${SAM_PATH}" \
        --output-dir "${OUTPUT_ROOT}/${name}" --variant "${variant}" \
        --warmup "${WARMUP}" --samples "${SAMPLES}" --seed "${SEED}" --allow-other-gpu
  done
}

echo "Stage 1/3: STAMP-2B and STAMP-7B timing on separate H100s"
run_stamp_group 2 "${STAMP2B_MODEL}" "${GPU0}" & pid0=$!
run_stamp_group 7 "${STAMP7B_MODEL}" "${GPU1}" & pid1=$!
failed=0
wait "${pid0}" || failed=1
wait "${pid1}" || failed=1
(( failed == 0 )) || exit 1

echo "Stage 2/3: Text4Seg timing and LISA timing"
run_text4seg_group "${GPU0}" & pid0=$!
run_logged "lisa_original" "${GPU1}" \
  conda run --no-capture-output -n "${LISA_CONDA_ENV}" \
  python -m efficiency_benchmark.run_lisa \
    --lisa-code-dir "${LISA_CODE_DIR}" --model-path "${LISA_MODEL}" \
    --vision-tower "${LISA_VISION_TOWER}" --sam-path "${SAM_PATH}" \
    --dataset-dir "${LISA_DATASET_DIR}" --output-dir "${OUTPUT_ROOT}/lisa_original" \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --seed "${SEED}" --allow-other-gpu & pid1=$!
failed=0
wait "${pid0}" || failed=1
wait "${pid1}" || failed=1
(( failed == 0 )) || exit 1

echo "Stage 3/3: building the extended efficiency table"
"${STAMP_ENV_PATH}/bin/python" -m efficiency_benchmark.summarize_extended \
  --input-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/efficiency_table.md"
echo "H100 efficiency suite complete: ${OUTPUT_ROOT}/efficiency_table.md"
