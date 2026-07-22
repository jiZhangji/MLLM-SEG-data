#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAMPLES="${EFFICIENCY_SAMPLES:-500}"
WARMUP="${EFFICIENCY_WARMUP:-20}"
SEED="${EFFICIENCY_SEED:-0}"
MIN_FREE_MB="${EFFICIENCY_MIN_FREE_MB:-20000}"
FORCE="${EFFICIENCY_FORCE:-0}"
OUTPUT_ROOT="${EFFICIENCY_OUTPUT_ROOT:-${ROOT}/outputs/freeref_efficiency_4090}"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
LISA_CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"

STAMP_CODE_DIR="${STAMP_CODE_DIR:-${ROOT}/code/STAMP}"
STAMP_MODEL="${STAMP7B_MODEL_PATH:-${ROOT}/models/STAMP-7B-lora}"
TEXT4SEG_CODE_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
TEXT4SEG_MODEL="${TEXT4SEG_P24_MODEL_PATH:-lmc22/text4seg-llava-7b-p24}"
VISION_TOWER_336="${TEXT4SEG_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
LISA_CODE_DIR="${LISA_CODE_DIR:-${ROOT}/code/third_party/lisa}"
LISA_MODEL="${LISA_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/lisa/LISA-7B-v1}"
LISA_VISION_TOWER="${LISA_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14}"
LISA_DATASET_DIR="${LISA_DATASET_DIR:-${ROOT}/data/lisa_paper_refer_seg}"
SAM_PATH="${SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
EVAL_JSON="${EFFICIENCY_EVAL_JSON:-${ROOT}/code/STAMP/playground/data/json_eval_baseline/refcoco_testA.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.freeref_efficiency_4090.lock"
  if ! flock -n 9; then
    echo "Another unified 4090 efficiency benchmark is running." >&2
    exit 0
  fi
fi

GPU_NAME="$(nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=name --format=csv,noheader | head -n1)"
if [[ "${GPU_NAME}" != *4090* ]]; then
  echo "ERROR: GPU ${CUDA_DEVICE} is ${GPU_NAME}; the paper protocol requires RTX 4090." >&2
  exit 1
fi
FREE_MB="$(nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
if [[ -z "${FREE_MB}" ]] || (( FREE_MB < MIN_FREE_MB )); then
  echo "ERROR: GPU ${CUDA_DEVICE} has ${FREE_MB:-unknown} MiB free; need ${MIN_FREE_MB} MiB." >&2
  exit 1
fi

for path in "${STAMP_CODE_DIR}" "${STAMP_MODEL}" "${TEXT4SEG_CODE_DIR}" "${VISION_TOWER_336}" "${LISA_CODE_DIR}" "${LISA_MODEL}" "${LISA_VISION_TOWER}" "${LISA_DATASET_DIR}" "${SAM_PATH}" "${EVAL_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required benchmark input is missing: ${path}" >&2
    exit 1
  fi
done

complete() {
  local summary="$1"
  [[ "${FORCE}" != "1" && -f "${summary}" ]] && \
    python -c 'import json,sys; raise SystemExit(0 if int(json.load(open(sys.argv[1])).get("samples",0)) == int(sys.argv[2]) else 1)' "${summary}" "${SAMPLES}"
}

run_logged() {
  local name="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${OUTPUT_ROOT}/logs/${name}.log"
  if complete "${output_dir}/summary.json"; then
    echo "SKIP complete ${name}"
    return
  fi
  echo "RUN ${name}; log=${log_path}"
  mkdir -p "${output_dir}"
  "$@" >"${log_path}" 2>&1
  echo "DONE ${name}"
}

cd "${SCRIPT_DIR}"
for variant in base freeref_gpu sam_h; do
  run_logged "stamp7b_${variant}" \
    "${STAMP_ENV_PATH}/bin/python" -m efficiency_benchmark.run_stamp \
      --root "${ROOT}" --stamp-code-dir "${STAMP_CODE_DIR}" --model "${STAMP_MODEL}" \
      --eval-json "${EVAL_JSON}" --sam-path "${SAM_PATH}" \
      --output-dir "${OUTPUT_ROOT}/stamp7b_${variant}" --variant "${variant}" \
      --warmup "${WARMUP}" --samples "${SAMPLES}" --seed "${SEED}"
done

for variant in base freeref_gpu; do
  run_logged "text4seg_${variant}" \
    conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
    python -m efficiency_benchmark.run_text4seg \
      --root "${ROOT}" --text4seg-code-dir "${TEXT4SEG_CODE_DIR}" \
      --model-path "${TEXT4SEG_MODEL}" --vision-tower "${VISION_TOWER_336}" \
      --eval-json "${EVAL_JSON}" --sam-path "${SAM_PATH}" \
      --output-dir "${OUTPUT_ROOT}/text4seg_${variant}" --variant "${variant}" \
      --warmup "${WARMUP}" --samples "${SAMPLES}" --seed "${SEED}"
done

run_logged "lisa_original" \
  conda run --no-capture-output -n "${LISA_CONDA_ENV}" \
  python -m efficiency_benchmark.run_lisa \
    --lisa-code-dir "${LISA_CODE_DIR}" --model-path "${LISA_MODEL}" \
    --vision-tower "${LISA_VISION_TOWER}" --sam-path "${SAM_PATH}" \
    --dataset-dir "${LISA_DATASET_DIR}" --output-dir "${OUTPUT_ROOT}/lisa_original" \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --seed "${SEED}"

python -m efficiency_benchmark.summarize \
  --input-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/efficiency_table.md"
echo "Unified RTX 4090 benchmark complete: ${OUTPUT_ROOT}/efficiency_table.md"
