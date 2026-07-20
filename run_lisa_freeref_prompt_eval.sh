#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FREEREF_REPO:-${SCRIPT_DIR}}"
LISA_DIR="${LISA_DIR:-${ROOT}/code/third_party/lisa}"
CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"
WEIGHTS_ROOT="${FREEREF_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
MODEL_PATH="${LISA_MODEL_PATH:-${WEIGHTS_ROOT}/lisa/LISA-7B-v1}"
VISION_TOWER="${LISA_VISION_TOWER:-${WEIGHTS_ROOT}/shared/clip-vit-large-patch14}"
EVAL_JSON_DIR="${LISA_EVAL_JSON_DIR:-${ROOT}/code/STAMP/playground/data/json_eval_baseline}"
SPLIT="${LISA_PROMPT_SPLIT:-refcoco_testA}"
LIMIT="${LISA_PROMPT_LIMIT:-16}"
OFFSET="${LISA_PROMPT_OFFSET:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PRECISION="${LISA_PRECISION:-bf16}"
MAX_EXPRESSIONS_PER_CALL="${LISA_MAX_EXPRESSIONS_PER_CALL:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"

safe_split="${SPLIT//+/plus}"
if [[ "${LIMIT}" == "0" ]]; then
  run_tag="full_o${OFFSET}"
else
  run_tag="n${LIMIT}_o${OFFSET}"
fi
OUTPUT_DIR="${LISA_PROMPT_OUTPUT_DIR:-${ROOT}/outputs/lisa_freeref_prompt/${safe_split}_${run_tag}}"
EVAL_JSON="${EVAL_JSON_DIR}/${SPLIT}.json"

export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/outputs" "${OUTPUT_DIR}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.lisa_freeref_prompt_${CUDA_DEVICE}.lock"
  if ! flock -n 9; then
    echo "Another LISA FreeRef-Prompt job already uses GPU ${CUDA_DEVICE}." >&2
    exit 0
  fi
fi

for required in \
  "${LISA_DIR}/model/LISA.py" \
  "${MODEL_PATH}/config.json" \
  "${VISION_TOWER}/config.json" \
  "${EVAL_JSON}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required local file is missing: ${required}" >&2
    exit 1
  fi
done

if ! conda run -n "${CONDA_ENV}" python -c \
  'import torch, transformers, scipy, skimage; assert torch.cuda.is_available()' \
  >/dev/null 2>&1; then
  echo "ERROR: conda environment ${CONDA_ENV} is unavailable or incomplete." >&2
  echo "Run run_lisa_freeref_eval.sh once with LISA_SETUP_ENV=1 to prepare it." >&2
  exit 1
fi

while true; do
  free_mb="$(nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
  if [[ -n "${free_mb}" && "${free_mb}" -ge "${MIN_FREE_MB}" ]]; then
    break
  fi
  echo "GPU ${CUDA_DEVICE} free: ${free_mb:-unknown} MiB; waiting 10 seconds..."
  sleep 10
done

echo "LISA FreeRef-Prompt quick evaluation"
echo "split=${SPLIT} limit=${LIMIT} offset=${OFFSET} gpu=${CUDA_DEVICE}"
echo "output=${OUTPUT_DIR}"
cd "${REPO}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.export_lisa_freeref_prompt \
    --lisa-code-dir "${LISA_DIR}" \
    --model-path "${MODEL_PATH}" \
    --vision-tower "${VISION_TOWER}" \
    --eval-json "${EVAL_JSON}" \
    --data-root "${ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --precision "${PRECISION}" \
    --max-expressions-per-call "${MAX_EXPRESSIONS_PER_CALL}" \
    --limit "${LIMIT}" \
    --offset "${OFFSET}" \
    --seed 0

PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.evaluate_lisa_freeref_prompt \
    --manifest "${OUTPUT_DIR}/manifest.jsonl" \
    --output-dir "${OUTPUT_DIR}"

echo "LISA FreeRef-Prompt evaluation completed."
echo "Summary: ${OUTPUT_DIR}/eval_summary.json"
cat "${OUTPUT_DIR}/comparison.md"
