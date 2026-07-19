#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO="${TRAINING_FREE_REPO:-${ROOT}/MLLM-SEG-data}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${ROOT}/code/STAMP}"
SAM_PATH="${SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
CONDA_ENV="${STAMP_CONDA_ENV:-STAMP}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MIN_FREE_MB="${SAMH_MIN_FREE_MB:-12000}"
OUTPUT_ROOT="${SAMH_OUTPUT_ROOT:-${ROOT}/outputs}"
COMBINED_DIR="${SAMH_COMBINED_DIR:-${OUTPUT_ROOT}/frozen_samh_full_comparison}"

mkdir -p "${OUTPUT_ROOT}" "${COMBINED_DIR}"
if [[ ! -f "${SAM_PATH}" ]] || (( $(stat -c '%s' "${SAM_PATH}" 2>/dev/null || echo 0) < 2000000000 )); then
  echo "ERROR: complete SAM-H checkpoint not found: ${SAM_PATH}" >&2
  exit 1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"${OUTPUT_ROOT}/.frozen_samh_full_eval.lock"
  if ! flock -n 9; then
    echo "Another frozen SAM-H full evaluation already holds the lock." >&2
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

declare -a JOBS=(
  "STAMP-2B|refcoco_val|refcoco_val_full_stamp2b"
  "STAMP-2B|refcoco_testA|refcoco_testA_full_stamp2b"
  "STAMP-2B|refcoco_testB|refcoco_testB_full_stamp2b"
  "STAMP-2B|refcoco+_val|refcocoplus_val_full_stamp2b"
  "STAMP-2B|refcoco+_testA|refcocoplus_testA_full_stamp2b"
  "STAMP-2B|refcoco+_testB|refcocoplus_testB_full_stamp2b"
  "STAMP-2B|refcocog_val|refcocog_val_full"
  "STAMP-2B|refcocog_test|refcocog_test_full"
  "STAMP-7B|refcoco_val|refcoco_val_full_stamp7b"
  "STAMP-7B|refcoco_testA|refcoco_testA_full_stamp7b"
  "STAMP-7B|refcoco_testB|refcoco_testB_full_stamp7b"
  "STAMP-7B|refcoco+_val|refcocoplus_val_full_stamp7b"
  "STAMP-7B|refcoco+_testA|refcocoplus_testA_full_stamp7b"
  "STAMP-7B|refcoco+_testB|refcocoplus_testB_full_stamp7b"
  "STAMP-7B|refcocog_val|refcocog_val_full_stamp7b"
  "STAMP-7B|refcocog_test|refcocog_test_full_stamp7b"
)

cd "${REPO}"
for job in "${JOBS[@]}"; do
  IFS='|' read -r MODEL_LABEL SPLIT_NAME DUMP_NAME <<<"${job}"
  INPUT_DIR="${ROOT}/outputs/refine_stamp_dumps/${DUMP_NAME}"
  SAFE_SPLIT="${SPLIT_NAME//+/plus}"
  SAFE_MODEL="${MODEL_LABEL,,}"
  OUTPUT_DIR="${OUTPUT_ROOT}/frozen_samh_${SAFE_MODEL}_${SAFE_SPLIT}"
  if [[ ! -d "${INPUT_DIR}" ]] || ! find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.pt' -print -quit | grep -q .; then
    echo "SKIP missing dumps: ${MODEL_LABEL} ${SPLIT_NAME} (${INPUT_DIR})"
    continue
  fi
  if [[ -f "${OUTPUT_DIR}/eval_summary.json" ]]; then
    echo "SKIP complete: ${MODEL_LABEL} ${SPLIT_NAME}"
    continue
  fi
  echo "RUN ${MODEL_LABEL} ${SPLIT_NAME}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m training_free_refine.eval_stamp_sam_h \
      --input-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --stamp-code-dir "${STAMP_CODE_DIR}" \
      --sam-path "${SAM_PATH}" \
      --model-label "${MODEL_LABEL}" \
      --split-name "${SPLIT_NAME}" \
      --point-count 10 \
      --cascade-steps 2 \
      --seed 0 \
      --save-visualizations 8
done

PYTHONPATH="${REPO}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m training_free_refine.summarize_sam_h \
    --root "${ROOT}" \
    --output-dir "${COMBINED_DIR}"

echo "Frozen SAM-H full evaluation completed."
echo "Summary: ${COMBINED_DIR}/combined_summary.md"
