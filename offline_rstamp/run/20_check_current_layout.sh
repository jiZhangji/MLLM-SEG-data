#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
DATA_ROOT="${DATA_ROOT:-${MLLM_SEG_ROOT}/data}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
RSTAMP_CODE_DIR="${RSTAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/R-STAMP}"
MODEL_ROOT="${MODEL_ROOT:-${MLLM_SEG_ROOT}/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MLLM_SEG_ROOT}/outputs}"
LOG_ROOT="${LOG_ROOT:-${MLLM_SEG_ROOT}/logs}"

echo "MLLM_SEG_ROOT=${MLLM_SEG_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "STAMP_CODE_DIR=${STAMP_CODE_DIR}"
echo "RSTAMP_CODE_DIR=${RSTAMP_CODE_DIR}"
echo "MODEL_ROOT=${MODEL_ROOT}"

mkdir -p "${MLLM_SEG_ROOT}/code" "${MODEL_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}"

required_dirs=(
  "${DATA_ROOT}"
  "${DATA_ROOT}/annotations/refcoco_family"
  "${DATA_ROOT}/annotations/grefcoco"
  "${DATA_ROOT}/datasets/reasonseg"
  "${DATA_ROOT}/shared/coco/train2014"
  "${DATA_ROOT}/shared/coco/annotations"
  "${STAMP_CODE_DIR}"
)

for path in "${required_dirs[@]}"; do
  if [[ -d "${path}" ]]; then
    echo "[OK] ${path}"
  else
    echo "[MISSING] ${path}"
  fi
done

echo
echo "Canonical data tree:"
find "${DATA_ROOT}" -maxdepth 3 -type d | sort

echo
echo "STAMP key files:"
for file in \
  "${STAMP_CODE_DIR}/run_seg_ref.py" \
  "${STAMP_CODE_DIR}/data/create_refcoco_new.py" \
  "${STAMP_CODE_DIR}/train/main_uni.py" \
  "${STAMP_CODE_DIR}/dataset/refer_seg_dataset.py"; do
  if [[ -f "${file}" ]]; then
    echo "[OK] ${file}"
  else
    echo "[MISSING] ${file}"
  fi
done

echo
echo "Legacy data copy:"
if [[ -d "${MLLM_SEG_ROOT}/MLLM-SEG-data/data" ]]; then
  du -sh "${MLLM_SEG_ROOT}/MLLM-SEG-data/data" || true
  echo "Legacy copy exists. If DATA_ROOT is already valid, you may remove it with:"
  echo "bash ${MLLM_SEG_ROOT}/MLLM-SEG-data/offline_rstamp/run/21_cleanup_legacy_data_copy.sh"
else
  echo "[OK] no legacy data copy at ${MLLM_SEG_ROOT}/MLLM-SEG-data/data"
fi

