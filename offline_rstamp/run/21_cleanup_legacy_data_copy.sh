#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
DATA_ROOT="${DATA_ROOT:-${MLLM_SEG_ROOT}/data}"
LEGACY_DATA="${MLLM_SEG_ROOT}/MLLM-SEG-data/data"

echo "DATA_ROOT=${DATA_ROOT}"
echo "LEGACY_DATA=${LEGACY_DATA}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: canonical DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

for path in \
  "${DATA_ROOT}/annotations/refcoco_family" \
  "${DATA_ROOT}/datasets/reasonseg" \
  "${DATA_ROOT}/shared/coco/train2014"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: canonical data seems incomplete, missing: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${LEGACY_DATA}" ]]; then
  echo "No legacy data directory found. Nothing to remove."
  exit 0
fi

echo "Legacy data size:"
du -sh "${LEGACY_DATA}" || true

echo
echo "About to delete ONLY this legacy data copy:"
echo "${LEGACY_DATA}"
echo
echo "This keeps the MLLM-SEG-data Git repository itself."
echo "Type DELETE to continue:"
read -r answer

if [[ "${answer}" != "DELETE" ]]; then
  echo "Abort."
  exit 1
fi

rm -rf "${LEGACY_DATA}"
echo "Removed: ${LEGACY_DATA}"

