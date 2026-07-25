#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${GSVA_REFER_SOURCE_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
GREF_SOURCE="${GSVA_GREF_SOURCE_ROOT:-${ROOT}/data/annotations/grefcoco}"
DATA_ROOT="${GSVA_DATA_ROOT:-${ROOT}/data/gsva_eval}"
TARGET="${DATA_ROOT}/refer_seg"

if [[ ! -d "${SOURCE_ROOT}/refcoco" ]]; then
  MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_lisa_paper_data.sh"
fi
for required in \
  "${SOURCE_ROOT}/refcoco/instances.json" \
  "${SOURCE_ROOT}/refcoco+/instances.json" \
  "${SOURCE_ROOT}/refcocog/instances.json" \
  "${SOURCE_ROOT}/images/mscoco/images/train2014"; do
  [[ -e "${required}" ]] || { echo "ERROR: missing GSVA data prerequisite: ${required}" >&2; exit 2; }
done
[[ -f "${GREF_SOURCE}/instances.json" ]] || {
  echo "ERROR: GSVA's official main.py constructs gRefCOCO even for standard RES evaluation." >&2
  echo "Missing ${GREF_SOURCE}/instances.json; prepare the canonical gRefCOCO annotations first." >&2
  exit 2
}
if [[ ! -f "${GREF_SOURCE}/grefs(unc).p" &&
      ! -f "${GREF_SOURCE}/grefs(unc).json" ]]; then
  echo "ERROR: missing grefs(unc).p/json below ${GREF_SOURCE}." >&2
  exit 2
fi
mkdir -p "${DATA_ROOT}"
if [[ -e "${TARGET}" || -L "${TARGET}" ]]; then
  if [[ "$(readlink -f "${TARGET}")" != "$(readlink -f "${SOURCE_ROOT}")" ]]; then
    echo "ERROR: refusing to replace existing GSVA data path: ${TARGET}" >&2
    exit 2
  fi
else
  ln -s "${SOURCE_ROOT}" "${TARGET}"
fi
if [[ -e "${TARGET}/grefcoco" || -L "${TARGET}/grefcoco" ]]; then
  if [[ "$(readlink -f "${TARGET}/grefcoco")" != "$(readlink -f "${GREF_SOURCE}")" ]]; then
    echo "ERROR: refusing to replace existing GSVA gRefCOCO path: ${TARGET}/grefcoco" >&2
    exit 2
  fi
else
  ln -s "${GREF_SOURCE}" "${TARGET}/grefcoco"
fi
echo "GSVA evaluation data layout is ready: ${TARGET} -> $(readlink -f "${TARGET}")"
echo "GSVA auxiliary gRefCOCO link: ${TARGET}/grefcoco -> $(readlink -f "${TARGET}/grefcoco")"
