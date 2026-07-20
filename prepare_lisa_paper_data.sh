#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
DATA_ROOT="${LISA_PAPER_DATA_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
COCO_TRAIN2014="${LISA_COCO_TRAIN2014:-${ROOT}/data/shared/coco/train2014}"

ANNOTATION_ROOTS=(
  "${ROOT}/LH/STAMP/playground/data/refer_seg"
  "${ROOT}/code/STAMP/playground/data/refer_seg"
  "${ROOT}/code/Text4Seg/ms-swift/data/refer_seg"
)
if [[ -n "${LISA_REFER_ANNOTATION_ROOT:-}" ]]; then
  ANNOTATION_ROOTS=("${LISA_REFER_ANNOTATION_ROOT}" "${ANNOTATION_ROOTS[@]}")
fi

link_once() {
  local source="$1"
  local target="$2"
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "$(readlink -f "${source}")" == "$(readlink -f "${target}")" ]]; then
      return
    fi
    echo "ERROR: refusing to replace existing LISA data path: ${target}" >&2
    exit 2
  fi
  ln -s "${source}" "${target}"
}

find_annotation() {
  local dataset="$1"
  local filename="$2"
  local root
  for root in "${ANNOTATION_ROOTS[@]}"; do
    if [[ -f "${root}/${dataset}/${filename}" ]]; then
      printf '%s\n' "${root}/${dataset}/${filename}"
      return 0
    fi
  done
  return 1
}

mkdir -p "${DATA_ROOT}"
for dataset in refcoco refcoco+ refcocog; do
  mkdir -p "${DATA_ROOT}/${dataset}"
  instances="$(find_annotation "${dataset}" instances.json || true)"
  if [[ -z "${instances}" ]]; then
    echo "ERROR: ${dataset}/instances.json was not found in the known annotation roots." >&2
    exit 2
  fi
  link_once "${instances}" "${DATA_ROOT}/${dataset}/instances.json"

  found_refs=0
  for refs_name in 'refs(unc).p' 'refs(umd).p' 'refs(google).p'; do
    refs="$(find_annotation "${dataset}" "${refs_name}" || true)"
    if [[ -n "${refs}" ]]; then
      link_once "${refs}" "${DATA_ROOT}/${dataset}/${refs_name}"
      found_refs=1
    fi
  done
  if (( found_refs == 0 )); then
    echo "ERROR: no REFER split annotation was found for ${dataset}." >&2
    exit 2
  fi
done

if [[ ! -d "${COCO_TRAIN2014}" ]]; then
  echo "ERROR: COCO train2014 directory is missing: ${COCO_TRAIN2014}" >&2
  exit 2
fi
link_once "${COCO_TRAIN2014}" "${DATA_ROOT}/images/mscoco/images/train2014"

echo "LISA official data layout is ready: ${DATA_ROOT}"
find "${DATA_ROOT}" -maxdepth 4 -type l -printf '%p -> %l\n' | sort
