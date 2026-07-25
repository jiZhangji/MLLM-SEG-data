#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SOURCE_ROOT="${RELA_REFER_SOURCE_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
DATA_ROOT="${RELA_DATA_ROOT:-${ROOT}/data/rela_refer}"
IMAGE_ROOT="${RELA_IMAGE_ROOT:-${ROOT}/data/shared/coco/train2014}"

link_once() {
  local source="$1" target="$2"
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "$(readlink -f "${source}")" == "$(readlink -f "${target}")" ]]; then
      return
    fi
    echo "ERROR: refusing to replace existing ReLA data path: ${target}" >&2
    exit 2
  fi
  ln -s "${source}" "${target}"
}

if [[ ! -d "${SOURCE_ROOT}/refcoco" ]]; then
  MLLM_SEG_ROOT="${ROOT}" bash "$(dirname "${BASH_SOURCE[0]}")/prepare_lisa_paper_data.sh"
fi
for dataset in refcoco refcoco+ refcocog; do
  for required in instances.json; do
    [[ -f "${SOURCE_ROOT}/${dataset}/${required}" ]] || {
      echo "ERROR: missing ${SOURCE_ROOT}/${dataset}/${required}" >&2
      exit 2
    }
  done
  link_once "${SOURCE_ROOT}/${dataset}" "${DATA_ROOT}/${dataset}"
done
[[ -d "${IMAGE_ROOT}" ]] || {
  echo "ERROR: COCO train2014 directory is missing: ${IMAGE_ROOT}" >&2
  exit 2
}
link_once "${IMAGE_ROOT}" "${DATA_ROOT}/images/train2014"

echo "ReLA standard REFER data layout is ready: ${DATA_ROOT}"
find "${DATA_ROOT}" -maxdepth 3 -type l -printf '%p -> %l\n' | sort
