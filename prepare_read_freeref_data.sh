#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${READ_REFER_SOURCE_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
DATA_ROOT="${READ_DATA_ROOT:-${ROOT}/data/read_eval}"
REFER_ROOT="${DATA_ROOT}/refer_seg"

if [[ ! -d "${SOURCE_ROOT}/refcoco" ]]; then
  MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_lisa_paper_data.sh"
fi
link_once() {
  local source="$1" target="$2"
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "$(readlink -f "${source}")" == "$(readlink -f "${target}")" ]]; then return; fi
    echo "ERROR: refusing to replace existing READ data path: ${target}" >&2
    exit 2
  fi
  ln -s "${source}" "${target}"
}
for dataset in refcoco refcoco+ refcocog; do
  mkdir -p "${REFER_ROOT}/${dataset}"
  link_once "${SOURCE_ROOT}/${dataset}/instances.json" "${REFER_ROOT}/${dataset}/instances.json"
done
for dataset in refcoco refcoco+; do
  source_refs="${SOURCE_ROOT}/${dataset}/refs(unc).p"
  [[ -f "${source_refs}" ]] || { echo "ERROR: missing ${source_refs}" >&2; exit 2; }
  link_once "${source_refs}" "${REFER_ROOT}/${dataset}/refs(unc).p"
  # READ/SESAME gives the standard annotations this internal split-by label.
  link_once "${source_refs}" "${REFER_ROOT}/${dataset}/refs(unc_exclude_unified).p"
done
source_refs="${SOURCE_ROOT}/refcocog/refs(umd).p"
[[ -f "${source_refs}" ]] || { echo "ERROR: missing ${source_refs}" >&2; exit 2; }
link_once "${source_refs}" "${REFER_ROOT}/refcocog/refs(umd).p"
link_once "${source_refs}" "${REFER_ROOT}/refcocog/refs(umd_exclude_unified).p"
link_once "${SOURCE_ROOT}/images/mscoco/images/train2014" \
  "${REFER_ROOT}/images/mscoco/images/train2014"

echo "READ standard RefCOCO evaluation data is ready: ${REFER_ROOT}"
echo "The *_exclude_unified names are READ's internal aliases for the standard annotation files."
