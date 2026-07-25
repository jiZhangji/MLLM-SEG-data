#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MLLM_SEG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
METHODS="rela polyformer uninext lisa gsva read"
STATUS_ROOT="${REMAINING_ASSET_STATUS_ROOT:-${ROOT}/outputs/freeref_remaining_six_assets}"
DOWNLOAD_STATUS_ROOT="${FREEREF_DOWNLOAD_STATUS_ROOT:-${ROOT}/outputs/freeref_weight_download}"
MIN_FREE_GB="${MIN_FREE_GB:-160}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${ROOT}/code/third_party" "${ROOT}/models/freeref_missing_methods" \
  "${ROOT}/outputs" "${STATUS_ROOT}"

echo "Preparing official code and released checkpoints for: ${METHODS}"
echo "Root: ${ROOT}"

ROOT="${ROOT}" \
TARGET_ROOT="${ROOT}/code/third_party" \
METHODS="${METHODS}" \
DRY_RUN="${DRY_RUN}" \
  bash "${SCRIPT_DIR}/prepare_universal_freeref_repos.sh"
repo_status="$?"

ROOT="${ROOT}" \
STATUS_ROOT="${DOWNLOAD_STATUS_ROOT}" \
METHODS="${METHODS}" \
DOWNLOAD_DATASETS=0 \
MIN_FREE_GB="${MIN_FREE_GB}" \
DRY_RUN="${DRY_RUN}" \
  bash "${SCRIPT_DIR}/download_missing_method_weights.sh"
download_status="$?"

report="${STATUS_ROOT}/asset_status.tsv"
printf 'method\trepository\tlarge_weight_files\tweight_gib\tmanual_items\n' > "${report}"
incomplete=0
for method in ${METHODS}; do
  repository="${ROOT}/code/third_party/${method}"
  weights="${ROOT}/models/freeref_missing_methods/${method}"
  repo_state=missing
  [[ -d "${repository}/.git" ]] && repo_state=complete
  count="$(find -L "${weights}" -type f -size +50M 2>/dev/null | wc -l)"
  bytes="$(find -L "${weights}" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s+0}')"
  gib="$(awk -v bytes="${bytes}" 'BEGIN {printf "%.2f", bytes/1073741824}')"
  manual=0
  if [[ -f "${DOWNLOAD_STATUS_ROOT}/manual_downloads.tsv" ]]; then
    manual="$(awk -F '\t' -v method="${method}" 'NR > 1 && $1 == method {n++} END {print n+0}' \
      "${DOWNLOAD_STATUS_ROOT}/manual_downloads.tsv")"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${method}" "${repo_state}" "${count}" "${gib}" "${manual}" >> "${report}"
  if [[ "${repo_state}" != complete || "${count}" -eq 0 || "${manual}" -gt 0 ]]; then
    incomplete=1
  fi
done

column -t -s $'\t' "${report}" 2>/dev/null || cat "${report}"
echo
echo "Downloader details: ${DOWNLOAD_STATUS_ROOT}/download_status.tsv"
echo "Manual queue: ${DOWNLOAD_STATUS_ROOT}/manual_downloads.tsv"
echo "Six-method report: ${report}"

if (( repo_status != 0 || download_status != 0 || incomplete != 0 )); then
  echo "Some official assets remain unavailable or require manual authorization." >&2
  echo "GSVA requires a licensed Vicuna/LLaVA merged base; UNINEXT may require a manually obtained checkpoint." >&2
  if [[ "${ALLOW_INCOMPLETE}" != 1 ]]; then
    exit 3
  fi
fi

echo "All automatically obtainable assets have been prepared."
