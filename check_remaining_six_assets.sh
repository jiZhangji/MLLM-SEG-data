#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MLLM_SEG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
METHODS="rela polyformer uninext lisa gsva read"
DOWNLOAD_STATUS_ROOT="${FREEREF_DOWNLOAD_STATUS_ROOT:-${ROOT}/outputs/freeref_weight_download}"

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
echo "===== Disk ====="
df -h "${ROOT}" | tail -n 1
echo "===== Official repositories and weights ====="
printf '%-12s %-10s %12s %12s\n' method repository files_gt_50m weight_gib
for method in ${METHODS}; do
  repository="${ROOT}/code/third_party/${method}"
  weights="${ROOT}/models/freeref_missing_methods/${method}"
  repo_state=missing
  [[ -d "${repository}/.git" ]] && repo_state=complete
  count="$(find -L "${weights}" -type f -size +50M 2>/dev/null | wc -l)"
  bytes="$(find -L "${weights}" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s+0}')"
  gib="$(awk -v bytes="${bytes}" 'BEGIN {printf "%.2f", bytes/1073741824}')"
  printf '%-12s %-10s %12s %12s\n' "${method}" "${repo_state}" "${count}" "${gib}"
done
echo "===== Selected download status ====="
if [[ -f "${DOWNLOAD_STATUS_ROOT}/download_status.tsv" ]]; then
  awk -F '\t' 'NR == 1 || $1 ~ /^(rela|polyformer|uninext|lisa|gsva|read)$/' \
    "${DOWNLOAD_STATUS_ROOT}/download_status.tsv" |
    { column -t -s $'\t' 2>/dev/null || cat; }
else
  echo "No download status file: ${DOWNLOAD_STATUS_ROOT}/download_status.tsv"
fi
echo "===== Manual blockers ====="
if [[ -f "${DOWNLOAD_STATUS_ROOT}/manual_downloads.tsv" ]]; then
  awk -F '\t' 'NR == 1 || $1 ~ /^(rela|polyformer|uninext|lisa|gsva|read)$/' \
    "${DOWNLOAD_STATUS_ROOT}/manual_downloads.tsv" |
    { column -t -s $'\t' 2>/dev/null || cat; }
else
  echo "No manual queue."
fi
echo "===== Active preparation ====="
pgrep -af '[p]repare_remaining_six_assets|[d]ownload_missing_method_weights|[g]it clone' || echo none
