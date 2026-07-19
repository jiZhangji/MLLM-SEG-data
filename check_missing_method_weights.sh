#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
STATUS_ROOT="${STATUS_ROOT:-${ROOT}/outputs/freeref_weight_download}"
STATUS_FILE="${STATUS_ROOT}/download_status.tsv"
MANUAL_FILE="${STATUS_ROOT}/manual_downloads.tsv"

echo "===== Active download ====="
pgrep -af '[d]ownload_missing_method_weights.sh|[h]f download|[g]down.*freeref_missing_methods|[m]odelscope.*download' || echo "none"

echo "===== Disk ====="
df -h "${ROOT}" | tail -n 1

echo "===== Latest artifact status ====="
if [[ -f "${STATUS_FILE}" ]]; then
  column -t -s $'\t' "${STATUS_FILE}" 2>/dev/null || cat "${STATUS_FILE}"
else
  echo "No status file yet: ${STATUS_FILE}"
fi

echo "===== Large files by method ====="
for method in hipie rela polyformer uninext pixellm lisa gsva read seg-zero segllm segagent shared; do
  directory="${WEIGHTS_ROOT}/${method}"
  count="$(find -L "${directory}" -type f -size +50M 2>/dev/null | wc -l)"
  bytes="$(find -L "${directory}" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s+0}')"
  awk -v method="${method}" -v count="${count}" -v bytes="${bytes}" \
    'BEGIN {printf "%-12s files>50MB: %-4d size: %.2f GiB\n", method, count, bytes/1073741824}'
done

echo "===== Manual or blocked items ====="
if [[ -f "${MANUAL_FILE}" && "$(wc -l < "${MANUAL_FILE}")" -gt 1 ]]; then
  column -t -s $'\t' "${MANUAL_FILE}" 2>/dev/null || cat "${MANUAL_FILE}"
else
  echo "none"
fi

echo "===== Log tail ====="
LOG_FILE="${LOG_FILE:-${ROOT}/outputs/freeref_missing_method_weights.log}"
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 30 "${LOG_FILE}"
else
  echo "No log yet: ${LOG_FILE}"
fi
