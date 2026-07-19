#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/universal_freeref}"

printf "%-34s %10s %10s\n" "run" "rows" "summary"
if [[ -d "${OUTPUT_ROOT}" ]]; then
  while IFS= read -r directory; do
    name="$(basename "${directory}")"
    [[ "${name}" == "combined" ]] && continue
    rows=0
    if [[ -f "${directory}/eval_rows.csv" ]]; then
      rows=$(( $(wc -l < "${directory}/eval_rows.csv") - 1 ))
    elif [[ -d "${directory}/row_cache" ]]; then
      rows=$(find "${directory}/row_cache" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
    fi
    summary="no"
    [[ -f "${directory}/eval_summary.json" ]] && summary="yes"
    printf "%-34s %10s %10s\n" "${name}" "${rows}" "${summary}"
  done < <(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
else
  echo "No output directory: ${OUTPUT_ROOT}"
fi

echo
echo "Active processes:"
if command -v pgrep >/dev/null 2>&1; then
  pgrep -af 'universal_freeref.evaluate|run_universal_freeref_eval' || echo "none"
else
  echo "process inspection unavailable (pgrep not installed)"
fi

echo
echo "Combined summary:"
if [[ -f "${OUTPUT_ROOT}/combined/comparison.md" ]]; then
  cat "${OUTPUT_ROOT}/combined/comparison.md"
else
  echo "not available"
fi
