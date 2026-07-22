#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${EFFICIENCY_OUTPUT_ROOT:-${ROOT}/outputs/freeref_efficiency_4090}"
SAMPLES="${EFFICIENCY_SAMPLES:-500}"

printf '%-25s %10s %12s\n' "experiment" "samples" "status"
for name in stamp7b_base stamp7b_freeref_gpu stamp7b_sam_h text4seg_base text4seg_freeref_gpu lisa_original; do
  summary="${OUTPUT_ROOT}/${name}/summary.json"
  if [[ -f "${summary}" ]]; then
    count="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("samples",0)))' "${summary}")"
    status="partial"
    [[ "${count}" == "${SAMPLES}" ]] && status="complete"
    printf '%-25s %10s %12s\n' "${name}" "${count}/${SAMPLES}" "${status}"
  else
    printf '%-25s %10s %12s\n' "${name}" "0/${SAMPLES}" "pending"
  fi
done

echo
echo "Active processes:"
pgrep -af 'efficiency_benchmark.run_(stamp|text4seg|lisa)|run_freeref_efficiency_4090' || echo "none"
echo
echo "Latest log:"
latest="$(find "${OUTPUT_ROOT}/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -n "${latest}" ]]; then
  echo "${latest}"
  tail -n 20 "${latest}"
else
  echo "none"
fi
