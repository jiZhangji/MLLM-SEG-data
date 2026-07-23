#!/usr/bin/env bash
set -u

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
LIMIT="${SAM_VIT_B_SAMPLES:-500}"
OUTPUT_ROOT="${SAM_VIT_B_OUTPUT_ROOT:-${ROOT}/outputs/sam_vit_b_n${LIMIT}}"

echo "Time: $(date -u '+%F %T UTC')"
echo "Complete: $([[ -f "${OUTPUT_ROOT}/COMPLETE" ]] && echo yes || echo no)"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo
for stage in accuracy timing; do
  for model in stamp7b text4seg_p24; do
    summary="${OUTPUT_ROOT}/${stage}/${model}/eval_summary.json"
    if [[ -f "${summary}" ]]; then
      samples="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("samples",0))' "${summary}" 2>/dev/null || echo '?')"
      status="complete (${samples})"
    else
      status=pending
    fi
    printf '%-34s %s\n' "${stage}/${model}" "${status}"
  done
done
echo
echo "Active processes:"
pgrep -af '[r]un_sam_vit_b_n500|eval_stamp_sam_h|eval_text4seg_sam_h' || echo none
echo
for log in "${OUTPUT_ROOT}"/logs/*.log; do
  [[ -f "${log}" ]] || continue
  line="$(tr '\r' '\n' <"${log}" | grep -E '[0-9]+%|Traceback|ERROR' | tail -n 1 || true)"
  [[ -n "${line}" ]] && printf '%s: %s\n' "$(basename "${log}")" "${line}"
done
if [[ -f "${OUTPUT_ROOT}/combined/sam_variant_comparison.md" ]]; then
  echo
  cat "${OUTPUT_ROOT}/combined/sam_variant_comparison.md"
fi
