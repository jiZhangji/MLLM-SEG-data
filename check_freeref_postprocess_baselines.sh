#!/usr/bin/env bash
set -u

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
LIMIT="${FREEREF_BASELINE_SAMPLES:-500}"
OUTPUT_ROOT="${FREEREF_BASELINE_OUTPUT_ROOT:-${ROOT}/outputs/freeref_postprocess_baselines_n${LIMIT}}"

echo "Time: $(date -u '+%F %T UTC')"
echo "Output: ${OUTPUT_ROOT}"
echo "Complete: $([[ -f "${OUTPUT_ROOT}/COMPLETE" ]] && echo yes || echo no)"
echo
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo
printf '%-42s %s\n' task status
for stage in accuracy timing; do
  for model in stamp7b text4seg_p24; do
    for branch in baselines sam_h; do
      task="${stage}/${model}/${branch}"
      summary="${OUTPUT_ROOT}/${task}/eval_summary.json"
      if [[ -f "${summary}" ]]; then
        samples="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("samples",0))' "${summary}" 2>/dev/null || echo '?')"
        status="complete (${samples})"
      else
        status=pending
      fi
      printf '%-42s %s\n' "${task}" "${status}"
    done
  done
done
echo
echo "Active processes:"
pgrep -af '[r]un_freeref_postprocess_baselines|eval_postprocess_baselines|eval_stamp_sam_h|eval_text4seg_sam_h' || echo none
echo
echo "Latest progress:"
for log in "${OUTPUT_ROOT}"/logs/*.log; do
  [[ -f "${log}" ]] || continue
  line="$(tr '\r' '\n' <"${log}" | grep -E '[0-9]+%|Traceback|ERROR' | tail -n 1 || true)"
  [[ -n "${line}" ]] && printf '%s: %s\n' "$(basename "${log}")" "${line}"
done
echo
if [[ -f "${OUTPUT_ROOT}/combined/postprocess_comparison.md" ]]; then
  cat "${OUTPUT_ROOT}/combined/postprocess_comparison.md"
fi
