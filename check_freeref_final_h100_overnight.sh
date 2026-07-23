#!/usr/bin/env bash
set -u

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-${ROOT}/outputs/freeref_final_h100_overnight}"

echo "Time: $(date -u '+%F %T UTC')"
echo "Stage: $(cat "${OUTPUT_ROOT}/current_stage.txt" 2>/dev/null || echo not-started)"
echo "Complete: $([[ -f "${OUTPUT_ROOT}/COMPLETE" ]] && echo yes || echo no)"
echo
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo

count_summaries() {
  local root="$1"
  [[ -d "${root}" ]] || { echo 0; return; }
  find "${root}" -type f -name eval_summary.json | wc -l
}

printf '%-38s %s\n' "SAM-B full splits" "$(count_summaries "${OUTPUT_ROOT}/sam_b")/16"
printf '%-38s %s\n' "Ablation/hyperparameter summaries" "$(count_summaries "${OUTPUT_ROOT}/studies")/55"
printf '%-38s %s\n' "Full post-processing summaries" "$(count_summaries "${OUTPUT_ROOT}/postprocess")/16"
timing_count=0
[[ -d "${OUTPUT_ROOT}/timing" ]] && timing_count="$(find "${OUTPUT_ROOT}/timing" -type f -name summary.json | wc -l)"
printf '%-38s %s\n' "Strictly serial timing summaries" "${timing_count}/18"
echo
echo "Active processes:"
pgrep -af '[r]un_freeref_final_h100_overnight|eval_stamp_sam_h|eval_stamp_dumps|eval_text4seg_outputs|eval_postprocess_baselines|efficiency_benchmark.run_' || echo none
echo
echo "Latest progress per active log:"
for log in "${OUTPUT_ROOT}"/logs/*.log; do
  [[ -f "${log}" ]] || continue
  line="$(tr '\r' '\n' <"${log}" | grep -E '[0-9]+%|Traceback|ERROR|DONE' | tail -n 1 || true)"
  [[ -n "${line}" ]] && printf '%s: %s\n' "$(basename "${log}")" "${line}"
done | tail -n 30
echo
if [[ -f "${OUTPUT_ROOT}/combined/timing/k_timing.md" ]]; then
  cat "${OUTPUT_ROOT}/combined/timing/k_timing.md"
elif [[ -f "${OUTPUT_ROOT}/combined/postprocess/postprocess_full.md" ]]; then
  cat "${OUTPUT_ROOT}/combined/postprocess/postprocess_full.md"
elif [[ -f "${OUTPUT_ROOT}/combined/sam_b/sam_full.md" ]]; then
  cat "${OUTPUT_ROOT}/combined/sam_b/sam_full.md"
fi
