#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${SEGAGENT_OUTPUT_ROOT:-${ROOT}/outputs/segagent_freeref}"
SPLITS="${SEGAGENT_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"

printf '%-20s %10s %10s %10s\n' split official_json manifest summary
for split in ${SPLITS}; do
  base="${OUTPUT_ROOT}/${split//+/plus}"
  official="$([[ -n "$(find "${base}/official" -maxdepth 1 -type f -name '*newresults*.json' -print -quit 2>/dev/null)" ]] && echo yes || echo no)"
  printf '%-20s %10s %10s %10s\n' "${split}" "${official}" \
    "$([[ -f "${base}/import/manifest.jsonl" ]] && echo yes || echo no)" \
    "$([[ -f "${base}/freeref/eval_summary.json" ]] && echo yes || echo no)"
done
echo "Combined summary: $([[ -f "${OUTPUT_ROOT}/combined/comparison.md" ]] && echo yes || echo no)"
echo "Active processes:"
pgrep -af '[r]un_segagent_freeref_full_eval|run_segagent_official|[u]niversal_freeref.import_segagent|[u]niversal_freeref.evaluate.*segagent' || echo none
