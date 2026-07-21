#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${PIXELLM_OUTPUT_ROOT:-${ROOT}/outputs/pixellm_public_freeref}"
SPLITS="${PIXELLM_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
EVAL_JSON_ROOT="${ROOT}/code/STAMP/playground/data/json_eval_baseline"

printf '%-20s %15s %10s %10s\n' split logits manifest summary
for split in ${SPLITS}; do
  base="${OUTPUT_ROOT}/${split//+/plus}"
  expected="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${EVAL_JSON_ROOT}/${split}.json" 2>/dev/null || echo '?')"
  count="$(find "${base}/official/pred_logits" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)"
  printf '%-20s %7s/%-7s %10s %10s\n' "${split}" "${count}" "${expected}" \
    "$([[ -f "${base}/official/manifest.jsonl" ]] && echo yes || echo no)" \
    "$([[ -f "${base}/freeref/eval_summary.json" ]] && echo yes || echo no)"
done
echo "Combined summary: $([[ -f "${OUTPUT_ROOT}/combined/comparison.md" ]] && echo yes || echo no)"
echo "Active processes:"
pgrep -af '[r]un_pixellm_freeref_full_eval|[e]xport_pixellm_masks|[u]niversal_freeref.evaluate.*pixellm' || echo none
