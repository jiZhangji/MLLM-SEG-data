#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
RESULTS_ROOT="${LISA_RESULTS_ROOT:-${ROOT}/outputs/lisa_official}"
REFINE_ROOT="${LISA_FREEREF_ROOT:-${ROOT}/outputs/universal_freeref_lisa}"
SPLITS="${LISA_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
EVAL_JSON_DIR="${LISA_EVAL_JSON_DIR:-${ROOT}/code/STAMP/playground/data/json_eval_baseline}"

printf '%-20s %12s %10s %10s\n' "split" "logits" "manifest" "summary"
for split in ${SPLITS}; do
  safe_split="${split//+/plus}"
  expected="?"
  eval_json="${EVAL_JSON_DIR}/${split}.json"
  if [[ -f "${eval_json}" ]]; then
    expected="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${eval_json}")"
  fi
  count="$(find "${RESULTS_ROOT}/${safe_split}/pred_logits" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)"
  manifest="$([[ -f "${RESULTS_ROOT}/${safe_split}/manifest.jsonl" ]] && echo yes || echo no)"
  summary="$([[ -f "${REFINE_ROOT}/${safe_split}/eval_summary.json" ]] && echo yes || echo no)"
  printf '%-20s %6s/%-5s %10s %10s\n' "${split}" "${count}" "${expected}" "${manifest}" "${summary}"
done

echo
echo "Combined summary: $([[ -f "${REFINE_ROOT}/combined/comparison.md" ]] && echo yes || echo no)"
echo "Active processes:"
pgrep -af '[r]un_lisa_freeref_eval|[e]xport_lisa_masks|universal_freeref.evaluate.*lisa' || echo "none"
