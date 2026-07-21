#!/usr/bin/env bash
set -u

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${LISA_OFFICIAL_FREEREF_SAM_OUTPUT_ROOT:-${ROOT}/outputs/lisa_official_freeref_before_sam}"

printf '%-25s %12s %12s %12s %12s %9s\n' "split" "baseline" "FreeRef" "+SAM-H" "FreeRef+SAM" "summary"
for spec in \
  'refcoco|unc|val 10834' 'refcoco|unc|testA 5657' 'refcoco|unc|testB 5095' \
  'refcoco+|unc|val 10758' 'refcoco+|unc|testA 5726' 'refcoco+|unc|testB 4889' \
  'refcocog|umd|val 4896' 'refcocog|umd|test 9602'; do
  read -r split expected <<<"${spec}"
  slug="${split//|/_}"
  out="${OUTPUT_ROOT}/${slug}"
  baseline="$(find "${out}/baseline_masks" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l || true)"
  freeref="$(find "${out}/freeref_masks" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l || true)"
  baseline_sam="$(find "${out}/baseline_sam_masks" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l || true)"
  freeref_sam="$(find "${out}/freeref_sam_masks" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l || true)"
  summary="$([[ -f "${out}/eval_summary.json" ]] && echo yes || echo no)"
  printf '%-25s %5s/%-6s %5s/%-6s %5s/%-6s %5s/%-6s %9s\n' \
    "${split}" "${baseline}" "${expected}" "${freeref}" "${expected}" \
    "${baseline_sam}" "${expected}" "${freeref_sam}" "${expected}" "${summary}"
done

echo "Combined summary: $([[ -f "${OUTPUT_ROOT}/combined/combined_summary.md" ]] && echo yes || echo no)"
echo
echo "Active processes:"
pgrep -af '[r]un_lisa_official_freeref_sam_h100|[e]val_lisa_official_freeref_sam' || echo none
