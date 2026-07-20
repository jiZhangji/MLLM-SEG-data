#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${LISA_PAPER_OUTPUT_ROOT:-${ROOT}/outputs/lisa_paper_reproduction}"

printf '%-28s %9s %9s %9s %9s\n' "split" "samples" "cIoU" "paper" "match"
for split_spec in \
  'refcoco|unc|val' 'refcoco|unc|testA' 'refcoco|unc|testB' \
  'refcoco+|unc|val' 'refcoco+|unc|testA' 'refcoco+|unc|testB' \
  'refcocog|umd|val' 'refcocog|umd|test'; do
  slug="${split_spec//|/_}"
  summary="${OUTPUT_ROOT}/${slug}/paper_reproduction_summary.json"
  if [[ -f "${summary}" ]]; then
    python - "${split_spec}" "${summary}" <<'PY'
import json
import sys

split, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    value = json.load(handle)
match = value.get("paper_match")
match_text = "yes" if match is True else "no" if match is False else "partial"
print(
    f"{split:<28} {value.get('samples', 0):>9} "
    f"{value.get('cIoU_percent', 0.0):>9.2f} "
    f"{value.get('paper_cIoU_percent', 0.0):>9.2f} {match_text:>9}"
)
PY
  else
    printf '%-28s %9s %9s %9s %9s\n' "${split_spec}" "-" "-" "-" "no"
  fi
done

echo
echo "Active processes:"
pgrep -af 'run_lisa_paper_reproduction|eval_lisa_paper_protocol' || echo "none"
