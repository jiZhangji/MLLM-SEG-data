#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
PAPER_OUTPUT_ROOT="${LISA_PAPER_OUTPUT_ROOT:-${ROOT}/outputs/lisa_paper_reproduction}"
FREEREF_OUTPUT_ROOT="${LISA_PAPER_FREEREF_OUTPUT_ROOT:-${ROOT}/outputs/lisa_paper_freeref}"

printf '%-28s %9s %9s %9s %9s %9s\n' "split" "samples" "paper" "match" "FreeRef" "delta"
for split_spec in \
  'refcoco|unc|val' 'refcoco|unc|testA' 'refcoco|unc|testB' \
  'refcoco+|unc|val' 'refcoco+|unc|testA' 'refcoco+|unc|testB' \
  'refcocog|umd|val' 'refcocog|umd|test'; do
  slug="${split_spec//|/_}"
  paper="${PAPER_OUTPUT_ROOT}/${slug}/paper_reproduction_summary.json"
  freeref="${FREEREF_OUTPUT_ROOT}/${slug}/eval_summary.json"
  python - "${split_spec}" "${paper}" "${freeref}" <<'PY'
import json
import sys
from pathlib import Path

split, paper_path, freeref_path = sys.argv[1:]
paper_file, freeref_file = Path(paper_path), Path(freeref_path)
paper = json.loads(paper_file.read_text(encoding="utf-8")) if paper_file.is_file() else {}
free = json.loads(freeref_file.read_text(encoding="utf-8")) if freeref_file.is_file() else {}
match = paper.get("paper_match")
match_text = "yes" if match is True else "no" if match is False else "-"
samples = paper.get("samples", "-")
paper_ciou = f'{paper["cIoU_percent"]:.2f}' if "cIoU_percent" in paper else "-"
free_ciou = f'{free["refined_cIoU"] * 100:.2f}' if "refined_cIoU" in free else "-"
delta = f'{free["cIoU_delta"] * 100:+.2f}' if "cIoU_delta" in free else "-"
print(f"{split:<28} {str(samples):>9} {paper_ciou:>9} {match_text:>9} {free_ciou:>9} {delta:>9}")
PY
done

echo
echo "Active processes:"
pgrep -af 'run_lisa_paper_freeref_eval|run_lisa_paper_reproduction|eval_lisa_paper_protocol|universal_freeref.evaluate' || echo "none"
