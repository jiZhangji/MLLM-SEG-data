#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
LIMIT="${POLYFORMER_LIMIT:-64}"
OFFSET="${POLYFORMER_OFFSET:-0}"
OUTPUT_ROOT="${POLYFORMER_OUTPUT_ROOT:-${ROOT}/outputs/polyformer_freeref_smoke_n${LIMIT}_o${OFFSET}}"

count_files() {
  local directory="$1" pattern="$2"
  find "${directory}" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | wc -l
}

echo "PolyFormer-L RefCOCO testA smoke:"
echo "  requested: ${LIMIT} expressions (offset ${OFFSET})"
echo "  masks: $(count_files "${OUTPUT_ROOT}/official/pred_masks" '*.png')/${LIMIT}"
echo "  export summary: $([[ -f "${OUTPUT_ROOT}/official/export_summary.json" ]] && echo yes || echo no)"
echo "  FreeRef summary: $([[ -f "${OUTPUT_ROOT}/freeref/eval_summary.json" ]] && echo yes || echo no)"
echo "  comparison: $([[ -f "${OUTPUT_ROOT}/comparison.md" ]] && echo yes || echo no)"
if [[ -f "${OUTPUT_ROOT}/comparison.md" ]]; then
  echo
  cat "${OUTPUT_ROOT}/comparison.md"
fi
echo
echo "Active processes:"
pgrep -af '[r]un_polyformer_freeref_smoke|[e]xport_polyformer_masks|[p]repare_polyformer_eval_data' || echo none
