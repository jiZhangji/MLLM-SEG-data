#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SPLIT="${LISA_PROMPT_SPLIT:-refcoco_testA}"
LIMIT="${LISA_PROMPT_LIMIT:-16}"
OFFSET="${LISA_PROMPT_OFFSET:-0}"
safe_split="${SPLIT//+/plus}"
if [[ "${LIMIT}" == "0" ]]; then
  run_tag="full_o${OFFSET}"
else
  run_tag="n${LIMIT}_o${OFFSET}"
fi
OUTPUT_DIR="${LISA_PROMPT_OUTPUT_DIR:-${ROOT}/outputs/lisa_freeref_prompt/${safe_split}_${run_tag}}"

completed="$(find "${OUTPUT_DIR}/metadata" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
echo "LISA FreeRef-Prompt ${SPLIT}: ${completed}/${LIMIT} artifacts"
echo "summary: $([[ -f "${OUTPUT_DIR}/eval_summary.json" ]] && echo yes || echo no)"
echo "output: ${OUTPUT_DIR}"
echo ""
echo "Active processes:"
pgrep -af 'run_lisa_freeref_prompt_eval|export_lisa_freeref_prompt' || echo "none"
if [[ -f "${OUTPUT_DIR}/comparison.md" ]]; then
  echo ""
  cat "${OUTPUT_DIR}/comparison.md"
fi
