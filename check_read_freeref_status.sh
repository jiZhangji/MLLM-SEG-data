#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${READ_FULL_OUTPUT_ROOT:-${ROOT}/outputs/read_freeref_full}"
echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== READ eight-split evaluation ====="
for split in \
  refcoco_val refcoco_testA refcoco_testB \
  refcocoplus_val refcocoplus_testA refcocoplus_testB \
  refcocog_val refcocog_test; do
  base="${OUTPUT_ROOT}/${split}"
  printf '%-22s manifest=%-3s summary=%-3s\n' "${split}" \
    "$([[ -s "${base}/official_export/manifest.jsonl" ]] && echo yes || echo no)" \
    "$([[ -s "${base}/freeref/eval_summary.json" ]] && echo yes || echo no)"
done
echo "===== Active READ processes ====="
pgrep -af '[r]un_read|[e]xport_read_masks' || echo none
