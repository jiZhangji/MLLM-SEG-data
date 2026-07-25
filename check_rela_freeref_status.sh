#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
MODEL_ROOT="${RELA_CLASSIC_MODEL_ROOT:-${ROOT}/models/freeref_missing_methods/rela/classic}"
OUTPUT_ROOT="${RELA_FULL_OUTPUT_ROOT:-${ROOT}/outputs/rela_freeref_full}"

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== ReLA training checkpoints ====="
for dataset in refcoco refcocoplus refcocog; do
  path="${MODEL_ROOT}/${dataset}/model_final.pth"
  printf '%-14s %-8s %s\n' "${dataset}" "$([[ -s "${path}" ]] && echo complete || echo missing)" "${path}"
done
echo "===== ReLA eight-split evaluation ====="
for split in \
  refcoco_val refcoco_testA refcoco_testB \
  refcocoplus_val refcocoplus_testA refcocoplus_testB \
  refcocog_val refcocog_test; do
  base="${OUTPUT_ROOT}/${split}"
  printf '%-22s official=%-3s manifest=%-3s summary=%-3s\n' "${split}" \
    "$([[ -s "${base}/official/inference/ref_seg_predictions.pth" ]] && echo yes || echo no)" \
    "$([[ -s "${base}/imported/manifest.jsonl" ]] && echo yes || echo no)" \
    "$([[ -s "${base}/freeref/eval_summary.json" ]] && echo yes || echo no)"
done
echo "===== Active ReLA processes ====="
pgrep -af '[t]rain_rela|[r]un_rela|[t]rain_net.py.*referring_swin_base' || echo none
echo "===== Latest logs ====="
find "${ROOT}/outputs" -path '*rela*logs/*.log' -type f -printf '%T@ %p\n' 2>/dev/null |
  sort -nr | head -n 4 | cut -d' ' -f2- | while read -r log; do
    echo "--- ${log}"
    tail -n 12 "${log}" || true
  done
