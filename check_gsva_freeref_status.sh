#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_ROOT="${GSVA_FULL_OUTPUT_ROOT:-${ROOT}/outputs/gsva_freeref_full}"
MODEL="${GSVA_MLLM_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/gsva/LLaVA-Lightning-7B-v1-1-merged}"

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== GSVA merged licensed base ====="
if [[ -f "${MODEL}/.freeref_merge_complete" && -f "${MODEL}/config.json" ]] &&
   find "${MODEL}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) \
     -size +100M -print -quit | grep -q .; then
  echo "complete ${MODEL}"
else
  echo "missing  ${MODEL}"
fi
echo "===== GSVA eight-split evaluation ====="
for split in \
  refcoco_val refcoco_testA refcoco_testB \
  refcocoplus_val refcocoplus_testA refcocoplus_testB \
  refcocog_val refcocog_test; do
  base="${OUTPUT_ROOT}/${split}"
  printf '%-22s manifest=%-3s summary=%-3s\n' "${split}" \
    "$([[ -s "${base}/official_export/manifest.jsonl" ]] && echo yes || echo no)" \
    "$([[ -s "${base}/freeref/eval_summary.json" ]] && echo yes || echo no)"
done
echo "===== Active GSVA processes ====="
pgrep -af '[r]un_gsva|[m]ain.py.*gsva-7b-ft-res|[d]eepspeed.*main.py' || echo none
