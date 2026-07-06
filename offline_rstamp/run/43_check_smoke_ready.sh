#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
MODEL_ROOT="${MODEL_ROOT:-${MLLM_SEG_ROOT}/models}"

echo "Checking smoke-training prerequisites..."

python - <<PY
import json, os
from pathlib import Path

root = Path("${MLLM_SEG_ROOT}")
stamp = Path("${STAMP_CODE_DIR}")
model_root = Path("${MODEL_ROOT}")

paths = [
    stamp / "train" / "main_uni.py",
    stamp / "playground" / "data" / "json_files_baseline",
    stamp / "playground" / "data" / "masks_baseline",
    stamp / "playground" / "data" / "json_files_rstamp",
    stamp / "playground" / "data" / "masks_rstamp",
]
for p in paths:
    print(("[OK] " if p.exists() else "[MISSING] ") + str(p))

for mode in ["baseline", "rstamp"]:
    p = stamp / "playground" / "data" / f"json_files_{mode}" / "refcoco_formatted_all_sentences_doubled_mp.json"
    if not p.exists():
        continue
    data = json.load(open(p))
    print(f"{mode}: {len(data)} refcoco samples")
    if data:
        item = data[0]
        print(f"  image exists: {os.path.exists(item['images'][0])}")
        print(f"  mask exists:  {os.path.exists(item['masks'][0])}")
        print(f"  has prior:    {'structured_prior_text' in item}")

for candidate in ["STAMP-2B-uni", "STAMP-7B-lora"]:
    p = model_root / candidate
    print(("[OK] " if p.exists() else "[MISSING] ") + str(p))
PY

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

echo "If main_uni.py has not been patched yet, run:"
echo "bash offline_rstamp/run/40_patch_stamp_local_training.sh"

