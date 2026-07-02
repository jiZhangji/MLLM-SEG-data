#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"

python - <<PY
from pathlib import Path

root = Path("${STAMP_CODE_DIR}")
targets = [
    root / "train" / "main_uni.py",
    root / "model" / "modeling_qwen2_vl.py",
    root / "model" / "qwen_changes.py",
]

for path in targets:
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_no_flash")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    new = text.replace("flash_attention_2", "sdpa")
    if new != text:
        path.write_text(new, encoding="utf-8")
        print(f"[PATCHED] {path}")
    else:
        print(f"[OK] no flash_attention_2 in {path}")
PY

echo "Remaining flash_attention_2 references:"
grep -R "flash_attention_2" -n "${STAMP_CODE_DIR}/train" "${STAMP_CODE_DIR}/model" || true

