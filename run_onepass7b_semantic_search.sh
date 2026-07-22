#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=""
EVAL_JSON="../code/STAMP/playground/data/json_eval_baseline/refcocog_val.json"
STAMP_CODE_DIR="../code/STAMP"
DATA_ROOT=".."
OUTPUT_DIR="../outputs/onepass7b_semantic_search_val200"
DEVICE="cuda"
LIMIT=200
BATCH_SIZE=8
NUM_WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --eval-json) EVAL_JSON="$2"; shift 2 ;;
    --stamp-code-dir) STAMP_CODE_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "--checkpoint is required" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
ALPHAS=(-2 -1 -0.5 -0.25 -0.1 0 0.1 0.25 0.5 1 2)

for MODE in strict_query semantic_anchor; do
  MODE_DIR="${OUTPUT_DIR}/${MODE}"
  python -m onepass_qwen7b.eval \
    --stamp-code-dir "${STAMP_CODE_DIR}" \
    --eval-json "${EVAL_JSON}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${MODE_DIR}" \
    --limit "${LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --prompt-mode "${MODE}" \
    --seg-fusion-sweep "${ALPHAS[@]}"
done

python - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for mode in ("strict_query", "semantic_anchor"):
    summary = json.loads((root / mode / "eval_summary.json").read_text())
    for item in summary["seg_fusion_sweep"]:
        rows.append({"prompt_mode": mode, **item})
rows.sort(key=lambda item: (item["cIoU"], item["gIoU"]), reverse=True)
with (root / "semantic_search.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "semantic_search.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("\nBest semantic configurations:")
for item in rows[:10]:
    print(
        f"{item['prompt_mode']:15s} alpha={item['alpha']:>5} "
        f"gIoU={item['gIoU']:.6f} cIoU={item['cIoU']:.6f}"
    )
print(f"\nFull table: {root / 'semantic_search.csv'}")
PY
