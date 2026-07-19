#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 MANIFEST.jsonl [MANIFEST2.jsonl ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/universal_freeref}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}"
cd "${SCRIPT_DIR}"

summaries=()
for manifest in "$@"; do
  manifest="$(realpath "${manifest}")"
  read -r method split < <(
    "${PYTHON_BIN}" - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    row = next(json.loads(line) for line in handle if line.strip() and not line.lstrip().startswith("#"))
print(str(row.get("method", "unknown")).replace(" ", "_"), str(row.get("split", "unknown")).replace(" ", "_"))
PY
  )
  output_dir="${OUTPUT_ROOT}/${method}_${split}"
  echo "RUN ${method} ${split} -> ${output_dir}"
  # EXTRA_ARGS is intentionally shell-expanded to support optional evaluator flags.
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -m universal_freeref.evaluate \
    --manifest "${manifest}" \
    --output-dir "${output_dir}" \
    ${EXTRA_ARGS}
  summaries+=("${method}_${split}=${output_dir}/eval_summary.json")
done

summary_args=()
for summary in "${summaries[@]}"; do
  summary_args+=(--summary "${summary}")
done
"${PYTHON_BIN}" -m universal_freeref.summarize \
  "${summary_args[@]}" \
  --output-dir "${OUTPUT_ROOT}/combined"

echo "Universal FreeRef evaluation complete: ${OUTPUT_ROOT}/combined/comparison.md"
