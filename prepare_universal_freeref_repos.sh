#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET_ROOT="${TARGET_ROOT:-${ROOT}/code/third_party}"
PYTHON_BIN="${PYTHON_BIN:-python}"
METHODS="${METHODS:-all}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${TARGET_ROOT}"

while IFS=$'\t' read -r method_id repo_url repo_status; do
  if [[ "${METHODS}" != "all" && " ${METHODS} " != *" ${method_id} "* ]]; then
    continue
  fi
  target="${TARGET_ROOT}/${method_id}"
  if [[ -d "${target}/.git" ]]; then
    echo "SKIP ${method_id}: already cloned at ${target}"
    continue
  fi
  echo "CLONE ${method_id}: ${repo_url}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    git clone --depth 1 "${repo_url}" "${target}"
  fi
done < <(
  ROOT_FOR_REGISTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["ROOT_FOR_REGISTRY"]) / "universal_freeref" / "methods.json"
for method in json.loads(path.read_text(encoding="utf-8")):
    if method.get("repo_url"):
        print(f"{method['id']}\t{method['repo_url']}\t{method['repo_status']}")
PY
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; no repositories were cloned."
else
  echo "Upstream source preparation complete: ${TARGET_ROOT}"
fi
echo "Checkpoints and per-repository environments are intentionally not installed by this script."
