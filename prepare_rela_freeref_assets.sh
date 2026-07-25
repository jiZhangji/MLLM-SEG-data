#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}" \
  bash "${SCRIPT_DIR}/prepare_rela_freeref_env.sh"
