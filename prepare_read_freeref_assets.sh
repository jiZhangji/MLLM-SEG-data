#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MLLM_SEG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_read_freeref_env.sh"
