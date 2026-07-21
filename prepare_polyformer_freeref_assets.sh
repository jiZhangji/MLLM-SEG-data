#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MLLM_SEG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
POLYFORMER_DIR="${POLYFORMER_DIR:-${ROOT}/code/third_party/polyformer}"
POLYFORMER_REVISION="${POLYFORMER_REVISION:-69fc728b2ec6a2b3595ec34db64074badcb19151}"
METHODS=polyformer bash "${SCRIPT_DIR}/prepare_universal_freeref_repos.sh"
git -C "${POLYFORMER_DIR}" fetch --depth 1 origin "${POLYFORMER_REVISION}"
git -C "${POLYFORMER_DIR}" checkout --detach "${POLYFORMER_REVISION}"
METHODS=polyformer DOWNLOAD_DATASETS=0 bash "${SCRIPT_DIR}/download_missing_method_weights.sh"
bash "${SCRIPT_DIR}/prepare_polyformer_freeref_env.sh"
echo "PolyFormer source ${POLYFORMER_REVISION}, checkpoints, BERT vocabulary, and isolated environment are ready."
