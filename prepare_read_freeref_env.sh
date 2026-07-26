#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READ_DIR="${READ_DIR:-${ROOT}/code/third_party/read}"
READ_REVISION="${READ_REVISION:-254924020ec4d33ace8575eb1e2a515390530594}"
CONDA_ENV="${READ_CONDA_ENV:-read-freeref}"
CONDA_BIN="${CONDA_BIN:-conda}"
HOST_TAG="${FREEREF_HOST_TAG:-$(hostname)}"
HOST_TAG="${HOST_TAG//[^A-Za-z0-9_.-]/_}"
READY_MARKER="${ROOT}/.cache/read-freeref-env-v1.${HOST_TAG}.ready"

METHODS=read ROOT="${ROOT}" TARGET_ROOT="${ROOT}/code/third_party" \
  bash "${SCRIPT_DIR}/prepare_universal_freeref_repos.sh"
git -C "${READ_DIR}" fetch --depth 1 origin "${READ_REVISION}"
git -C "${READ_DIR}" checkout --detach "${READ_REVISION}"
METHODS=read ROOT="${ROOT}" DOWNLOAD_DATASETS=0 \
  bash "${SCRIPT_DIR}/download_missing_method_weights.sh" || true

if ! "${CONDA_BIN}" env list --json | python -c \
  'import json,os,sys; name=sys.argv[1]; paths=json.load(sys.stdin)["envs"]; raise SystemExit(0 if any(os.path.basename(p)==name for p in paths) else 1)' \
  "${CONDA_ENV}"; then
  "${CONDA_BIN}" create -y -n "${CONDA_ENV}" python=3.9 pip=23.3 setuptools=68.2.2 wheel
fi
if [[ ! -f "${READY_MARKER}" ]]; then
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
    numpy==1.23.5 transformers==4.31.0 tokenizers==0.13.3 \
    accelerate==0.21.0 peft==0.4.0 sentencepiece==0.1.99 \
    einops==0.4.1 opencv-python-headless==4.8.0.76 \
    Pillow==10.4.0 pycocotools==2.0.7 scipy==1.10.1 \
    scikit-image==0.21.0 scikit-learn==1.3.2 \
    matplotlib==3.7.5 tqdm==4.67.1 requests==2.32.3 \
    safetensors==0.4.5 protobuf==4.25.5
fi
PYTHONPATH="${READ_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
import torch
import transformers
from model.READ import READForCausalLM
from dataloaders.test_dataset import TestReferDataset

print(
    "READ environment ready:",
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "transformers", transformers.__version__,
    "model", READForCausalLM.__name__,
    "dataset", TestReferDataset.__name__,
)
PY
mkdir -p "$(dirname "${READY_MARKER}")"
touch "${READY_MARKER}"
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_read_freeref_data.sh"
echo "READ source, full public checkpoint, runtime, and standard evaluation data are ready."
