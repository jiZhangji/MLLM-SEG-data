#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSVA_DIR="${GSVA_DIR:-${ROOT}/code/third_party/gsva}"
GSVA_REVISION="${GSVA_REVISION:-618dc7957cd239442d1ecd45c8d575cf91b9ca8e}"
CONDA_ENV="${GSVA_CONDA_ENV:-gsva-freeref}"
CONDA_BIN="${CONDA_BIN:-conda}"
READY_MARKER="${ROOT}/.cache/gsva-freeref-env-v1.ready"

METHODS=gsva ROOT="${ROOT}" TARGET_ROOT="${ROOT}/code/third_party" \
  bash "${SCRIPT_DIR}/prepare_universal_freeref_repos.sh"
git -C "${GSVA_DIR}" fetch --depth 1 origin "${GSVA_REVISION}"
git -C "${GSVA_DIR}" checkout --detach "${GSVA_REVISION}"
METHODS=gsva ROOT="${ROOT}" DOWNLOAD_DATASETS=0 \
  bash "${SCRIPT_DIR}/download_missing_method_weights.sh" || true

if ! "${CONDA_BIN}" env list --json | python -c \
  'import json,os,sys; name=sys.argv[1]; paths=json.load(sys.stdin)["envs"]; raise SystemExit(0 if any(os.path.basename(p)==name for p in paths) else 1)' \
  "${CONDA_ENV}"; then
  "${CONDA_BIN}" create -y -n "${CONDA_ENV}" python=3.10 pip=24.2 setuptools=75 wheel
fi
if [[ ! -f "${READY_MARKER}" ]]; then
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
  env DS_BUILD_OPS=0 \
    "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
    bitsandbytes==0.43.3 deepspeed==0.15.0 \
    numpy==1.26.4 transformers==4.44.2 tokenizers==0.19.1 \
    accelerate==0.34.2 peft==0.4.0 sentencepiece==0.2.0 \
    einops==0.4.1 opencv-python-headless==4.10.0.84 \
    Pillow==10.4.0 pycocotools==2.0.7 scipy==1.13.1 \
    scikit-image==0.24.0 scikit-learn==1.5.2 \
    PyYAML==6.0.2 termcolor==2.4.0 timm==0.6.13 \
    tqdm==4.67.1 protobuf==5.28.3 ninja==1.11.1.1
fi
PYTHONPATH="${GSVA_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
import deepspeed
import torch
import transformers
from model import LisaGSVAForCausalLM

print(
    "GSVA environment ready:",
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "transformers", transformers.__version__,
    "deepspeed", deepspeed.__version__,
    "model", LisaGSVAForCausalLM.__name__,
)
PY
mkdir -p "$(dirname "${READY_MARKER}")"
touch "${READY_MARKER}"
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_gsva_freeref_data.sh"
echo "GSVA source, public weights, runtime, and evaluation data links are ready."
