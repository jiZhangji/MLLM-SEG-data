#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
POLYFORMER_DIR="${POLYFORMER_DIR:-${ROOT}/code/third_party/polyformer}"
CONDA_ENV="${POLYFORMER_CONDA_ENV:-polyformer-freeref}"
CONDA_BIN="${CONDA_BIN:-conda}"

if [[ ! -f "${POLYFORMER_DIR}/evaluate.py" || ! -f "${POLYFORMER_DIR}/fairseq/setup.py" ]]; then
  echo "ERROR: official PolyFormer source is missing: ${POLYFORMER_DIR}" >&2
  echo "Run METHODS=polyformer bash prepare_universal_freeref_repos.sh first." >&2
  exit 1
fi

if ! "${CONDA_BIN}" env list --json | python -c \
  'import json,os,sys; name=sys.argv[1]; paths=json.load(sys.stdin)["envs"]; raise SystemExit(0 if any(os.path.basename(p)==name for p in paths) else 1)' \
  "${CONDA_ENV}"; then
  "${CONDA_BIN}" create -y -n "${CONDA_ENV}" python=3.8 pip=23.3 setuptools=68.2.2 wheel
fi

echo "Installing the H100/H200-compatible PolyFormer inference runtime into ${CONDA_ENV}"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
  torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
  numpy==1.23.5 scipy==1.10.1 scikit-image==0.21.0 \
  opencv-python-headless==4.10.0.84 Pillow==10.4.0 \
  timm==0.6.13 ftfy==6.0.3 tensorboardX==2.6.2.2 \
  pycocotools==2.0.7 einops==0.8.1 tqdm==4.67.1 \
  hydra-core==1.0.7 omegaconf==2.0.6 regex sacrebleu==1.5.1 \
  bitarray==2.9.3 cython==0.29.37 cffi

env -u CUDA_HOME MAX_JOBS="${MAX_JOBS:-4}" "${CONDA_BIN}" run -n "${CONDA_ENV}" \
  python -m pip install --no-deps --no-build-isolation -e "${POLYFORMER_DIR}/fairseq"

(
  cd "${POLYFORMER_DIR}"
  PYTHONPATH="${POLYFORMER_DIR}:${PYTHONPATH:-}" "${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
import cv2
import fairseq
import numpy
import skimage
import torch
import torchvision
from models.polyformer import PolyFormerModel

print(
    "PolyFormer environment ready:",
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "numpy", numpy.__version__,
    "opencv", cv2.__version__,
)
PY
)

echo "Environment preparation completed. No evaluation was launched."
