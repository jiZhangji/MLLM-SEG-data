#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELA_DIR="${RELA_DIR:-${ROOT}/code/third_party/rela}"
RELA_REVISION="${RELA_REVISION:-3ca955198da8ace68f8980b6fcbada0791cc51c2}"
DETECTRON2_DIR="${RELA_DETECTRON2_DIR:-${ROOT}/code/third_party/detectron2-v0.6}"
CONDA_ENV="${RELA_CONDA_ENV:-rela-freeref}"
CONDA_BIN="${CONDA_BIN:-conda}"
WEIGHTS_ROOT="${RELA_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
SWIN_PTH="${RELA_SWIN_PTH:-${WEIGHTS_ROOT}/rela/swin_base_patch4_window12_384_22k.pth}"
SWIN_D2="${RELA_SWIN_D2:-${WEIGHTS_ROOT}/rela/swin_base_patch4_window12_384_22k.pkl}"
HOST_TAG="${FREEREF_HOST_TAG:-$(hostname)}"
HOST_TAG="${HOST_TAG//[^A-Za-z0-9_.-]/_}"
READY_MARKER="${ROOT}/.cache/rela-freeref-env-v2.${HOST_TAG}.ready"

METHODS=rela ROOT="${ROOT}" TARGET_ROOT="${ROOT}/code/third_party" \
  bash "${SCRIPT_DIR}/prepare_universal_freeref_repos.sh"
git -C "${RELA_DIR}" fetch --depth 1 origin "${RELA_REVISION}"
git -C "${RELA_DIR}" checkout --detach "${RELA_REVISION}"
METHODS=rela ROOT="${ROOT}" DOWNLOAD_DATASETS=0 \
  bash "${SCRIPT_DIR}/download_missing_method_weights.sh" || true

if [[ ! -d "${DETECTRON2_DIR}/.git" ]]; then
  git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2.git \
    "${DETECTRON2_DIR}"
fi
if ! "${CONDA_BIN}" env list --json | python -c \
  'import json,os,sys; name=sys.argv[1]; paths=json.load(sys.stdin)["envs"]; raise SystemExit(0 if any(os.path.basename(p)==name for p in paths) else 1)' \
  "${CONDA_ENV}"; then
  "${CONDA_BIN}" create -y -n "${CONDA_ENV}" python=3.9 pip=23.3 setuptools=68.2.2 wheel
fi

if [[ ! -f "${READY_MARKER}" ]]; then
  # ReLA builds Detectron2 and MS-Deformable-Attention from source.  Keep nvcc
  # aligned with the cu118 PyTorch wheel instead of relying on an arbitrary
  # system CUDA toolkit (often CUDA 12.x on H100/H200 servers).
  "${CONDA_BIN}" install -y -n "${CONDA_ENV}" -c nvidia cuda-nvcc=11.8
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
    torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
    numpy==1.23.5 scipy==1.10.1 scikit-image==0.21.0 \
    shapely==1.8.5.post1 h5py==3.10.0 cython==0.29.37 \
    transformers==4.30.2 timm==0.6.13 pycocotools==2.0.7 \
    opencv-python-headless==4.10.0.84 Pillow==10.4.0 ninja==1.11.1.1
  RELA_CUDA_HOME="${RELA_CUDA_HOME:-$("${CONDA_BIN}" run -n "${CONDA_ENV}" \
    python -c 'import pathlib,shutil; print(pathlib.Path(shutil.which("nvcc")).resolve().parents[1])' |
    tail -n 1)}"
  env CUDA_HOME="${RELA_CUDA_HOME}" FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-8}" \
    "${CONDA_BIN}" run -n "${CONDA_ENV}" \
    python -m pip install --no-build-isolation -e "${DETECTRON2_DIR}"
  (
    cd "${RELA_DIR}/gres_model/modeling/pixel_decoder/ops"
    env CUDA_HOME="${RELA_CUDA_HOME}" FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-8}" \
      "${CONDA_BIN}" run -n "${CONDA_ENV}" python setup.py build install
  )
fi

if [[ ! -f "${SWIN_D2}" ]]; then
  [[ -f "${SWIN_PTH}" ]] || {
    echo "ERROR: missing Swin-B source checkpoint: ${SWIN_PTH}" >&2
    exit 2
  }
  "${CONDA_BIN}" run -n "${CONDA_ENV}" \
    python "${RELA_DIR}/tools/convert-pretrained-swin-model-to-d2.py" \
    "${SWIN_PTH}" "${SWIN_D2}"
fi

PYTHONPATH="${RELA_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
import detectron2
import torch
from gres_model.modeling.pixel_decoder.ops.modules.ms_deform_attn import MSDeformAttn
from gres_model import ReferEvaluator

print(
    "ReLA environment ready:",
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "detectron2", detectron2.__file__,
    "evaluator", ReferEvaluator.__name__,
    "op", MSDeformAttn.__name__,
)
PY
mkdir -p "$(dirname "${READY_MARKER}")"
touch "${READY_MARKER}"

MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_rela_freeref_data.sh"
echo "ReLA source, runtime, data links, and training initialization are ready."
