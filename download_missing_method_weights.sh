#!/usr/bin/env bash
set -uo pipefail

# Download the official checkpoints needed to evaluate the methods that are
# still missing from the FreeRef comparison. STAMP and Text4Seg are excluded
# because their predictions have already been evaluated.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
STATUS_ROOT="${STATUS_ROOT:-${ROOT}/outputs/freeref_weight_download}"
TOOLS_ENV="${TOOLS_ENV:-${ROOT}/.cache/freeref-download-tools}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
METHODS="${METHODS:-all}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
MIN_WEIGHT_BYTES="${MIN_WEIGHT_BYTES:-50000000}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"

ALL_METHODS=(
  hipie rela polyformer uninext pixellm lisa gsva read seg-zero segllm segagent
)

mkdir -p "${WEIGHTS_ROOT}" "${STATUS_ROOT}"

STATUS_FILE="${STATUS_ROOT}/download_status.tsv"
MANUAL_FILE="${STATUS_ROOT}/manual_downloads.tsv"
PLAN_FILE="${STATUS_ROOT}/download_plan.tsv"
INVENTORY_FILE="${STATUS_ROOT}/weights_inventory.tsv"
LOCK_FILE="${STATUS_ROOT}/download.lock"

printf 'method\tartifact\tstatus\ttarget\tsource\tdetail\n' > "${STATUS_FILE}"
printf 'method\tartifact\ttarget\tsource\treason\n' > "${MANUAL_FILE}"
printf 'method\tartifact\ttarget\tsource\n' > "${PLAN_FILE}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

clean_field() {
  printf '%s' "$*" | tr '\t\r\n' '   '
}

record_status() {
  local method="$1" artifact="$2" status="$3" target="$4" source="$5" detail="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(clean_field "${method}")" \
    "$(clean_field "${artifact}")" \
    "$(clean_field "${status}")" \
    "$(clean_field "${target}")" \
    "$(clean_field "${source}")" \
    "$(clean_field "${detail}")" >> "${STATUS_FILE}"
}

record_manual() {
  local method="$1" artifact="$2" target="$3" source="$4" reason="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(clean_field "${method}")" \
    "$(clean_field "${artifact}")" \
    "$(clean_field "${target}")" \
    "$(clean_field "${source}")" \
    "$(clean_field "${reason}")" >> "${MANUAL_FILE}"
}

record_plan() {
  printf '%s\t%s\t%s\t%s\n' \
    "$(clean_field "$1")" \
    "$(clean_field "$2")" \
    "$(clean_field "$3")" \
    "$(clean_field "$4")" >> "${PLAN_FILE}"
}

selected() {
  local method="$1"
  [[ "${METHODS}" == "all" || " ${METHODS} " == *" ${method} "* ]]
}

any_selected() {
  local method
  for method in "$@"; do
    if selected "${method}"; then
      return 0
    fi
  done
  return 1
}

marker_for() {
  local kind="$1" target="$2"
  if [[ "${kind}" == "dir" ]]; then
    printf '%s/.freeref_download_complete' "${target}"
  else
    printf '%s.freeref_download_complete' "${target}"
  fi
}

validate_artifact() {
  local kind="$1" target="$2"
  if [[ "${kind}" == "file" ]]; then
    [[ -f "${target}" && "$(stat -c '%s' "${target}" 2>/dev/null || printf 0)" -ge "${MIN_WEIGHT_BYTES}" ]]
    return
  fi
  [[ -d "${target}" ]] || return 1
  find "${target}" -type f -size "+${MIN_WEIGHT_BYTES}c" -print -quit 2>/dev/null | grep -q .
}

run_artifact() {
  local method="$1" artifact="$2" kind="$3" target="$4" source="$5"
  shift 5
  local marker
  marker="$(marker_for "${kind}" "${target}")"

  record_plan "${method}" "${artifact}" "${target}" "${source}"

  if [[ "${FORCE}" != "1" && -f "${marker}" ]] && validate_artifact "${kind}" "${target}"; then
    log "SKIP ${method}/${artifact}: complete"
    record_status "${method}" "${artifact}" "complete" "${target}" "${source}" "completion marker and weight files found"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "PLAN ${method}/${artifact} <- ${source}"
    record_status "${method}" "${artifact}" "planned" "${target}" "${source}" "dry run"
    return 0
  fi

  mkdir -p "$(dirname "${target}")"
  [[ "${kind}" == "dir" ]] && mkdir -p "${target}"
  log "DOWNLOAD ${method}/${artifact}"

  if "$@"; then
    if validate_artifact "${kind}" "${target}"; then
      touch "${marker}"
      record_status "${method}" "${artifact}" "complete" "${target}" "${source}" "downloaded and size-validated"
      log "DONE ${method}/${artifact}"
      return 0
    fi
    record_status "${method}" "${artifact}" "invalid" "${target}" "${source}" "download command returned success, but no weight file passed validation"
    log "INVALID ${method}/${artifact}: no file larger than ${MIN_WEIGHT_BYTES} bytes"
    return 1
  fi

  record_status "${method}" "${artifact}" "failed" "${target}" "${source}" "download command failed; rerun this script to retry"
  log "FAILED ${method}/${artifact}"
  return 1
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Another checkpoint download is already using ${LOCK_FILE}" >&2
    exit 2
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN=python
    else
      echo "Python was not found. Set PYTHON_BIN to a Python 3 executable." >&2
      exit 2
    fi
  fi

  available_kb="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"
  required_kb="$((MIN_FREE_GB * 1024 * 1024))"
  if [[ -n "${available_kb}" && "${available_kb}" -lt "${required_kb}" ]]; then
    echo "Only $((available_kb / 1024 / 1024)) GiB is free under ${ROOT}; ${MIN_FREE_GB} GiB is required." >&2
    echo "Free space or rerun with a lower MIN_FREE_GB after checking capacity." >&2
    exit 2
  fi

  tools_marker="${TOOLS_ENV}/.freeref_download_tools_v1"
  if [[ ! -f "${tools_marker}" ]]; then
    log "Preparing isolated download tools in ${TOOLS_ENV}"
    "${PYTHON_BIN}" -m venv "${TOOLS_ENV}"
    "${TOOLS_ENV}/bin/python" -m pip install --disable-pip-version-check --upgrade pip
    "${TOOLS_ENV}/bin/python" -m pip install --disable-pip-version-check \
      'huggingface_hub>=0.27' \
      'gdown>=5.2' \
      'modelscope>=1.20' \
      'requests>=2.31'
    touch "${tools_marker}"
  fi
fi

TOOLS_PY="${TOOLS_ENV}/bin/python"
HF_BIN="${TOOLS_ENV}/bin/hf"
GDOWN_BIN="${TOOLS_ENV}/bin/gdown"

hf_snapshot() {
  local repo_id="$1" target="$2"
  "${HF_BIN}" download "${repo_id}" --local-dir "${target}"
}

hf_dataset_snapshot() {
  local repo_id="$1" target="$2"
  "${HF_BIN}" download "${repo_id}" --repo-type dataset --local-dir "${target}"
}

gdrive_file() {
  local file_id="$1" target="$2"
  "${GDOWN_BIN}" --fuzzy --continue "https://drive.google.com/file/d/${file_id}/view" -O "${target}"
}

gdrive_folder() {
  local url="$1" target="$2"
  "${GDOWN_BIN}" --folder --remaining-ok "${url}" -O "${target}"
}

download_url_file() {
  local url="$1" target="$2"
  curl -L --fail --retry 10 --retry-all-errors --connect-timeout 30 \
    --continue-at - --output "${target}" "${url}"
}

onedrive_folder() {
  local url="$1" target="$2" archive="$3"
  ONEDRIVE_URL="${url}" ONEDRIVE_TARGET="${target}" ONEDRIVE_ARCHIVE="${archive}" \
    "${TOOLS_PY}" - <<'PY'
import os
import zipfile
from pathlib import Path

import requests

url = os.environ["ONEDRIVE_URL"].split("?", 1)[0] + "?download=1"
target = Path(os.environ["ONEDRIVE_TARGET"])
archive = Path(os.environ["ONEDRIVE_ARCHIVE"])
target.mkdir(parents=True, exist_ok=True)
archive.parent.mkdir(parents=True, exist_ok=True)

headers = {}
existing = archive.stat().st_size if archive.exists() else 0
if existing:
    headers["Range"] = f"bytes={existing}-"

with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise RuntimeError("OneDrive returned an HTML page instead of a checkpoint archive")
    append = existing > 0 and response.status_code == 206
    mode = "ab" if append else "wb"
    with archive.open(mode) as output:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                output.write(chunk)

if not zipfile.is_zipfile(archive):
    raise RuntimeError(f"OneDrive response is not a ZIP archive: {archive}")
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(target)
PY
}

modelscope_snapshot() {
  local model_id="$1" target="$2"
  MODELSCOPE_MODEL_ID="${model_id}" MODELSCOPE_TARGET="${target}" \
    "${TOOLS_PY}" - <<'PY'
import os
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    os.environ["MODELSCOPE_MODEL_ID"],
    local_dir=os.environ["MODELSCOPE_TARGET"],
)
PY
}

prepare_shared_sam_h() {
  local source_path="${ROOT}/models/SAM/sam_vit_h_4b8939.pth"
  local target="${WEIGHTS_ROOT}/shared/sam_vit_h_4b8939.pth"
  local marker
  marker="$(marker_for file "${target}")"
  record_plan shared sam-h "${target}" "${source_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    record_status shared sam-h planned "${target}" "${source_path}" "reuse existing SAM-H or download the official checkpoint"
    log "PLAN shared/sam-h"
    return 0
  fi

  mkdir -p "$(dirname "${target}")"
  if [[ -f "${source_path}" && "$(stat -c '%s' "${source_path}" 2>/dev/null || printf 0)" -ge "${MIN_WEIGHT_BYTES}" ]]; then
    if [[ ! -e "${target}" ]]; then
      ln -s "${source_path}" "${target}"
    fi
    touch "${marker}"
    record_status shared sam-h complete "${target}" "${source_path}" "reused the existing server checkpoint"
    return 0
  fi

  run_artifact shared sam-h file "${target}" \
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' \
    download_url_file \
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' \
    "${target}"
}

log "Root: ${ROOT}"
log "Weights: ${WEIGHTS_ROOT}"
log "Methods: ${METHODS}"

if any_selected pixellm lisa gsva read segagent; then
  prepare_shared_sam_h || true
fi

if any_selected pixellm lisa gsva; then
  run_artifact shared clip-vit-large-patch14 dir \
    "${WEIGHTS_ROOT}/shared/clip-vit-large-patch14" \
    'https://huggingface.co/openai/clip-vit-large-patch14' \
    hf_snapshot openai/clip-vit-large-patch14 \
    "${WEIGHTS_ROOT}/shared/clip-vit-large-patch14" || true
fi

if any_selected read segagent; then
  run_artifact shared clip-vit-large-patch14-336 dir \
    "${WEIGHTS_ROOT}/shared/clip-vit-large-patch14-336" \
    'https://huggingface.co/openai/clip-vit-large-patch14-336' \
    hf_snapshot openai/clip-vit-large-patch14-336 \
    "${WEIGHTS_ROOT}/shared/clip-vit-large-patch14-336" || true
fi

if any_selected rela uninext; then
  run_artifact shared bert-base-uncased dir \
    "${WEIGHTS_ROOT}/shared/bert-base-uncased" \
    'https://huggingface.co/google-bert/bert-base-uncased' \
    hf_snapshot google-bert/bert-base-uncased \
    "${WEIGHTS_ROOT}/shared/bert-base-uncased" || true
fi

if selected hipie; then
  run_artifact hipie official-checkpoints dir \
    "${WEIGHTS_ROOT}/hipie/HIPIE" \
    'https://huggingface.co/KonstantinosKK/HIPIE' \
    hf_snapshot KonstantinosKK/HIPIE \
    "${WEIGHTS_ROOT}/hipie/HIPIE" || true
fi

if selected rela; then
  rela_url='https://drive.google.com/drive/folders/1Jw7GKiN-Y2tgLL6ueOKOKfikiWVOl2-n?usp=drive_link'
  if ! run_artifact rela official-checkpoints dir \
    "${WEIGHTS_ROOT}/rela/official_checkpoints" "${rela_url}" \
    gdrive_folder "${rela_url}" "${WEIGHTS_ROOT}/rela/official_checkpoints"; then
    record_manual rela official-checkpoints "${WEIGHTS_ROOT}/rela/official_checkpoints" \
      "${rela_url}" "Google Drive folder download failed; open the link and place the official checkpoints here"
  fi
fi

if selected polyformer; then
  run_artifact polyformer polyformer-l-refcoco file \
    "${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco.pt" \
    'https://drive.google.com/file/d/15P6m5RI6HAQE2QXQXMAjw_oBsaPii7b3/view' \
    gdrive_file 15P6m5RI6HAQE2QXQXMAjw_oBsaPii7b3 \
    "${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco.pt" || true
  run_artifact polyformer polyformer-l-refcoco-plus file \
    "${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco+.pt" \
    'https://drive.google.com/file/d/1lUCv7dUPctEz4vEpPr7aI8A8ZmfYCB8y/view' \
    gdrive_file 1lUCv7dUPctEz4vEpPr7aI8A8ZmfYCB8y \
    "${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco+.pt" || true
  run_artifact polyformer polyformer-l-refcocog file \
    "${WEIGHTS_ROOT}/polyformer/polyformer_l_refcocog.pt" \
    'https://drive.google.com/file/d/1upjK4YmtQT9b6qcA3yj3DXKnOuI52Pxv/view' \
    gdrive_file 1upjK4YmtQT9b6qcA3yj3DXKnOuI52Pxv \
    "${WEIGHTS_ROOT}/polyformer/polyformer_l_refcocog.pt" || true
fi

if selected uninext; then
  uninext_url='https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/Et6GBDgKgPZDn5zp49yKwDYBd50EBTxaKs7R6Yuck_lf7g?e=818rMm'
  if ! run_artifact uninext image-joint-convnext-large dir \
    "${WEIGHTS_ROOT}/uninext/image_joint_convnext_large" "${uninext_url}" \
    onedrive_folder "${uninext_url}" \
    "${WEIGHTS_ROOT}/uninext/image_joint_convnext_large" \
    "${WEIGHTS_ROOT}/uninext/image_joint_convnext_large.zip"; then
    record_manual uninext image-joint-convnext-large \
      "${WEIGHTS_ROOT}/uninext/image_joint_convnext_large" "${uninext_url}" \
      "SharePoint folder could not be exported automatically; download the Stage-2 ConvNeXt-Large folder"
  fi
fi

if selected pixellm; then
  run_artifact pixellm pixellm-7b dir \
    "${WEIGHTS_ROOT}/pixellm/PixelLM-7B" \
    'https://huggingface.co/maverickrzw/PixelLM-7B' \
    hf_snapshot maverickrzw/PixelLM-7B \
    "${WEIGHTS_ROOT}/pixellm/PixelLM-7B" || true
fi

if selected lisa; then
  run_artifact lisa lisa-7b-v1 dir \
    "${WEIGHTS_ROOT}/lisa/LISA-7B-v1" \
    'https://huggingface.co/xinlai/LISA-7B-v1' \
    hf_snapshot xinlai/LISA-7B-v1 \
    "${WEIGHTS_ROOT}/lisa/LISA-7B-v1" || true
fi

if selected gsva; then
  gsva_url='https://1drv.ms/f/s!ApI0vb6wPqmtku1kOKbVTJkwa6jG_Q?e=vaG9Gj'
  if ! run_artifact gsva official-7b-checkpoints dir \
    "${WEIGHTS_ROOT}/gsva/official_checkpoints" "${gsva_url}" \
    onedrive_folder "${gsva_url}" \
    "${WEIGHTS_ROOT}/gsva/official_checkpoints" \
    "${WEIGHTS_ROOT}/gsva/official_checkpoints.zip"; then
    record_manual gsva official-7b-checkpoints \
      "${WEIGHTS_ROOT}/gsva/official_checkpoints" "${gsva_url}" \
      'OneDrive folder export failed; use the official Tsinghua mirror: https://cloud.tsinghua.edu.cn/d/1423fb16fdb9445e8155/'
  fi
  run_artifact gsva llava-lightning-7b-delta dir \
    "${WEIGHTS_ROOT}/gsva/LLaVA-Lightning-7B-delta-v1-1" \
    'https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1' \
    hf_snapshot liuhaotian/LLaVA-Lightning-7B-delta-v1-1 \
    "${WEIGHTS_ROOT}/gsva/LLaVA-Lightning-7B-delta-v1-1" || true
  record_manual gsva vicuna-7b-base \
    "${WEIGHTS_ROOT}/gsva/LLaVA-Lightning-7B-v1-1-merged" \
    'Official LLaVA legacy model-zoo instructions' \
    'The released GSVA-7B bin requires the licensed LLaMA/Vicuna-7B base merged with the downloaded LLaVA delta'
fi

if selected read; then
  run_artifact read read-llava-v1.5-7b-fprefcoco dir \
    "${WEIGHTS_ROOT}/read/READ-LLaVA-v1.5-7B-for-fprefcoco" \
    'https://huggingface.co/rui-qian/READ-LLaVA-v1.5-7B-for-fprefcoco' \
    hf_snapshot rui-qian/READ-LLaVA-v1.5-7B-for-fprefcoco \
    "${WEIGHTS_ROOT}/read/READ-LLaVA-v1.5-7B-for-fprefcoco" || true
fi

if selected seg-zero; then
  run_artifact seg-zero seg-zero-7b dir \
    "${WEIGHTS_ROOT}/seg-zero/Seg-Zero-7B" \
    'https://huggingface.co/Ricky06662/Seg-Zero-7B' \
    hf_snapshot Ricky06662/Seg-Zero-7B \
    "${WEIGHTS_ROOT}/seg-zero/Seg-Zero-7B" || true
  run_artifact seg-zero sam2-hiera-large dir \
    "${WEIGHTS_ROOT}/seg-zero/sam2-hiera-large" \
    'https://huggingface.co/facebook/sam2-hiera-large' \
    hf_snapshot facebook/sam2-hiera-large \
    "${WEIGHTS_ROOT}/seg-zero/sam2-hiera-large" || true
  if [[ "${DOWNLOAD_DATASETS}" == "1" ]]; then
    run_artifact seg-zero reasonseg-val-dataset dir \
      "${WEIGHTS_ROOT}/seg-zero/datasets/ReasonSeg_val" \
      'https://huggingface.co/datasets/Ricky06662/ReasonSeg_val' \
      hf_dataset_snapshot Ricky06662/ReasonSeg_val \
      "${WEIGHTS_ROOT}/seg-zero/datasets/ReasonSeg_val" || true
  fi
fi

if selected segllm; then
  record_plan segllm official-checkpoint "${WEIGHTS_ROOT}/segllm" 'not publicly verified'
  record_status segllm official-checkpoint blocked "${WEIGHTS_ROOT}/segllm" \
    'not publicly verified' 'No executable official repository/checkpoint has been verified'
  record_manual segllm official-checkpoint "${WEIGHTS_ROOT}/segllm" \
    'author release required' 'SegLLM cannot be reproduced from a verified official checkpoint at present'
fi

if selected segagent; then
  run_artifact segagent segagent-model dir \
    "${WEIGHTS_ROOT}/segagent/SegAgent-Model" \
    'https://www.modelscope.cn/models/zzzmmz/SegAgent-Model' \
    modelscope_snapshot zzzmmz/SegAgent-Model \
    "${WEIGHTS_ROOT}/segagent/SegAgent-Model" || true

  simpleclick_url='https://drive.google.com/drive/folders/1qpK0gtAPkVMF7VC42UA9XF4xMWr5KJmL?usp=sharing'
  if ! run_artifact segagent simpleclick-models dir \
    "${WEIGHTS_ROOT}/segagent/simpleclick_models" "${simpleclick_url}" \
    gdrive_folder "${simpleclick_url}" \
    "${WEIGHTS_ROOT}/segagent/simpleclick_models"; then
    record_manual segagent simpleclick-models \
      "${WEIGHTS_ROOT}/segagent/simpleclick_models" "${simpleclick_url}" \
      'Download the official folder and ensure cocolvis_vit_large.pth is present'
  fi

  if [[ "${DOWNLOAD_DATASETS}" == "1" ]]; then
    run_artifact segagent segagent-dataset dir \
      "${WEIGHTS_ROOT}/segagent/SegAgent-Dataset" \
      'https://www.modelscope.cn/models/zzzmmz/SegAgent-Dataset' \
      modelscope_snapshot zzzmmz/SegAgent-Dataset \
      "${WEIGHTS_ROOT}/segagent/SegAgent-Dataset" || true
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  printf 'bytes\tpath\n' > "${INVENTORY_FILE}"
  find "${WEIGHTS_ROOT}" -type f -size +1M -printf '%s\t%p\n' 2>/dev/null \
    | sort -nr >> "${INVENTORY_FILE}"
else
  printf 'bytes\tpath\n' > "${INVENTORY_FILE}"
fi

complete_count="$(awk -F '\t' 'NR > 1 && $3 == "complete" {n++} END {print n+0}' "${STATUS_FILE}")"
failed_count="$(awk -F '\t' 'NR > 1 && ($3 == "failed" || $3 == "invalid") {n++} END {print n+0}' "${STATUS_FILE}")"
blocked_count="$(awk -F '\t' 'NR > 1 && $3 == "blocked" {n++} END {print n+0}' "${STATUS_FILE}")"
manual_count="$(awk 'NR > 1 {n++} END {print n+0}' "${MANUAL_FILE}")"

log "Finished: complete=${complete_count}, failed=${failed_count}, blocked=${blocked_count}, manual=${manual_count}"
log "Status: ${STATUS_FILE}"
log "Manual queue: ${MANUAL_FILE}"
log "Inventory: ${INVENTORY_FILE}"

# Download failures are reported in the status files but do not stop later
# methods from downloading. A non-zero exit makes nohup/CI status truthful.
if [[ "${DRY_RUN}" != "1" && "${failed_count}" -gt 0 ]]; then
  exit 1
fi
