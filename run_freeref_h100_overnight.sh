#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_OUTPUT="${FREEREF_OVERNIGHT_OUTPUT:-${ROOT}/outputs/freeref_h100_overnight}"
SKIP_TEXT4SEG="${OVERNIGHT_SKIP_TEXT4SEG_FULL:-0}"
SKIP_STAMP2_REPAIR="${OVERNIGHT_SKIP_STAMP2_REPAIR:-0}"
SKIP_EFFICIENCY="${OVERNIGHT_SKIP_EFFICIENCY:-0}"
SKIP_STUDIES="${OVERNIGHT_SKIP_STUDIES:-0}"

if [[ -n "${FREEREF_H100_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${FREEREF_H100_GPUS}"
else
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,name --format=csv,noheader |
      awk -F, 'tolower($2) ~ /h100/ {gsub(/ /,"",$1); print $1}' | head -n 2
  )
fi
if (( ${#GPUS[@]} != 2 )); then
  echo "ERROR: exactly two H100 GPUs are required; detected: ${GPUS[*]:-none}" >&2
  exit 1
fi
export FREEREF_H100_GPUS="${GPUS[0]} ${GPUS[1]}"

mkdir -p "${MASTER_OUTPUT}"
exec > >(tee -a "${MASTER_OUTPUT}/overnight.log") 2>&1
if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.freeref_h100_overnight.lock"
  if ! flock -n 9; then
    echo "Another FreeRef H100 overnight suite is already running."
    exit 0
  fi
fi

stage() {
  printf '\n===== %s | %s =====\n' "$(date -u '+%F %T UTC')" "$1"
  echo "$1" >"${MASTER_OUTPUT}/current_stage.txt"
}

wait_for_existing_text4seg() {
  while pgrep -f '[r]un_text4seg_public_p24_full_eval' >/dev/null; do
    echo "Existing Text4Seg scheduler is active; waiting 60 seconds."
    sleep 60
  done
}

cd "${SCRIPT_DIR}"
echo "FreeRef overnight suite on physical H100 GPUs: ${FREEREF_H100_GPUS}"

if [[ "${SKIP_TEXT4SEG}" != "1" ]]; then
  stage "Text4Seg-p24 remaining full evaluation"
  combined="${ROOT}/outputs/text4seg_public_p24_freeref_samh_full_comparison/combined_summary.json"
  if [[ -f "${combined}" ]]; then
    echo "SKIP complete Text4Seg combined result: ${combined}"
  else
    wait_for_existing_text4seg
    TEXT4SEG_P24_CUDA_DEVICES="${FREEREF_H100_GPUS}" \
    TEXT4SEG_P24_PARALLEL_JOBS=4 \
    TEXT4SEG_P24_MIN_FREE_MB="${TEXT4SEG_P24_MIN_FREE_MB:-30000}" \
      bash run_text4seg_public_p24_full_eval.sh
  fi
fi

if [[ "${SKIP_STAMP2_REPAIR}" != "1" ]]; then
  stage "STAMP-2B two-sample repair check"
  if python - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = (
    ("refcoco_testA", "stamp_official_samh_stamp-2b_refcoco_testA"),
    ("refcoco+_testB", "stamp_official_samh_stamp-2b_refcoplus_testB"),
)
complete = True
for split, directory in specs:
    expected = len(json.loads((root / "code/STAMP/playground/data/json_eval_baseline" / f"{split}.json").read_text()))
    path = root / "outputs" / directory / "eval_summary.json"
    samples = json.loads(path.read_text()).get("samples", 0) if path.is_file() else 0
    complete &= int(samples) == expected
raise SystemExit(0 if complete else 1)
PY
  then
    echo "SKIP complete STAMP-2B repaired SAM-H summaries"
  else
    STAMP2B_REPAIR_GPU="${GPUS[0]}" bash repair_stamp2b_missing_samples.sh
  fi
fi

if [[ "${SKIP_EFFICIENCY}" != "1" ]]; then
  stage "Extended H100 end-to-end efficiency"
  EFFICIENCY_OUTPUT_ROOT="${ROOT}/outputs/freeref_efficiency_h100" \
    bash run_freeref_efficiency_h100.sh
fi

if [[ "${SKIP_STUDIES}" != "1" ]]; then
  stage "Component ablations and sensitivity studies"
  bash run_freeref_paper_studies.sh
fi

stage "COMPLETE"
touch "${MASTER_OUTPUT}/COMPLETE"
echo "All requested overnight experiments completed."
