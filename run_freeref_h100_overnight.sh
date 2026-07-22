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

finish_text4seg() {
  local combined="${ROOT}/outputs/text4seg_public_p24_freeref_samh_full_comparison/combined_summary.json"
  while pgrep -f '[r]un_text4seg_public_p24_full_eval' >/dev/null; do
    echo "Text4Seg scheduler remains active; parallel tasks continue."
    sleep 60
  done
  if [[ ! -f "${combined}" ]]; then
    TEXT4SEG_P24_CUDA_DEVICES="${FREEREF_H100_GPUS}" \
    TEXT4SEG_P24_PARALLEL_JOBS=4 \
    TEXT4SEG_P24_MIN_FREE_MB="${TEXT4SEG_P24_MIN_FREE_MB:-30000}" \
      bash run_text4seg_public_p24_full_eval.sh
  fi
}

cd "${SCRIPT_DIR}"
echo "FreeRef parallel overnight suite on physical H100 GPUs: ${FREEREF_H100_GPUS}"
stage "Parallel: Text4Seg + STAMP-2B repair + H100 timing + paper studies"
declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

launch_job() {
  local name="$1"
  shift
  echo "LAUNCH ${name}"
  "$@" &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("${name}")
}

if [[ "${SKIP_TEXT4SEG}" != "1" ]]; then
  combined="${ROOT}/outputs/text4seg_public_p24_freeref_samh_full_comparison/combined_summary.json"
  if [[ -f "${combined}" ]]; then
    echo "SKIP complete Text4Seg combined result: ${combined}"
  else
    launch_job text4seg finish_text4seg
  fi
fi

if [[ "${SKIP_STAMP2_REPAIR}" != "1" ]]; then
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
    launch_job stamp2b_repair env \
      STAMP2B_REPAIR_GPU="${GPUS[0]}" \
      STAMP2B_REPAIR_SAMH_PARALLEL_JOBS=1 \
      STAMP2B_REPAIR_SAMH_MIN_FREE_MB=16000 \
      bash repair_stamp2b_missing_samples.sh
  fi
fi

if [[ "${SKIP_EFFICIENCY}" != "1" ]]; then
  launch_job efficiency env \
    EFFICIENCY_OUTPUT_ROOT="${ROOT}/outputs/freeref_efficiency_h100" \
    bash run_freeref_efficiency_h100.sh
fi

if [[ "${SKIP_STUDIES}" != "1" ]]; then
  launch_job studies bash run_freeref_paper_studies.sh
fi

failed=0
for index in "${!JOB_PIDS[@]}"; do
  if wait "${JOB_PIDS[$index]}"; then
    echo "DONE ${JOB_NAMES[$index]}"
  else
    echo "ERROR ${JOB_NAMES[$index]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  stage "FAILED"
  exit 1
fi

stage "COMPLETE"
touch "${MASTER_OUTPUT}/COMPLETE"
echo "All requested overnight experiments completed."
