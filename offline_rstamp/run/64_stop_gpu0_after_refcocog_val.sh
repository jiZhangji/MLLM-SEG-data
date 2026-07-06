#!/usr/bin/env bash
set -euo pipefail

# Monitor the RefCOCOg val report and stop GPU0 evaluation after val finishes.
#
# Use case:
#   - Terminal A/GPU0 runs:
#       CUDA_VISIBLE_DEVICES=0 bash offline_rstamp/run/61_eval_refcocog_official_2b_text_prior.sh
#   - Terminal B/GPU1 runs refcocog_test separately.
#   - This script runs in Terminal C and kills the GPU0 eval process once the
#     val report exists, so GPU0 will not duplicate the test split.
#
# It only targets processes whose environment contains CUDA_VISIBLE_DEVICES=0
# and whose command line is related to this RefCOCOg prior evaluation.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
VAL_REPORT="${VAL_REPORT:-${MLLM_SEG_ROOT}/outputs/eval_refcocog_official_2b_text_prior/refcocog_val/smoke_iou_comparison.md}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GRACE_SECONDS="${GRACE_SECONDS:-20}"

echo "Monitoring val report:"
echo "${VAL_REPORT}"
echo "Polling every ${POLL_SECONDS}s. Will stop GPU0 eval after report appears."

while [ ! -s "${VAL_REPORT}" ]; do
  date "+[%F %T] waiting for val report..."
  sleep "${POLL_SECONDS}"
done

date "+[%F %T] val report found."
echo "Waiting ${GRACE_SECONDS}s to let file writes finish..."
sleep "${GRACE_SECONDS}"

echo "Searching for GPU0 RefCOCOg eval processes..."

matched=0
for env_file in /proc/[0-9]*/environ; do
  pid="$(echo "${env_file}" | awk -F/ '{print $3}')"
  cmd_file="/proc/${pid}/cmdline"
  [ -r "${env_file}" ] || continue
  [ -r "${cmd_file}" ] || continue

  env_text="$(tr '\0' '\n' < "${env_file}" 2>/dev/null || true)"
  cmd_text="$(tr '\0' ' ' < "${cmd_file}" 2>/dev/null || true)"

  if echo "${env_text}" | grep -qx "CUDA_VISIBLE_DEVICES=0"; then
    if echo "${cmd_text}" | grep -Eq "61_eval_refcocog_official_2b_text_prior|eval_smoke_iou.py"; then
      echo "Stopping PID ${pid}: ${cmd_text}"
      kill "${pid}" 2>/dev/null || true
      matched=$((matched + 1))
    fi
  fi
done

if [ "${matched}" -eq 0 ]; then
  echo "No matching GPU0 eval process found. It may have already stopped or moved to another environment."
else
  echo "Sent SIGTERM to ${matched} matching process(es)."
fi

echo "Done. You can inspect:"
echo "cat \"${VAL_REPORT}\""
