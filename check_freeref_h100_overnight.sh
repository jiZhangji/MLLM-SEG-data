#!/usr/bin/env bash
set -u

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_OUTPUT="${FREEREF_OVERNIGHT_OUTPUT:-${ROOT}/outputs/freeref_h100_overnight}"
EFFICIENCY_ROOT="${EFFICIENCY_OUTPUT_ROOT:-${ROOT}/outputs/freeref_efficiency_h100}"
STUDY_SAMPLES="${PAPER_STUDY_SAMPLES:-500}"
STUDY_ROOT="${PAPER_STUDY_OUTPUT_ROOT:-${ROOT}/outputs/freeref_paper_studies_n${STUDY_SAMPLES}}"

echo "Time: $(date -u '+%F %T UTC')"
echo "Stage: $(cat "${MASTER_OUTPUT}/current_stage.txt" 2>/dev/null || echo not-started)"
echo "Complete: $([[ -f "${MASTER_OUTPUT}/COMPLETE" ]] && echo yes || echo no)"
echo
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo
echo "===== Text4Seg full accuracy ====="
bash "${SCRIPT_DIR}/check_text4seg_public_p24_status.sh" || true

echo
echo "===== H100 timing ====="
timing_tasks=(
  stamp2b_base stamp2b_freeref_gpu stamp2b_sam_h stamp2b_freeref_sam_h
  stamp7b_base stamp7b_freeref_gpu stamp7b_sam_h stamp7b_freeref_sam_h
  text4seg_base text4seg_freeref_gpu text4seg_sam_h text4seg_freeref_sam_h
  lisa_original
)
timing_done=0
for name in "${timing_tasks[@]}"; do
  if [[ -f "${EFFICIENCY_ROOT}/${name}/summary.json" ]]; then
    ((timing_done += 1))
    status=complete
  else
    status=pending
  fi
  printf '%-32s %s\n' "${name}" "${status}"
done
echo "Timing summaries: ${timing_done}/${#timing_tasks[@]}"

echo
echo "===== Paper studies ====="
study_done="$(find "${STUDY_ROOT}" -mindepth 4 -maxdepth 4 -name eval_summary.json -type f 2>/dev/null | wc -l)"
echo "Study summaries: ${study_done}/26"
echo "Combined table: $([[ -f "${STUDY_ROOT}/combined/paper_studies.md" ]] && echo yes || echo no)"

echo
echo "===== Active processes ====="
pgrep -af '[r]un_freeref_h100_overnight|[r]un_freeref_efficiency_h100|[r]un_freeref_paper_studies|efficiency_benchmark.run_|training_free_refine.eval_|[r]un_text4seg_public_p24' || echo none

echo
echo "===== Master log tail ====="
tail -n 40 "${MASTER_OUTPUT}/overnight.log" 2>/dev/null || echo none
