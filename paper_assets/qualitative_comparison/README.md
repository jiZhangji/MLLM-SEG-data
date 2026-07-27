# FreeRef Qualitative Comparison Figures

This utility generates two paired, publication-ready qualitative figures from
the completed RefCOCO validation outputs.

## Main-table transfer figure

Columns:

```text
Input | GT | STAMP | +FreeRef | Text4Seg | +FreeRef | PixelLM | +FreeRef
```

All eight panels in a row use the same RefCOCO instance and prompt. The figure
therefore visualizes the cross-model and cross-interface transfer claim from
the main table rather than mixing unrelated published examples.

## Post-processing comparison figure

Columns:

```text
Input | GT | Coarse | DenseCRF | Guided Filter | FBS | SLIC Avg. | FreeRef
```

Every post-processor receives the same saved STAMP-7B probability map. The
implementations and default parameters are imported directly from
`training_free_refine.eval_postprocess_baselines` and
`training_free_refine.postprocess_baselines`.

## Server usage

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data" || exit 1

MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/qualitative_comparison/run_qualitative_figures.sh
```

Default output:

```text
$ROOT/outputs/freeref_qualitative_figures/
  main_table_qualitative.{png,pdf,svg}
  postprocess_qualitative.{png,pdf,svg}
  main_table_qualitative_rows.csv
  postprocess_qualitative_rows.csv
  qualitative_manifest.json
  main_table_panels/
  postprocess_panels/
  freeref_qualitative_figures.zip
```

The default selector ranks paired examples deterministically and removes
duplicate source images. To force samples after reviewing the first output:

```bash
QUALITATIVE_MAIN_SAMPLE_IDS="1765 7226 5011" \
QUALITATIVE_POST_SAMPLE_IDS="1765 7226 5011" \
MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/qualitative_comparison/run_qualitative_figures.sh
```

Prediction masks use one shared blue color, ground truth uses green, and
FreeRef is emphasized only by its header and border. Sample-level IoU and bIoU
are stored in the CSV files rather than printed over the images.

For a larger candidate pool, figures are automatically paginated so that PDF
pages and raster previews remain manageable:

```bash
QUALITATIVE_SAMPLE_COUNT=24 \
QUALITATIVE_ROWS_PER_PAGE=4 \
QUALITATIVE_CANDIDATE_POOL=512 \
QUALITATIVE_DPI=180 \
QUALITATIVE_OUTPUT_DIR="$ROOT/outputs/freeref_qualitative_candidates_n24" \
MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/qualitative_comparison/run_qualitative_figures.sh
```

This produces six main-table pages, six post-processing pages, and all 384
full-resolution independent panels.

To mine difficult examples that FreeRef recovers especially well, enable the
hard-recovery ranking. The defaults require mean base IoU at most 0.78, mean
final IoU at least 0.72, mean IoU gain at least 0.04, and at least two of the
three main-table models to improve by that amount:

```bash
QUALITATIVE_MAIN_SELECTION_MODE=hard_recovery \
QUALITATIVE_POST_SELECTION_MODE=hard_recovery \
QUALITATIVE_SAMPLE_COUNT=36 \
QUALITATIVE_ROWS_PER_PAGE=4 \
QUALITATIVE_CANDIDATE_POOL=512 \
MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/qualitative_comparison/run_qualitative_figures.sh
```

The output CSV records the selection score, mean base/final IoU, mean IoU
gain, and corresponding boundary-IoU values. These fields make the qualitative
selection auditable rather than relying on visual preference alone.
