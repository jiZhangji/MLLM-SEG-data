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
