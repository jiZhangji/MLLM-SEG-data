# FreeRef Framework Figure from Real Experiments

The paper framework figure is rendered from one real evaluation sample. The
pipeline does **not** invent a photograph or use a generative model:

1. read an existing `eval_rows.csv`;
2. rank representative successful examples using recorded IoU and boundary-IoU;
3. recompute FreeRef only for a small candidate pool;
4. export the real input, base prediction, uncertainty, SLIC labels, regional
   solution, refined prediction, and changed pixels;
5. place those arrays into the three-panel paper diagram.

Ground truth is used only to rank and verify representative candidates. It is
not consumed by FreeRef and is not drawn in the main framework figure.

## Run on the experiment server

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
git pull origin main
conda activate STAMP
bash paper_assets/framework/run_real_framework_figure.sh
```

Generate and upload the viewable outputs to the existing Hugging Face repo:

```bash
FRAMEWORK_UPLOAD_HF=1 \
bash paper_assets/framework/run_real_framework_figure.sh
```

The default upload target is:

```text
https://huggingface.co/shimiandeshu/MLLM-SEG/tree/main/paper_assets/framework_runs/stamp
```

Only the PNG/PDF/SVG, candidate sheet, metrics/manifest, and component PNGs are
uploaded. The larger `selected_real_sample.npz` is intentionally excluded.
Override the destination with `FRAMEWORK_HF_REPO_ID` and
`FRAMEWORK_HF_PATH`.

The default source is the real STAMP-7B RefCOCOg validation run:

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/training_free_refine_stamp7b_refcocog_val_full/eval_rows.csv
```

The outputs are written to:

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/framework_figure_real/stamp/
```

Important files:

- `framework_candidate_contact_sheet.png`: real candidate examples for review;
- `framework_candidates.csv`: candidate ranking and metrics;
- `selected_real_sample.json`: exact sample, source row, configuration, and metrics;
- `selected_real_sample.npz`: real arrays used by the renderer;
- `freeref_framework_real.pdf`: paper-ready vector figure;
- `freeref_framework_real.svg`: editable vector figure;
- `freeref_framework_real.png`: 300-DPI preview;
- `freeref_framework_real_components/`: individual real image/map assets,
  including the input image, GT mask, baseline mask, FreeRef mask, and their
  colored overlays.

## Select a different real example

Use another candidate rank from the contact sheet:

```bash
FRAMEWORK_SAMPLE_RANK=2 \
bash paper_assets/framework/run_real_framework_figure.sh
```

Select a named sample directly:

```bash
FRAMEWORK_SAMPLE_NAME="exact_sample_name" \
bash paper_assets/framework/run_real_framework_figure.sh
```

Rank by the strongest boundary improvement instead of the default
representative-success policy:

```bash
FRAMEWORK_SELECTION=best_boundary \
bash paper_assets/framework/run_real_framework_figure.sh
```

Use the real Text4Seg run:

```bash
FRAMEWORK_KIND=text4seg \
bash paper_assets/framework/run_real_framework_figure.sh
```

Paths can be overridden with `FRAMEWORK_ROWS`, `FRAMEWORK_OUTPUT_DIR`,
`FRAMEWORK_PYTHON`, and `FRAMEWORK_CANDIDATE_POOL`.

## Download for local inspection

Open the PNG directly:

```text
https://huggingface.co/shimiandeshu/MLLM-SEG/resolve/main/paper_assets/framework_runs/stamp/freeref_framework_real.png
```

Or download all viewable files locally:

```bash
hf download shimiandeshu/MLLM-SEG \
  --include "paper_assets/framework_runs/stamp/*" \
  --local-dir ./freeref_framework_from_server
```

## Renderer-only command

After a real bundle has been exported:

```bash
python paper_assets/framework/generate_framework_figure.py \
  --sample-bundle /path/to/selected_real_sample.npz \
  --output-dir /path/to/output \
  --stem freeref_framework_real
```

`--demo` exists only as a deterministic layout smoke test for developers. It is
never called by `run_real_framework_figure.sh` and must not be used for the
paper figure.
