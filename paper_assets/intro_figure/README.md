# FreeRef Introduction Motivation Figure

This utility selects a real, paired RefCOCO sample from completed STAMP-7B
and PixelLM evaluations and renders the three-row introduction figure:

1. STAMP soft probability as a VLM localization heatmap and its predicted box.
2. PixelLM as the learned-mask-decoder paradigm.
3. STAMP as the native-mask-token paradigm.

The selector requires both methods to localize the same target correctly, then
prefers samples with a large gap between region IoU and boundary IoU. Red
callouts are placed deterministically on connected boundary-error regions.
Ground truth is used only for candidate ranking, box-IoU verification, and
diagnostic callout placement. It is not used to produce the model heatmap or
coarse masks.

## Server Usage

Pull the code separately:

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data" || exit 1
git pull --rebase --autostash https://github.com/jiZhangji/MLLM-SEG-data.git main
```

Then generate the figure. This is a CPU post-processing job and does not load
the STAMP or PixelLM model:

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data" || exit 1

MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/intro_figure/run_intro_motivation_figure.sh
```

Default outputs:

```text
$ROOT/outputs/freeref_intro_motivation/
  freeref_intro_motivation.png
  freeref_intro_motivation.pdf
  freeref_intro_motivation.svg
  intro_candidate_contact_sheet.png
  intro_candidates.csv
  intro_figure_manifest.json
```

To force a sample after reviewing the contact sheet:

```bash
INTRO_SAMPLE_ID=1234 \
MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/intro_figure/run_intro_motivation_figure.sh
```

Every input path can be overridden:

```bash
INTRO_STAMP_ROWS=/absolute/path/eval_rows.csv \
INTRO_PIXELLM_ROWS=/absolute/path/eval_rows.csv \
INTRO_PIXELLM_MANIFEST=/absolute/path/manifest.jsonl \
INTRO_OUTPUT_DIR=/absolute/path/output \
MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/intro_figure/run_intro_motivation_figure.sh
```
