# FreeRef Introduction Motivation Figure

This utility selects one real RefCOCO instance shared by completed STAMP-7B,
Text4Seg-p24, and PixelLM evaluations and renders a compact introduction figure:

1. The upper row shows the input, a soft localization response, and the box
   enclosing its predicted foreground.
2. Rows `(a)`, `(b)`, and `(c)` show the same image, the model's coarse mask
   overlay, and small grid cells over locally uncertain boundary regions.
3. The paper figure deliberately omits method names and metric text. The rows
   correspond to PixelLM, STAMP, and Text4Seg, respectively; this mapping and
   its interpretation belong in the caption or surrounding paragraph.

The selector requires all three methods to localize the same target correctly,
then prefers samples with a large gap between region IoU and boundary IoU. The
red translucent overlay is the coarse foreground. The small yellow grid cells
are selected from model-derived uncertainty around the coarse boundary. Ground
truth is used only for candidate ranking and box-IoU verification; it is not
used to produce the heatmap, mask overlays, or uncertainty cells.

Suggested paper caption:

> MLLMs can identify the referred object but remain limited in local spatial
> recovery. The upper row illustrates accurate semantic localization. The
> lower rows show coarse outputs from (a) a learned mask-decoder interface,
> (b) native visual mask tokens, and (c) textual mask tokens. Red denotes the
> predicted foreground and the overlaid cells denote locally uncertain patches.
> Method names are stated only in the text: PixelLM, STAMP, and Text4Seg are
> used as the corresponding representative outputs.

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
INTRO_TEXT4SEG_ROWS=/absolute/path/eval_rows.csv \
INTRO_PIXELLM_ROWS=/absolute/path/eval_rows.csv \
INTRO_PIXELLM_MANIFEST=/absolute/path/manifest.jsonl \
INTRO_OUTPUT_DIR=/absolute/path/output \
MLLM_SEG_ROOT="$ROOT" \
bash paper_assets/intro_figure/run_intro_motivation_figure.sh
```
