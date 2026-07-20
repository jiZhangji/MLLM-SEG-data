# Training-Free Uncertainty Graph Refinement

This package refines STAMP mask-token probabilities without training an
adapter. It consumes only the RGB image, STAMP mask logits, and the token-grid
shape. Ground-truth masks are used by the evaluator only and never by the
refinement algorithm.

## Method

1. Reconstruct the STAMP foreground probability with the existing softmax and
   bilinear upsampling protocol.
2. Compute uncertainty as `1 - abs(2 * p - 1)`.
3. Build an RGB/Lab-aware SLIC superpixel adjacency graph.
4. Solve the sparse graph-Laplacian energy
   `sum C_k(q_k-p_k)^2 + lambda * sum w_kl(q_k-q_l)^2`, where
   `C_k=(1-u_k)^2+eps` and confident foreground/background superpixels are
   strengthened as seeds.
5. Fuse the graph solution only where STAMP is uncertain:
   `p_refined=(1-u)*p + u*q`.

There are no learned parameters, checkpoints, backpropagation steps, or uses of
STAMP `mask_hidden`.

## Evaluation

```bash
python -m training_free_refine.eval_stamp_dumps \
  --input-dir ../outputs/refine_stamp_dumps/refcocog_val_200 \
  --output-dir ../outputs/training_free_refine_on_val200
```

The evaluator writes `eval_summary.json`, `eval_rows.csv`, and optional
visualizations. Each visualization contains, from left to right: RGB image, GT
overlay, coarse STAMP overlay, uncertainty, and refined overlay.

Run the full validation split before test. Fix the configuration after
validation and evaluate test only once to preserve the training-free protocol.

To run unit tests, complete RefCOCOg val, complete test, and the final combined
comparison sequentially with one script:

```bash
bash run_training_free_refine_full_eval.sh
```

The combined report is written to
`../outputs/training_free_refine_refcocog_full_comparison/combined_summary.md`.

For the released STAMP-7B LoRA checkpoint, the resumable full val/test runner
uses separate 7B dump and result directories:

```bash
bash run_training_free_stamp7b_full_eval.sh
```

Both Text4Seg and STAMP-7B exporters reuse completed per-sample files. Inspect
their exact counts and active processes with:

```bash
bash check_training_free_eval_status.sh
```

## Text4Seg integration

The shared graph core also accepts full-image probabilities and hard masks.
For official Text4Seg outputs, hard-mask uncertainty is constructed from the
distance to the coarse boundary, so refinement remains training-free and does
not read GT. The evaluator consumes Text4Seg's official sibling files:

```text
*_pred_mask.png
*_gt_mask.png
*_image.png
*_sam_mask.png (optional comparison)
```

The complete server runner clones Text4Seg, creates a clean H200-compatible
environment, downloads SAM-H for comparison, runs Text4Seg on the exact flat
JSON already used by STAMP, and evaluates coarse/training-free/SAM masks:

```bash
bash run_text4seg_training_free_eval.sh
```

It defaults to complete RefCOCOg val(U) with the public p24 demonstration
checkpoint. Set `TEXT4SEG_EVAL_JSON`, result paths, and optionally
`TEXT4SEG_EVAL_LIMIT` when starting a smoke or test(U) run. `p24` means a
24-by-24 semantic-descriptor grid, not 24 visual input tokens. It is not the
paper-default p16 protocol and must be labeled separately in result tables.
The released Text4Seg checkpoint records the author's relative CLIP path; the
runner maps it to the equivalent Hugging Face identifier
`openai/clip-vit-large-patch14-336` before model construction.

To evaluate that same public p24 checkpoint on all eight RefCOCO, RefCOCO+,
and RefCOCOg splits, while reusing already completed split outputs, run:

```bash
bash run_text4seg_public_p24_full_eval.sh
bash check_text4seg_public_p24_status.sh
```

These results are a public-checkpoint paired transfer experiment, not a
reproduction of the private ms-swift checkpoints used in Text4Seg Table 1.
The full runner reports four branches: Text4Seg, Text4Seg + FreeRef,
Text4Seg + frozen SAM-H, and Text4Seg + FreeRef + frozen SAM-H. It defaults to
four concurrent split workers on one large-memory GPU. Set
`TEXT4SEG_P24_PARALLEL_JOBS=2` if four model replicas exceed available memory.
Each worker writes a separate log below
`outputs/text4seg_public_p24_worker_logs/`. Failed parallel workers are retried
serially with resumable artifacts, including failures caused by transient OOM.

For the released LLaVA-1.5/Vicuna-7B p16 Table-4 checkpoint family, place the
checkpoint locally and run:

```bash
bash run_text4seg_llava7b_p16_full_eval.sh
```

The exporter validates that the configured descriptor grid agrees with the
checkpoint's `p16`, `p24`, or `p32` suffix. The released p16 checkpoint is not
the private ms-swift checkpoint used for the Text4Seg Table-1 rows copied by
STAMP. See `TEXT4SEG_CONFIG_ALIGNMENT_CN.md` for the exact row mapping and the
distinction between a released-checkpoint paired evaluation and a Table-1
reproduction.

## Publication-quality visualization

Analyze complete STAMP-7B and Text4Seg val results, generate uncertainty-aware
sample panels, aggregate diagnostics, and a cross-model comparison with:

```bash
bash run_training_free_visualizations.sh
```

The visualization step reuses saved dumps and masks and does not rerun either
base model. Use `TRAINING_FREE_VIS_LIMIT=8` for a quick layout smoke test. See
`TRAINING_FREE_VISUALIZATION_GUIDE_CN.md` for panel semantics, output paths,
selection protocol, tests, and full server commands.
