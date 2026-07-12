# OnePass Qwen2-VL-7B

This package trains query-driven referring segmentation from an untrained
Qwen2-VL-7B base checkpoint. It reuses STAMP's source-level Hybrid Attention
mechanism but does not initialize from STAMP segmentation weights.

## Input and attention

```text
[merged image tokens, user instruction, SEG, MASK_1:P]
  -> one Transformer forward
  -> shared scalar classifier on MASK hidden states
  -> patch mask
```

The number of MASK queries equals the number of merged image tokens entering
the language Transformer. Each query is constructed as:

```text
M_i = E_MASK + W_v V_i + E_row(row_i) + E_col(col_i)
```

`E_row` and `E_col` start at zero. `W_v` starts as identity. SEG and MASK are
new trainable query embeddings initialized from semantic averages of the base
language embedding table.

The image/text/SEG prefix remains causal. Every MASK query can attend to the
complete prefix and all MASK queries, giving bidirectional MASK-to-MASK
interaction through STAMP's Hybrid Attention mask.

## Fair initialization

- Load a base Qwen2-VL-7B checkpoint, not `STAMP-7B-uni`.
- Add fresh SEG/MASK tokenizer entries.
- Randomly initialize the scalar mask classifier.
- Train query embeddings, row/column positions, visual projection, classifier,
  and language-Transformer LoRA adapters.
- Keep the vision encoder and original 7B weights frozen.

The loader rejects checkpoints whose tokenizer already contains SEG/MASK by
default, preventing accidental STAMP-weight initialization.

## Training and inference

Training uses weighted BCE plus Dice and supports DDP, gradient accumulation,
gradient checkpointing, atomic step/epoch checkpoints, and mid-epoch resume.
Inference uses exactly the same strict-query input and never calls `generate()`
or a second model forward.

Start with a 16-sample overfit test before any full-data run.

