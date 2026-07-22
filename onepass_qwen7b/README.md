# OnePass Qwen2-VL-7B

This package trains query-driven referring segmentation from an untrained
Qwen2-VL-7B base checkpoint. It reuses STAMP's source-level Hybrid Attention
mechanism. The default path does not initialize from STAMP segmentation
weights; `--stamp-adapter` enables a separate STAMP-LoRA warm-start experiment.

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

## STAMP-LoRA warm start

Pass `--stamp-adapter /path/to/STAMP-7B-lora` to initialize every compatible
language-model LoRA A/B tensor from the trained STAMP PEFT adapter. The loader
reads `adapter_config.json` and automatically adopts its rank, alpha, dropout,
target modules, and RS-LoRA scaling. By default it also initializes the
same-shaped scalar mask classifier; use `--no-stamp-init-classifier` for a
LoRA-only initialization ablation.

The OnePass-specific SEG/MASK queries, visual projection, and row/column
embeddings remain OnePass modules and are optimized together with the loaded
LoRA tensors. Saved OnePass checkpoints contain the complete trained LoRA,
query, and classifier state, so evaluation does not need to reload the source
STAMP adapter.

## Optional SEG grounding fine-tuning

`--use-seg-grounding` adds a small side head without replacing the existing
mask classifier. It projects the contextual SEG state and every spatial MASK
state into a shared space and supervises their similarity with the token-grid
ground-truth mask. Training uses:

```text
L = L_mask + lambda_seg * L_seg
final_logits = raw_classifier_logits + tanh(alpha) * seg_logits
```

The fusion parameter `alpha` starts at zero, so enabling this branch initially
preserves the parent OnePass output exactly. `--init-onepass-checkpoint` loads a
completed parent model while resetting epochs, optimizer, and LR scheduling;
this differs from `--resume`, which strictly continues an interrupted run.

`run_onepass7b_seg_grounding_2gpu.sh` polls both visible GPUs every 10 seconds
until they simultaneously meet the free-memory threshold. It then validates a
CUDA allocation and a two-rank NCCL all-reduce, retrying if resources change,
and starts full DDP fine-tuning only after every check passes.

## Training and inference

Training uses weighted BCE plus Dice and supports DDP, gradient accumulation,
gradient checkpointing, atomic step/epoch checkpoints, and mid-epoch resume.
Inference uses the prompt mode recorded by training and never calls `generate()`
or a second model forward.

Start with a 16-sample overfit test before any full-data run.

## Semantic-path calibration

Older SEG-grounding checkpoints learned a fusion coefficient close to zero, so
their final prediction bypassed the explicit SEG branch. The evaluator now
supports two backward-compatible diagnostics:

- `--prompt-mode semantic_anchor` inserts the same deterministic assistant-side
  `The referred object is <|seg|>.` context used by STAMP's classification pass,
  while retaining one model call and zero autoregressive generation calls.
- `--seg-fusion-sweep ...` evaluates `raw_logits + alpha * seg_logits` for many
  fixed coefficients from one forward pass.

Use `run_onepass7b_semantic_search.sh --checkpoint CHECKPOINT` to search both
prompt modes and several coefficients on a validation subset. For new training,
`--seg-fusion-alpha 0.25 --prompt-mode semantic_anchor` keeps the semantic path
active instead of learning its coefficient from zero.
