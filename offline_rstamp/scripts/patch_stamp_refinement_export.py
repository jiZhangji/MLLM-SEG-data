#!/usr/bin/env python3
"""Patch STAMP GenerativeSegmenter to export Phase-2 refinement tensors.

This appends a non-invasive monkey patch to `segment_predictor.py`. The original
`generate_with_segmentation()` API is left unchanged. The new method is:

    generate_with_refinement_outputs(image, prompt)

and the batched Phase-1 generation API is:

    generate_batch_with_refinement_outputs(images, prompts)

and returns:

    {
        "mask_logits": Tensor[1, N, 2],
        "mask_hidden": Tensor[1, N, D],
        "grid_hw": (grid_h, grid_w),
        "response_text": str,
    }

STAMP's current `bi_logits` is a single binary foreground logit. For selector
code expecting two-class logits, we export `[-logit, +logit]`; the foreground
threshold remains equivalent to `bi_logits > 0`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "# >>> Refine-STAMP Phase-2 export patch"
PATCH_END_MARKER = "# <<< Refine-STAMP Phase-2 export patch"


PATCH_CODE = r'''

# >>> Refine-STAMP Phase-2 export patch
@torch.no_grad()
def _refine_stamp_generate_with_refinement_outputs(self, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [{"image": image}, {"text": prompt}]}]
    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self.processor(text=[text], images=[image], return_tensors="pt")
    merge_size = self.processor.image_processor.merge_size

    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    image_grid_thw = inputs.get("image_grid_thw").to(self.device)
    attention_mask_raw = inputs["attention_mask"].to(self.device)

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        eos_token_id=self.eos_token_id,
        pad_token_id=self.tokenizer.pad_token_id,
    )

    sequence = outputs.sequences[0]
    full_past_key_values = outputs.past_key_values
    seg_indices = torch.where(sequence == self.seg_token_id)[0].tolist()

    generated_ids = sequence[prompt_len:]
    response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    if not seg_indices:
        raise RuntimeError("No <|seg|> token generated; cannot export refinement tensors.")

    num_patches = find_image_patch_info(self.image_pad_id, inputs["input_ids"])

    # Export the first segmentation task. Multi-<seg> support can be added later
    # by returning a list, but Refine-STAMP Phase 1 uses single-object samples.
    idx = seg_indices[0]
    sliced_len = idx + 1
    attention_mask = attention_mask_raw[:, :sliced_len]
    legacy_cache = full_past_key_values.to_legacy_cache()
    past_key_values_sliced = tuple(
        (
            key_layer[:, :, :sliced_len, :],
            value_layer[:, :, :sliced_len, :],
        )
        for key_layer, value_layer in legacy_cache
    )
    past_key_values_sliced = DynamicCache.from_legacy_cache(past_key_values_sliced)

    mask_query_ids = torch.full((1, num_patches), self.mask_token_id, dtype=torch.long, device=self.device)
    mask_query_attention_mask = torch.ones(
        (1, num_patches + sliced_len - attention_mask[0].sum()),
        dtype=torch.long,
        device=self.device,
    )
    mask_query_attention_mask = torch.cat((attention_mask, mask_query_attention_mask), dim=1)
    mask_grid_thw = image_grid_thw[-1].clone().unsqueeze(0)

    mask_pre_ids = sequence.clone().unsqueeze(0)
    mask_ids = torch.cat([mask_pre_ids[0, :sliced_len], mask_query_ids[0]], dim=0).unsqueeze(0)

    seg_forward_outputs = self.model(
        input_ids=mask_ids,
        attention_mask=mask_query_attention_mask,
        image_grid_thw=image_grid_thw,
        pixel_values=inputs["pixel_values"],
        past_key_values=past_key_values_sliced,
        return_dict=True,
        do_classification=True,
        output_hidden_states=True,
    )

    binary_logits = seg_forward_outputs.bi_logits[:, -num_patches:]
    if binary_logits.ndim == 3 and binary_logits.shape[-1] == 1:
        binary_logits = binary_logits.squeeze(-1)
    mask_logits = torch.stack([-binary_logits, binary_logits], dim=-1)

    hidden_states = getattr(seg_forward_outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > 0:
        mask_hidden = hidden_states[-1][:, -num_patches:, :]
        mask_hidden_is_fallback = False
    elif hasattr(seg_forward_outputs, "last_hidden_state"):
        mask_hidden = seg_forward_outputs.last_hidden_state[:, -num_patches:, :]
        mask_hidden_is_fallback = False
    else:
        # Phase-1 selector quality does not use mask_hidden, but the dump format
        # keeps the field present. If this fallback appears, patch the model
        # forward to return hidden states before training the local refiner.
        mask_hidden = binary_logits.unsqueeze(-1)
        mask_hidden_is_fallback = True

    h_grid, w_grid = mask_grid_thw[0, 1:]
    h_grid, w_grid = int(h_grid / merge_size), int(w_grid / merge_size)

    return {
        "mask_logits": mask_logits,
        "mask_hidden": mask_hidden,
        "grid_hw": (h_grid, w_grid),
        "response_text": response_text,
        "num_patches": int(num_patches),
        "mask_hidden_is_fallback": mask_hidden_is_fallback,
    }


GenerativeSegmenter.generate_with_refinement_outputs = _refine_stamp_generate_with_refinement_outputs


@torch.no_grad()
def _refine_stamp_generate_batch_with_refinement_outputs(self, images, prompts):
    """Batch autoregressive generation, then export each sample independently.

    Qwen2-VL already supports batched generation. STAMP's released predictor
    only indexes sequence zero, so this method keeps one image/query per sample,
    selects each sample's KV cache and visual patches, and performs the mask
    classification forward separately. The expensive autoregressive phase is
    shared while variable image grids remain exact.
    """
    images = list(images)
    prompts = list(prompts)
    if len(images) != len(prompts) or not images:
        raise ValueError("images and prompts must be non-empty lists with equal length.")

    texts = []
    for image, prompt in zip(images, prompts):
        messages = [{"role": "user", "content": [{"image": image}, {"text": prompt}]}]
        texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    original_padding_side = self.tokenizer.padding_side
    self.tokenizer.padding_side = "left"
    try:
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
    finally:
        self.tokenizer.padding_side = original_padding_side

    merge_size = self.processor.image_processor.merge_size
    inputs = {key: value.to(self.device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    image_grid_thw = inputs["image_grid_thw"]
    attention_mask_raw = inputs["attention_mask"]

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        eos_token_id=self.eos_token_id,
        pad_token_id=self.tokenizer.pad_token_id,
    )
    full_past_key_values = outputs.past_key_values
    legacy_cache = full_past_key_values.to_legacy_cache()
    rope_owner = getattr(self.model, "model", None)
    batched_rope_deltas = getattr(rope_owner, "rope_deltas", None)
    if torch.is_tensor(batched_rope_deltas):
        batched_rope_deltas = batched_rope_deltas.clone()

    raw_patch_counts = [int(grid.prod().item()) for grid in image_grid_thw]
    raw_patch_offsets = [0]
    for count in raw_patch_counts:
        raw_patch_offsets.append(raw_patch_offsets[-1] + count)

    batch_outputs = []
    for batch_index in range(len(images)):
        response_text = None
        try:
            sequence = outputs.sequences[batch_index]
            generated = sequence[input_width:]
            generated_seg = torch.where(generated == self.seg_token_id)[0]
            response_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if generated_seg.numel() == 0:
                raise RuntimeError("No <|seg|> token generated; cannot export refinement tensors.")

            idx = input_width + int(generated_seg[0].item())
            sliced_len = idx + 1
            sample_input_ids = inputs["input_ids"][batch_index : batch_index + 1]
            num_patches = find_image_patch_info(self.image_pad_id, sample_input_ids)
            sample_attention = attention_mask_raw[batch_index : batch_index + 1]
            generated_attention_len = sliced_len - input_width + num_patches
            generated_attention = torch.ones(
                (1, generated_attention_len), dtype=sample_attention.dtype, device=self.device
            )
            mask_attention = torch.cat((sample_attention, generated_attention), dim=1)

            sample_cache = tuple(
                (
                    key_layer[batch_index : batch_index + 1, :, :sliced_len, :],
                    value_layer[batch_index : batch_index + 1, :, :sliced_len, :],
                )
                for key_layer, value_layer in legacy_cache
            )
            sample_cache = DynamicCache.from_legacy_cache(sample_cache)

            mask_query_ids = torch.full(
                (num_patches,), self.mask_token_id, dtype=torch.long, device=self.device
            )
            mask_ids = torch.cat((sequence[:sliced_len], mask_query_ids), dim=0).unsqueeze(0)
            sample_grid = image_grid_thw[batch_index : batch_index + 1]
            pixel_start = raw_patch_offsets[batch_index]
            pixel_end = raw_patch_offsets[batch_index + 1]
            sample_pixels = inputs["pixel_values"][pixel_start:pixel_end]

            if torch.is_tensor(batched_rope_deltas) and batched_rope_deltas.shape[0] == len(images):
                rope_owner.rope_deltas = batched_rope_deltas[batch_index : batch_index + 1].clone()

            seg_forward_outputs = self.model(
                input_ids=mask_ids,
                attention_mask=mask_attention,
                image_grid_thw=sample_grid,
                pixel_values=sample_pixels,
                past_key_values=sample_cache,
                return_dict=True,
                do_classification=True,
                output_hidden_states=True,
            )

            binary_logits = seg_forward_outputs.bi_logits[:, -num_patches:]
            if binary_logits.ndim == 3 and binary_logits.shape[-1] == 1:
                binary_logits = binary_logits.squeeze(-1)
            mask_logits = torch.stack((-binary_logits, binary_logits), dim=-1)

            hidden_states = getattr(seg_forward_outputs, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) > 0:
                mask_hidden = hidden_states[-1][:, -num_patches:, :]
                mask_hidden_is_fallback = False
            elif hasattr(seg_forward_outputs, "last_hidden_state"):
                mask_hidden = seg_forward_outputs.last_hidden_state[:, -num_patches:, :]
                mask_hidden_is_fallback = False
            else:
                mask_hidden = binary_logits.unsqueeze(-1)
                mask_hidden_is_fallback = True

            h_grid, w_grid = sample_grid[0, 1:]
            h_grid, w_grid = int(h_grid / merge_size), int(w_grid / merge_size)
            batch_outputs.append(
                {
                    "mask_logits": mask_logits,
                    "mask_hidden": mask_hidden,
                    "grid_hw": (h_grid, w_grid),
                    "response_text": response_text,
                    "num_patches": int(num_patches),
                    "mask_hidden_is_fallback": mask_hidden_is_fallback,
                }
            )
        except Exception as exc:
            batch_outputs.append(
                {
                    "batch_export_error": f"{type(exc).__name__}: {exc}",
                    "batch_response_text": response_text,
                    "batch_index": batch_index,
                }
            )

    if rope_owner is not None:
        rope_owner.rope_deltas = batched_rope_deltas

    return batch_outputs


GenerativeSegmenter.generate_batch_with_refinement_outputs = _refine_stamp_generate_batch_with_refinement_outputs
# <<< Refine-STAMP Phase-2 export patch
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--target", choices=["segment_predictor", "segment_predictor_cache", "both"], default="segment_predictor")
    args = parser.parse_args()

    root = args.stamp_code_dir.expanduser().resolve()
    targets = []
    if args.target in {"segment_predictor", "both"}:
        targets.append(root / "segment_predictor.py")
    if args.target in {"segment_predictor_cache", "both"}:
        targets.append(root / "segment_predictor_cache.py")

    patched = []
    for path in targets:
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if PATCH_MARKER in text:
            start = text.index(PATCH_MARKER)
            end = text.find(PATCH_END_MARKER, start)
            if end == -1:
                raise SystemExit(f"Found patch start but not end marker in {path}")
            end += len(PATCH_END_MARKER)
            text = text[:start].rstrip() + PATCH_CODE + text[end:].lstrip("\n")
            path.write_text(text.rstrip() + "\n", encoding="utf-8")
            print(f"[OK] refreshed existing patch: {path}")
            patched.append(str(path))
            continue
        path.write_text(text.rstrip() + PATCH_CODE + "\n", encoding="utf-8")
        print(f"[OK] patched: {path}")
        patched.append(str(path))

    if not patched:
        raise SystemExit("No files patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
