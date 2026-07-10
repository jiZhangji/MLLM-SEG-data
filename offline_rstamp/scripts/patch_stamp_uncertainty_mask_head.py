#!/usr/bin/env python3
"""Patch official STAMP with a trainable uncertainty-aware mask head.

The patched second-stage path is:

    mask_hidden -> frozen STAMP classifier -> base logit
                -> learned error probability + residual correction
                -> final mask logit

The backbone and original classifier remain frozen by default. Training uses
the official online STAMP segmentation path, adds an explicit uncertainty
target, and saves only the small dynamic head.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


LEGACY_MAIN = "# >>> Uncertainty-aware mask head patch"
LEGACY_QWEN = "# >>> Uncertainty-aware mask head qwen patch"
MARKER_MAIN = "# >>> Uncertainty-aware mask head patch v2"
MARKER_QWEN = "# >>> Uncertainty-aware mask head qwen patch v2"
MARKER_TRAINER = "# >>> Uncertainty-aware mask head trainer patch v2"
MARKER_INFERENCE = "# >>> Uncertainty-aware mask head inference patch v2"


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_uncertainty_mask_head")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find expected block in {path}:\n{old[:500]}")
    return text.replace(old, new, 1)


def copy_module(tool_repo: Path, stamp_code_dir: Path) -> None:
    src = tool_repo / "online_token_refine"
    dst = stamp_code_dir / "online_token_refine"
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[OK] copied {src} -> {dst}")


def _head_setup_block() -> str:
    return f'''        {MARKER_MAIN}
        if os.environ.get("STAMP_UNCERTAINTY_MASK_HEAD", "0") == "1":
            from online_token_refine import UncertaintyAwareMaskHead, attach_uncertainty_head_from_checkpoint
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = self.model.config.text_config.hidden_size
            checkpoint_path = os.environ.get("STAMP_UNCERTAINTY_HEAD_CHECKPOINT", "").strip()
            if checkpoint_path:
                attach_uncertainty_head_from_checkpoint(self.model, checkpoint_path)
                if IS_MAIN_PROCESS:
                    print(f"--- Loaded uncertainty head: {{checkpoint_path}} ---")
            else:
                self.model.uncertainty_mask_head = UncertaintyAwareMaskHead(
                    token_dim=int(hidden_size),
                    hidden_size=int(os.environ.get("STAMP_REFINE_HIDDEN_SIZE", "128")),
                    use_uncertainty_gate=os.environ.get("STAMP_REFINE_USE_GATE", "1") == "1",
                )
                self.model.uncertainty_mask_head.initialize_from_classifier(self.model.classifier)
            if os.environ.get("STAMP_FREEZE_FOR_UNCERTAINTY_HEAD", "1") == "1":
                for name, param in self.model.named_parameters():
                    param.requires_grad = name.startswith("uncertainty_mask_head.")
                if os.environ.get("STAMP_REFINE_TRAIN_BASE", "0") != "1":
                    for param in self.model.uncertainty_mask_head.base_classifier.parameters():
                        param.requires_grad = False
            if IS_MAIN_PROCESS:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"--- Uncertainty-aware mask head enabled: trainable={{trainable}} / total={{total}} ---")
                print("--- STAMP backbone frozen; LoRA is not enabled for this run ---")
        # <<< Uncertainty-aware mask head patch v2
'''


def patch_main_uni(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup_once(path)

    setup = _head_setup_block()
    if MARKER_MAIN not in text:
        if LEGACY_MAIN in text:
            pattern = re.compile(
                r"        # >>> Uncertainty-aware mask head patch\n.*?"
                r"        # <<< Uncertainty-aware mask head patch\n",
                flags=re.S,
            )
            text, count = pattern.subn(setup, text, count=1)
            if count != 1:
                raise RuntimeError(f"Could not upgrade legacy head setup in {path}")
        else:
            old = '        self.model.mask_token_id = mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|mask|>")\n'
            text = replace_once(text, old, old + "\n" + setup, path)

    if "import torchvision.transforms.functional as TVF" not in text:
        text = text.replace("from PIL import Image\n", "from PIL import Image\nimport torchvision.transforms.functional as TVF\n", 1)

    old_images = """        for i, example in enumerate(examples):
            imgs = [Image.open(m) for m in example['images']]
            # to tensor
            all_images.append(imgs)
"""
    new_images = """        for i, example in enumerate(examples):
            if os.environ.get("STAMP_SAVE_SEG_VIS", "0") == "1":
                imgs = [Image.open(m) for m in example['images']]
                all_images.append(imgs)
            else:
                all_images.append([])
"""
    text = text.replace(old_images, new_images, 1)
    old_masks = """                masks = [Image.open(m) for m in example['masks']]
                # to tensor
                all_masks.append(masks)
"""
    new_masks = """                masks = [TVF.pil_to_tensor(Image.open(m).convert("L")).float().div_(255.0) for m in example['masks']]
                all_masks.append(masks)
"""
    text = text.replace(old_masks, new_masks, 1)

    text = text.replace(
        '            print("--- Configuring PEFT (LoRA) ---")',
        '            print("--- LoRA config is not attached; training uncertainty head only ---")',
        1,
    )

    if "dataloader_num_workers=int(os.environ.get(\"STAMP_NUM_WORKERS\"" not in text:
        needle = "            gradient_checkpointing_kwargs={'use_reentrant': False},\n"
        insert = (
            needle
            + '            dataloader_num_workers=int(os.environ.get("STAMP_NUM_WORKERS", "8")),\n'
            + "            dataloader_pin_memory=True,\n"
            + '            dataloader_persistent_workers=int(os.environ.get("STAMP_NUM_WORKERS", "8")) > 0,\n'
            + '            save_strategy="steps" if os.environ.get("STAMP_SAVE_FULL_CHECKPOINTS", "0") == "1" else "no",\n'
        )
        text = replace_once(text, needle, insert, path)

    trainer_old = """        trainer = SegmentationSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.processor,
            data_collator=self.collator,
        )
"""
    trainer_new = """        from online_token_refine import UncertaintyHeadCheckpointCallback
        trainer = SegmentationSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.processor,
            data_collator=self.collator,
            callbacks=[UncertaintyHeadCheckpointCallback(
                save_steps=int(os.environ.get("STAMP_SAVE_STEPS", "0")),
            )],
        )
"""
    if "UncertaintyHeadCheckpointCallback" not in text:
        text = replace_once(text, trainer_old, trainer_new, path)

    final_old = """        final_model_path = f"{self.output_dir}/final_model"
        trainer.save_model(final_model_path)
        if trainer.is_world_process_zero():
            print(f"--- Model saved to {final_model_path} ---")
"""
    final_new = """        if trainer.is_world_process_zero():
            from online_token_refine import save_uncertainty_head
            head_path = save_uncertainty_head(
                self.model,
                f"{self.output_dir}/uncertainty_mask_head.pt",
                step=trainer.state.global_step,
            )
            print(f"--- Uncertainty head saved to {head_path} ---")
        if os.environ.get("STAMP_SAVE_FULL_MODEL", "0") == "1":
            final_model_path = f"{self.output_dir}/final_model"
            trainer.save_model(final_model_path)
            if trainer.is_world_process_zero():
                print(f"--- Full model saved to {final_model_path} ---")
"""
    if "STAMP_SAVE_FULL_MODEL" not in text:
        text = replace_once(text, final_old, final_new, path)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched uncertainty head setup/training: {path}")


def patch_qwen_changes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup_once(path)

    if "learned_uncertainty: Optional[torch.FloatTensor]" not in text:
        dataclass_needle = "    bi_logits: Optional[torch.FloatTensor] = None\n"
        dataclass_fields = (
            dataclass_needle
            + "    base_binary_logits: Optional[torch.FloatTensor] = None\n"
            + "    learned_uncertainty: Optional[torch.FloatTensor] = None\n"
            + "    logit_uncertainty: Optional[torch.FloatTensor] = None\n"
            + "    delta_binary_logits: Optional[torch.FloatTensor] = None\n"
        )
        text = replace_once(text, dataclass_needle, dataclass_fields, path)

    new_tail = f'''            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=None,
                output_hidden_states=False,
                position_ids=position_ids,
                seg_mask=seg_mask,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state  {MARKER_QWEN}
            base_binary_logits = None
            learned_uncertainty = None
            logit_uncertainty = None
            delta_binary_logits = None
            if hasattr(self, "uncertainty_mask_head"):
                head_outputs = self.uncertainty_mask_head(last_hidden_state)
                logits = head_outputs["refined_binary_logits"]
                base_binary_logits = head_outputs["base_binary_logits"]
                learned_uncertainty = head_outputs["learned_uncertainty"]
                logit_uncertainty = head_outputs["logit_uncertainty"]
                delta_binary_logits = head_outputs["delta_binary_logits"]
            else:
                logits = self.classifier(last_hidden_state)

            return CustomModelOutput(
                bi_logits=logits,
                base_binary_logits=base_binary_logits,
                learned_uncertainty=learned_uncertainty,
                logit_uncertainty=logit_uncertainty,
                delta_binary_logits=delta_binary_logits,
                attentions=outputs.attentions,
            )
'''
    if MARKER_QWEN not in text:
        pattern = re.compile(
            r"            outputs = self\.model\(\n"
            r".*?"
            r"            return CustomModelOutput\(\n"
            r".*?"
            r"            \)\n",
            flags=re.S,
        )
        text, count = pattern.subn(new_tail, text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not patch classification forward in {path}")

    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched uncertainty head forward: {path}")


def patch_seg_trainer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_TRAINER in text:
        print(f"[SKIP] trainer already at v2: {path}")
        return
    backup_once(path)

    compute_needle = "    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):\n"
    compute_insert = compute_needle + f'''        {MARKER_TRAINER}
        raw_model = model.module if hasattr(model, "module") else model
        if os.environ.get("STAMP_FREEZE_FOR_UNCERTAINTY_HEAD", "1") == "1" and hasattr(raw_model, "uncertainty_mask_head"):
            raw_model.model.eval()
            raw_model.classifier.eval()
            raw_model.uncertainty_mask_head.train()
'''
    text = replace_once(text, compute_needle, compute_insert, path)

    initial_loss_old = "        loss_seg = torch.tensor(0.0, device=model.device)\n"
    initial_loss_new = (
        initial_loss_old
        + "        loss_uncertainty = torch.tensor(0.0, device=model.device)\n"
        + "        loss_delta_reg = torch.tensor(0.0, device=model.device)\n"
    )
    text = replace_once(text, initial_loss_old, initial_loss_new, path)

    call_old = """            seg_logits = model(input_ids=batch_input_ids, attention_mask=batch_attn_mask, pixel_values=batch_pixel_values,
                  image_grid_thw=batch_grid_thw, output_hidden_states=True, do_classification=True).bi_logits
            mask_preds = seg_logits[batch_idx_mask]
"""
    call_new = """            seg_outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attn_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_grid_thw,
                output_hidden_states=False,
                do_classification=True,
            )
            mask_preds = seg_outputs.bi_logits[batch_idx_mask]
            base_mask_preds = seg_outputs.base_binary_logits[batch_idx_mask]
            learned_uncertainty_preds = seg_outputs.learned_uncertainty[batch_idx_mask]
            delta_mask_preds = seg_outputs.delta_binary_logits[batch_idx_mask]
"""
    text = replace_once(text, call_old, call_new, path)

    losses_old = """            start_p = 0
            num_gt = 0
            loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)
"""
    losses_new = """            start_p = 0
            num_gt = 0
            loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)
"""
    text = replace_once(text, losses_old, losses_new, path)

    gt_old = """                    gt_mask = T.ToTensor()(gt_mask)[0].unsqueeze(0).unsqueeze(0).to(mask_preds.device)
"""
    gt_new = """                    if torch.is_tensor(gt_mask):
                        gt_mask = gt_mask[:1].unsqueeze(0).to(mask_preds.device, non_blocking=True)
                    else:
                        gt_mask = T.ToTensor()(gt_mask)[0].unsqueeze(0).unsqueeze(0).to(mask_preds.device)
"""
    text = replace_once(text, gt_old, gt_new, path)

    slice_old = """                    mask_pred = mask_preds[start_p: start_p + num_p]
                    start_p += num_p
                    gt_mask = gt_mask.view(-1)
                    binary_gt_labels = (gt_mask > 0.5).long()
                    num_pos = (binary_gt_labels == 1).sum().item()


                    loss_seg_ = loss_fn_weighted(mask_pred.float(), binary_gt_labels.unsqueeze(1).float())
"""
    slice_new = """                    token_slice = slice(start_p, start_p + num_p)
                    mask_pred = mask_preds[token_slice]
                    base_mask_pred = base_mask_preds[token_slice]
                    learned_uncertainty_pred = learned_uncertainty_preds[token_slice]
                    delta_mask_pred = delta_mask_preds[token_slice]
                    start_p += num_p
                    gt_mask = gt_mask.view(-1)
                    binary_gt_labels = (gt_mask > 0.5).long()

                    target = binary_gt_labels.unsqueeze(1).float()
                    loss_seg_ = loss_fn_weighted(mask_pred.float(), target)
                    uncertainty_target = (torch.sigmoid(base_mask_pred.detach().float()) - target).abs()
                    loss_uncertainty += F.binary_cross_entropy(
                        learned_uncertainty_pred.float().clamp(1e-5, 1.0 - 1e-5),
                        uncertainty_target,
                    )
                    loss_delta_reg += ((1.0 - uncertainty_target) * delta_mask_pred.float().pow(2)).mean()
"""
    text = replace_once(text, slice_old, slice_new, path)

    total_old = """            loss_seg = loss_seg / num_gt
        total_loss = loss_lm + loss_seg

        self._metrics['train']['loss_lm'].append(loss_lm.item())
        self._metrics['train']['loss_seg'].append(loss_seg.item())
"""
    total_new = """            if num_gt > 0:
                loss_seg = loss_seg / num_gt
                loss_uncertainty = loss_uncertainty / num_gt
                loss_delta_reg = loss_delta_reg / num_gt
        uncertainty_weight = float(os.environ.get("STAMP_UNCERTAINTY_LOSS_WEIGHT", "0.2"))
        delta_reg_weight = float(os.environ.get("STAMP_DELTA_REG_WEIGHT", "0.01"))
        total_loss = loss_lm + loss_seg + uncertainty_weight * loss_uncertainty + delta_reg_weight * loss_delta_reg

        self._metrics['train']['loss_lm'].append(loss_lm.item())
        self._metrics['train']['loss_seg'].append(loss_seg.item())
        self._metrics['train']['loss_uncertainty'].append(loss_uncertainty.item())
        self._metrics['train']['loss_delta_reg'].append(loss_delta_reg.item())
"""
    text = replace_once(text, total_old, total_new, path)

    log_old = """                logs['loss_seg'] = round(mean_loss_seg, 4)

                # CRITICAL: Overwrite the 'loss' key with the correct sum
                logs['loss'] = round(mean_loss_lm + mean_loss_seg, 4)
"""
    log_new = """                logs['loss_seg'] = round(mean_loss_seg, 4)
                uncertainty_losses = self._metrics['train']['loss_uncertainty']
                delta_reg_losses = self._metrics['train']['loss_delta_reg']
                mean_loss_uncertainty = 0.0
                mean_loss_delta_reg = 0.0
                if uncertainty_losses:
                    mean_loss_uncertainty = sum(uncertainty_losses) / len(uncertainty_losses)
                    logs['loss_uncertainty'] = round(mean_loss_uncertainty, 4)
                if delta_reg_losses:
                    mean_loss_delta_reg = sum(delta_reg_losses) / len(delta_reg_losses)
                    logs['loss_delta_reg'] = round(mean_loss_delta_reg, 4)

                # CRITICAL: Overwrite the 'loss' key with the correct sum
                uncertainty_weight = float(os.environ.get("STAMP_UNCERTAINTY_LOSS_WEIGHT", "0.2"))
                delta_reg_weight = float(os.environ.get("STAMP_DELTA_REG_WEIGHT", "0.01"))
                logs['loss'] = round(
                    mean_loss_lm
                    + mean_loss_seg
                    + uncertainty_weight * mean_loss_uncertainty
                    + delta_reg_weight * mean_loss_delta_reg,
                    4,
                )
"""
    text = replace_once(text, log_old, log_new, path)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched uncertainty supervision: {path}")


def patch_inference_loader(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if MARKER_INFERENCE in text:
        print(f"[SKIP] inference loader already at v2: {path}")
        return
    backup_once(path)
    needle = "        self.model.mask_token_id = self.mask_token_id\n"
    insert = needle + f'''        {MARKER_INFERENCE}
        uncertainty_checkpoint = os.environ.get("STAMP_UNCERTAINTY_HEAD_CHECKPOINT", "").strip()
        if uncertainty_checkpoint:
            from online_token_refine import attach_uncertainty_head_from_checkpoint
            attach_uncertainty_head_from_checkpoint(self.model, uncertainty_checkpoint)
            self.model.uncertainty_mask_head.eval()
            print(f"Loaded uncertainty mask head from '{{uncertainty_checkpoint}}'.")
        # <<< Uncertainty-aware mask head inference patch v2
'''
    text = replace_once(text, needle, insert, path)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched uncertainty head inference loading: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-repo", required=True, type=Path)
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    args = parser.parse_args()

    tool_repo = args.tool_repo.expanduser().resolve()
    stamp_code_dir = args.stamp_code_dir.expanduser().resolve()
    copy_module(tool_repo, stamp_code_dir)
    patch_main_uni(stamp_code_dir / "train" / "main_uni.py")
    patch_qwen_changes(stamp_code_dir / "model" / "qwen_changes.py")
    patch_seg_trainer(stamp_code_dir / "train" / "seg_trainer.py")
    patch_inference_loader(stamp_code_dir / "segment_predictor.py")
    patch_inference_loader(stamp_code_dir / "segment_predictor_cache.py")
    print("[DONE] Uncertainty-aware mask head v2 patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
