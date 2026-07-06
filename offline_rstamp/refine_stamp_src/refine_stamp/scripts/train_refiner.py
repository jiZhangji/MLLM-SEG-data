from __future__ import annotations

import torch
from torch.optim import AdamW

from refine_stamp.losses.refinement_losses import crop_gt_masks, refinement_loss


def build_optimizer(model, learning_rate: float = 1e-4, weight_decay: float = 1e-4) -> AdamW:
    trainable_parameters = [
        parameter
        for parameter in model.local_refiner.parameters()
        if parameter.requires_grad
    ]
    return AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(
    model,
    dataloader,
    optimizer: AdamW,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    model.stamp.eval()
    model.assert_stamp_frozen()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    num_steps = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        gt_masks = batch["masks"].to(device, non_blocking=True)
        texts = batch["texts"]

        outputs = model(images=images, texts=texts)
        local_targets = crop_gt_masks(
            gt_masks=gt_masks,
            boxes=outputs["boxes"],
            output_size=outputs["local_logits"].shape[-1],
        )
        losses = refinement_loss(outputs["local_logits"], local_targets)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.local_refiner.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += float(losses["loss"].item())
        total_bce += float(losses["loss_bce"].item())
        total_dice += float(losses["loss_dice"].item())
        num_steps += 1

    denominator = max(num_steps, 1)
    return {
        "loss": total_loss / denominator,
        "loss_bce": total_bce / denominator,
        "loss_dice": total_dice / denominator,
    }
