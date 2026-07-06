# Refine-STAMP：基于不确定性与边界选择的局部高分辨率精修

## 1. 项目目标

在尽量保留 STAMP 原有优势的前提下，提升其不使用 SAM 时的分割边界质量。

本项目不再继续沿用“给 STAMP 增加结构化文本先验”的方案，原因是：

- 在未对齐的低分辨率和非官方 prompt 设置下，结构化提示主要减少了 no-mask / empty-mask；
- 在官方分辨率、官方 prompt 和官方 checkpoint 对齐后，简单 text-only prior 基本没有提升；
- 这说明“结构化 prompt”更多是在修复早期评测流程中的触发问题，而不是稳定提升标准设置下的分割质量。

因此，新的研究方向直接针对 STAMP 更明确的弱点：

> STAMP 的 All-mask Prediction 能快速找到目标，但 patch 级输出较粗，特别是在边界、小目标和细长结构上仍有提升空间。

---

## 2. 核心创新思路

### 2.1 一句话描述

先让 STAMP 快速生成粗 mask，再根据 STAMP 自身输出的置信度和边界信息，只选择少量困难 patch，进行局部高分辨率精修。

### 2.2 与原始 STAMP 的区别

原始 STAMP：

```text
Image + Query
    ↓
Phase 1：生成文本并输出 <SEG>
    ↓
Phase 2：所有 [MASK] token 并行预测
    ↓
Patch-level coarse mask
    ↓
可选 SAM 精修
```

Refine-STAMP：

```text
Image + Query
    ↓
Frozen STAMP
    ↓
Patch-level coarse mask + FG probability + mask hidden states
    ↓
Uncertainty / Boundary Selector
    ↓
只选择困难 patch
    ↓
Local High-resolution Refiner
    ↓
Final fine-grained mask
```

### 2.3 预期创新点

#### 创新点 1：利用 STAMP 自身输出进行精修区域选择

不使用额外检测器，也不依赖外部 bbox。

选择依据包括：

- 前景概率接近 0.5 的不确定 patch；
- 粗 mask 前景和背景交界处的边界 patch；
- 后续可扩展到小目标、细长结构和局部不一致区域。

#### 创新点 2：动态局部计算

不是对整张图运行一个完整 mask decoder，而是只对少量困难区域计算。

例如：

- 总 patch 数：1024；
- 精修 patch 数：64；
- 实际精修比例：约 6.25%。

目标是：

- 比原始 STAMP 更精细；
- 比 STAMP + SAM 更轻；
- 比全图高分辨率 decoder 更快。

#### 创新点 3：保留 STAMP 原始能力

第一版中完全冻结 STAMP，只训练局部精修模块。

这样可以：

- 避免破坏 `<SEG>` 触发；
- 避免灾难性遗忘；
- 保留原始对话能力；
- 保留 All-mask Prediction 的高速特性；
- 让新模块只学习“如何修边界”。

---

## 3. 初步研究假设

需要通过实验验证以下假设。

### 假设 H1

STAMP 不使用 SAM 时，主要误差之一来自 patch 级分辨率不足，而不是目标理解失败。

### 假设 H2

STAMP 自身的前景概率可以反映局部不确定性。

### 假设 H3

边界 patch 和高不确定 patch 比随机 patch 更值得精修。

### 假设 H4

只精修少量困难 patch，可以获得接近全图精修的收益，同时显著降低额外开销。

### 假设 H5

冻结 STAMP、只训练局部 refiner，比直接微调整个 STAMP 更稳定。

---

## 4. 最小可行版本（MVP）

第一版不要追求复杂结构，只实现以下流程：

```text
1. 冻结 STAMP
2. 从 Phase 2 取出：
   - mask_logits
   - mask_hidden
   - grid_hw
3. 计算 uncertainty map
4. 计算 boundary map
5. 选择 Top-K patch
6. 从原图裁剪对应区域
7. 用一个小 CNN 预测局部高分辨率 mask
8. 将局部结果拼接回粗 mask
9. 评估总体 IoU、Boundary IoU、速度和显存
```

建议第一版参数：

```text
输入分辨率：896 × 896
Patch 网格：动态读取，常见为 32 × 32
Top-K：64
局部 crop：64 × 64
局部输出：16 × 16
Refiner：轻量 CNN
训练方式：只训练 refiner
损失：BCE + Dice
```

---

## 5. 建议代码目录

```text
refine_stamp/
├── README.md
├── configs/
│   └── refine_stamp_mvp.yaml
├── models/
│   ├── stamp_wrapper.py
│   ├── patch_selector.py
│   ├── local_refiner.py
│   └── refine_stamp.py
├── losses/
│   └── refinement_losses.py
├── data/
│   └── dataset_adapter.py
├── utils/
│   ├── geometry.py
│   ├── stitch.py
│   ├── visualization.py
│   └── metrics.py
├── scripts/
│   ├── inspect_stamp_outputs.py
│   ├── visualize_selector.py
│   ├── train_refiner.py
│   ├── eval_refiner.py
│   └── run_ablation.py
└── tests/
    ├── test_patch_coordinates.py
    ├── test_selector.py
    └── test_stitch.py
```

---

## 6. STAMP 需要暴露的中间结果

需要在 STAMP 的 Phase 2 中找到类似逻辑：

```python
z_mask = outputs.last_hidden_state[:, -num_mask_tokens:, :]
mask_logits = self.mask_classifier(z_mask)
```

增加一个用于 refinement 的接口，返回：

```python
{
    "mask_logits": mask_logits,  # [B, N, 2]
    "mask_hidden": z_mask,       # [B, N, D]
    "grid_hw": (grid_h, grid_w),
}
```

注意事项：

- `grid_hw` 不能写死为 `(32, 32)`；
- 必须从当前图像 token 数量或视觉 processor 中读取；
- 如果图像为动态分辨率，`grid_h * grid_w` 必须等于 `N`；
- STAMP 在训练 refiner 时必须保持 `eval()` 和 `requires_grad=False`。

---

## 7. 模块一：Frozen STAMP Wrapper

```python
# models/stamp_wrapper.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FrozenSTAMP(nn.Module):
    def __init__(self, stamp_model: nn.Module) -> None:
        super().__init__()
        self.stamp = stamp_model
        self.freeze_stamp()

    def freeze_stamp(self) -> None:
        self.stamp.eval()
        for parameter in self.stamp.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Expected output:
            mask_logits: [B, N, 2]
            mask_hidden: [B, N, D]
            grid_hw: tuple[int, int]
        """
        outputs = self.stamp.forward_for_refinement(
            images=images,
            texts=texts,
            **kwargs,
        )

        required_keys = {"mask_logits", "mask_hidden", "grid_hw"}
        missing = required_keys - outputs.keys()
        if missing:
            raise KeyError(f"Missing STAMP outputs: {sorted(missing)}")

        return outputs
```

Codex 实现时需要根据实际 STAMP 仓库替换：

```python
self.stamp.forward_for_refinement(...)
```

---

## 8. 模块二：Patch Selector

### 8.1 不确定性

对每个 patch 的前景概率：

```math
p_i = P(FG | z_i)
```

定义不确定性：

```math
u_i = 1 - |2p_i - 1|
```

含义：

- `p = 0.5` 时不确定性最大；
- `p = 0` 或 `p = 1` 时不确定性最小。

### 8.2 边界检测

将粗 mask 转为二值图。

如果一个 patch 的 3×3 邻域同时包含前景和背景，则认为它位于边界附近。

### 8.3 综合得分

```math
score_i = uncertainty_i + λ * boundary_i
```

第一版建议：

```text
boundary_weight = 0.5
top_k = 64
```

### 8.4 初步代码

```python
# models/patch_selector.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSelector(nn.Module):
    def __init__(
        self,
        top_k: int = 64,
        boundary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.top_k = top_k
        self.boundary_weight = boundary_weight

    @torch.no_grad()
    def forward(
        self,
        mask_logits: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            mask_logits: [B, N, 2]
            grid_hw: (grid_h, grid_w)

        Returns:
            selected_ids: [B, K]
            uncertainty_map: [B, 1, H, W]
            boundary_map: [B, 1, H, W]
            score_map: [B, 1, H, W]
        """
        if mask_logits.ndim != 3 or mask_logits.shape[-1] != 2:
            raise ValueError(
                f"mask_logits must have shape [B, N, 2], got {mask_logits.shape}"
            )

        batch_size, num_tokens, _ = mask_logits.shape
        grid_h, grid_w = grid_hw

        if grid_h * grid_w != num_tokens:
            raise ValueError(
                f"grid_hw={grid_hw} does not match num_tokens={num_tokens}"
            )

        fg_prob = torch.softmax(mask_logits, dim=-1)[..., 1]
        fg_prob = fg_prob.reshape(batch_size, 1, grid_h, grid_w)

        uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)

        coarse_binary = (fg_prob >= 0.5).float()

        local_max = F.max_pool2d(
            coarse_binary,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        local_min = -F.max_pool2d(
            -coarse_binary,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        boundary = (local_max - local_min).clamp(0.0, 1.0)

        score = uncertainty + self.boundary_weight * boundary

        flat_score = score.flatten(start_dim=1)
        k = min(self.top_k, num_tokens)

        selected_ids = torch.topk(
            flat_score,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        return {
            "selected_ids": selected_ids,
            "uncertainty_map": uncertainty,
            "boundary_map": boundary,
            "score_map": score,
            "fg_prob_map": fg_prob,
        }
```

---

## 9. 模块三：Patch 坐标转换

需要把 patch id 转换成原图中的矩形区域。

例如：

```text
Image: 896 × 896
Grid: 32 × 32
Each patch: 28 × 28 pixels
```

patch id 转行列：

```python
row = patch_id // grid_w
col = patch_id % grid_w
```

### 初步代码

```python
# utils/geometry.py

from __future__ import annotations

import torch


def patch_ids_to_boxes(
    selected_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Args:
        selected_ids: [B, K]
        grid_hw: (grid_h, grid_w)
        image_hw: (image_h, image_w)

    Returns:
        boxes: [B, K, 4], format: x1, y1, x2, y2
    """
    if selected_ids.ndim != 2:
        raise ValueError(
            f"selected_ids must have shape [B, K], got {selected_ids.shape}"
        )

    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw

    rows = selected_ids // grid_w
    cols = selected_ids % grid_w

    patch_h = image_h / grid_h
    patch_w = image_w / grid_w

    x1 = cols.float() * patch_w
    y1 = rows.float() * patch_h
    x2 = (cols.float() + 1) * patch_w
    y2 = (rows.float() + 1) * patch_h

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes
```

---

## 10. 模块四：Local Patch Refiner

### 10.1 输入

每个被选中的 patch 使用两类信息：

1. 原图对应区域的 RGB crop；
2. 对应 `[MASK]` token 的 hidden state。

这样：

- RGB crop 提供边缘和纹理；
- mask hidden state 提供与用户 query 相关的语义信息。

### 10.2 第一版网络

第一版使用小 CNN：

```text
RGB crop: 64 × 64
Mask token hidden: D
    ↓
Linear projection: D → 32
    ↓
扩展为 32 × 64 × 64
    ↓
与 RGB 拼接
    ↓
Conv blocks
    ↓
Local mask logits: 16 × 16
```

### 10.3 初步代码

```python
# models/local_refiner.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from utils.geometry import patch_ids_to_boxes


class LocalPatchRefiner(nn.Module):
    def __init__(
        self,
        token_dim: int,
        token_channels: int = 32,
        crop_size: int = 64,
        output_size: int = 16,
    ) -> None:
        super().__init__()

        self.crop_size = crop_size
        self.output_size = output_size

        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )

        input_channels = 3 + token_channels

        self.refiner = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask_hidden: torch.Tensor,
        selected_ids: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]
            mask_hidden: [B, N, D]
            selected_ids: [B, K]
            grid_hw: (grid_h, grid_w)

        Returns:
            local_logits: [B, K, 1, S, S]
            boxes: [B, K, 4]
        """
        if images.ndim != 4:
            raise ValueError(f"images must be [B, 3, H, W], got {images.shape}")

        batch_size, _, image_h, image_w = images.shape
        _, top_k = selected_ids.shape
        hidden_dim = mask_hidden.shape[-1]

        boxes = patch_ids_to_boxes(
            selected_ids=selected_ids,
            grid_hw=grid_hw,
            image_hw=(image_h, image_w),
        )

        box_list = [boxes[index] for index in range(batch_size)]

        image_crops = roi_align(
            input=images,
            boxes=box_list,
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            aligned=True,
        )

        gather_ids = selected_ids.unsqueeze(-1).expand(
            -1,
            -1,
            hidden_dim,
        )

        selected_hidden = torch.gather(
            mask_hidden,
            dim=1,
            index=gather_ids,
        )

        token_features = self.token_projection(selected_hidden)
        token_features = token_features.reshape(
            batch_size * top_k,
            -1,
            1,
            1,
        )
        token_features = token_features.expand(
            -1,
            -1,
            self.crop_size,
            self.crop_size,
        )

        features = torch.cat([image_crops, token_features], dim=1)
        local_logits = self.refiner(features)

        local_logits = F.interpolate(
            local_logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        local_logits = local_logits.reshape(
            batch_size,
            top_k,
            1,
            self.output_size,
            self.output_size,
        )

        return {
            "local_logits": local_logits,
            "boxes": boxes,
        }
```

---

## 11. 完整模型封装

```python
# models/refine_stamp.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.patch_selector import PatchSelector
from models.local_refiner import LocalPatchRefiner


class RefineSTAMP(nn.Module):
    def __init__(
        self,
        frozen_stamp: nn.Module,
        token_dim: int,
        top_k: int = 64,
        boundary_weight: float = 0.5,
        crop_size: int = 64,
        output_size: int = 16,
    ) -> None:
        super().__init__()

        self.stamp = frozen_stamp

        self.selector = PatchSelector(
            top_k=top_k,
            boundary_weight=boundary_weight,
        )

        self.local_refiner = LocalPatchRefiner(
            token_dim=token_dim,
            token_channels=32,
            crop_size=crop_size,
            output_size=output_size,
        )

    def train(self, mode: bool = True) -> "RefineSTAMP":
        super().train(mode)
        self.stamp.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        **stamp_kwargs: Any,
    ) -> dict[str, Any]:
        with torch.no_grad():
            stamp_outputs = self.stamp(
                images=images,
                texts=texts,
                **stamp_kwargs,
            )

        mask_logits = stamp_outputs["mask_logits"]
        mask_hidden = stamp_outputs["mask_hidden"]
        grid_hw = stamp_outputs["grid_hw"]

        selection = self.selector(
            mask_logits=mask_logits,
            grid_hw=grid_hw,
        )

        refinement = self.local_refiner(
            images=images,
            mask_hidden=mask_hidden,
            selected_ids=selection["selected_ids"],
            grid_hw=grid_hw,
        )

        return {
            **stamp_outputs,
            **selection,
            **refinement,
        }
```

---

## 12. 局部 GT mask 生成

```python
# losses/refinement_losses.py

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def crop_gt_masks(
    gt_masks: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Args:
        gt_masks: [B, 1, H, W]
        boxes: [B, K, 4]

    Returns:
        local_targets: [B, K, 1, output_size, output_size]
    """
    batch_size, top_k, _ = boxes.shape
    box_list = [boxes[index] for index in range(batch_size)]

    targets = roi_align(
        input=gt_masks.float(),
        boxes=box_list,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        aligned=True,
    )

    targets = targets.reshape(
        batch_size,
        top_k,
        1,
        output_size,
        output_size,
    )

    return (targets >= 0.5).float()


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=2)
    targets = targets.flatten(start_dim=2)

    intersection = (probabilities * targets).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def refinement_loss(
    local_logits: torch.Tensor,
    local_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(
        local_logits,
        local_targets,
    )
    dice = dice_loss(local_logits, local_targets)
    total = bce + dice

    return {
        "loss": total,
        "loss_bce": bce,
        "loss_dice": dice,
    }
```

---

## 13. 拼接最终 mask

### 13.1 基础方法

1. 将 STAMP 粗前景概率上采样到原图大小；
2. 将每个局部预测 resize 到对应 patch 尺寸；
3. 用局部预测替换或融合原粗 mask。

### 13.2 初步代码

```python
# utils/stitch.py

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def stitch_masks(
    coarse_logits: torch.Tensor,
    local_logits: torch.Tensor,
    selected_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    blend_weight: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        final_prob: [B, 1, H, W]
        final_mask: [B, 1, H, W]
    """
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be in [0, 1].")

    batch_size = coarse_logits.shape[0]
    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw

    coarse_prob = torch.softmax(coarse_logits, dim=-1)[..., 1]
    coarse_prob = coarse_prob.reshape(
        batch_size,
        1,
        grid_h,
        grid_w,
    )

    final_prob = F.interpolate(
        coarse_prob,
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )

    local_prob = torch.sigmoid(local_logits)

    patch_h = image_h / grid_h
    patch_w = image_w / grid_w
    top_k = selected_ids.shape[1]

    for batch_index in range(batch_size):
        for patch_index in range(top_k):
            patch_id = int(selected_ids[batch_index, patch_index].item())

            row = patch_id // grid_w
            col = patch_id % grid_w

            y1 = round(row * patch_h)
            y2 = round((row + 1) * patch_h)
            x1 = round(col * patch_w)
            x2 = round((col + 1) * patch_w)

            if y2 <= y1 or x2 <= x1:
                continue

            refined_patch = F.interpolate(
                local_prob[batch_index, patch_index].unsqueeze(0),
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0]

            old_patch = final_prob[
                batch_index,
                :,
                y1:y2,
                x1:x2,
            ]

            final_prob[
                batch_index,
                :,
                y1:y2,
                x1:x2,
            ] = (
                blend_weight * refined_patch
                + (1.0 - blend_weight) * old_patch
            )

    final_mask = final_prob >= 0.5
    return final_prob, final_mask
```

### 13.3 后续改进

如果出现明显方块接缝，可尝试：

- patch crop 扩大到原 patch 的 1.5 倍；
- 使用重叠 crop；
- 使用中心权重或 Gaussian blending；
- 让 refiner 输出 residual，而不是完整局部 mask；
- 对最终 mask 做轻量边缘平滑。

---

## 14. 训练流程

```python
# scripts/train_refiner.py

from __future__ import annotations

import torch
from torch.optim import AdamW

from losses.refinement_losses import (
    crop_gt_masks,
    refinement_loss,
)


def train_one_epoch(
    model,
    dataloader,
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    model.stamp.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    num_steps = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        gt_masks = batch["masks"].to(device, non_blocking=True)
        texts = batch["texts"]

        outputs = model(
            images=images,
            texts=texts,
        )

        local_targets = crop_gt_masks(
            gt_masks=gt_masks,
            boxes=outputs["boxes"],
            output_size=outputs["local_logits"].shape[-1],
        )

        losses = refinement_loss(
            local_logits=outputs["local_logits"],
            local_targets=local_targets,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            model.local_refiner.parameters(),
            max_norm=1.0,
        )

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


def build_optimizer(model) -> AdamW:
    trainable_parameters = [
        parameter
        for parameter in model.local_refiner.parameters()
        if parameter.requires_grad
    ]

    return AdamW(
        trainable_parameters,
        lr=1e-4,
        weight_decay=1e-4,
    )
```

---

## 15. 建议配置文件

```yaml
# configs/refine_stamp_mvp.yaml

model:
  top_k: 64
  boundary_weight: 0.5
  token_channels: 32
  crop_size: 64
  output_size: 16
  blend_weight: 0.8

training:
  epochs: 10
  batch_size: 2
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0
  amp: true

data:
  image_size: 896
  num_workers: 4

evaluation:
  threshold: 0.5
  report_boundary_iou: true
  report_small_object_iou: true
  report_latency: true
  report_memory: true
```

---

## 16. 实验设计

### 16.1 基线

必须包含：

```text
A. STAMP without SAM
B. STAMP + SAM
C. STAMP + Random-K refinement
D. STAMP + Uncertainty refinement
E. STAMP + Boundary refinement
F. STAMP + Uncertainty + Boundary refinement
G. STAMP + Full-grid refinement
```

### 16.2 消融实验

#### Patch 选择方式

```text
Random
Uncertainty only
Boundary only
Uncertainty + Boundary
Oracle GT boundary
```

#### Top-K

```text
K = 16
K = 32
K = 64
K = 128
K = 256
```

#### 局部输出尺寸

```text
8 × 8
16 × 16
32 × 32
```

#### 融合方式

```text
Full replacement
Fixed blend
Residual correction
Gaussian blending
```

#### 输入特征

```text
RGB only
Mask hidden only
RGB + Mask hidden
后续：ViT shallow feature + Mask hidden
```

### 16.3 指标

除 cIoU 和 gIoU 外，必须报告：

```text
Boundary IoU / Boundary F-score
Small-object IoU
Thin-object subset IoU
Inference latency
Peak GPU memory
Trainable parameter count
Refined patch ratio
```

---

## 17. 推荐实验顺序

### 阶段 0：确认原始 STAMP 输出

目标：

- 成功取到 `mask_logits`；
- 成功取到 `mask_hidden`；
- 成功读取动态 `grid_hw`；
- 与原始 STAMP 最终 mask 一致。

完成标准：

```text
重新构造出的 coarse mask 与原 STAMP 输出一致。
```

### 阶段 1：只实现 Selector

不训练 refiner。

需要可视化：

```text
原图
GT mask
STAMP coarse mask
Uncertainty map
Boundary map
Selected patches
```

完成标准：

```text
被选择区域大部分位于目标边界附近，而不是随机背景。
```

### 阶段 2：Oracle Selector

使用 GT 边界选择 patch，只训练 local refiner。

目的：

```text
验证局部精修是否有上限。
```

判断：

- 如果 Oracle selector 都不能提升 Boundary IoU，说明 refiner 设计有问题；
- 如果 Oracle selector 能提升，而预测 selector 不行，说明 selector 设计有问题。

### 阶段 3：公平 Selector

替换为：

```text
Uncertainty + Predicted Boundary
```

### 阶段 4：速度优化

第一版 RGB crop 成功后，再考虑：

```text
ViT shallow feature
    ↓
ROI Align
    ↓
Local refiner
```

这样可减少 RGB crop 的重复编码。

---

## 18. 初步成功标准

建议设置一个明确的 go / no-go 标准。

### 值得继续

满足以下任意一组：

```text
cIoU 提升 ≥ 1.0
Boundary IoU 提升 ≥ 3.0
额外延迟 ≤ 20%
```

或：

```text
性能接近 STAMP + SAM
但参数、显存或延迟显著更低
```

### 建议停止或调整

出现以下情况：

```text
Oracle selector 也几乎无提升
Boundary IoU 无提升
额外延迟接近或超过 SAM
随机区域与 uncertainty 区域效果相同
```

---

## 19. 关键风险

### 19.1 坐标错位

Qwen2-VL 可能对图像做：

- resize；
- dynamic resolution；
- padding；
- token reordering。

因此必须确保：

```text
原图
STAMP patch grid
GT mask
ROI crop
```

使用完全一致的坐标系。

建议实现：

```python
tests/test_patch_coordinates.py
```

对 patch 0、中心 patch、最后一个 patch 进行可视化检查。

### 19.2 grid_hw 写死

不能假设所有图像都是 32×32。

必须验证：

```python
grid_h * grid_w == num_mask_tokens
```

### 19.3 STAMP 被意外训练

每个 epoch 前检查：

```python
for name, parameter in model.stamp.named_parameters():
    assert parameter.requires_grad is False
```

### 19.4 Selector 只选择背景

可能原因：

- logits 未校准；
- 大面积背景概率接近 0.5；
- boundary map 计算异常。

建议：

- 可视化；
- 加前景邻域约束；
- 限制一部分 patch 必须来自 predicted boundary；
- 一部分来自 uncertainty。

### 19.5 Patch 接缝

第一版直接替换可能产生方块边缘。

解决方向：

- 重叠 crop；
- 扩大 ROI；
- Gaussian blending；
- residual refinement。

---

## 20. Codex 实现任务清单

建议按以下顺序交给 Codex。

### Task 1：定位 STAMP Phase 2

要求：

- 找到 `mask hidden states` 和 `mask classifier`；
- 新增 `forward_for_refinement()`；
- 返回 `mask_logits / mask_hidden / grid_hw`；
- 保证不影响原有推理接口。

### Task 2：实现中间结果检查脚本

脚本：

```text
scripts/inspect_stamp_outputs.py
```

输出：

```text
mask_logits.shape
mask_hidden.shape
grid_hw
num_mask_tokens
coarse foreground ratio
```

### Task 3：实现 Selector 与可视化

脚本：

```text
scripts/visualize_selector.py
```

保存：

```text
image.png
coarse_mask.png
uncertainty.png
boundary.png
selected_patches.png
```

### Task 4：实现 Oracle boundary selector

仅用于诊断，不进入最终方法。

### Task 5：实现 LocalPatchRefiner

要求：

- 支持 RGB crop；
- 支持 mask hidden；
- 支持动态 batch；
- 支持动态 grid_hw；
- 有 shape 检查和单元测试。

### Task 6：实现训练

要求：

- 冻结 STAMP；
- 只保存 refiner checkpoint；
- 记录 BCE / Dice / 总 loss；
- 支持 AMP；
- 支持 resume。

### Task 7：实现最终拼接与评测

要求：

- 复用原 STAMP 评测脚本；
- 增加 Boundary IoU；
- 记录推理延迟和显存；
- 输出可视化案例。

### Task 8：实现消融配置

至少支持：

```text
selector=random
selector=uncertainty
selector=boundary
selector=hybrid
selector=oracle
```

---

## 21. 推荐先让 Codex 回答的问题

在正式修改代码前，让 Codex 先分析仓库并回答：

```text
1. STAMP Phase 2 的入口文件和函数在哪里？
2. mask token hidden states 在哪一行生成？
3. mask classifier 在哪里？
4. 当前 grid_h / grid_w 是如何获得的？
5. 图像 resize 和 padding 在哪里完成？
6. GT mask 是否已经与 processor 后图像对齐？
7. 当前评测脚本如何从 patch prediction 恢复 mask？
8. 是否可以不改原接口，通过 forward hook 获取 hidden states？
```

在这些问题明确后，再开始修改。

---

## 22. 最终论文故事草案

### 问题

STAMP 通过 All-mask Prediction 实现高效并行分割，但 patch-level mask 在边界、小目标和细长结构上仍较粗。使用 SAM 可以精修，但增加了额外模型依赖和推理开销。

### 方法

提出 Refine-STAMP：

1. 利用 STAMP 的 patch-level 前景概率估计不确定性；
2. 根据粗 mask 检测边界区域；
3. 动态选择少量困难 patch；
4. 融合局部高分辨率视觉信息和 mask token 语义；
5. 只对困难区域进行轻量精修；
6. 保留原始 STAMP 作为稳定基础。

### 核心贡献

```text
1. Uncertainty-aware patch selection
2. Boundary-focused local refinement
3. Dynamic sparse computation
4. SAM-free refinement while preserving STAMP efficiency
```

### 一句话创新

> Refine-STAMP uses the uncertainty and boundary cues produced by All-mask Prediction to dynamically refine only difficult regions, improving fine-grained mask quality without relying on a heavyweight external segmenter.

中文：

> Refine-STAMP 利用 All-mask Prediction 输出的不确定性和边界信息，动态选择困难区域进行局部高分辨率精修，在不依赖大型外部分割模型的情况下提升细粒度掩码质量。

---

## 23. 当前最重要的原则

```text
先证明“局部精修有上限”，再优化 selector。
先做 Oracle selector，再做预测 selector。
先冻结 STAMP，再考虑联合训练。
先保证坐标正确，再看模型效果。
先看 Boundary IoU，再看总体 cIoU。
```
