from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .refiner import TrainingFreeRefineConfig


def _gpu_modules():
    try:
        import cupy as cp
        from cucim.core.operations.morphology import distance_transform_edt
        from cucim.skimage.color import rgb2lab
        from cupyx.scipy import sparse
        from cupyx.scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
        from cupyx.scipy.sparse.linalg import cg, spsolve
    except ImportError as exc:
        raise RuntimeError(
            "GPU FreeRef requires cupy-cuda12x and cucim-cu12. "
            "Run prepare_freeref_efficiency_env.sh in the target conda environment."
        ) from exc
    return cp, rgb2lab, sparse, cg, spsolve, binary_dilation, binary_erosion, gaussian_filter, distance_transform_edt


def _as_cupy(value: np.ndarray | torch.Tensor, cp, *, dtype=None):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous()
        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=False)
        array = cp.from_dlpack(tensor)
    else:
        array = cp.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _as_torch(value) -> torch.Tensor:
    return torch.from_dlpack(value)


def stamp_probability_gpu(
    mask_logits: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    logits = mask_logits.detach().float()
    if not logits.is_cuda:
        logits = logits.cuda(non_blocking=False)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError(
            f"mask_logits must have shape [N, 2] or [1, N, 2], got {tuple(logits.shape)}"
        )
    grid_h, grid_w = (int(value) for value in grid_hw)
    if grid_h * grid_w != logits.shape[0]:
        raise ValueError(f"grid_hw={grid_hw} does not match N={logits.shape[0]}.")
    foreground = torch.softmax(logits, dim=-1)[..., 1].reshape(1, 1, grid_h, grid_w)
    return F.interpolate(foreground, size=image_hw, mode="bilinear", align_corners=False)[0, 0]


def _segment_means(values, labels, count: int, cp):
    flat_labels = labels.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1]) if values.ndim == 3 else values.reshape(-1)
    sizes = cp.bincount(flat_labels, minlength=count).astype(cp.float32)
    sizes = cp.maximum(sizes, 1.0)
    if flat_values.ndim == 1:
        return cp.bincount(flat_labels, weights=flat_values, minlength=count) / sizes
    channels = [
        cp.bincount(flat_labels, weights=flat_values[:, index], minlength=count)
        for index in range(flat_values.shape[1])
    ]
    return cp.stack(channels, axis=1) / sizes[:, None]


def _adjacent_pairs(labels, segment_count: int, cp):
    first = cp.concatenate((labels[:, :-1].reshape(-1), labels[:-1, :].reshape(-1)))
    second = cp.concatenate((labels[:, 1:].reshape(-1), labels[1:, :].reshape(-1)))
    keep = first != second
    first = first[keep]
    second = second[keep]
    if first.size == 0:
        return cp.empty((0, 2), dtype=cp.int64)
    low = cp.minimum(first, second).astype(cp.int64, copy=False)
    high = cp.maximum(first, second).astype(cp.int64, copy=False)
    encoded = cp.unique(low * segment_count + high)
    return cp.stack((encoded // segment_count, encoded % segment_count), axis=1)


def _slic_gpu(image_rgb, n_segments: int, compactness: float, sigma: float, modules):
    """SLIC on CUDA using each pixel's 3x3 neighboring center cells."""
    cp, rgb2lab, *_, gaussian_filter, _ = modules
    height, width = image_rgb.shape[:2]
    grid_y = max(1, int(round((n_segments * height / width) ** 0.5)))
    grid_x = max(1, int(round(n_segments / grid_y)))
    center_y = (cp.arange(grid_y, dtype=cp.float32) + 0.5) * height / grid_y
    center_x = (cp.arange(grid_x, dtype=cp.float32) + 0.5) * width / grid_x
    yy, xx = cp.meshgrid(
        cp.arange(height, dtype=cp.float32),
        cp.arange(width, dtype=cp.float32),
        indexing="ij",
    )
    base_y = cp.minimum((yy * grid_y / height).astype(cp.int32), grid_y - 1)
    base_x = cp.minimum((xx * grid_x / width).astype(cp.int32), grid_x - 1)
    offsets_y = cp.asarray((-1, -1, -1, 0, 0, 0, 1, 1, 1), dtype=cp.int32)
    offsets_x = cp.asarray((-1, 0, 1, -1, 0, 1, -1, 0, 1), dtype=cp.int32)
    candidate_y = cp.clip(base_y[..., None] + offsets_y, 0, grid_y - 1)
    candidate_x = cp.clip(base_x[..., None] + offsets_x, 0, grid_x - 1)
    candidate_ids = candidate_y * grid_x + candidate_x
    smoothed = (
        gaussian_filter(image_rgb, sigma=(sigma, sigma, 0.0)) if sigma > 0 else image_rgb
    )
    lab = rgb2lab(smoothed).astype(cp.float32, copy=False)
    initial_y, initial_x = cp.meshgrid(center_y, center_x, indexing="ij")
    centers_y = initial_y.reshape(-1)
    centers_x = initial_x.reshape(-1)
    nearest_y = cp.clip(cp.rint(centers_y).astype(cp.int32), 0, height - 1)
    nearest_x = cp.clip(cp.rint(centers_x).astype(cp.int32), 0, width - 1)
    centers_lab = lab[nearest_y, nearest_x].copy()
    segment_count = grid_y * grid_x
    step = (height * width / segment_count) ** 0.5
    spatial_weight = (compactness / max(step, 1e-6)) ** 2
    labels = base_y * grid_x + base_x
    flat_lab = lab.reshape(-1, 3)
    flat_y = yy.reshape(-1)
    flat_x = xx.reshape(-1)
    flat_candidates = candidate_ids.reshape(-1, 9)
    for _ in range(10):
        color_delta = flat_lab[:, None, :] - centers_lab[flat_candidates]
        spatial = (
            cp.square(flat_y[:, None] - centers_y[flat_candidates])
            + cp.square(flat_x[:, None] - centers_x[flat_candidates])
        )
        distance = cp.sum(cp.square(color_delta), axis=2) + spatial_weight * spatial
        labels = cp.take_along_axis(
            flat_candidates, cp.argmin(distance, axis=1)[:, None], axis=1
        )[:, 0]
        counts = cp.bincount(labels, minlength=segment_count).astype(cp.float32)
        valid = counts > 0
        divisor = cp.maximum(counts, 1.0)
        for channel in range(3):
            updated = cp.bincount(
                labels, weights=flat_lab[:, channel], minlength=segment_count
            ) / divisor
            centers_lab[:, channel] = cp.where(valid, updated, centers_lab[:, channel])
        updated_y = cp.bincount(labels, weights=flat_y, minlength=segment_count) / divisor
        updated_x = cp.bincount(labels, weights=flat_x, minlength=segment_count) / divisor
        centers_y = cp.where(valid, updated_y, centers_y)
        centers_x = cp.where(valid, updated_x, centers_x)
    return labels.reshape(height, width).astype(cp.int64, copy=False), lab


def _boundary_uncertainty(mask, sigma: float, modules):
    cp, _, _, _, _, binary_dilation, binary_erosion, _, distance_transform_edt = modules
    if sigma <= 0:
        raise ValueError("boundary_sigma must be positive.")
    hard = mask.astype(cp.bool_, copy=False)
    if not bool(hard.any()) or bool(hard.all()):
        return cp.zeros(hard.shape, dtype=cp.float32)
    boundary = cp.logical_xor(binary_dilation(hard), binary_erosion(hard))
    distance = distance_transform_edt(~boundary, float64_distances=False)
    return cp.exp(-0.5 * cp.square(distance.astype(cp.float32) / sigma))


class GpuTrainingFreeUncertaintyRefiner:
    """CUDA implementation of FreeRef with no learned or fitted parameters."""

    def __init__(self, config: TrainingFreeRefineConfig | None = None) -> None:
        self.config = config or TrainingFreeRefineConfig()

    def refine(
        self,
        image: np.ndarray | torch.Tensor,
        mask_logits: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, Any]:
        image_hw = (int(image.shape[0]), int(image.shape[1]))
        coarse = stamp_probability_gpu(mask_logits, grid_hw, image_hw)
        return self.refine_probability(image, coarse)

    def refine_hard_mask(
        self,
        image: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        boundary_sigma: float = 8.0,
    ) -> dict[str, Any]:
        modules = _gpu_modules()
        cp = modules[0]
        coarse = _as_cupy(mask, cp, dtype=cp.float32)
        uncertainty = _boundary_uncertainty(
            coarse >= self.config.threshold, boundary_sigma, modules
        )
        return self._refine_cupy(image, coarse, uncertainty, modules)

    def refine_probability(
        self,
        image: np.ndarray | torch.Tensor,
        probability: np.ndarray | torch.Tensor,
        uncertainty: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        modules = _gpu_modules()
        cp = modules[0]
        coarse = _as_cupy(probability, cp, dtype=cp.float32)
        uncertainty_map = (
            cp.clip(1.0 - cp.abs(2.0 * coarse - 1.0), 0.0, 1.0)
            if uncertainty is None
            else cp.clip(_as_cupy(uncertainty, cp, dtype=cp.float32), 0.0, 1.0)
        )
        return self._refine_cupy(image, coarse, uncertainty_map, modules)

    def _refine_cupy(self, image, coarse, uncertainty_map, modules) -> dict[str, Any]:
        cp, _, sparse, cg, spsolve, *_ = modules
        image_rgb = _as_cupy(image, cp)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image must have shape [H, W, 3], got {image_rgb.shape}")
        image_rgb = image_rgb.astype(cp.float32, copy=False)
        if float(image_rgb.max()) > 1.0:
            image_rgb = image_rgb / 255.0
        image_rgb = cp.clip(image_rgb, 0.0, 1.0)
        image_hw = tuple(int(value) for value in image_rgb.shape[:2])
        if coarse.shape != image_hw or uncertainty_map.shape != image_hw:
            raise ValueError(
                f"probability and uncertainty must match image shape {image_hw}; "
                f"got {coarse.shape} and {uncertainty_map.shape}."
            )
        if not bool(cp.isfinite(coarse).all()) or not bool(cp.isfinite(uncertainty_map).all()):
            raise ValueError("probability or uncertainty contains non-finite values.")
        coarse = cp.clip(coarse, 0.0, 1.0)

        labels, lab = _slic_gpu(
            image_rgb,
            n_segments=self.config.n_segments,
            compactness=self.config.compactness,
            sigma=self.config.slic_sigma,
            modules=modules,
        )
        segment_count = int(labels.max().item()) + 1
        mean_color = _segment_means(lab, labels, segment_count, cp)
        mean_probability = _segment_means(coarse, labels, segment_count, cp)
        mean_uncertainty = _segment_means(uncertainty_map, labels, segment_count, cp)
        pairs = _adjacent_pairs(labels, segment_count, cp)

        if pairs.size:
            color_distance = cp.linalg.norm(
                mean_color[pairs[:, 0]] - mean_color[pairs[:, 1]], axis=1
            )
            positive = color_distance[color_distance > 1e-8]
            color_scale = cp.median(positive) if positive.size else cp.asarray(1.0, cp.float32)
            weights = cp.exp(-cp.square(color_distance / cp.maximum(color_scale, 1e-6)))
            weights = cp.clip(weights, 1e-4, 1.0)
            rows = cp.concatenate((pairs[:, 0], pairs[:, 1]))
            cols = cp.concatenate((pairs[:, 1], pairs[:, 0]))
            values = cp.concatenate((weights, weights))
            affinity = sparse.csr_matrix(
                (values, (rows, cols)), shape=(segment_count, segment_count)
            )
        else:
            color_scale = cp.asarray(1.0, cp.float32)
            affinity = sparse.csr_matrix((segment_count, segment_count), dtype=cp.float32)

        degree = cp.asarray(affinity.sum(axis=1)).reshape(-1)
        laplacian = sparse.diags(degree) - affinity
        confidence = cp.power(
            cp.clip(1.0 - mean_uncertainty, 0.0, 1.0), self.config.confidence_power
        ) + 1e-4
        targets = mean_probability.copy()
        foreground = mean_probability >= self.config.foreground_seed
        background = mean_probability <= self.config.background_seed
        confidence[foreground | background] += self.config.seed_strength
        targets[foreground] = 1.0
        targets[background] = 0.0
        system = (
            sparse.diags(confidence)
            + self.config.graph_lambda * laplacian
            + sparse.eye(segment_count, dtype=cp.float32) * 1e-8
        ).tocsr()
        rhs = confidence * targets
        try:
            solution, solver_info = cg(
                system,
                rhs,
                rtol=self.config.solver_tolerance,
                atol=0.0,
                maxiter=self.config.solver_max_iterations,
            )
        except TypeError:
            solution, solver_info = cg(
                system,
                rhs,
                tol=self.config.solver_tolerance,
                atol=0.0,
                maxiter=self.config.solver_max_iterations,
            )
        if solver_info != 0 or not bool(cp.isfinite(solution).all()):
            solution = spsolve(system, rhs)
        solution = cp.clip(solution, 0.0, 1.0)
        graph_probability = solution[labels]
        fusion = cp.power(uncertainty_map, self.config.fusion_power)
        refined = cp.clip((1.0 - fusion) * coarse + fusion * graph_probability, 0.0, 1.0)
        return {
            "coarse_probability": _as_torch(coarse),
            "uncertainty_map": _as_torch(uncertainty_map),
            "superpixels": _as_torch(labels),
            "graph_probability": _as_torch(graph_probability),
            "refined_probability": _as_torch(refined),
            "refined_mask": _as_torch(refined >= self.config.threshold),
            "diagnostics": {
                "backend": "cucim-cupy-local-slic",
                "segments": float(segment_count),
                "edges": float(pairs.shape[0]),
                "color_scale": float(color_scale.item()),
                "foreground_seeds": float(foreground.sum().item()),
                "background_seeds": float(background.sum().item()),
                "solver_info": float(solver_info),
                "config": asdict(self.config),
            },
        }
