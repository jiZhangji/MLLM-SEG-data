from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.sparse.linalg import cg
from skimage.color import rgb2lab
from skimage.segmentation import slic


@dataclass(frozen=True)
class TrainingFreeRefineConfig:
    n_segments: int = 1024
    compactness: float = 10.0
    slic_sigma: float = 1.0
    graph_lambda: float = 1.0
    confidence_power: float = 2.0
    fusion_power: float = 1.0
    foreground_seed: float = 0.9
    background_seed: float = 0.1
    seed_strength: float = 50.0
    threshold: float = 0.5
    solver_tolerance: float = 1e-5
    solver_max_iterations: int = 500

    def __post_init__(self) -> None:
        if self.n_segments <= 1:
            raise ValueError("n_segments must be greater than one.")
        if self.compactness <= 0 or self.graph_lambda < 0:
            raise ValueError("compactness must be positive and graph_lambda non-negative.")
        if self.confidence_power <= 0 or self.fusion_power <= 0:
            raise ValueError("confidence_power and fusion_power must be positive.")
        if not 0 <= self.background_seed < self.foreground_seed <= 1:
            raise ValueError("Seed thresholds must satisfy 0 <= background < foreground <= 1.")
        if self.seed_strength < 0:
            raise ValueError("seed_strength must be non-negative.")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie in (0, 1).")


def stamp_probability(
    mask_logits: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    logits = mask_logits.detach().float().cpu()
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError(f"mask_logits must have shape [N, 2] or [1, N, 2], got {tuple(logits.shape)}")
    grid_h, grid_w = (int(value) for value in grid_hw)
    if grid_h * grid_w != logits.shape[0]:
        raise ValueError(f"grid_hw={grid_hw} does not match N={logits.shape[0]}.")
    foreground = torch.softmax(logits, dim=-1)[..., 1].reshape(1, 1, grid_h, grid_w)
    return F.interpolate(foreground, size=image_hw, mode="bilinear", align_corners=False)[0, 0]


def probability_uncertainty(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return np.clip(1.0 - np.abs(2.0 * probability - 1.0), 0.0, 1.0)


def boundary_uncertainty(mask: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Construct uncertainty around a hard mask boundary without learned scores."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    hard_mask = np.asarray(mask, dtype=bool)
    if hard_mask.ndim != 2:
        raise ValueError(f"mask must have shape [H, W], got {hard_mask.shape}")
    if not hard_mask.any() or hard_mask.all():
        return np.zeros(hard_mask.shape, dtype=np.float64)
    boundary = np.logical_xor(binary_dilation(hard_mask), binary_erosion(hard_mask))
    distance = distance_transform_edt(~boundary)
    return np.exp(-0.5 * np.square(distance / sigma))


def _segment_means(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    flat_labels = labels.reshape(-1)
    flat_values = values.reshape(-1, *values.shape[2:]) if values.ndim == 3 else values.reshape(-1)
    sizes = np.bincount(flat_labels, minlength=count).astype(np.float64).clip(min=1.0)
    if flat_values.ndim == 1:
        sums = np.bincount(flat_labels, weights=flat_values, minlength=count)
        return sums / sizes
    channels = [np.bincount(flat_labels, weights=flat_values[:, index], minlength=count) for index in range(flat_values.shape[1])]
    return np.stack(channels, axis=1) / sizes[:, None]


def _adjacent_pairs(labels: np.ndarray) -> np.ndarray:
    horizontal = np.stack([labels[:, :-1].reshape(-1), labels[:, 1:].reshape(-1)], axis=1)
    vertical = np.stack([labels[:-1, :].reshape(-1), labels[1:, :].reshape(-1)], axis=1)
    pairs = np.concatenate([horizontal, vertical], axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)


def _solve_graph(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
    config: TrainingFreeRefineConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    segment_count = int(labels.max()) + 1
    lab = rgb2lab(image_rgb)
    mean_color = _segment_means(lab, labels, segment_count)
    mean_probability = _segment_means(probability, labels, segment_count)
    mean_uncertainty = _segment_means(uncertainty, labels, segment_count)
    pairs = _adjacent_pairs(labels)

    if pairs.size:
        color_distance = np.linalg.norm(mean_color[pairs[:, 0]] - mean_color[pairs[:, 1]], axis=1)
        positive = color_distance[color_distance > 1e-8]
        color_scale = float(np.median(positive)) if positive.size else 1.0
        weights = np.exp(-np.square(color_distance / max(color_scale, 1e-6)))
        weights = np.clip(weights, 1e-4, 1.0)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        values = np.concatenate([weights, weights])
        affinity = sparse.csr_matrix((values, (rows, cols)), shape=(segment_count, segment_count))
    else:
        color_scale = 1.0
        affinity = sparse.csr_matrix((segment_count, segment_count), dtype=np.float64)

    degree = np.asarray(affinity.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree) - affinity
    confidence = np.power(np.clip(1.0 - mean_uncertainty, 0.0, 1.0), config.confidence_power) + 1e-4
    targets = mean_probability.copy()

    foreground = mean_probability >= config.foreground_seed
    background = mean_probability <= config.background_seed
    confidence[foreground | background] += config.seed_strength
    targets[foreground] = 1.0
    targets[background] = 0.0

    system = sparse.diags(confidence) + config.graph_lambda * laplacian + sparse.eye(segment_count) * 1e-8
    rhs = confidence * targets
    try:
        solution, solver_info = cg(
            system,
            rhs,
            rtol=config.solver_tolerance,
            atol=0.0,
            maxiter=config.solver_max_iterations,
        )
    except TypeError:
        solution, solver_info = cg(
            system,
            rhs,
            tol=config.solver_tolerance,
            atol=0.0,
            maxiter=config.solver_max_iterations,
        )
    if solver_info != 0 or not np.isfinite(solution).all():
        solution = sparse.linalg.spsolve(system, rhs)
    solution = np.clip(np.asarray(solution), 0.0, 1.0)
    graph_probability = solution[labels]
    diagnostics = {
        "segments": float(segment_count),
        "edges": float(pairs.shape[0]),
        "color_scale": color_scale,
        "foreground_seeds": float(foreground.sum()),
        "background_seeds": float(background.sum()),
        "solver_info": float(solver_info),
    }
    return graph_probability, diagnostics


class TrainingFreeUncertaintyRefiner:
    """Image-aware refinement with no learned or fitted parameters."""

    def __init__(self, config: TrainingFreeRefineConfig | None = None) -> None:
        self.config = config or TrainingFreeRefineConfig()

    def refine(
        self,
        image: np.ndarray,
        mask_logits: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, object]:
        image_hw = (int(image.shape[0]), int(image.shape[1]))
        coarse = stamp_probability(mask_logits, grid_hw, image_hw).numpy()
        return self.refine_probability(image, coarse)

    def refine_hard_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        boundary_sigma: float = 8.0,
    ) -> dict[str, object]:
        coarse = np.asarray(mask, dtype=np.float64)
        return self.refine_probability(
            image,
            coarse,
            uncertainty=boundary_uncertainty(coarse >= self.config.threshold, boundary_sigma),
        )

    def refine_probability(
        self,
        image: np.ndarray,
        probability: np.ndarray | torch.Tensor,
        uncertainty: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, object]:
        image_rgb = np.asarray(image)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image must have shape [H, W, 3], got {image_rgb.shape}")
        if image_rgb.dtype != np.float32 and image_rgb.dtype != np.float64:
            image_rgb = image_rgb.astype(np.float32) / 255.0
        else:
            image_rgb = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)

        image_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
        if isinstance(probability, torch.Tensor):
            probability = probability.detach().float().cpu().numpy()
        coarse = np.asarray(probability, dtype=np.float64)
        if coarse.shape != image_hw:
            raise ValueError(f"probability must match image shape {image_hw}, got {coarse.shape}")
        if not np.isfinite(coarse).all():
            raise ValueError("probability contains non-finite values.")
        coarse = np.clip(coarse, 0.0, 1.0)
        if uncertainty is None:
            uncertainty_map = probability_uncertainty(coarse)
        else:
            if isinstance(uncertainty, torch.Tensor):
                uncertainty = uncertainty.detach().float().cpu().numpy()
            uncertainty_map = np.asarray(uncertainty, dtype=np.float64)
            if uncertainty_map.shape != image_hw:
                raise ValueError(f"uncertainty must match image shape {image_hw}, got {uncertainty_map.shape}")
            if not np.isfinite(uncertainty_map).all():
                raise ValueError("uncertainty contains non-finite values.")
            uncertainty_map = np.clip(uncertainty_map, 0.0, 1.0)
        labels = slic(
            image_rgb,
            n_segments=self.config.n_segments,
            compactness=self.config.compactness,
            sigma=self.config.slic_sigma,
            start_label=0,
            channel_axis=-1,
            enforce_connectivity=True,
        ).astype(np.int64)
        graph_probability, diagnostics = _solve_graph(
            labels,
            image_rgb,
            coarse,
            uncertainty_map,
            self.config,
        )
        fusion = np.power(uncertainty_map, self.config.fusion_power)
        refined = np.clip((1.0 - fusion) * coarse + fusion * graph_probability, 0.0, 1.0)
        return {
            "coarse_probability": torch.from_numpy(coarse.astype(np.float32)),
            "uncertainty_map": torch.from_numpy(uncertainty_map.astype(np.float32)),
            "superpixels": torch.from_numpy(labels),
            "graph_probability": torch.from_numpy(graph_probability.astype(np.float32)),
            "refined_probability": torch.from_numpy(refined.astype(np.float32)),
            "refined_mask": torch.from_numpy(refined >= self.config.threshold),
            "diagnostics": diagnostics,
        }
