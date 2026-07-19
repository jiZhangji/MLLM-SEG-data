from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import wilcoxon


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int, int]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return (float(intersection / union) if union else 1.0), intersection, union


def boundary_iou(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    pred_boundary = np.logical_xor(prediction, binary_erosion(prediction))
    target_boundary = np.logical_xor(target, binary_erosion(target))
    if tolerance > 0:
        pred_boundary = binary_dilation(pred_boundary, iterations=tolerance)
        target_boundary = binary_dilation(target_boundary, iterations=tolerance)
    union = int(np.logical_or(pred_boundary, target_boundary).sum())
    return float(np.logical_and(pred_boundary, target_boundary).sum() / union) if union else 1.0


def paired_statistics(deltas: list[float], bootstrap_samples: int, seed: int = 0) -> dict[str, float | None]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        return {"delta_ci95_low": None, "delta_ci95_high": None, "wilcoxon_p": None}
    if bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        means = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            means[index] = values[rng.integers(0, values.size, size=values.size)].mean()
        low, high = np.quantile(means, [0.025, 0.975])
    else:
        low = high = float(values.mean())
    nonzero = values[np.abs(values) > 1e-12]
    if nonzero.size:
        try:
            p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        except ValueError:
            p_value = None
    else:
        p_value = 1.0
    return {
        "delta_ci95_low": float(low),
        "delta_ci95_high": float(high),
        "wilcoxon_p": p_value,
    }
