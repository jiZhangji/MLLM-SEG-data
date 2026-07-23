from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.segmentation import slic


@dataclass(frozen=True)
class PostprocessBaselineConfig:
    threshold: float = 0.5
    hard_mask_confidence: float = 0.95
    densecrf_iterations: int = 5
    densecrf_gaussian_sxy: float = 3.0
    densecrf_gaussian_compat: float = 3.0
    densecrf_bilateral_sxy: float = 80.0
    densecrf_bilateral_srgb: float = 13.0
    densecrf_bilateral_compat: float = 10.0
    guided_radius: int = 8
    guided_epsilon: float = 1e-3
    n_segments: int = 1024
    compactness: float = 10.0
    slic_sigma: float = 1.0

    def __post_init__(self) -> None:
        if not 0.5 < self.hard_mask_confidence < 1.0:
            raise ValueError("hard_mask_confidence must lie in (0.5, 1).")
        if self.densecrf_iterations <= 0:
            raise ValueError("densecrf_iterations must be positive.")
        if self.guided_radius < 1 or self.guided_epsilon <= 0:
            raise ValueError("guided_radius and guided_epsilon must be positive.")
        if self.n_segments <= 1 or self.compactness <= 0 or self.slic_sigma < 0:
            raise ValueError("Invalid SLIC configuration.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def hard_mask_probability(mask: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    return np.where(hard, confidence, 1.0 - confidence).astype(np.float32)


def _validate_inputs(image: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_rgb = np.asarray(image)
    values = np.asarray(probability, dtype=np.float32)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {image_rgb.shape}")
    if values.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"probability must match image shape {image_rgb.shape[:2]}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("probability contains non-finite values.")
    if image_rgb.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(image_rgb)) <= 1.0 else 1.0
        image_rgb = np.clip(image_rgb * scale, 0, 255).astype(np.uint8)
    return image_rgb, np.clip(values, 0.0, 1.0)


def densecrf_probability(
    image: np.ndarray,
    probability: np.ndarray,
    config: PostprocessBaselineConfig,
) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError as exc:
        raise RuntimeError(
            "DenseCRF requires pydensecrf. Run prepare_freeref_baseline_env.sh first."
        ) from exc

    image_rgb, foreground = _validate_inputs(image, probability)
    height, width = foreground.shape
    foreground = np.clip(foreground, 1e-6, 1.0 - 1e-6)
    class_probabilities = np.stack((1.0 - foreground, foreground), axis=0)
    unary = unary_from_softmax(np.ascontiguousarray(class_probabilities, dtype=np.float32))
    model = dcrf.DenseCRF2D(width, height, 2)
    model.setUnaryEnergy(unary)
    model.addPairwiseGaussian(
        sxy=config.densecrf_gaussian_sxy,
        compat=config.densecrf_gaussian_compat,
    )
    model.addPairwiseBilateral(
        sxy=config.densecrf_bilateral_sxy,
        srgb=config.densecrf_bilateral_srgb,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=config.densecrf_bilateral_compat,
    )
    posterior = np.asarray(model.inference(config.densecrf_iterations), dtype=np.float32)
    return np.clip(posterior.reshape(2, height, width)[1], 0.0, 1.0)


def guided_filter_probability(
    image: np.ndarray,
    probability: np.ndarray,
    config: PostprocessBaselineConfig,
) -> np.ndarray:
    image_rgb, foreground = _validate_inputs(image, probability)
    image_float = image_rgb.astype(np.float32) / 255.0
    guide = (
        0.299 * image_float[..., 0]
        + 0.587 * image_float[..., 1]
        + 0.114 * image_float[..., 2]
    )
    size = 2 * config.guided_radius + 1
    mean_guide = uniform_filter(guide, size=size, mode="reflect")
    mean_probability = uniform_filter(foreground, size=size, mode="reflect")
    correlation_guide = uniform_filter(guide * guide, size=size, mode="reflect")
    correlation_cross = uniform_filter(guide * foreground, size=size, mode="reflect")
    variance_guide = correlation_guide - mean_guide * mean_guide
    covariance = correlation_cross - mean_guide * mean_probability
    coefficient = covariance / (variance_guide + config.guided_epsilon)
    intercept = mean_probability - coefficient * mean_guide
    mean_coefficient = uniform_filter(coefficient, size=size, mode="reflect")
    mean_intercept = uniform_filter(intercept, size=size, mode="reflect")
    return np.clip(mean_coefficient * guide + mean_intercept, 0.0, 1.0).astype(np.float32)


def slic_region_average_probability(
    image: np.ndarray,
    probability: np.ndarray,
    config: PostprocessBaselineConfig,
) -> np.ndarray:
    image_rgb, foreground = _validate_inputs(image, probability)
    labels = slic(
        image_rgb.astype(np.float32) / 255.0,
        n_segments=config.n_segments,
        compactness=config.compactness,
        sigma=config.slic_sigma,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    ).astype(np.int64)
    count = int(labels.max()) + 1
    flat_labels = labels.reshape(-1)
    sizes = np.bincount(flat_labels, minlength=count).clip(min=1)
    sums = np.bincount(flat_labels, weights=foreground.reshape(-1), minlength=count)
    means = sums / sizes
    return np.clip(means[labels], 0.0, 1.0).astype(np.float32)
