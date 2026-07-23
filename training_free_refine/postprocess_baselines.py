from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from scipy import sparse
from scipy.sparse.linalg import cg
from skimage.color import rgb2lab
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
    fbs_sigma_spatial: float = 16.0
    fbs_sigma_luma: float = 8.0
    fbs_sigma_chroma: float = 8.0
    fbs_lambda: float = 128.0
    fbs_iterations: int = 25
    fbs_max_tolerance: float = 1e-5
    fbs_confidence_floor: float = 1e-3
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
        if min(
            self.fbs_sigma_spatial,
            self.fbs_sigma_luma,
            self.fbs_sigma_chroma,
            self.fbs_lambda,
            self.fbs_max_tolerance,
        ) <= 0:
            raise ValueError("Fast Bilateral Solver scales, lambda, and tolerance must be positive.")
        if self.fbs_iterations <= 0 or not 0 < self.fbs_confidence_floor <= 1:
            raise ValueError("Invalid Fast Bilateral Solver iteration/confidence configuration.")
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
    # PyDenseCRF's Cython memoryview requires writable C-contiguous RGB storage.
    image_rgb = np.array(image_rgb, dtype=np.uint8, order="C", copy=True)
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


def fast_bilateral_solver_probability(
    image: np.ndarray,
    probability: np.ndarray,
    config: PostprocessBaselineConfig,
) -> np.ndarray:
    image_rgb, foreground = _validate_inputs(image, probability)
    height, width = foreground.shape
    lab = rgb2lab(image_rgb.astype(np.float32) / 255.0)
    yy, xx = np.indices((height, width), dtype=np.float64)
    coordinates = np.stack(
        (
            xx / config.fbs_sigma_spatial,
            yy / config.fbs_sigma_spatial,
            lab[..., 0] / config.fbs_sigma_luma,
            lab[..., 1] / config.fbs_sigma_chroma,
            lab[..., 2] / config.fbs_sigma_chroma,
        ),
        axis=-1,
    )
    coordinates = np.floor(coordinates).reshape(-1, 5).astype(np.int32)
    vertices, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    pixel_count = inverse.size
    vertex_count = vertices.shape[0]
    splat = sparse.csr_matrix(
        (np.ones(pixel_count), (inverse, np.arange(pixel_count))),
        shape=(vertex_count, pixel_count),
    )

    lookup = {tuple(vertex): index for index, vertex in enumerate(vertices)}
    blur_rows = list(range(vertex_count))
    blur_cols = list(range(vertex_count))
    for index, vertex in enumerate(vertices):
        for dimension in range(vertices.shape[1]):
            neighbor = vertex.copy()
            neighbor[dimension] += 1
            target = lookup.get(tuple(neighbor))
            if target is not None:
                blur_rows.extend((index, target))
                blur_cols.extend((target, index))
    blur = sparse.csr_matrix(
        (np.ones(len(blur_rows)), (blur_rows, blur_cols)),
        shape=(vertex_count, vertex_count),
    )

    mass = np.asarray(splat @ np.ones(pixel_count)).reshape(-1)
    normalization = np.ones(vertex_count, dtype=np.float64)
    for _ in range(10):
        blurred = np.asarray(blur @ normalization).reshape(-1)
        normalization = np.sqrt(
            np.maximum(normalization * mass, 1e-12) / np.maximum(blurred, 1e-12)
        )
    normalized_mass = normalization * np.asarray(blur @ normalization).reshape(-1)
    diagonal_normalization = sparse.diags(normalization)
    smoothness = config.fbs_lambda * (
        sparse.diags(normalized_mass)
        - diagonal_normalization @ blur @ diagonal_normalization
    )

    source = foreground.reshape(-1).astype(np.float64)
    confidence = np.maximum(
        np.abs(2.0 * source - 1.0), config.fbs_confidence_floor
    ).astype(np.float64)
    data_diagonal = np.asarray(splat @ confidence).reshape(-1)
    system = smoothness + sparse.diags(data_diagonal) + sparse.eye(vertex_count) * 1e-8
    rhs = np.asarray(splat @ (confidence * source)).reshape(-1)
    try:
        solution, info = cg(
            system,
            rhs,
            rtol=config.fbs_max_tolerance,
            atol=0.0,
            maxiter=config.fbs_iterations,
        )
    except TypeError:
        solution, info = cg(
            system,
            rhs,
            tol=config.fbs_max_tolerance,
            atol=0.0,
            maxiter=config.fbs_iterations,
        )
    if info < 0 or not np.isfinite(solution).all():
        solution = sparse.linalg.spsolve(system, rhs)
    result = np.asarray(splat.T @ solution).reshape(height, width)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


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
