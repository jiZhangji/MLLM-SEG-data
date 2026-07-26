from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#20313B"
MUTED = "#5D6B73"
LINE = "#9AA8AE"
PANEL = "#F5F7F8"
BLUE = "#2E6FBB"
CYAN = "#2497A8"
GREEN = "#4E9A68"
YELLOW = "#E6B84A"
CORAL = "#DC6B52"
PURPLE = "#7965A8"
WHITE = "#FFFFFF"
PROBABILITY_CMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the FreeRef framework figure from exported experiment arrays."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sample-bundle",
        type=Path,
        help="NPZ bundle exported from a real evaluation sample.",
    )
    source.add_argument(
        "--demo",
        action="store_true",
        help="Use the synthetic layout smoke test (never used for the paper figure).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    parser.add_argument("--stem", default="freeref_framework")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_sample_bundle(path: Path) -> dict[str, Any]:
    """Load and validate arrays exported by select_real_framework_sample.py."""
    required = {
        "scene",
        "target",
        "hard",
        "p",
        "u",
        "u_hard",
        "r",
        "refined",
        "changed",
        "superpixels",
    }
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        sample: dict[str, Any] = {key: np.asarray(payload[key]) for key in required}
        for key in ("name", "query", "model"):
            sample[key] = str(payload[key].item()) if key in payload.files else ""

    scene = np.asarray(sample["scene"])
    if scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError(f"scene must have shape [H, W, 3], got {scene.shape}")
    if scene.dtype.kind in "ui":
        scene = scene.astype(np.float32) / 255.0
    sample["scene"] = np.clip(scene.astype(np.float32), 0.0, 1.0)
    shape = scene.shape[:2]
    for key in ("target", "hard", "p", "u", "u_hard", "r", "refined", "changed"):
        value = np.asarray(sample[key])
        if value.shape != shape:
            raise ValueError(f"{key} must match image shape {shape}, got {value.shape}")
        sample[key] = np.clip(value.astype(np.float32), 0.0, 1.0)
    labels = np.asarray(sample["superpixels"])
    if labels.shape != shape:
        raise ValueError(f"superpixels must match image shape {shape}, got {labels.shape}")
    if not np.isfinite(sample["p"]).all() or not np.isfinite(sample["refined"]).all():
        raise ValueError("Probability fields contain non-finite values.")
    sample["superpixels"] = labels.astype(np.int64)
    return sample


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = WHITE,
    edge: str = LINE,
    radius: float = 0.008,
    linewidth: float = 0.9,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    width: float = 1.1,
    connection: str = "arc3,rad=0",
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=width,
            color=color,
            connectionstyle=connection,
            shrinkA=1,
            shrinkB=1,
            zorder=zorder,
        )
    )


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8,
    color: str = INK,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: int = 5,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size * 0.72,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=1.12,
        zorder=zorder,
    )


def section_title(
    ax: plt.Axes,
    x: float,
    y: float,
    marker: str,
    text: str,
    *,
    size: float = 11.3,
) -> None:
    label(ax, x, y, marker, size=10.4, color=WHITE, weight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.013, y - 0.013),
            0.026,
            0.026,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor=INK,
            edgecolor=INK,
            linewidth=0,
            zorder=3,
        )
    )
    label(ax, x + 0.020, y, text, size=size, weight="bold", ha="left")


def inset(fig: plt.Figure, rect: tuple[float, float, float, float]) -> plt.Axes:
    axis = fig.add_axes(rect)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return axis


def smooth(field: np.ndarray, passes: int = 4) -> np.ndarray:
    result = field.astype(float)
    for _ in range(passes):
        padded = np.pad(result, 1, mode="edge")
        result = (
            padded[:-2, :-2]
            + 2 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2 * padded[1:-1, :-2]
            + 4 * padded[1:-1, 1:-1]
            + 2 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return result


def shift(array: np.ndarray, dy: int, dx: int, fill: bool = False) -> np.ndarray:
    result = np.full_like(array, fill)
    src_y0 = max(0, -dy)
    src_y1 = array.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = array.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = array.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = array.shape[1] - max(0, -dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return result


def boundary_permission(mask: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    eroded = mask.copy()
    dilated = mask.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        eroded &= shift(mask, dy, dx, fill=False)
        dilated |= shift(mask, dy, dx, fill=False)
    boundary = dilated ^ eroded
    permission = boundary.astype(float)
    radius = int(2.5 * sigma)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            value = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            permission = np.maximum(permission, value * shift(boundary, dy, dx, fill=False))
    return permission


def build_demo() -> dict[str, Any]:
    """Synthetic smoke test for layout development only."""
    height, width = 84, 112
    yy, xx = np.mgrid[0:height, 0:width]
    scene = np.zeros((height, width, 3), dtype=float)
    sky = np.clip(0.86 - 0.0025 * yy, 0, 1)
    scene[..., 0] = 0.72 * sky + 0.12
    scene[..., 1] = 0.84 * sky + 0.08
    scene[..., 2] = 0.90 * sky + 0.06
    scene[58:, :, :] = np.array([0.60, 0.66, 0.61])
    scene[62:, :, :] *= np.linspace(1.0, 0.75, height - 62)[:, None, None]

    umbrella = ((xx - 55) / 27) ** 2 + ((yy - 30) / 13) ** 2 <= 1
    umbrella &= yy <= 34 + 0.11 * np.abs(xx - 55)
    head = (xx - 55) ** 2 + (yy - 43) ** 2 <= 5.5**2
    body = ((xx - 55) / 10.5) ** 2 + ((yy - 59) / 18) ** 2 <= 1
    leg_left = (np.abs(xx - (51 - 0.12 * (yy - 67))) < 2.2) & (yy >= 66) & (yy < 83)
    leg_right = (np.abs(xx - (60 + 0.10 * (yy - 67))) < 2.2) & (yy >= 66) & (yy < 83)
    arm = (np.abs(yy - (53 - 0.20 * (xx - 55))) < 2.2) & (xx >= 55) & (xx <= 76)
    target = umbrella | head | body | leg_left | leg_right | arm

    scene[umbrella] = np.array([0.16, 0.23, 0.27])
    scene[head] = np.array([0.68, 0.48, 0.36])
    scene[body] = np.array([0.78, 0.13, 0.15])
    scene[leg_left | leg_right] = np.array([0.18, 0.20, 0.23])
    scene[arm] = np.array([0.74, 0.20, 0.19])
    scene[:, 79:81, :] = np.array([0.50, 0.42, 0.31])

    base = target.copy()
    base &= ~(((xx - 78) ** 2 + (yy - 52) ** 2) < 8**2)
    base |= (((xx - 37) / 7) ** 2 + ((yy - 32) / 8) ** 2 <= 1)
    probability = np.clip(0.05 + 0.90 * smooth(base.astype(float), passes=5), 0, 1)
    probability += 0.08 * np.sin(xx / 8.0) * np.exp(-((yy - 44) / 24.0) ** 2)
    probability = np.clip(probability, 0, 1)
    uncertainty = 1.0 - np.abs(2.0 * probability - 1.0)
    hard_permission = boundary_permission(base, sigma=4.0)

    candidate = smooth(target.astype(float), passes=3)
    candidate = np.clip(candidate + 0.05 * smooth(scene[..., 1] - scene[..., 0], 2), 0, 1)
    refined = (1.0 - uncertainty) * probability + uncertainty * candidate
    changed = np.abs(refined - probability)
    superpixels, _ = voronoi_superpixels(height, width)

    return {
        "scene": scene,
        "target": target.astype(float),
        "hard": base.astype(float),
        "p": probability,
        "u": uncertainty,
        "u_hard": hard_permission,
        "r": candidate,
        "refined": refined,
        "changed": changed,
        "superpixels": superpixels,
        "name": "layout-smoke-test",
        "query": "the woman in red holding the umbrella",
        "model": "synthetic demo",
    }


def voronoi_superpixels(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    rows, cols = 7, 9
    centers = []
    for row in range(rows):
        for col in range(cols):
            cy = (row + 0.5) * height / rows + rng.uniform(-2.2, 2.2)
            cx = (col + 0.5) * width / cols + rng.uniform(-2.2, 2.2)
            centers.append((cy, cx))
    centers_array = np.asarray(centers)
    yy, xx = np.mgrid[0:height, 0:width]
    distances = (
        (yy[..., None] - centers_array[:, 0]) ** 2
        + (xx[..., None] - centers_array[:, 1]) ** 2
    )
    labels = np.argmin(distances, axis=-1)
    return labels, centers_array


def region_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def region_graph(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actual region ids, centroids, and unique SLIC adjacency edges."""
    region_ids, inverse = np.unique(labels.astype(np.int64), return_inverse=True)
    compact = inverse.reshape(labels.shape)
    count = len(region_ids)
    yy, xx = np.indices(labels.shape)
    sizes = np.bincount(compact.ravel(), minlength=count).clip(min=1)
    centers = np.column_stack(
        (
            np.bincount(compact.ravel(), weights=yy.ravel(), minlength=count) / sizes,
            np.bincount(compact.ravel(), weights=xx.ravel(), minlength=count) / sizes,
        )
    )
    horizontal = np.column_stack((compact[:, :-1].ravel(), compact[:, 1:].ravel()))
    vertical = np.column_stack((compact[:-1, :].ravel(), compact[1:, :].ravel()))
    edges = np.concatenate((horizontal, vertical), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size:
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    return compact, centers, edges


def pool_regions(
    values: np.ndarray,
    compact_labels: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.bincount(compact_labels.ravel(), minlength=count).clip(min=1)
    means = (
        np.bincount(
            compact_labels.ravel(),
            weights=np.asarray(values, dtype=float).ravel(),
            minlength=count,
        )
        / sizes
    )
    return means[compact_labels], means


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Small NumPy sRGB-to-CIELAB conversion used only to draw real affinities."""
    values = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    ).T
    xyz /= np.asarray([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=-1,
    )


def region_affinities(
    scene: np.ndarray,
    compact_labels: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    count = int(compact_labels.max()) + 1
    sizes = np.bincount(compact_labels.ravel(), minlength=count).clip(min=1)
    lab = rgb_to_lab(scene).reshape(-1, 3)
    means = np.column_stack(
        [
            np.bincount(
                compact_labels.ravel(),
                weights=lab[:, channel],
                minlength=count,
            )
            / sizes
            for channel in range(3)
        ]
    )
    if not len(edges):
        return np.empty(0, dtype=float)
    distances = np.linalg.norm(means[edges[:, 0]] - means[edges[:, 1]], axis=1)
    positive = distances[distances > 1e-8]
    scale = float(np.median(positive)) if positive.size else 1.0
    return np.clip(np.exp(-np.square(distances / max(scale, 1e-6))), 1e-4, 1.0)


def show_map(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    array: np.ndarray,
    *,
    cmap: str | colors.Colormap = "viridis",
    title: str = "",
    border: str = LINE,
    contour: np.ndarray | None = None,
) -> plt.Axes:
    axis = inset(fig, rect)
    axis.imshow(array, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    if contour is not None:
        axis.contour(contour, levels=[0.5], colors=[WHITE], linewidths=0.8)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(border)
        spine.set_linewidth(0.8)
    if title:
        axis.set_title(title, fontsize=5.2, color=INK, pad=1.5, fontweight="semibold")
    return axis


def save_component_images(
    output_dir: Path,
    demo: dict[str, Any],
) -> list[Path]:
    """Export the experiment-derived raster ingredients used by the main figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_map, centers, _ = region_graph(demo["superpixels"])
    boundary = region_boundary(labels_map)
    pooled_p, _ = pool_regions(demo["p"], labels_map, len(centers))
    pooled_u, _ = pool_regions(demo["u"], labels_map, len(centers))

    slic_scene = demo["scene"].copy()
    slic_scene[boundary] = np.array(colors.to_rgb(WHITE))

    base_mask = demo["p"] >= 0.5
    final_mask = demo["refined"] >= 0.5
    changes = np.zeros((*base_mask.shape, 3), dtype=float)
    changes[:] = np.array(colors.to_rgb("#ECEFF1"))
    changes[base_mask & final_mask] = np.array(colors.to_rgb("#9AA5AB"))
    changes[(~base_mask) & final_mask] = np.array(colors.to_rgb(GREEN))
    changes[base_mask & (~final_mask)] = np.array(colors.to_rgb(CORAL))

    def overlay_mask(mask: np.ndarray, color: str, alpha: float = 0.48) -> np.ndarray:
        result = demo["scene"].copy()
        selected = np.asarray(mask, dtype=bool)
        overlay_color = np.asarray(colors.to_rgb(color), dtype=float)
        result[selected] = (
            (1.0 - alpha) * result[selected] + alpha * overlay_color
        )
        return np.clip(result, 0.0, 1.0)

    components: tuple[tuple[str, np.ndarray, str | None], ...] = (
        ("input_scene.png", demo["scene"], None),
        ("ground_truth_mask.png", demo["target"], "gray"),
        ("baseline_segmentation_mask.png", base_mask.astype(float), "gray"),
        ("freeref_segmentation_mask.png", final_mask.astype(float), "gray"),
        ("ground_truth_overlay.png", overlay_mask(demo["target"] >= 0.5, GREEN), None),
        ("baseline_segmentation_overlay.png", overlay_mask(base_mask, CORAL), None),
        ("freeref_segmentation_overlay.png", overlay_mask(final_mask, BLUE), None),
        ("soft_probability.png", demo["p"], PROBABILITY_CMAP),
        ("soft_intervention.png", demo["u"], "magma"),
        ("hard_mask.png", demo["hard"], "gray"),
        ("boundary_permission.png", demo["u_hard"], "magma"),
        ("regional_probability.png", pooled_p, PROBABILITY_CMAP),
        ("regional_intervention.png", pooled_u, "magma"),
        ("slic_superpixels.png", slic_scene, None),
        ("regional_field.png", demo["r"], PROBABILITY_CMAP),
        ("refined_probability.png", demo["refined"], PROBABILITY_CMAP),
        ("changed_pixels.png", changes, None),
    )
    outputs: list[Path] = []
    metadata = {"Software": "FreeRef framework figure generator"}
    for filename, array, cmap in components:
        output = output_dir / filename
        save_args: dict[str, object] = {
            "origin": "upper",
            "metadata": metadata,
        }
        if cmap is not None:
            save_args.update({"cmap": cmap, "vmin": 0, "vmax": 1})
        plt.imsave(output, np.clip(array, 0, 1), **save_args)
        outputs.append(output)
    return outputs


def draw_panel_a(fig: plt.Figure, ax: plt.Axes, demo: dict[str, Any]) -> None:
    x0, x1 = 0.015, 0.342
    box(ax, x0, 0.075, x1 - x0, 0.875, face="#F5F8FA", edge="#CBD6DC", radius=0.012)
    section_title(ax, x0 + 0.020, 0.918, "a", "Unified Output Adaptation")

    query = " ".join(str(demo.get("query") or "the referred object").split())
    query = textwrap.shorten(query, width=50, placeholder="...")
    query = textwrap.fill(query, width=17)
    label(ax, x0 + 0.052, 0.846, f'Refer to:\n"{query}"', size=7.0, ha="left")
    scene_ax = inset(fig, (x0 + 0.018, 0.500, 0.105, 0.285))
    scene_ax.imshow(demo["scene"])
    for spine in scene_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#7C8C94")
        spine.set_linewidth(0.8)
    box(ax, x0 + 0.018, 0.390, 0.105, 0.070, face="#E8EEF1", edge="#758891")
    label(ax, x0 + 0.0705, 0.425, "Any Frozen\nMLLM Segmenter", size=7.5, weight="bold")
    box(ax, x0 + 0.018, 0.278, 0.105, 0.060, face=WHITE, edge=INK)
    label(ax, x0 + 0.0705, 0.308, "Output  $O$", size=7.2, weight="bold")
    arrow(ax, (x0 + 0.0705, 0.500), (x0 + 0.0705, 0.463))
    arrow(ax, (x0 + 0.0705, 0.389), (x0 + 0.0705, 0.340))

    label(ax, x0 + 0.210, 0.855, "Soft probability", size=8.5, weight="bold")
    show_map(fig, (x0 + 0.142, 0.665, 0.092, 0.145), demo["p"], cmap=PROBABILITY_CMAP, title="$p$")
    show_map(fig, (x0 + 0.244, 0.665, 0.092, 0.145), demo["u"], cmap="magma", title="Intervention field  $u$")
    label(ax, x0 + 0.290, 0.642, r"$u=1-|2p-1|$", size=7.3)

    label(ax, x0 + 0.210, 0.588, "Hard mask", size=8.5, weight="bold")
    show_map(fig, (x0 + 0.142, 0.406, 0.092, 0.145), demo["hard"], cmap="gray", title="$p=M$")
    show_map(fig, (x0 + 0.244, 0.406, 0.092, 0.145), demo["u_hard"], cmap="magma", title="Boundary permission  $u$")
    label(ax, x0 + 0.290, 0.382, r"$u=e^{-d(x,\partial M)^2/(2\sigma^2)}$", size=6.8)

    arrow(ax, (x0 + 0.124, 0.308), (x0 + 0.140, 0.726), connection="arc3,rad=-0.20")
    arrow(ax, (x0 + 0.124, 0.308), (x0 + 0.140, 0.475), connection="arc3,rad=-0.08")
    box(ax, x0 + 0.145, 0.170, 0.187, 0.095, face="#E7F0FA", edge=BLUE)
    label(ax, x0 + 0.2385, 0.225, "Unified black-box interface", size=8.2, color=BLUE, weight="bold")
    label(ax, x0 + 0.2385, 0.193, r"$(p,u)=\mathcal{A}(O)$", size=9.8)
    arrow(ax, (x0 + 0.2385, 0.403), (x0 + 0.2385, 0.267), color=BLUE)
    ax.add_patch(Rectangle((x0 + 0.146, 0.110), 0.185, 0.018, facecolor="none", edgecolor=LINE, linewidth=0.6))
    gradient = np.linspace(0, 1, 256)[None, :]
    grad_ax = inset(fig, (x0 + 0.148, 0.112, 0.181, 0.014))
    grad_ax.imshow(gradient, aspect="auto", cmap="magma")
    label(ax, x0 + 0.145, 0.092, "protected  $u=0$", size=6.5, ha="left", color=MUTED)
    label(ax, x0 + 0.332, 0.092, "$u=1$  editable", size=6.5, ha="right", color=MUTED)


def draw_panel_b(fig: plt.Figure, ax: plt.Axes, demo: dict[str, Any]) -> None:
    x0, x1 = 0.352, 0.744
    box(ax, x0, 0.075, x1 - x0, 0.875, face="#FBF8F1", edge="#D7CDAE", radius=0.012)
    section_title(ax, x0 + 0.020, 0.918, "b", "Reliability-Weighted Inference")
    label(ax, x0 + 0.020, 0.855, "Semantic evidence - what to preserve", size=8.8, weight="bold", ha="left", color=CORAL)
    label(ax, x0 + 0.020, 0.485, "Image structure - where to propagate", size=8.8, weight="bold", ha="left", color=GREEN)

    labels_map, centers, edges = region_graph(demo["superpixels"])
    boundary = region_boundary(labels_map)
    pooled_p, region_p = pool_regions(demo["p"], labels_map, len(centers))
    pooled_u, region_u = pool_regions(demo["u"], labels_map, len(centers))
    affinities = region_affinities(demo["scene"], labels_map, edges)

    show_map(fig, (x0 + 0.018, 0.670, 0.092, 0.145), pooled_p, cmap=PROBABILITY_CMAP, title="Regional $\\bar p$")
    show_map(fig, (x0 + 0.118, 0.670, 0.092, 0.145), pooled_u, cmap="magma", title="Regional $\\bar u$")

    graph_ax = inset(fig, (x0 + 0.218, 0.650, 0.092, 0.175))
    graph_ax.set_xlim(0, demo["p"].shape[1])
    graph_ax.set_ylim(demo["p"].shape[0], 0)
    graph_ax.set_aspect("equal")
    graph_ax.set_title("Region graph", fontsize=5.2, color=INK, pad=1.5, fontweight="semibold")
    stride = max(1, int(np.ceil(len(edges) / 2400)))
    for edge_index in range(0, len(edges), stride):
        first, second = edges[edge_index]
        graph_ax.plot(
            [centers[first, 1], centers[second, 1]],
            [centers[first, 0], centers[second, 0]],
            color="#C8B897",
            lw=0.15 + 0.35 * float(affinities[edge_index]),
            alpha=0.55,
        )
    node_size = 10 if len(centers) < 180 else 3.0
    graph_ax.scatter(
        centers[:, 1],
        centers[:, 0],
        c=region_p,
        cmap=PROBABILITY_CMAP,
        vmin=0,
        vmax=1,
        s=node_size,
        edgecolor=INK,
        linewidth=0.12,
    )

    box(ax, x0 + 0.310, 0.633, 0.074, 0.202, face=WHITE, edge="#C7B988")
    label(ax, x0 + 0.347, 0.810, r"anchors $t_i$", size=7.4, weight="bold")
    for center_y, color, text in (
        (0.765, BLUE, "FG anchor"),
        (0.704, "#7D8790", "BG anchor"),
        (0.643, CORAL, "editable"),
    ):
        ax.scatter([x0 + 0.322], [center_y], s=32, c=[color], edgecolors=INK, linewidths=0.4, zorder=6)
        label(ax, x0 + 0.332, center_y, text, size=5.7, ha="left")
    label(ax, x0 + 0.347, 0.616, r"$a_i\uparrow$ as $\bar u_i\downarrow$", size=6.0)

    slic_ax = inset(fig, (x0 + 0.018, 0.270, 0.105, 0.170))
    slic_scene = demo["scene"].copy()
    slic_scene[boundary] = np.array(colors.to_rgb(WHITE))
    slic_ax.imshow(slic_scene)
    slic_ax.set_title("SLIC superpixels", fontsize=5.2, color=INK, pad=1.5, fontweight="semibold")
    for spine in slic_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(LINE)
        spine.set_linewidth(0.8)

    adj_ax = inset(fig, (x0 + 0.132, 0.270, 0.100, 0.170))
    adj_ax.imshow(demo["scene"], alpha=0.22)
    adj_ax.set_xlim(0, demo["p"].shape[1])
    adj_ax.set_ylim(demo["p"].shape[0], 0)
    adj_ax.set_aspect("equal")
    adj_ax.set_title("Region adjacency", fontsize=5.2, color=INK, pad=1.5, fontweight="semibold")
    neighbors: list[set[int]] = [set() for _ in range(len(centers))]
    for first, second in edges:
        neighbors[int(first)].add(int(second))
        neighbors[int(second)].add(int(first))
    seed = int(np.argmax(region_u))
    selected_set = {seed, *neighbors[seed]}
    frontier = list(selected_set)
    while len(selected_set) < 8 and frontier:
        current = frontier.pop(0)
        for neighbor in neighbors[current]:
            if neighbor not in selected_set:
                selected_set.add(neighbor)
                frontier.append(neighbor)
            if len(selected_set) >= 16:
                break
    selected_ids = np.asarray(sorted(selected_set)[:16], dtype=np.int64)
    selected_lookup = set(int(value) for value in selected_ids)
    for edge_index, (first, second) in enumerate(edges):
        if int(first) not in selected_lookup or int(second) not in selected_lookup:
            continue
        adj_ax.plot(
            [centers[first, 1], centers[second, 1]],
            [centers[first, 0], centers[second, 0]],
            color=GREEN,
            lw=0.35 + 1.8 * float(affinities[edge_index]),
            alpha=0.85,
        )
    adj_ax.scatter(
        centers[selected_ids, 1],
        centers[selected_ids, 0],
        s=18,
        c=region_u[selected_ids],
        cmap="magma",
        vmin=0,
        vmax=1,
        edgecolor=INK,
        linewidth=0.35,
    )
    if selected_ids.size:
        y_values = centers[selected_ids, 0]
        x_values = centers[selected_ids, 1]
        y_pad = max(5.0, 0.22 * max(float(np.ptp(y_values)), 1.0))
        x_pad = max(5.0, 0.22 * max(float(np.ptp(x_values)), 1.0))
        adj_ax.set_xlim(max(0.0, float(x_values.min() - x_pad)), min(demo["p"].shape[1], float(x_values.max() + x_pad)))
        adj_ax.set_ylim(min(demo["p"].shape[0], float(y_values.max() + y_pad)), max(0.0, float(y_values.min() - y_pad)))

    box(ax, x0 + 0.236, 0.275, 0.058, 0.155, face=WHITE, edge="#AFC6B5")
    label(ax, x0 + 0.265, 0.404, r"Affinity $w_{ij}$", size=5.6, weight="bold")
    for y, line_width, text in ((0.365, 2.4, "strong"), (0.312, 0.6, "weak")):
        ax.scatter([x0 + 0.248, x0 + 0.268], [y, y], s=20, c=[GREEN, GREEN], edgecolors=INK, linewidths=0.35, zorder=5)
        ax.plot([x0 + 0.250, x0 + 0.266], [y, y], color=GREEN, lw=line_width, zorder=4)
        label(ax, x0 + 0.275, y, text, size=5.0, ha="left")
    label(ax, x0 + 0.268, 0.286, r"$L=D-W$", size=7.8, weight="bold")

    box(ax, x0 + 0.310, 0.307, 0.072, 0.114, face="#FFF4D7", edge=YELLOW)
    label(ax, x0 + 0.346, 0.398, "Sparse SPD", size=5.8, weight="bold")
    label(ax, x0 + 0.346, 0.369, r"$H=A+\lambda L+\delta I$", size=5.5)
    label(ax, x0 + 0.346, 0.346, r"$Hq^*=At$", size=5.8, weight="bold")
    label(ax, x0 + 0.346, 0.322, "unique | no learning", size=5.1, color=MUTED)
    arrow(ax, (x0 + 0.306, 0.736), (x0 + 0.346, 0.424), color=CORAL, connection="arc3,rad=0.18")
    arrow(ax, (x0 + 0.298, 0.347), (x0 + 0.311, 0.360), color=GREEN)

    q_ax = show_map(fig, (x0 + 0.306, 0.112, 0.072, 0.145), demo["r"], cmap=PROBABILITY_CMAP, title="Regional $q^*$")
    q_ax.contour(demo["r"], levels=[0.5], colors=[CORAL], linewidths=0.8)
    arrow(ax, (x0 + 0.349, 0.306), (x0 + 0.339, 0.258), color=PURPLE)


def draw_panel_c(fig: plt.Figure, ax: plt.Axes, demo: dict[str, Any]) -> None:
    x0, x1 = 0.754, 0.988
    box(ax, x0, 0.075, x1 - x0, 0.875, face="#F8F5FA", edge="#CFC5D8", radius=0.012)
    section_title(
        ax,
        x0 + 0.020,
        0.918,
        "c",
        "Bounded Pixel\nReconstruction",
        size=8.8,
    )

    show_map(fig, (x0 + 0.014, 0.680, 0.064, 0.130), demo["p"], cmap=PROBABILITY_CMAP, title="Original $p$")
    show_map(fig, (x0 + 0.086, 0.680, 0.064, 0.130), demo["u"], cmap="magma", title="Gate $u$")
    show_map(fig, (x0 + 0.158, 0.680, 0.064, 0.130), demo["r"], cmap=PROBABILITY_CMAP, title="Lifted $r$")
    arrow(ax, (x0 + 0.078, 0.745), (x0 + 0.084, 0.745), color=PURPLE)
    arrow(ax, (x0 + 0.150, 0.745), (x0 + 0.156, 0.745), color=PURPLE)

    box(ax, x0 + 0.018, 0.535, 0.198, 0.090, face="#EEE8F4", edge=PURPLE)
    label(ax, x0 + 0.117, 0.601, "Fusion - pixel-wise gate", size=8.0, color=PURPLE, weight="bold")
    label(ax, x0 + 0.117, 0.566, r"$\alpha=u^\beta,\quad \hat p=(1-\alpha)p+\alpha r$", size=8.0)
    arrow(ax, (x0 + 0.117, 0.680), (x0 + 0.117, 0.627), color=PURPLE)

    base_mask = demo["p"] >= 0.5
    final_mask = demo["refined"] >= 0.5
    changes = np.zeros((*base_mask.shape, 3), dtype=float)
    changes[:] = np.array(colors.to_rgb("#ECEFF1"))
    corrected = (~base_mask) & final_mask
    removed = base_mask & (~final_mask)
    unchanged = base_mask & final_mask
    changes[unchanged] = np.array(colors.to_rgb("#9AA5AB"))
    changes[corrected] = np.array(colors.to_rgb(GREEN))
    changes[removed] = np.array(colors.to_rgb(CORAL))

    show_map(fig, (x0 + 0.014, 0.315, 0.064, 0.135), base_mask.astype(float), cmap="gray", title="Base")
    change_ax = inset(fig, (x0 + 0.086, 0.315, 0.064, 0.135))
    change_ax.imshow(changes)
    change_ax.set_title("Changed", fontsize=5.2, color=INK, pad=1.5, fontweight="semibold")
    for spine in change_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(LINE)
        spine.set_linewidth(0.8)
    show_map(fig, (x0 + 0.158, 0.315, 0.064, 0.135), final_mask.astype(float), cmap="gray", title="FreeRef")
    arrow(ax, (x0 + 0.117, 0.533), (x0 + 0.117, 0.452), color=PURPLE)

    contour_ax = inset(fig, (x0 + 0.018, 0.160, 0.110, 0.125))
    contour_ax.imshow(demo["scene"])
    contour_ax.contour(demo["p"], levels=[0.5], colors=[CORAL], linewidths=1.0)
    contour_ax.contour(demo["refined"], levels=[0.5], colors=[GREEN], linewidths=1.0)
    contour_ax.set_title("Contour update", fontsize=5.2, color=INK, pad=1.5, fontweight="semibold")
    for spine in contour_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(LINE)
        spine.set_linewidth(0.8)
    ax.plot([x0 + 0.140, x0 + 0.157], [0.235, 0.235], color=CORAL, lw=1.6)
    label(ax, x0 + 0.162, 0.235, "original", size=6.1, ha="left")
    ax.plot([x0 + 0.140, x0 + 0.157], [0.200, 0.200], color=GREEN, lw=1.6)
    label(ax, x0 + 0.162, 0.200, "refined", size=6.1, ha="left")
    box(ax, x0 + 0.018, 0.092, 0.198, 0.052, face=WHITE, edge=PURPLE)
    label(ax, x0 + 0.117, 0.118, r"$|\hat p(x)-p(x)|\leq u(x)^\beta$", size=6.4, weight="bold")


def build_figure(sample: dict[str, Any]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "freeref-framework-v1",
            "axes.unicode_minus": False,
        }
    )
    # Match an AAAI two-column text block so typography is not downscaled in LaTeX.
    fig = plt.figure(figsize=(7.05, 3.525), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_panel_a(fig, ax, sample)
    draw_panel_b(fig, ax, sample)
    draw_panel_c(fig, ax, sample)

    box(ax, 0.015, 0.018, 0.973, 0.040, face=INK, edge=INK, radius=0.008)
    footer = "OUTPUT ONLY   |   NO INTERNAL FEATURES   |   NO GRADIENTS   |   NO LEARNED MASK DECODER"
    text = ax.text(
        0.501,
        0.038,
        footer,
        color=WHITE,
        fontsize=7.7,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=8,
    )
    text.set_path_effects([pe.withStroke(linewidth=0.2, foreground=WHITE)])
    return fig


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = build_demo() if args.demo else load_sample_bundle(args.sample_bundle)
    fig = build_figure(sample)
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        output = output_dir / f"{args.stem}.{suffix}"
        save_args: dict[str, object] = {
            "bbox_inches": None,
            "pad_inches": 0,
            "facecolor": WHITE,
        }
        if suffix == "pdf":
            save_args["metadata"] = {
                "Creator": "FreeRef framework figure generator",
                "CreationDate": None,
                "ModDate": None,
            }
        elif suffix == "svg":
            save_args["metadata"] = {
                "Creator": "FreeRef framework figure generator",
                "Date": None,
            }
        else:
            save_args["metadata"] = {"Software": "FreeRef framework figure generator"}
        if suffix == "png":
            save_args["dpi"] = args.dpi
        fig.savefig(output, **save_args)
        if suffix == "svg":
            svg = output.read_text(encoding="utf-8")
            output.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        outputs.append(output)
    plt.close(fig)
    component_outputs = save_component_images(
        output_dir / f"{args.stem}_components",
        sample,
    )
    outputs.extend(component_outputs)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
