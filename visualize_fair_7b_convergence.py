from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("../outputs/fair_7b_onepass6_stamp4_e2_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize fair OnePass/STAMP 7B training convergence."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--recent-window", type=int, default=200)
    parser.add_argument("--image-name", default="training_convergence.png")
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if "global_step" in row and "loss" in row:
            rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    # A resumed run can append the same optimizer step again. Keep its latest row.
    deduplicated = {int(row["global_step"]): row for row in rows}
    return sorted(deduplicated.values(), key=lambda row: int(row["global_step"]))


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else float("nan")


def slope_per_100_steps(rows: list[dict[str, Any]], key: str = "loss") -> float:
    points = [(float(row["global_step"]), float(row[key])) for row in rows if key in row]
    if len(points) < 2:
        return float("nan")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return 0.0
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    return 100.0 * numerator / denominator


def summarize(name: str, path: Path, rows: list[dict[str, Any]], recent_window: int) -> None:
    print(f"\n================ {name} ================")
    print(f"history: {path}")
    print(f"records: {len(rows)}")
    print(f"step range: {rows[0]['global_step']} -> {rows[-1]['global_step']}")

    for epoch in sorted({int(row["epoch"]) for row in rows}):
        epoch_rows = [row for row in rows if int(row["epoch"]) == epoch]
        window = min(100, max(1, len(epoch_rows) // 4))
        minimum = min(float(row["loss"]) for row in epoch_rows)
        print(
            f"epoch {epoch}: n={len(epoch_rows)}, "
            f"mean={average(epoch_rows, 'loss'):.6f}, "
            f"first={average(epoch_rows[:window], 'loss'):.6f}, "
            f"last={average(epoch_rows[-window:], 'loss'):.6f}, "
            f"min={minimum:.6f}"
        )

    window = min(recent_window, max(1, len(rows) // 2))
    previous = rows[-2 * window : -window]
    recent = rows[-window:]
    previous_loss = average(previous, "loss")
    recent_loss = average(recent, "loss")
    decrease = (
        100.0 * (previous_loss - recent_loss) / previous_loss
        if previous and previous_loss != 0
        else float("nan")
    )
    print(f"comparison window: {window} records")
    print(f"previous loss: {previous_loss:.6f}")
    print(f"recent loss:   {recent_loss:.6f}")
    print(f"loss decrease: {decrease:.3f}%")
    print(f"recent slope per 100 steps: {slope_per_100_steps(recent):.6f}")
    for key in ("loss_lm", "loss_mask", "loss_bce", "loss_dice"):
        if any(key in row for row in recent):
            print(f"recent {key}: {average(recent, key):.6f}")

    if any("loss_lm" in row for row in rows):
        values = [float(row["loss_lm"]) for row in rows]
        print(f"loss_lm min/max/mean: {min(values):.10g} / {max(values):.10g} / {mean(values):.10g}")
        print(f"loss_lm nonzero records: {sum(abs(value) > 1e-10 for value in values)}")
    print(f"final learning rates: {rows[-1].get('learning_rates', {})}")


def moving_average(rows: list[dict[str, Any]], key: str, window: int):
    import numpy as np

    points = [(int(row["global_step"]), float(row[key])) for row in rows if key in row]
    x = np.asarray([point[0] for point in points])
    y = np.asarray([point[1] for point in points])
    if len(y) < window:
        return x, y
    kernel = np.ones(window, dtype=float) / float(window)
    return x[window - 1 :], np.convolve(y, kernel, mode="valid")


def add_epoch_lines(axis, rows: list[dict[str, Any]]) -> None:
    epochs = sorted({int(row["epoch"]) for row in rows})
    for epoch in epochs[1:]:
        boundary = min(int(row["global_step"]) for row in rows if int(row["epoch"]) == epoch)
        axis.axvline(boundary, color="black", linestyle="--", alpha=0.35)


def plot_histories(
    histories: dict[str, list[dict[str, Any]]],
    output_path: Path,
    smooth_window: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib and numpy.") from exc

    colors = {"OnePass": "#2563eb", "STAMP": "#dc2626"}
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))

    axis = axes[0, 0]
    for name, rows in histories.items():
        x = [int(row["global_step"]) for row in rows]
        y = [float(row["loss"]) for row in rows]
        axis.plot(x, y, color=colors[name], alpha=0.08)
        smooth_x, smooth_y = moving_average(rows, "loss", smooth_window)
        axis.plot(smooth_x, smooth_y, color=colors[name], linewidth=2, label=name)
    axis.set_title(f"Total loss ({smooth_window}-step average)")
    axis.set_ylabel("Loss")
    axis.legend()

    axis = axes[0, 1]
    smooth_x, smooth_y = moving_average(histories["OnePass"], "loss", smooth_window)
    axis.plot(smooth_x, smooth_y, color=colors["OnePass"], linewidth=2, label="OnePass mask")
    smooth_x, smooth_y = moving_average(histories["STAMP"], "loss_mask", smooth_window)
    axis.plot(smooth_x, smooth_y, color=colors["STAMP"], linewidth=2, label="STAMP mask")
    axis.set_title("Segmentation loss")
    axis.set_ylabel("Mask loss")
    axis.legend()

    axis = axes[0, 2]
    stamp_rows = histories["STAMP"]
    x = [int(row["global_step"]) for row in stamp_rows if "loss_lm" in row]
    y = [max(float(row["loss_lm"]), 1e-10) for row in stamp_rows if "loss_lm" in row]
    axis.plot(x, y, color="#7c3aed", linewidth=1.5)
    axis.set_yscale("log")
    axis.set_title("STAMP language loss")
    axis.set_ylabel("LM loss (log scale)")

    axis = axes[1, 0]
    for name, rows in histories.items():
        key = "loss" if name == "OnePass" else "loss_mask"
        smooth_x, smooth_y = moving_average(rows, key, min(50, smooth_window))
        tail = min(800, len(smooth_x))
        axis.plot(smooth_x[-tail:], smooth_y[-tail:], color=colors[name], linewidth=2, label=name)
    axis.set_title("Final-stage segmentation loss")
    axis.set_ylabel("Mask loss")
    axis.legend()

    axis = axes[1, 1]
    for name, rows in histories.items():
        x = []
        y = []
        for row in rows:
            rates = row.get("learning_rates", {})
            if rates:
                x.append(int(row["global_step"]))
                y.append(max(float(value) for value in rates.values()))
        axis.plot(x, y, color=colors[name], linewidth=2, label=name)
    axis.set_title("Learning-rate schedule")
    axis.set_ylabel("Maximum group LR")
    axis.legend()

    axis = axes[1, 2]
    methods = list(histories)
    all_epochs = sorted({int(row["epoch"]) for rows in histories.values() for row in rows})
    width = 0.35
    positions = np.arange(len(all_epochs))
    for offset, name in enumerate(methods):
        rows = histories[name]
        key = "loss" if name == "OnePass" else "loss_mask"
        values = [average([row for row in rows if int(row["epoch"]) == epoch], key) for epoch in all_epochs]
        axis.bar(positions + (offset - 0.5) * width, values, width=width, color=colors[name], label=name)
    axis.set_xticks(positions, [f"Epoch {epoch}" for epoch in all_epochs])
    axis.set_title("Mean segmentation loss by epoch")
    axis.set_ylabel("Mean mask loss")
    axis.legend()

    for axis in axes.flat[:5]:
        add_epoch_lines(axis, histories["OnePass"])
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].set_xlabel("Epoch")

    figure.suptitle("Fair OnePass vs STAMP 7B Training Convergence", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.smooth_window <= 0 or args.recent_window <= 0:
        raise ValueError("smooth-window and recent-window must be positive.")
    history_paths = {
        "OnePass": args.output_root / "onepass7b" / "train_history.jsonl",
        "STAMP": args.output_root / "stamp7b_native" / "train_history.jsonl",
    }
    histories = {name: load_history(path) for name, path in history_paths.items()}
    for name, rows in histories.items():
        summarize(name, history_paths[name], rows, args.recent_window)

    output_path = args.output_root / args.image_name
    plot_histories(histories, output_path, args.smooth_window)
    print(f"\nConvergence plot written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
