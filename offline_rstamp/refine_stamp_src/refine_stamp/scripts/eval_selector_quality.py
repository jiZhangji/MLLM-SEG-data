from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from refine_stamp.models.patch_selector import PatchSelector
from refine_stamp.utils.selector_eval import (
    coarse_mask_metrics,
    logits_to_coarse_mask,
    resize_binary_to_grid,
    selector_quality_metrics,
)
from refine_stamp.utils.visualization import save_image
from refine_stamp.scripts.visualize_selector import load_payload, visualize_one


SELECTOR_MODES = ("random", "uncertainty", "boundary", "hybrid")


def find_inputs(input_path: Path | None, input_dir: Path | None, limit: int) -> list[Path]:
    if input_path is None and input_dir is None:
        raise SystemExit("Pass --input or --input-dir.")
    if input_path is not None:
        paths = [input_path]
    else:
        paths = sorted(input_dir.glob("*.pt"))  # type: ignore[union-attr]
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError("No .pt payloads found.")
    return paths


def load_gt_mask(payload: dict[str, Any]) -> torch.Tensor:
    if "gt_mask" in payload:
        mask = payload["gt_mask"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        return (mask.float() > 0).float()

    mask_path = payload.get("mask_path") or payload.get("gt_mask_path")
    if not mask_path:
        raise KeyError("Payload must contain gt_mask, mask_path or gt_mask_path.")
    arr = np.array(Image.open(mask_path).convert("L"))
    return torch.from_numpy((arr > 0).astype(np.float32))


def evaluate_payload(
    path: Path,
    *,
    top_k: int,
    boundary_weight: float,
    visualize: bool,
    output_dir: Path,
    visualized_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_payload(path)
    mask_logits = payload["mask_logits"].float()
    grid_hw = tuple(int(x) for x in payload["grid_hw"])
    gt_mask = load_gt_mask(payload)
    if gt_mask.ndim == 2:
        gt_mask_4d = gt_mask.unsqueeze(0).unsqueeze(0)
    elif gt_mask.ndim == 3:
        gt_mask_4d = gt_mask.unsqueeze(0)
    else:
        gt_mask_4d = gt_mask

    coarse_grid_mask = logits_to_coarse_mask(mask_logits, grid_hw)
    gt_grid_mask = resize_binary_to_grid(gt_mask_4d, grid_hw)
    coarse_metrics = coarse_mask_metrics(
        coarse_grid_mask=coarse_grid_mask,
        gt_mask=gt_mask_4d,
        grid_hw=grid_hw,
    )

    rows: list[dict[str, Any]] = []
    name = str(payload.get("name") or path.stem)
    for mode in SELECTOR_MODES:
        selector = PatchSelector(top_k=top_k, boundary_weight=boundary_weight, mode=mode)
        selection = selector(mask_logits=mask_logits, grid_hw=grid_hw)
        quality = selector_quality_metrics(
            selected_ids=selection["selected_ids"],
            score_map=selection["score_map"],
            coarse_grid_mask=coarse_grid_mask,
            gt_grid_mask=gt_grid_mask,
        )
        rows.append(
            {
                "sample": name,
                "path": str(path),
                "selector": mode,
                "grid_h": grid_hw[0],
                "grid_w": grid_hw[1],
                "top_k": min(top_k, mask_logits.shape[1]),
                **coarse_metrics,
                **quality,
            }
        )

    vis_report: dict[str, Any] = {}
    if visualize and visualized_count >= 0:
        vis_dir = output_dir / "visualizations" / name
        vis_report = visualize_one(
            payload=payload,
            output_dir=vis_dir,
            top_k=top_k,
            boundary_weight=boundary_weight,
            selector_mode="hybrid",
        )

    return rows, vis_report


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["selector"])].append(row)

    summary = []
    numeric_keys = [
        "coarse_iou",
        "coarse_boundary_iou",
        "coarse_fg_ratio",
        "gt_fg_ratio",
        "selected_gt_boundary_rate",
        "selected_error_rate",
        "selected_fg_rate",
        "selected_gt_fg_rate",
        "score_mean_on_selected",
    ]
    for selector, items in grouped.items():
        out: dict[str, Any] = {"selector": selector, "samples": len(items)}
        for key in numeric_keys:
            out[key] = float(np.mean([float(item[key]) for item in items]))
        summary.append(out)
    return sorted(summary, key=lambda x: str(x["selector"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Refine-STAMP Phase 1 Selector Quality",
        "",
        "| Selector | Samples | Coarse IoU | Coarse Boundary IoU | GT Boundary Hit | Error Hit | Pred FG Hit | GT FG Hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {selector} | {samples} | {coarse_iou:.4f} | {coarse_boundary_iou:.4f} | "
            "{selected_gt_boundary_rate:.4f} | {selected_error_rate:.4f} | "
            "{selected_fg_rate:.4f} | {selected_gt_fg_rate:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `GT Boundary Hit`: fraction of selected patches overlapping the GT boundary map.",
            "- `Error Hit`: fraction of selected patches overlapping coarse-mask errors.",
            "- Useful selectors should beat `random` on boundary/error targeting before refiner training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--visualize-limit", type=int, default=8)
    args = parser.parse_args()

    paths = find_inputs(args.input, args.input_dir, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    vis_reports = []
    for index, path in enumerate(paths):
        rows, vis_report = evaluate_payload(
            path,
            top_k=args.top_k,
            boundary_weight=args.boundary_weight,
            visualize=index < args.visualize_limit,
            output_dir=args.output_dir,
            visualized_count=index,
        )
        all_rows.extend(rows)
        if vis_report:
            vis_reports.append(vis_report)

    summary = summarize(all_rows)
    write_csv(args.output_dir / "selector_quality_rows.csv", all_rows)
    write_csv(args.output_dir / "selector_quality_summary.csv", summary)
    write_markdown(args.output_dir / "selector_quality_summary.md", summary)

    report = {
        "inputs": [str(p) for p in paths],
        "rows_csv": str(args.output_dir / "selector_quality_rows.csv"),
        "summary_csv": str(args.output_dir / "selector_quality_summary.csv"),
        "summary_md": str(args.output_dir / "selector_quality_summary.md"),
        "visualizations": vis_reports,
        "summary": summary,
    }
    (args.output_dir / "selector_quality_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
