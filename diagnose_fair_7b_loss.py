from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from onepass_qwen7b.checkpoint import load_checkpoint, read_checkpoint_config
from onepass_qwen7b.data import OnePass7BDataset, collate_samples, grid_targets, prepare_batch
from onepass_qwen7b.model import segmentation_loss
from onepass_qwen7b.runtime import configure_cudnn, load_base_onepass_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose the fair 7B loss gap and probe OnePass gradient/sensitivity paths."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--train-json", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--onepass-checkpoint", type=Path)
    parser.add_argument("--probe-samples", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--recent-window", type=int, default=200)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--attn-implementation")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-existing-segmentation-tokens", action="store_true")
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
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
        raise ValueError(f"No history records found in {path}")
    deduplicated = {int(row["global_step"]): row for row in rows}
    return sorted(deduplicated.values(), key=lambda row: int(row["global_step"]))


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else float("nan")


def loss_gap_for_rows(
    onepass_rows: list[dict[str, Any]],
    stamp_rows: list[dict[str, Any]],
    bce_weight: float,
    dice_weight: float,
) -> dict[str, float]:
    onepass_bce = average(onepass_rows, "loss_bce")
    stamp_bce = average(stamp_rows, "loss_bce")
    onepass_dice = average(onepass_rows, "loss_dice")
    stamp_dice = average(stamp_rows, "loss_dice")
    bce_gap = onepass_bce - stamp_bce
    dice_gap = onepass_dice - stamp_dice
    weighted_bce_gap = bce_weight * bce_gap
    weighted_dice_gap = dice_weight * dice_gap
    reconstructed_gap = weighted_bce_gap + weighted_dice_gap
    measured_gap = average(onepass_rows, "loss") - average(stamp_rows, "loss_mask")
    return {
        "onepass_loss": average(onepass_rows, "loss"),
        "stamp_mask_loss": average(stamp_rows, "loss_mask"),
        "measured_mask_loss_gap": measured_gap,
        "onepass_bce": onepass_bce,
        "stamp_bce": stamp_bce,
        "raw_bce_gap": bce_gap,
        "weighted_bce_gap": weighted_bce_gap,
        "onepass_dice": onepass_dice,
        "stamp_dice": stamp_dice,
        "raw_dice_gap": dice_gap,
        "weighted_dice_gap": weighted_dice_gap,
        "reconstructed_mask_loss_gap": reconstructed_gap,
        "bce_gap_fraction": weighted_bce_gap / reconstructed_gap if reconstructed_gap else float("nan"),
        "dice_gap_fraction": weighted_dice_gap / reconstructed_gap if reconstructed_gap else float("nan"),
    }


def history_diagnostics(
    output_root: Path,
    recent_window: int,
    bce_weight: float,
    dice_weight: float,
) -> dict[str, Any]:
    onepass = load_history(output_root / "onepass7b" / "train_history.jsonl")
    stamp = load_history(output_root / "stamp7b_native" / "train_history.jsonl")
    onepass_by_step = {int(row["global_step"]): row for row in onepass}
    stamp_by_step = {int(row["global_step"]): row for row in stamp}
    common_steps = sorted(set(onepass_by_step) & set(stamp_by_step))
    if not common_steps:
        raise ValueError("OnePass and STAMP histories do not share optimizer steps.")
    aligned_onepass = [onepass_by_step[step] for step in common_steps]
    aligned_stamp = [stamp_by_step[step] for step in common_steps]
    window = min(recent_window, len(common_steps))
    epochs = sorted(
        set(int(row["epoch"]) for row in aligned_onepass)
        & set(int(row["epoch"]) for row in aligned_stamp)
    )
    by_epoch = {}
    for epoch in epochs:
        onepass_epoch = [row for row in aligned_onepass if int(row["epoch"]) == epoch]
        stamp_epoch = [row for row in aligned_stamp if int(row["epoch"]) == epoch]
        by_epoch[str(epoch)] = loss_gap_for_rows(
            onepass_epoch, stamp_epoch, bce_weight, dice_weight
        )
    return {
        "aligned_steps": len(common_steps),
        "recent_window": window,
        "recent": loss_gap_for_rows(
            aligned_onepass[-window:], aligned_stamp[-window:], bce_weight, dice_weight
        ),
        "by_epoch": by_epoch,
    }


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    query_builder = model.query_builder
    return {
        "seg_embedding": [query_builder.seg_embedding],
        "mask_embedding": [query_builder.mask_embedding],
        "visual_projection": list(query_builder.visual_projection.parameters()),
        "position_embeddings": [
            *query_builder.row_embedding.parameters(),
            *query_builder.col_embedding.parameters(),
        ],
        "classifier": list(model.mask_classifier.parameters()),
        "lora": list(model.lora_parameters()),
    }


def gradient_statistics(parameters: Iterable[torch.nn.Parameter]) -> dict[str, float]:
    gradient_square_sum = 0.0
    parameter_square_sum = 0.0
    parameter_count = 0
    gradient_element_count = 0
    active_gradient_count = 0
    for parameter in parameters:
        values = parameter.detach().float()
        parameter_square_sum += float(values.square().sum().item())
        parameter_count += values.numel()
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradient_square_sum += float(gradient.square().sum().item())
        gradient_element_count += gradient.numel()
        active_gradient_count += int(gradient.ne(0).sum().item())
    gradient_norm = math.sqrt(gradient_square_sum)
    gradient_rms = math.sqrt(gradient_square_sum / max(gradient_element_count, 1))
    active_gradient_rms = math.sqrt(gradient_square_sum / max(active_gradient_count, 1))
    parameter_rms = math.sqrt(parameter_square_sum / max(parameter_count, 1))
    return {
        "gradient_norm": gradient_norm,
        "gradient_rms": gradient_rms,
        "active_gradient_rms": active_gradient_rms,
        "parameter_rms": parameter_rms,
        "relative_gradient_rms": gradient_rms / max(parameter_rms, 1e-12),
        "active_gradient_fraction": active_gradient_count / max(parameter_count, 1),
    }


def average_stat_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean(float(row[key]) for row in rows if key in row) for key in keys}


@contextmanager
def zero_parameters(parameters: list[torch.nn.Parameter]):
    backups = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, backup in zip(parameters, backups):
                parameter.copy_(backup)


def forward_loss(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    targets: list[torch.Tensor],
    bce_weight: float,
    dice_weight: float,
    device: torch.device,
    dtype: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    autocast_enabled = device.type == "cuda" and dtype != "fp32"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=autocast_enabled,
    ):
        outputs = model(
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            pixel_values=prepared["pixel_values"],
            image_grid_thw=prepared["image_grid_thw"],
        )
        return segmentation_loss(
            outputs.mask_logits,
            targets,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )


def model_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.probe_samples <= 0:
        return {}
    checkpoint = args.onepass_checkpoint or args.output_root / "onepass7b" / "onepass_qwen7b.pt"
    config = read_checkpoint_config(checkpoint)
    base_model = args.base_model or config.get("base_model")
    if not base_model:
        raise ValueError("base-model is required and was not found in the checkpoint config.")
    dtype = args.dtype or str(config.get("dtype", "bf16"))
    attn_implementation = args.attn_implementation or str(config.get("attn_implementation", "sdpa"))
    min_pixels = args.min_pixels or int(config.get("min_pixels", 802816))
    max_pixels = args.max_pixels or int(config.get("max_pixels", 1003520))
    device = torch.device(args.device)
    configure_cudnn(args.disable_cudnn, device)
    dataset = OnePass7BDataset(
        args.train_json,
        args.data_root,
        limit=args.probe_samples,
        offset=args.offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_samples,
    )
    model, processor = load_base_onepass_model(
        stamp_code_dir=args.stamp_code_dir,
        base_model=base_model,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_grid_height=int(config.get("max_grid_height", 512)),
        max_grid_width=int(config.get("max_grid_width", 512)),
        use_lora=bool(config.get("use_lora", True)),
        lora_rank=int(config.get("lora_rank", 16)),
        lora_alpha=float(config.get("lora_alpha", 32.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        use_rslora=bool(config.get("use_rslora", False)),
        train_visual_projection=bool(config.get("train_visual_projection", True)),
        train_position_embeddings=bool(config.get("train_position_embeddings", True)),
        train_classifier=bool(config.get("train_classifier", True)),
        allow_existing_segmentation_tokens=args.allow_existing_segmentation_tokens,
        gradient_checkpointing=False,
    )
    load_checkpoint(model, checkpoint)
    model.eval()
    groups = parameter_groups(model)
    ablations = {
        "zero_seg_embedding": groups["seg_embedding"],
        "zero_mask_embedding": groups["mask_embedding"],
        "zero_visual_projection": groups["visual_projection"],
        "zero_position_embeddings": groups["position_embeddings"],
    }
    loss_rows: dict[str, list[dict[str, float]]] = {
        "baseline": [],
        **{name: [] for name in ablations},
    }
    gradient_rows: dict[str, list[dict[str, float]]] = {name: [] for name in groups}

    for samples in tqdm(loader, desc="OnePass loss-path probe", dynamic_ncols=True):
        prepared = prepare_batch(samples, processor, device)
        targets = grid_targets(samples, prepared["grid_shapes"], device)
        model.zero_grad(set_to_none=True)
        loss, parts = forward_loss(
            model, prepared, targets, args.bce_weight, args.dice_weight, device, dtype
        )
        loss.backward()
        loss_rows["baseline"].append(parts)
        for name, parameters in groups.items():
            gradient_rows[name].append(gradient_statistics(parameters))
        model.zero_grad(set_to_none=True)

        for name, parameters in ablations.items():
            with zero_parameters(parameters), torch.no_grad():
                _, ablated_parts = forward_loss(
                    model, prepared, targets, args.bce_weight, args.dice_weight, device, dtype
                )
            loss_rows[name].append(ablated_parts)

    averaged_losses = {name: average_stat_rows(rows) for name, rows in loss_rows.items()}
    baseline_loss = averaged_losses["baseline"]["loss"]
    for name, values in averaged_losses.items():
        values["loss_delta"] = values["loss"] - baseline_loss
        values["loss_delta_percent"] = (
            100.0 * values["loss_delta"] / baseline_loss if baseline_loss else float("nan")
        )
    averaged_gradients = {name: average_stat_rows(rows) for name, rows in gradient_rows.items()}
    seg_rms = averaged_gradients["seg_embedding"]["gradient_rms"]
    mask_rms = averaged_gradients["mask_embedding"]["gradient_rms"]
    return {
        "checkpoint": str(checkpoint),
        "samples": len(dataset),
        "batch_size": args.batch_size,
        "loss_sensitivity": averaged_losses,
        "gradient_groups": averaged_gradients,
        "seg_to_mask_gradient_rms_ratio": seg_rms / max(mask_rms, 1e-30),
        "interpretation_note": (
            "A small zero_seg_embedding loss delta plus a small SEG-to-MASK gradient ratio supports, "
            "but does not by itself prove, weak SEG-query usage."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    recent = result["history"]["recent"]
    lines = [
        "# Fair 7B Loss Diagnostics",
        "",
        "## Recent Training-Loss Gap",
        "",
        "| Component | OnePass | STAMP | Raw gap | Weighted gap | Gap share |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| BCE | {recent['onepass_bce']:.6f} | {recent['stamp_bce']:.6f} | "
            f"{recent['raw_bce_gap']:.6f} | {recent['weighted_bce_gap']:.6f} | "
            f"{100.0 * recent['bce_gap_fraction']:.2f}% |"
        ),
        (
            f"| Dice | {recent['onepass_dice']:.6f} | {recent['stamp_dice']:.6f} | "
            f"{recent['raw_dice_gap']:.6f} | {recent['weighted_dice_gap']:.6f} | "
            f"{100.0 * recent['dice_gap_fraction']:.2f}% |"
        ),
        "",
        f"Recent mask-loss gap: `{recent['measured_mask_loss_gap']:.6f}`.",
    ]
    probe = result.get("probe") or {}
    if probe:
        lines.extend(
            [
                "",
                "## OnePass Component Sensitivity",
                "",
                "| Intervention | Loss | Delta | Delta % | BCE | Dice |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in probe["loss_sensitivity"].items():
            lines.append(
                f"| {name} | {values['loss']:.6f} | {values['loss_delta']:.6f} | "
                f"{values['loss_delta_percent']:.2f}% | {values['loss_bce']:.6f} | "
                f"{values['loss_dice']:.6f} |"
            )
        lines.extend(
            [
                "",
                "## Gradient Flow",
                "",
                "| Parameter group | Gradient norm | Gradient RMS | Active RMS | Relative RMS | Active fraction |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in probe["gradient_groups"].items():
            lines.append(
                f"| {name} | {values['gradient_norm']:.6g} | {values['gradient_rms']:.6g} | "
                f"{values['active_gradient_rms']:.6g} | {values['relative_gradient_rms']:.6g} | "
                f"{100.0 * values['active_gradient_fraction']:.2f}% |"
            )
        lines.extend(
            [
                "",
                f"SEG/MASK gradient-RMS ratio: `{probe['seg_to_mask_gradient_rms_ratio']:.6g}`.",
                "",
                probe["interpretation_note"],
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.probe_samples < 0 or args.offset < 0:
        raise ValueError("probe-samples and offset must be non-negative.")
    if min(args.batch_size, args.recent_window) <= 0:
        raise ValueError("batch-size and recent-window must be positive.")
    if args.bce_weight < 0 or args.dice_weight < 0 or args.bce_weight + args.dice_weight <= 0:
        raise ValueError("Loss weights must be non-negative and not both zero.")
    result = {
        "history": history_diagnostics(
            args.output_root,
            args.recent_window,
            args.bce_weight,
            args.dice_weight,
        ),
        "probe": model_probe(args),
    }
    output_dir = args.output_root / "loss_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "loss_diagnostics.json"
    markdown_path = output_dir / "loss_diagnostics.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    print(f"JSON written to: {json_path}")
    print(f"Markdown written to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
