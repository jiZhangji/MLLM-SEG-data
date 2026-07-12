from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from token_refine.data import resolve_path
from token_refine.model import MaskTokenRefinementAdapter


PANEL_NAMES = ("Image", "Ground truth", "STAMP coarse", "Token Adapter", "Change map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize STAMP coarse masks against Token Adapter refinements."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing dump .pt files.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Token Adapter checkpoint.")
    parser.add_argument("--rows-csv", required=True, type=Path, help="eval_rows.csv used to rank examples.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-best", type=int, default=8)
    parser.add_argument("--num-worst", type=int, default=4)
    parser.add_argument("--num-neutral", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def delta_value(row: dict[str, str]) -> float:
    for key in ("original_delta", "delta"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError("CSV must contain original_delta or delta.")


def select_rows(
    rows: list[dict[str, str]],
    num_best: int,
    num_worst: int,
    num_neutral: int,
) -> list[tuple[str, dict[str, str]]]:
    if min(num_best, num_worst, num_neutral) < 0:
        raise ValueError("Selection counts must be non-negative.")
    ranked = sorted(rows, key=delta_value)
    selected: list[tuple[str, dict[str, str]]] = []
    used: set[str] = set()

    def add(category: str, candidates: list[dict[str, str]], count: int) -> None:
        for row in candidates:
            identity = row.get("path") or row.get("name") or json.dumps(row, sort_keys=True)
            if identity in used:
                continue
            used.add(identity)
            selected.append((category, row))
            if sum(1 for current, _ in selected if current == category) >= count:
                break

    add("best", list(reversed(ranked)), num_best)
    add("worst", ranked, num_worst)
    add("neutral", sorted(rows, key=lambda row: abs(delta_value(row))), num_neutral)
    return selected


def load_model(checkpoint_path: Path, device: torch.device) -> MaskTokenRefinementAdapter:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = MaskTokenRefinementAdapter(
        token_dim=int(checkpoint["token_dim"]),
        hidden_size=int(config.get("hidden_size", 128)),
        use_uncertainty_gate=str(config.get("use_uncertainty_gate", "True")).lower() != "false",
        trainable_logit_calibration=str(config.get("trainable_logit_calibration", "False")).lower() == "true",
    ).to(device)
    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has neither 'model' nor 'model_state_dict'.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def resolve_dump_path(row: dict[str, str], input_dir: Path) -> Path:
    raw = Path(row.get("path", ""))
    candidates = [raw, input_dir / raw.name]
    name = row.get("name")
    if name:
        candidates.append(input_dir / f"{name}.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve dump for row name={name!r}, path={str(raw)!r} in {input_dir}")


def payload_path(payload: dict[str, Any], dump_path: Path, keys: tuple[str, ...]) -> Path:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if value:
            path = resolve_path(str(value), dump_path)
            if path.exists():
                return path
    raise KeyError(f"Dump {dump_path} has no valid path in keys {keys}; available keys={sorted(payload)}")


def logits_probability(logits: torch.Tensor, grid_hw: tuple[int, int], image_hw: tuple[int, int]) -> np.ndarray:
    foreground = torch.softmax(logits.float(), dim=-1)[..., 1].reshape(1, 1, *grid_hw)
    resized = F.interpolate(foreground, size=image_hw, mode="bilinear", align_corners=False)
    return resized.squeeze().detach().cpu().numpy()


def resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    output = base.copy()
    selected = mask.astype(bool)
    output[selected] = output[selected] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def change_map(image: Image.Image, gt: np.ndarray, coarse: np.ndarray, refined: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) * 0.34
    coarse_correct = coarse == gt
    refined_correct = refined == gt
    fixed = (~coarse_correct) & refined_correct
    harmed = coarse_correct & (~refined_correct)
    remaining_error = (~coarse_correct) & (~refined_correct)
    base[fixed] = (35, 220, 90)       # green: corrected by adapter
    base[harmed] = (240, 55, 55)      # red: newly incorrect
    base[remaining_error] = (245, 185, 45)  # yellow: still incorrect
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def fit_panel(image: Image.Image, panel_size: int) -> Image.Image:
    width, height = image.size
    scale = min(panel_size / max(width, 1), panel_size / max(height, 1))
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (panel_size, panel_size), "white")
    offset = ((panel_size - resized.width) // 2, (panel_size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def render_comparison(
    image: Image.Image,
    gt: np.ndarray,
    coarse: np.ndarray,
    refined: np.ndarray,
    title: str,
    panel_size: int,
) -> Image.Image:
    panels = [
        image.convert("RGB"),
        overlay_mask(image, gt, (45, 210, 95)),
        overlay_mask(image, coarse, (45, 145, 245)),
        overlay_mask(image, refined, (185, 75, 235)),
        change_map(image, gt, coarse, refined),
    ]
    header_height = 72
    label_height = 34
    gap = 8
    width = len(panels) * panel_size + (len(panels) - 1) * gap
    canvas = Image.new("RGB", (width, header_height + panel_size + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill="black", font=font)
    draw.text(
        (10, 34),
        "Change map: green=fixed, red=new error, yellow=remaining error",
        fill=(55, 55, 55),
        font=font,
    )
    for index, (name, panel) in enumerate(zip(PANEL_NAMES, panels)):
        x = index * (panel_size + gap)
        fitted = fit_panel(panel, panel_size)
        canvas.paste(fitted, (x, header_height))
        label_x = x + max(0, (panel_size - text_width(draw, name, font)) // 2)
        draw.text((label_x, header_height + panel_size + 9), name, fill="black", font=font)
    return canvas


def safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned.strip("_") or "sample"


def make_contact_sheet(images: list[tuple[str, Image.Image]], output_path: Path) -> None:
    if not images:
        return
    max_width = max(image.width for _, image in images)
    total_height = sum(image.height + 28 for _, image in images)
    sheet = Image.new("RGB", (max_width, total_height), (235, 237, 240))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = 0
    for category, image in images:
        draw.text((8, y + 7), category.upper(), fill="black", font=font)
        y += 28
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output_path)


def write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['category'])}</td>"
            f"<td>{html.escape(record['name'])}</td>"
            f"<td>{record['coarse_iou']:.4f}</td>"
            f"<td>{record['refined_iou']:.4f}</td>"
            f"<td>{record['delta']:+.4f}</td>"
            f"<td><a href=\"{html.escape(record['file'])}\">view</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Token Adapter visualization</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:7px 10px}}th{{background:#eee}}</style></head>
<body><h1>STAMP coarse vs Token Adapter</h1>
<p>Green pixels in the change map were fixed; red pixels became wrong; yellow pixels remain wrong.</p>
<table><thead><tr><th>Group</th><th>Sample</th><th>Coarse IoU</th><th>Adapter IoU</th><th>Delta</th><th>Image</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.panel_size < 64:
        raise ValueError("panel-size must be at least 64.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = read_rows(args.rows_csv)
    selected = select_rows(rows, args.num_best, args.num_worst, args.num_neutral)
    if not selected:
        raise ValueError("No samples selected. Increase at least one selection count.")
    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    sheet_images: list[tuple[str, Image.Image]] = []
    with torch.inference_mode():
        for order, (category, row) in enumerate(selected):
            dump_path = resolve_dump_path(row, args.input_dir)
            payload = torch.load(dump_path, map_location="cpu", weights_only=False)
            image_path = payload_path(payload, dump_path, ("image_path", "image", "images"))
            mask_path = payload_path(payload, dump_path, ("mask_path", "mask", "masks"))
            image = Image.open(image_path).convert("RGB")
            gt_image = Image.open(mask_path).convert("L")
            if gt_image.size != image.size:
                gt_image = gt_image.resize(image.size, Image.Resampling.NEAREST)
            gt = np.asarray(gt_image) > 0

            mask_logits = payload["mask_logits"].squeeze(0).float().unsqueeze(0).to(device)
            mask_hidden = payload["mask_hidden"].squeeze(0).float().unsqueeze(0).to(device)
            grid_hw = tuple(int(value) for value in payload["grid_hw"])
            outputs = model(mask_hidden, mask_logits)
            image_hw = (image.height, image.width)
            coarse_prob = logits_probability(mask_logits, grid_hw, image_hw)
            refined_prob = logits_probability(outputs["refined_logits"], grid_hw, image_hw)
            coarse = coarse_prob >= args.threshold
            refined = refined_prob >= args.threshold

            coarse_iou = float(row.get("original_coarse_iou") or row.get("coarse_iou") or 0.0)
            refined_iou = float(row.get("original_refined_iou") or row.get("refined_iou") or 0.0)
            delta = refined_iou - coarse_iou
            name = row.get("name") or dump_path.stem
            title = (
                f"{category.upper()} | {name} | STAMP IoU={coarse_iou:.4f} | "
                f"Adapter IoU={refined_iou:.4f} | Delta={delta:+.4f}"
            )
            comparison = render_comparison(image, gt, coarse, refined, title, args.panel_size)
            filename = f"{order:03d}_{category}_{safe_name(name)}.png"
            comparison.save(args.output_dir / filename)
            sheet_images.append((category, comparison))
            records.append(
                {
                    "category": category,
                    "name": name,
                    "dump_path": str(dump_path),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "coarse_iou": coarse_iou,
                    "refined_iou": refined_iou,
                    "delta": delta,
                    "file": filename,
                }
            )
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    make_contact_sheet(sheet_images, args.output_dir / "contact_sheet.png")
    write_index(args.output_dir, records)
    with (args.output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(
            {
                "samples": len(records),
                "num_best": sum(record["category"] == "best" for record in records),
                "num_worst": sum(record["category"] == "worst" for record in records),
                "num_neutral": sum(record["category"] == "neutral" for record in records),
                "threshold": args.threshold,
                "checkpoint": str(args.checkpoint),
                "rows_csv": str(args.rows_csv),
                "contact_sheet": str(args.output_dir / "contact_sheet.png"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Visualizations written to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
