#!/usr/bin/env python3
"""Evaluate baseline STAMP vs prior-prompt R-STAMP on prepared smoke data.

This script intentionally evaluates the JSON/mask files produced by
`prepare_stamp_training_data.py`, so it does not depend on the original REFER
pickle layout. It is meant for quick evidence collection after smoke training:
same image, same GT mask, same sample order; baseline uses the original query,
R-STAMP uses the query plus `structured_prior_text`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


DEFAULT_ROOT = Path("/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG")
DEFAULT_DATASET_JSON = "refcocog_formatted_all_sentences_doubled_mp.json"


@dataclass
class EvalRow:
    run: str
    index: int
    image: str
    mask: str
    query: str
    has_prior: bool
    pred_found: bool
    pred_nonzero: int
    gt_nonzero: int
    intersection: int
    union: int
    iou: float


def setup_stamp_imports(stamp_code_dir: Path) -> None:
    stamp_code_dir = stamp_code_dir.resolve()
    if str(stamp_code_dir) not in sys.path:
        sys.path.insert(0, str(stamp_code_dir))


def load_segmenter_class(stamp_code_dir: Path):
    setup_stamp_imports(stamp_code_dir)
    try:
        from segment_predictor import GenerativeSegmenter  # type: ignore
    except Exception:
        from segment_predictor_cache import GenerativeSegmenter  # type: ignore
    return GenerativeSegmenter


def load_items(path: Path, limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")
    if limit is not None and limit > 0:
        data = data[:limit]
    return data


def extract_text_from_messages(item: dict[str, Any]) -> str:
    messages = item.get("messages") or item.get("conversations") or []
    if not messages:
        return item.get("query") or item.get("text") or ""
    first = messages[0]
    content = first.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(x for x in texts if x)
    return str(content)


def build_query(item: dict[str, Any], use_prior: bool) -> str:
    query = extract_text_from_messages(item).strip()
    prior = str(item.get("structured_prior_text") or "").strip()
    if use_prior and prior and prior not in query:
        return f"{prior}\n\nUser request: {query}"
    return query


def first_existing_path(paths: list[str], root: Path | None = None) -> Path:
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
        if root is not None:
            q = root / raw
            if q.exists():
                return q
    raise FileNotFoundError(f"No existing path found among: {paths}")


def get_image_path(item: dict[str, Any], root: Path) -> Path:
    values = item.get("images") or item.get("image") or item.get("image_path")
    if isinstance(values, list):
        candidates = [str(x) for x in values]
    else:
        candidates = [str(values)]
    return first_existing_path(candidates, root)


def get_mask_path(item: dict[str, Any], root: Path) -> Path:
    values = item.get("masks") or item.get("mask") or item.get("mask_path")
    if isinstance(values, list):
        candidates = [str(x) for x in values]
    else:
        candidates = [str(values)]
    return first_existing_path(candidates, root)


def binary_mask_from_file(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"))
    return torch.from_numpy((arr > 0).astype(np.uint8))


def binary_mask_from_prediction(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.detach().float().cpu()
    if mask.ndim > 2:
        mask = mask.squeeze()
    if tuple(mask.shape[-2:]) != (height, width):
        mask = F.interpolate(
            mask.reshape(1, 1, mask.shape[-2], mask.shape[-1]),
            size=(height, width),
            mode="nearest",
        ).reshape(height, width)
    return (mask > 0).to(torch.uint8)


def compute_iou(pred: torch.Tensor, gt: torch.Tensor) -> tuple[int, int, int, float]:
    pred = pred.bool()
    gt = gt.bool()
    inter = int(torch.logical_and(pred, gt).sum().item())
    union = int(torch.logical_or(pred, gt).sum().item())
    if union == 0:
        return inter, union, int(pred.sum().item()), 1.0
    return inter, union, int(pred.sum().item()), float(inter / union)


def evaluate_run(
    *,
    run_name: str,
    model_path: Path,
    json_path: Path,
    root: Path,
    stamp_code_dir: Path,
    use_prior: bool,
    limit: int,
    min_pixels: int,
    max_pixels: int,
) -> list[EvalRow]:
    GenerativeSegmenter = load_segmenter_class(stamp_code_dir)
    items = load_items(json_path, limit)

    print(f"\n=== Evaluating {run_name} ===")
    print(f"model_path={model_path}")
    print(f"json_path={json_path}")
    print(f"samples={len(items)}")
    print(f"use_prior={use_prior}")

    segmenter = GenerativeSegmenter(
        str(model_path),
        device_map="cuda",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    rows: list[EvalRow] = []
    for idx, item in enumerate(tqdm(items, desc=run_name)):
        image_path = get_image_path(item, root)
        mask_path = get_mask_path(item, root)
        image = Image.open(image_path).convert("RGB")
        gt = binary_mask_from_file(mask_path)
        height, width = gt.shape[-2:]
        query = build_query(item, use_prior=use_prior)

        pred_found = False
        pred = torch.zeros((height, width), dtype=torch.uint8)
        try:
            with torch.inference_mode():
                masks, _response_text = segmenter.generate_with_segmentation(image, query)
            if masks is not None and len(masks) > 0:
                pred_found = True
                pred = binary_mask_from_prediction(masks[0], height, width)
        except Exception as exc:
            print(f"[WARN] {run_name} idx={idx} failed: {type(exc).__name__}: {exc}")

        inter, union, pred_nonzero, iou = compute_iou(pred, gt)
        rows.append(
            EvalRow(
                run=run_name,
                index=idx,
                image=str(image_path),
                mask=str(mask_path),
                query=query,
                has_prior=bool(item.get("structured_prior_text")),
                pred_found=pred_found,
                pred_nonzero=pred_nonzero,
                gt_nonzero=int(gt.bool().sum().item()),
                intersection=inter,
                union=union,
                iou=iou,
            )
        )

    del segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "mean_iou": None,
            "no_mask_rate": None,
            "empty_pred_rate": None,
        }
    ious = np.array([r.iou for r in rows], dtype=np.float64)
    no_mask = np.array([not r.pred_found for r in rows], dtype=np.float64)
    empty = np.array([r.pred_nonzero == 0 for r in rows], dtype=np.float64)
    return {
        "num_samples": len(rows),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou_25": float(np.quantile(ious, 0.25)),
        "iou_75": float(np.quantile(ious, 0.75)),
        "no_mask_rate": float(no_mask.mean()),
        "empty_pred_rate": float(empty.mean()),
        "iou_ge_0_5_rate": float((ious >= 0.5).mean()),
    }


def write_rows_csv(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else [f.name for f in EvalRow.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_report(path: Path, report: dict[str, Any]) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    s = report["summary"]
    b = s.get("baseline", {})
    r = s.get("rstamp", {})
    c = report.get("comparison", {})
    lines = [
        "# Smoke IoU Evaluation Report",
        "",
        "> Same prepared samples are used for both runs. Baseline uses the original query; R-STAMP uses the query plus `structured_prior_text`.",
        "",
        "| Metric | Baseline | R-STAMP | Delta (R-B) |",
        "|---|---:|---:|---:|",
        f"| mean_iou | {fmt(b.get('mean_iou'))} | {fmt(r.get('mean_iou'))} | {fmt(c.get('mean_iou_delta'))} |",
        f"| median_iou | {fmt(b.get('median_iou'))} | {fmt(r.get('median_iou'))} | {fmt(c.get('median_iou_delta'))} |",
        f"| iou_ge_0_5_rate | {fmt(b.get('iou_ge_0_5_rate'))} | {fmt(r.get('iou_ge_0_5_rate'))} | {fmt(c.get('iou_ge_0_5_rate_delta'))} |",
        f"| no_mask_rate | {fmt(b.get('no_mask_rate'))} | {fmt(r.get('no_mask_rate'))} | {fmt(c.get('no_mask_rate_delta'))} |",
        f"| empty_pred_rate | {fmt(b.get('empty_pred_rate'))} | {fmt(r.get('empty_pred_rate'))} | {fmt(c.get('empty_pred_rate_delta'))} |",
        "",
        "## Files",
        "",
        f"- baseline_csv: `{report['files']['baseline_csv']}`",
        f"- rstamp_csv: `{report['files']['rstamp_csv']}`",
        f"- json: `{report['files']['json']}`",
        "",
        "## Interpretation guide",
        "",
        "- If R-STAMP has higher `mean_iou` and lower `no_mask_rate`, the structured prior is giving useful segmentation signal.",
        "- If R-STAMP only improves training `loss_seg` but not eval IoU, the current prompt-injection prior may be overfitting or not aligned with inference.",
        "- This is still a smoke result. Use it for direction validation, not final SOTA claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stamp-code-dir", type=Path, default=None)
    parser.add_argument("--baseline-model", type=Path, default=None)
    parser.add_argument("--rstamp-model", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--rstamp-json", type=Path, default=None)
    parser.add_argument("--dataset-json-name", default=DEFAULT_DATASET_JSON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    stamp_code_dir = (args.stamp_code_dir or root / "code" / "STAMP").expanduser().resolve()
    baseline_model = (args.baseline_model or root / "outputs" / "smoke_baseline_1x48g" / "final_model").expanduser().resolve()
    rstamp_model = (args.rstamp_model or root / "outputs" / "smoke_rstamp_1x48g" / "final_model").expanduser().resolve()
    baseline_json = (
        args.baseline_json
        or stamp_code_dir / "playground" / "data" / "json_files_baseline" / args.dataset_json_name
    ).expanduser().resolve()
    rstamp_json = (
        args.rstamp_json
        or stamp_code_dir / "playground" / "data" / "json_files_rstamp" / args.dataset_json_name
    ).expanduser().resolve()
    output_dir = (args.output_dir or root / "outputs" / "smoke_eval_iou").expanduser().resolve()

    if os.environ.get("STAMP_DISABLE_CUDNN", "0") == "1":
        torch.backends.cudnn.enabled = False

    required = [stamp_code_dir, baseline_model, rstamp_model, baseline_json, rstamp_json]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(str(p))

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = evaluate_run(
        run_name="baseline",
        model_path=baseline_model,
        json_path=baseline_json,
        root=root,
        stamp_code_dir=stamp_code_dir,
        use_prior=False,
        limit=args.limit,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    rstamp_rows = evaluate_run(
        run_name="rstamp",
        model_path=rstamp_model,
        json_path=rstamp_json,
        root=root,
        stamp_code_dir=stamp_code_dir,
        use_prior=True,
        limit=args.limit,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    baseline_summary = summarize(baseline_rows)
    rstamp_summary = summarize(rstamp_rows)
    comparison = {}
    for key in ["mean_iou", "median_iou", "iou_ge_0_5_rate", "no_mask_rate", "empty_pred_rate"]:
        if baseline_summary.get(key) is not None and rstamp_summary.get(key) is not None:
            comparison[f"{key}_delta"] = rstamp_summary[key] - baseline_summary[key]

    baseline_csv = output_dir / "baseline_per_sample.csv"
    rstamp_csv = output_dir / "rstamp_per_sample.csv"
    json_path = output_dir / "smoke_iou_comparison.json"
    md_path = output_dir / "smoke_iou_comparison.md"
    write_rows_csv(baseline_csv, baseline_rows)
    write_rows_csv(rstamp_csv, rstamp_rows)

    report = {
        "root": str(root),
        "limit": args.limit,
        "summary": {
            "baseline": baseline_summary,
            "rstamp": rstamp_summary,
        },
        "comparison": comparison,
        "files": {
            "baseline_csv": str(baseline_csv),
            "rstamp_csv": str(rstamp_csv),
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(md_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"MD:   {md_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
