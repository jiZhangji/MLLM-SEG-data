from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage import draw


@dataclass
class ExportState:
    output_dir: Path
    method: str
    split: str
    tsv: Path
    patch_size: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    intersections: int = 0
    unions: int = 0
    model_seconds: float = 0.0


STATE: ExportState | None = None


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freeref-polyformer-code-dir", type=Path, required=True)
    parser.add_argument("--freeref-output-dir", type=Path, required=True)
    parser.add_argument("--freeref-tsv", type=Path, required=True)
    parser.add_argument("--freeref-bert-dir", type=Path, required=True)
    parser.add_argument("--freeref-method", default="PolyFormer-L-official")
    parser.add_argument("--freeref-split", required=True)
    parser.add_argument("--freeref-patch-size", type=int, default=512)
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    return args, remaining


def _artifact_key(instance_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id).strip("._") or "sample"
    digest = hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{digest}"


def _atomic_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    Image.fromarray(values).save(temporary, format="PNG")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _polygons_from_generation(generation: Any, patch_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    values = np.asarray(generation, dtype=np.float64)
    values = values[values != -1]
    if values.size < 4:
        return np.zeros(4, dtype=np.float64), []
    box = values[:4].copy()
    polygon_values = np.append(values[4:], 2.0)
    separator_indices = np.flatnonzero(polygon_values == 2.0)
    polygons: list[np.ndarray] = []
    previous = 0
    for current in separator_indices:
        points = polygon_values[previous:current]
        points = points[: points.size - points.size % 2]
        if points.size >= 6:
            polygons.append((points * patch_size).reshape(-1, 2))
        previous = int(current) + 1
    return box, polygons


def _rasterize(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for polygon in polygons:
        if polygon.shape[0] < 3:
            continue
        coordinates = np.stack([polygon[:, 1], polygon[:, 0]], axis=1)
        try:
            mask |= draw.polygon2mask(shape, coordinates)
        except (TypeError, ValueError, IndexError):
            continue
    return mask


def _patch_image_to_uint8(value: torch.Tensor) -> np.ndarray:
    image = ((value.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
    return image.permute(1, 2, 0).round().byte().numpy()


def _iou(mask: np.ndarray, target: np.ndarray) -> tuple[float, int, int]:
    intersection = int(np.logical_and(mask, target).sum())
    union = int(np.logical_or(mask, target).sum())
    return (intersection / union if union else 0.0), intersection, union


def export_eval_refcoco(task: Any, generator: Any, models: Any, sample: dict[str, Any], **kwargs: Any):
    del generator
    if STATE is None:
        raise RuntimeError("PolyFormer export state has not been initialized.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated = task.inference_step(models, sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    STATE.model_seconds += time.perf_counter() - started

    iou_scores: list[float] = []
    f_scores: list[float] = []
    intersections: list[int] = []
    unions: list[int] = []
    boxes = []
    polygon_counts = []
    polygon_lengths = []

    for index, generation in enumerate(generated):
        instance_id = str(sample["id"][index])
        artifact_key = _artifact_key(instance_id)
        box, polygons = _polygons_from_generation(generation, STATE.patch_size)
        box[::2] *= float(sample["w"][index].cpu())
        box[1::2] *= float(sample["h"][index].cpu())
        boxes.append(box)
        polygon_counts.append(len(polygons))
        polygon_lengths.append(sum(int(polygon.size) for polygon in polygons))

        target = np.asarray(sample["label"][index]) > 0
        prediction = _rasterize(polygons, target.shape)
        score, intersection, union = _iou(prediction, target)
        precision = (intersection + 1e-6) / (int(prediction.sum()) + 1e-6)
        recall = (intersection + 1e-6) / (int(target.sum()) + 1e-6)
        f_score = 2.0 * precision * recall / max(precision + recall, 1e-12)
        iou_scores.append(score)
        f_scores.append(f_score)
        intersections.append(intersection)
        unions.append(union)
        STATE.ious.append(score)
        STATE.intersections += intersection
        STATE.unions += union

        image_path = STATE.output_dir / "images" / f"{artifact_key}.png"
        gt_path = STATE.output_dir / "gt_masks" / f"{artifact_key}.png"
        prediction_path = STATE.output_dir / "pred_masks" / f"{artifact_key}.png"
        polygon_path = STATE.output_dir / "polygons" / f"{artifact_key}.json"
        _atomic_png(image_path, _patch_image_to_uint8(sample["net_input"]["patch_images"][index]))
        _atomic_png(gt_path, target.astype(np.uint8) * 255)
        _atomic_png(prediction_path, prediction.astype(np.uint8) * 255)
        _atomic_json(
            polygon_path,
            {
                "instance_id": instance_id,
                "polygons_xy": [polygon.tolist() for polygon in polygons],
                "box_xyxy": box.tolist(),
            },
        )
        STATE.rows.append(
            {
                "name": artifact_key,
                "method": STATE.method,
                "split": STATE.split,
                "instance_id": instance_id,
                "image": str(image_path),
                "gt_mask": str(gt_path),
                "prediction": str(prediction_path),
                "prediction_kind": "mask",
                "threshold": 0.5,
                "protocol": "polyformer_official_refer_polygon_rasterization",
                "polygon_json": str(polygon_path),
            }
        )

    box_tensor = torch.tensor(np.stack(boxes), dtype=torch.float32)
    box_tensor = box_tensor.to(sample["region_coords"].device)
    ap_scores = task._calculate_ap_score(box_tensor, sample["region_coords"].float())
    result_dir = Path(kwargs["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iou_scores": torch.tensor(iou_scores),
            "ap_scores": ap_scores.detach().cpu(),
            "n_poly_pred": polygon_counts,
            "n_poly_gt": sample["n_poly"],
            "poly_len": polygon_lengths,
            "uniq_id": sample["id"],
        },
        result_dir / f"{_artifact_key(str(sample['id'][0]))}.pt",
    )
    results = [{"uniq_id": sample_id} for sample_id in sample["id"].tolist()]
    return (
        results,
        torch.tensor(iou_scores),
        torch.tensor(f_scores),
        ap_scores.detach().cpu(),
        torch.tensor(intersections),
        torch.tensor(unions),
    )


def _ordered_rows(rows: list[dict[str, Any]], tsv: Path) -> list[dict[str, Any]]:
    by_id = {str(row["instance_id"]): row for row in rows}
    ordered = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        instance_id = line.split("\t", 1)[0]
        if instance_id not in by_id:
            raise RuntimeError(f"PolyFormer did not export TSV instance {instance_id!r}.")
        ordered.append(by_id[instance_id])
    if len(ordered) != len(by_id):
        raise RuntimeError("PolyFormer exported duplicate or unexpected instance IDs.")
    return ordered


def main() -> int:
    wrapper, official_argv = parse_wrapper_args(sys.argv[1:])
    if wrapper.freeref_patch_size <= 0:
        raise ValueError("freeref-patch-size must be positive.")
    code_dir = wrapper.freeref_polyformer_code_dir.expanduser().resolve()
    output_dir = wrapper.freeref_output_dir.expanduser().resolve()
    tsv = wrapper.freeref_tsv.expanduser().resolve()
    bert_dir = wrapper.freeref_bert_dir.expanduser().resolve()
    required = {
        "PolyFormer evaluate.py": code_dir / "evaluate.py",
        "PolyFormer TSV": tsv,
        "local BERT vocabulary": bert_dir / "vocab.txt",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("PolyFormer export inputs are incomplete:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code_dir))
    os.chdir(code_dir)

    # The released stack predates NumPy 1.24 and PyTorch 2.6.
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    original_torch_load = torch.load

    def trusted_checkpoint_load(*args: Any, **kwargs: Any):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load  # type: ignore[assignment]

    from bert.tokenization_bert import BertTokenizer  # type: ignore[import-not-found]

    original_from_pretrained = BertTokenizer.from_pretrained.__func__

    @classmethod
    def local_bert_from_pretrained(cls: Any, name: str, *args: Any, **kwargs: Any):
        source = str(bert_dir) if name == "bert-base-uncased" else name
        return original_from_pretrained(cls, source, *args, **kwargs)

    BertTokenizer.from_pretrained = local_bert_from_pretrained  # type: ignore[method-assign]

    from utils import eval_utils as official_eval_utils  # type: ignore[import-not-found]
    import evaluate as official_evaluate  # type: ignore[import-not-found]

    global STATE
    STATE = ExportState(
        output_dir=output_dir,
        method=wrapper.freeref_method,
        split=wrapper.freeref_split,
        tsv=tsv,
        patch_size=wrapper.freeref_patch_size,
    )
    official_eval_utils.eval_refcoco = export_eval_refcoco
    sys.argv = [str(code_dir / "evaluate.py"), *official_argv]
    official_evaluate.cli_main()

    rows = _ordered_rows(STATE.rows, tsv)
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )
    sample_count = len(rows)
    try:
        code_revision = subprocess.check_output(
            ["git", "-C", str(code_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = "unknown"
    summary = {
        "source": "polyformer_official_refer_polygon_rasterization",
        "method": wrapper.freeref_method,
        "split": wrapper.freeref_split,
        "samples": sample_count,
        "coarse_mean_iou": float(np.mean(STATE.ious)),
        "coarse_cIoU": STATE.intersections / max(STATE.unions, 1),
        "model_seconds": STATE.model_seconds,
        "model_seconds_per_sample": STATE.model_seconds / max(sample_count, 1),
        "polyformer_code_revision": code_revision,
        "tsv": str(tsv),
        "manifest": str(manifest),
    }
    _atomic_json(output_dir / "export_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
