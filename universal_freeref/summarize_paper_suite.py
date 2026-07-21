from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcoco+_val",
    "refcoco+_testA",
    "refcoco+_testB",
    "refcocog_val",
    "refcocog_test",
)

EXPECTED = {
    "refcoco_val": 10834,
    "refcoco_testA": 5657,
    "refcoco_testB": 5095,
    "refcoco+_val": 10758,
    "refcoco+_testA": 5726,
    "refcoco+_testB": 4889,
    "refcocog_val": 4896,
    "refcocog_test": 9602,
}

METHODS = (
    ("STAMP-2B", "Qwen2-2B"),
    ("STAMP-7B", "Qwen2-7B"),
    ("Text4Seg-p24", "Vicuna-7B"),
    ("PixelLM-7B-public", "Vicuna-7B"),
    ("SegAgent-SimpleClick", "Qwen-7B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a provenance-aware FreeRef paper table.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def candidate_paths(root: Path, method: str, split: str) -> list[Path]:
    safe = split.replace("+", "plus")
    if method.startswith("STAMP-"):
        size = "2b" if "2B" in method else "7b"
        dataset = split.replace("refcoco+", "refcocoplus")
        values = [
            root / "outputs" / f"training_free_refine_stamp{size}_{dataset}_full" / "eval_summary.json",
            root / "outputs" / f"training_free_refine_stamp{size}_{split}_full" / "eval_summary.json",
        ]
        if size == "2b":
            values.append(root / "outputs" / f"training_free_refine_{split}_full" / "eval_summary.json")
        return values
    if method == "Text4Seg-p24":
        return [root / "outputs" / f"text4seg_training_free_{split}" / "eval_summary.json"]
    if method == "PixelLM-7B-public":
        return [root / "outputs" / "pixellm_public_freeref" / safe / "freeref" / "eval_summary.json"]
    if method == "SegAgent-SimpleClick":
        return [root / "outputs" / "segagent_freeref" / safe / "freeref" / "eval_summary.json"]
    return []


def load_result(root: Path, method: str, split: str) -> dict[str, Any] | None:
    for path in candidate_paths(root, method, split):
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        samples = int(value.get("samples", 0))
        expected = EXPECTED[split]
        complete = samples == expected or (
            method == "STAMP-2B"
            and split in {"refcoco_testA", "refcoco+_testB"}
            and samples == expected - 1
        )
        return {
            "samples": samples,
            "expected": expected,
            "complete": complete,
            "coarse_miou": float(value["coarse_mean_iou"]) * 100.0,
            "refined_miou": float(value["refined_mean_iou"]) * 100.0,
            "delta_miou": float(value["mean_iou_delta"]) * 100.0,
            "coarse_ciou": float(value["coarse_cIoU"]) * 100.0,
            "refined_ciou": float(value["refined_cIoU"]) * 100.0,
            "delta_ciou": float(value["cIoU_delta"]) * 100.0,
            "source": str(path.resolve()),
            "protocol": str(value.get("source", "unknown")),
        }
    return None


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def method_average(results: dict[str, dict[str, Any] | None], key: str) -> float | None:
    values = [result[key] for result in results.values() if result is not None and result["complete"]]
    return sum(values) / len(values) if len(values) == len(SPLITS) else None


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, Any] = {"splits": list(SPLITS), "methods": {}}
    lines = [
        "# FreeRef Paired Paper Results",
        "",
        "Values are paired measurements from local saved predictions. They are not copied paper-table values.",
        "",
        "| Method | LLM | Variant | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val(U) | test(U) | Avg. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, llm in METHODS:
        results = {split: load_result(root, method, split) for split in SPLITS}
        inventory["methods"][method] = results
        for variant, key in (("Original", "coarse_miou"), ("+ FreeRef", "refined_miou")):
            scores = [results[split][key] if results[split] is not None and results[split]["complete"] else None for split in SPLITS]
            average = method_average(results, key)
            lines.append(
                "| " + " | ".join(
                    [method, llm, variant, *[format_score(value) for value in scores], format_score(average)]
                ) + " |"
            )

    lines.extend(
        [
            "",
            "## Completion",
            "",
            "| Method | Split | Samples | Expected | Status | mIoU Delta | cIoU Delta | Source |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for method, _ in METHODS:
        for split in SPLITS:
            result = inventory["methods"][method][split]
            if result is None:
                lines.append(f"| {method} | {split} | - | {EXPECTED[split]} | missing | - | - | - |")
                continue
            status = "complete" if result["complete"] else "partial"
            lines.append(
                f"| {method} | {split} | {result['samples']} | {result['expected']} | {status} | "
                f"{result['delta_miou']:+.2f} | {result['delta_ciou']:+.2f} | `{result['source']}` |"
            )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "paper_results.md").write_text(markdown, encoding="utf-8")
    (output_dir / "paper_results.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
