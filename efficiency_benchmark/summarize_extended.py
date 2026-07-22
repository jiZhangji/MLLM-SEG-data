from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = (
    ("stamp2b_base", "STAMP-2B", "Base", "-", "0"),
    ("stamp2b_freeref_gpu", "STAMP-2B", "+FreeRef-GPU", "No", "0"),
    ("stamp2b_sam_h", "STAMP-2B", "+SAM-H", "No", "SAM-H"),
    ("stamp2b_freeref_sam_h", "STAMP-2B", "+FreeRef-GPU -> SAM-H", "No", "SAM-H"),
    ("stamp7b_base", "STAMP-7B", "Base", "-", "0"),
    ("stamp7b_freeref_gpu", "STAMP-7B", "+FreeRef-GPU", "No", "0"),
    ("stamp7b_sam_h", "STAMP-7B", "+SAM-H", "No", "SAM-H"),
    ("stamp7b_freeref_sam_h", "STAMP-7B", "+FreeRef-GPU -> SAM-H", "No", "SAM-H"),
    ("text4seg_base", "Text4Seg-p24", "Base", "-", "0"),
    ("text4seg_freeref_gpu", "Text4Seg-p24", "+FreeRef-GPU", "No", "0"),
    ("text4seg_sam_h", "Text4Seg-p24", "+SAM-H", "No", "SAM-H"),
    (
        "text4seg_freeref_sam_h",
        "Text4Seg-p24",
        "+FreeRef-GPU -> SAM-H",
        "No",
        "SAM-H",
    ),
    ("lisa_original", "LISA-7B", "Original", "Yes", "learned decoder"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the extended single-GPU E2E suite.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = []
    missing = []
    for directory, method, variant, extra_train, extra_params in ROWS:
        path = args.input_root / directory / "summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        values.append((method, variant, extra_train, extra_params, summary))
    if missing:
        raise FileNotFoundError("Missing benchmark summaries:\n" + "\n".join(missing))

    sample_counts = {int(summary["samples"]) for *_, summary in values}
    warmups = {int(summary["warmup"]) for *_, summary in values}
    devices = {str(summary["device"]) for *_, summary in values}
    if len(sample_counts) != 1 or len(warmups) != 1:
        raise ValueError("All rows must use identical measured and warm-up sample counts.")
    if len(devices) != 1:
        raise ValueError("All rows must use the same GPU model.")

    measured = sample_counts.pop()
    warmup = warmups.pop()
    device = devices.pop()
    lines = [
        f"# Unified {device} End-to-End Efficiency",
        "",
        f"Batch size 1; {warmup} warm-up samples; {measured} measured samples.",
        "Model loading, dataset image decoding, and result writes are excluded. GPU work is synchronized per sample.",
        "",
        "| Method | Variant | Extra Train | Extra Params | Device | E2E Mean (s) | Median (s) | P95 (s) | FPS | Peak GPU (GiB) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method, variant, extra_train, extra_params, summary in values:
        lines.append(
            f"| {method} | {variant} | {extra_train} | {extra_params} | {summary['device']} | "
            f"{summary['e2e_mean_seconds']:.4f} | {summary['e2e_median_seconds']:.4f} | "
            f"{summary['e2e_p95_seconds']:.4f} | {summary['fps']:.3f} | "
            f"{summary['peak_gpu_gib']:.2f} |"
        )
    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
