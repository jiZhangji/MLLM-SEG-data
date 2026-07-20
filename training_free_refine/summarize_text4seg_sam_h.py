from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine paired Text4Seg/FreeRef/SAM-H summaries.")
    parser.add_argument("--summary", action="append", default=[], metavar="SPLIT=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}"


def main() -> int:
    args = parse_args()
    if not args.summary:
        raise ValueError("At least one --summary SPLIT=PATH is required.")
    rows: list[dict[str, object]] = []
    for specification in args.summary:
        split, separator, path_text = specification.partition("=")
        if not separator:
            raise ValueError(f"Invalid --summary value: {specification}")
        path = Path(path_text)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol") != "text4seg_public_p24_paired_frozen_sam_h_v1":
            raise ValueError(f"Unexpected protocol in {path}: {value.get('protocol')}")
        value["split"] = split
        rows.append(value)

    lines = [
        "# Text4Seg Public-p24 + FreeRef + Frozen SAM-H",
        "",
        "All four branches use the same saved Text4Seg masks; both SAM-H branches use paired seeds.",
        "",
        "| Split | N | Text4Seg mIoU | +FreeRef | +SAM-H | +FreeRef+SAM-H | Text4Seg cIoU | +FreeRef | +SAM-H | +FreeRef+SAM-H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {samples} | {coarse_miou} | {freeref_miou} | {coarse_sam_miou} | "
            "{freeref_sam_miou} | {coarse_ciou} | {freeref_ciou} | {coarse_sam_ciou} | "
            "{freeref_sam_ciou} |".format(
                split=row["split"],
                samples=row["samples"],
                coarse_miou=pct(row["coarse_mean_iou"]),
                freeref_miou=pct(row["freeref_mean_iou"]),
                coarse_sam_miou=pct(row["coarse_sam_mean_iou"]),
                freeref_sam_miou=pct(row["freeref_sam_mean_iou"]),
                coarse_ciou=pct(row["coarse_cIoU"]),
                freeref_ciou=pct(row["freeref_cIoU"]),
                coarse_sam_ciou=pct(row["coarse_sam_cIoU"]),
                freeref_sam_ciou=pct(row["freeref_sam_cIoU"]),
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(
            {"protocol": "text4seg_public_p24_paired_frozen_sam_h_v1", "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

