from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine official STAMP SAM-H summaries.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pct(value: object) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.results_root.glob("stamp_official_samh_stamp*/eval_summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol") != "stamp_official_frozen_sam_h_v1":
            continue
        value["source"] = str(path)
        rows.append(value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "| Model | Split | N | Coarse mIoU | FreeRef mIoU | Coarse->SAM-H mIoU | "
        "FreeRef->SAM-H mIoU | Coarse cIoU | FreeRef cIoU | Coarse->SAM-H cIoU | FreeRef->SAM-H cIoU |"
    )
    divider = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Official STAMP Frozen SAM-H Comparison",
        "",
        "STAMP rows use the released STAMP frozen-SAM-H prompting helpers on paired saved dumps.",
        "This isolates post-processing but is not a full reproduction of STAMP model inference.",
        "",
        header,
        divider,
    ]
    for row in rows:
        lines.append(
            "| {model} | {split} | {samples} | {coarse_miou} | {freeref_miou} | {coarse_sam_miou} | "
            "{freeref_sam_miou} | {coarse_ciou} | {freeref_ciou} | {coarse_sam_ciou} | {freeref_sam_ciou} |".format(
                model=row.get("model", "-"),
                split=row.get("split", "-"),
                samples=row.get("samples", "-"),
                coarse_miou=pct(row.get("coarse_mean_iou")),
                freeref_miou=pct(row.get("freeref_mean_iou")),
                coarse_sam_miou=pct(row.get("coarse_sam_mean_iou")),
                freeref_sam_miou=pct(row.get("freeref_sam_mean_iou")),
                coarse_ciou=pct(row.get("coarse_cIoU")),
                freeref_ciou=pct(row.get("freeref_cIoU")),
                coarse_sam_ciou=pct(row.get("coarse_sam_cIoU")),
                freeref_sam_ciou=pct(row.get("freeref_sam_cIoU")),
            )
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "combined_summary.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(
            {"protocol": "stamp_official_frozen_sam_h_v1_on_paired_dumps", "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
