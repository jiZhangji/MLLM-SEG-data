from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_lisa_official_freeref_sam import BRANCHES, PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine official REFER latent FreeRef summaries.")
    parser.add_argument("--summary", action="append", default=[], metavar="SPLIT=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summary:
        raise ValueError("At least one --summary SPLIT=PATH is required.")
    rows = []
    for specification in args.summary:
        split, separator, path_text = specification.partition("=")
        if not separator:
            raise ValueError(f"Invalid summary specification: {specification}")
        value = json.loads(Path(path_text).read_text(encoding="utf-8"))
        if value.get("protocol") != PROTOCOL:
            raise ValueError(f"Unexpected protocol in {path_text}: {value.get('protocol')}")
        value["table_split"] = split
        rows.append(value)

    labels = {
        "baseline": "LISA",
        "latent_sam": "+latent prompt+SAM-H",
        "freeref_sam": "+latent FreeRef+SAM-H",
    }
    lines = [
        "# Public LISA-7B-v1 + Latent FreeRef Before Native SAM-H",
        "",
        "Official REFER loader; each compared method path uses one native SAM-H decode.",
        "The latent FreeRef path does not consume the baseline LISA mask.",
        "",
        "| Split | N | " + " | ".join(f"{labels[b]} mIoU | {labels[b]} cIoU" for b in BRANCHES) + " |",
        "|---|---:|" + "---:|---:|" * len(BRANCHES),
    ]
    for row in rows:
        values = []
        for branch in BRANCHES:
            values.extend(
                [
                    f"{row[f'{branch}_mean_iou'] * 100:.2f}",
                    f"{row[f'{branch}_cIoU'] * 100:.2f}",
                ]
            )
        lines.append(f"| {row['table_split']} | {row['samples']} | " + " | ".join(values) + " |")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps({"protocol": PROTOCOL, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
