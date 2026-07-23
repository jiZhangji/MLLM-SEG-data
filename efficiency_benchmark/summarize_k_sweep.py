from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = (("stamp2b", "STAMP-2B"), ("stamp7b", "STAMP-7B"), ("text4seg_p24", "Text4Seg-p24"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize serial end-to-end K timing runs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for slug, label in MODELS:
        for variant, k in [("base", None), *[(f"k_{value}", value) for value in args.k_values]]:
            path = args.input_root / slug / variant / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": label,
                    "configuration": "Base" if k is None else f"K={k}",
                    "K": k,
                    "device": value["device"],
                    "samples": int(value["samples"]),
                    "mean_seconds": float(value["e2e_mean_seconds"]),
                    "median_seconds": float(value["e2e_median_seconds"]),
                    "p95_seconds": float(value["e2e_p95_seconds"]),
                    "samples_per_second": float(value["fps"]),
                    "summary": str(path),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "k_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "k_timing.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Serial H100 End-to-End K Timing",
        "",
        "| Model | Configuration | GPU | Mean (s/sample) | Median | P95 | Samples/s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {configuration} | {device} | {mean_seconds:.4f} | "
            "{median_seconds:.4f} | {p95_seconds:.4f} | {samples_per_second:.3f} |".format(
                **row
            )
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "k_timing.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
