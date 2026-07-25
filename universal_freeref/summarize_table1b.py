from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .export_utils import atomic_write_json


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a nine-number Table 1(b) row from eight paired summaries.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--summary", action="append", default=[], metavar="SPLIT=PATH", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eligibility",
        choices=("paper_candidate", "diagnostic_only"),
        default="paper_candidate",
    )
    parser.add_argument(
        "--expected-baseline",
        nargs=8,
        type=float,
        metavar=(
            "R_VAL",
            "R_TESTA",
            "R_TESTB",
            "R+_VAL",
            "R+_TESTA",
            "R+_TESTB",
            "RG_VAL",
            "RG_TEST",
        ),
        help="Paper-reported cIoU percentages used to gate a reproduced paired row.",
    )
    parser.add_argument(
        "--baseline-gate-tolerance",
        type=float,
        default=2.0,
        help="Maximum allowed absolute split error in percentage points (default: 2.0).",
    )
    parser.add_argument("--note", default="")
    return parser.parse_args()


def load_summaries(values: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--summary must be SPLIT=PATH, received {value!r}.")
        split, path_text = value.split("=", 1)
        if split not in SPLITS:
            raise ValueError(f"Unknown Table 1(b) split {split!r}.")
        path = Path(path_text).expanduser().resolve()
        result[split] = json.loads(path.read_text(encoding="utf-8"))
    missing = [split for split in SPLITS if split not in result]
    if missing:
        raise ValueError(f"Missing Table 1(b) summaries: {missing}.")
    return result


def metric_row(summaries: dict[str, dict[str, Any]], key: str) -> list[float]:
    values = [100.0 * float(summaries[split][key]) for split in SPLITS]
    return values + [sum(values) / len(values)]


def main() -> int:
    args = parse_args()
    if args.baseline_gate_tolerance < 0:
        raise ValueError("--baseline-gate-tolerance must be non-negative.")
    summaries = load_summaries(args.summary)
    baseline = metric_row(summaries, "coarse_cIoU")
    freeref = metric_row(summaries, "refined_cIoU")
    expected = list(args.expected_baseline) if args.expected_baseline is not None else None
    baseline_errors = (
        [observed - reference for observed, reference in zip(baseline[:8], expected)]
        if expected is not None
        else None
    )
    baseline_gate_passed = (
        max(abs(error) for error in baseline_errors) <= args.baseline_gate_tolerance
        if baseline_errors is not None
        else None
    )
    eligibility = args.eligibility
    if eligibility == "paper_candidate" and baseline_gate_passed is False:
        eligibility = "diagnostic_only"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": args.method,
        "eligibility": eligibility,
        "eligibility_requested": args.eligibility,
        "note": args.note,
        "columns": list(SPLITS) + ["average"],
        "baseline_cIoU_percent": baseline,
        "freeref_cIoU_percent": freeref,
        "delta_points": [after - before for before, after in zip(baseline, freeref)],
        "expected_baseline_cIoU_percent": expected,
        "baseline_errors_points": baseline_errors,
        "baseline_gate_tolerance_points": args.baseline_gate_tolerance,
        "baseline_gate_passed": baseline_gate_passed,
    }
    atomic_write_json(output_dir / "table1b_row.json", report)
    header = ["method", *SPLITS, "average", "eligibility"]
    lines = ["\t".join(header)]
    for suffix, values in (("baseline", baseline), ("+ FreeRef (ours)", freeref)):
        lines.append(
            "\t".join(
                [f"{args.method} {suffix}", *[f"{value:.1f}" for value in values], eligibility]
            )
        )
    (output_dir / "table1b_row.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown = [
        f"# {args.method} Table 1(b) row artifact",
        "",
        f"Eligibility: `{eligibility}`",
        "",
        "| Row | " + " | ".join((*SPLITS, "Avg.")) + " |",
        "|---|" + "|".join(["---:"] * 9) + "|",
        "| Baseline | " + " | ".join(f"{value:.1f}" for value in baseline) + " |",
        "| + FreeRef (ours) | " + " | ".join(f"{value:.1f}" for value in freeref) + " |",
    ]
    if expected is not None:
        markdown.extend(
            (
                "",
                "| Baseline gate | Value |",
                "|---|---:|",
                "| Paper reference | " + ", ".join(f"{value:.1f}" for value in expected) + " |",
                "| Maximum absolute split error | "
                + f"{max(abs(error) for error in baseline_errors):.2f} pp |",
                f"| Tolerance | {args.baseline_gate_tolerance:.2f} pp |",
                f"| Passed | {'yes' if baseline_gate_passed else 'no'} |",
            )
        )
    if args.note:
        markdown.extend(("", f"Note: {args.note}"))
    (output_dir / "table1b_row.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
