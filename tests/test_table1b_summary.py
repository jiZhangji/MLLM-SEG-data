from __future__ import annotations

import json
import sys
from pathlib import Path

from universal_freeref import summarize_table1b


def test_table1b_summary_builds_nine_number_rows(tmp_path: Path, monkeypatch) -> None:
    args: list[str] = [
        "summarize_table1b",
        "--method",
        "Example",
        "--output-dir",
        str(tmp_path / "out"),
        "--expected-baseline",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
    ]
    for index, split in enumerate(summarize_table1b.SPLITS):
        summary = tmp_path / f"{index}.json"
        summary.write_text(
            json.dumps({"coarse_cIoU": 0.5 + index / 100, "refined_cIoU": 0.51 + index / 100})
        )
        args.extend(["--summary", f"{split}={summary}"])
    monkeypatch.setattr(sys, "argv", args)
    assert summarize_table1b.main() == 0
    report = json.loads((tmp_path / "out" / "table1b_row.json").read_text())
    assert len(report["baseline_cIoU_percent"]) == 9
    assert len(report["freeref_cIoU_percent"]) == 9
    assert report["baseline_cIoU_percent"][-1] == 53.5
    assert report["freeref_cIoU_percent"][-1] == 54.5
    assert report["baseline_gate_passed"] is True
    assert report["eligibility"] == "paper_candidate"


def test_table1b_failed_baseline_gate_downgrades_candidate(tmp_path: Path, monkeypatch) -> None:
    args: list[str] = [
        "summarize_table1b",
        "--method",
        "Mismatch",
        "--output-dir",
        str(tmp_path / "out"),
        "--expected-baseline",
        *(["80"] * 8),
    ]
    for index, split in enumerate(summarize_table1b.SPLITS):
        summary = tmp_path / f"{index}.json"
        summary.write_text(json.dumps({"coarse_cIoU": 0.5, "refined_cIoU": 0.55}))
        args.extend(["--summary", f"{split}={summary}"])
    monkeypatch.setattr(sys, "argv", args)
    assert summarize_table1b.main() == 0
    report = json.loads((tmp_path / "out" / "table1b_row.json").read_text())
    assert report["baseline_gate_passed"] is False
    assert report["eligibility_requested"] == "paper_candidate"
    assert report["eligibility"] == "diagnostic_only"
