from __future__ import annotations

import json
from pathlib import Path

from efficiency_benchmark.common import select_indices, write_report
from efficiency_benchmark.summarize import ROWS, main as summarize_main


def test_select_indices_is_paired_and_reproducible() -> None:
    first = select_indices(100, warmup=10, samples=20, seed=7)
    second = select_indices(100, warmup=10, samples=20, seed=7)
    assert first == second
    assert len(first[0]) == 10
    assert len(first[1]) == 20
    assert not set(first[0]) & set(first[1])


def test_write_report_computes_end_to_end_statistics(tmp_path: Path) -> None:
    summary = write_report(
        tmp_path,
        {"method": "test", "variant": "base"},
        [
            {"index": 0, "model_seconds": 1.0, "total_seconds": 1.0},
            {"index": 1, "model_seconds": 3.0, "total_seconds": 3.0},
        ],
        peak_gpu_gib=4.5,
        memory_backend="test",
    )
    assert summary["e2e_mean_seconds"] == 2.0
    assert summary["e2e_median_seconds"] == 2.0
    assert summary["fps"] == 0.5
    assert summary["peak_gpu_gib"] == 4.5


def test_summarize_requires_and_renders_all_rows(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    for directory, method, variant, *_ in ROWS:
        output = root / directory
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "variant": variant,
                    "samples": 500,
                    "warmup": 20,
                    "e2e_mean_seconds": 1.0,
                    "e2e_median_seconds": 0.9,
                    "e2e_p95_seconds": 1.2,
                    "fps": 1.0,
                    "peak_gpu_gib": 10.0,
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "table.md"
    monkeypatch.setattr(
        "sys.argv", ["summarize", "--input-root", str(root), "--output", str(output)]
    )
    assert summarize_main() == 0
    text = output.read_text(encoding="utf-8")
    assert "STAMP-7B | +FreeRef-GPU" in text
    assert "Text4Seg-p24 | Base" in text
    assert "LISA-7B | Original" in text
