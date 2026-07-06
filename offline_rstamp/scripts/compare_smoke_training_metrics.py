#!/usr/bin/env python3
"""Compare baseline vs R-STAMP smoke training metrics.

This script reads HuggingFace Trainer `trainer_state.json` files and writes a
small comparison report. These are training metrics, not validation IoU metrics.
Use them to verify that both runs completed and to inspect early optimization
signals.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG")
RUNS = ["smoke_baseline_1x48g", "smoke_rstamp_1x48g"]


def find_state_file(root: Path, run_name: str) -> Path | None:
    files = sorted(glob.glob(str(root / "outputs" / run_name / "**" / "trainer_state.json"), recursive=True))
    if not files:
        return None
    # Prefer the last checkpoint by path order; smoke runs normally have one.
    return Path(files[-1])


def load_losses(state_file: Path) -> list[dict[str, Any]]:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return [x for x in data.get("log_history", []) if "loss" in x]


def pct_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first == 0:
        return None
    return (last - first) / first * 100.0


def get_metric(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def summarize_run(root: Path, run_name: str) -> dict[str, Any]:
    state_file = find_state_file(root, run_name)
    out: dict[str, Any] = {
        "run": run_name,
        "state_file": str(state_file) if state_file else None,
        "num_loss_logs": 0,
        "final_model_exists": (root / "outputs" / run_name / "final_model" / "model.safetensors").exists(),
    }
    if not state_file:
        return out
    losses = load_losses(state_file)
    out["num_loss_logs"] = len(losses)
    out["first"] = losses[0] if losses else None
    out["last"] = losses[-1] if losses else None
    for key in ["loss", "loss_lm", "loss_seg", "mean_token_accuracy", "entropy", "grad_norm"]:
        first = get_metric(out["first"], key)
        last = get_metric(out["last"], key)
        out[f"first_{key}"] = first
        out[f"last_{key}"] = last
        out[f"{key}_pct_change"] = pct_change(first, last)
    return out


def compare(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b = summary.get("smoke_baseline_1x48g", {})
    r = summary.get("smoke_rstamp_1x48g", {})
    out: dict[str, Any] = {}
    for key in ["loss", "loss_lm", "loss_seg", "mean_token_accuracy", "entropy", "grad_norm"]:
        b_last = b.get(f"last_{key}")
        r_last = r.get(f"last_{key}")
        if b_last is None or r_last is None:
            continue
        out[f"last_{key}_baseline"] = b_last
        out[f"last_{key}_rstamp"] = r_last
        out[f"last_{key}_delta_rstamp_minus_baseline"] = r_last - b_last
        out[f"last_{key}_relative_delta_pct"] = ((r_last - b_last) / b_last * 100.0) if b_last != 0 else None
    return out


def write_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    fields = [
        "run",
        "state_file",
        "num_loss_logs",
        "final_model_exists",
        "first_loss",
        "last_loss",
        "loss_pct_change",
        "first_loss_lm",
        "last_loss_lm",
        "loss_lm_pct_change",
        "first_loss_seg",
        "last_loss_seg",
        "loss_seg_pct_change",
        "first_mean_token_accuracy",
        "last_mean_token_accuracy",
        "mean_token_accuracy_pct_change",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in RUNS:
            row = summary.get(run, {})
            writer.writerow({k: row.get(k) for k in fields})


def write_markdown(path: Path, summary: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)

    lines = [
        "# Smoke Training Metrics Comparison",
        "",
        "> These are training metrics from 50-sample smoke runs. They are not validation IoU/cIoU/gIoU.",
        "",
        "| Metric | Baseline last | R-STAMP last | Delta (R-B) | Relative delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["loss", "loss_lm", "loss_seg", "mean_token_accuracy", "entropy", "grad_norm"]:
        b = comparison.get(f"last_{key}_baseline")
        r = comparison.get(f"last_{key}_rstamp")
        d = comparison.get(f"last_{key}_delta_rstamp_minus_baseline")
        p = comparison.get(f"last_{key}_relative_delta_pct")
        lines.append(f"| {key} | {fmt(b)} | {fmt(r)} | {fmt(d)} | {fmt(p)}% |")

    lines.extend(["", "## Per-run first/last", ""])
    for run in RUNS:
        row = summary.get(run, {})
        lines.extend([
            f"### {run}",
            "",
            f"- state_file: `{row.get('state_file')}`",
            f"- final_model_exists: `{row.get('final_model_exists')}`",
            f"- num_loss_logs: `{row.get('num_loss_logs')}`",
            f"- first loss: `{fmt(row.get('first_loss'))}`",
            f"- last loss: `{fmt(row.get('last_loss'))}`",
            f"- first loss_seg: `{fmt(row.get('first_loss_seg'))}`",
            f"- last loss_seg: `{fmt(row.get('last_loss_seg'))}`",
            f"- first mean_token_accuracy: `{fmt(row.get('first_mean_token_accuracy'))}`",
            f"- last mean_token_accuracy: `{fmt(row.get('last_mean_token_accuracy'))}`",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or root / "outputs").expanduser().resolve()

    summary = {run: summarize_run(root, run) for run in RUNS}
    comparison = compare(summary)
    report = {"root": str(root), "runs": summary, "comparison": comparison}

    json_path = output_dir / "smoke_training_metrics_comparison.json"
    csv_path = output_dir / "smoke_training_metrics_comparison.csv"
    md_path = output_dir / "smoke_training_metrics_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, summary)
    write_markdown(md_path, summary, comparison)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

