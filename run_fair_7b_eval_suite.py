from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OnePass and native STAMP on RefCOCOg val/test.")
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--val-json", required=True, type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--onepass-checkpoint", required=True, type=Path)
    parser.add_argument("--stamp-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--onepass-batch-size", type=int, default=8)
    parser.add_argument("--stamp-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_pair(tasks: list[dict[str, Any]], skip_existing: bool) -> None:
    running: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    for task in tasks:
        summary = task["output_dir"] / "eval_summary.json"
        if skip_existing and summary.exists():
            print(f"[skip] {task['name']}: {summary}", flush=True)
            continue
        task["output_dir"].mkdir(parents=True, exist_ok=True)
        task["log_path"].parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = task["gpu"]
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = task["log_path"].open("a", encoding="utf-8")
        print(f"[start] {task['name']} on GPU {task['gpu']}", flush=True)
        print(" ".join(task["command"]), flush=True)
        process = subprocess.Popen(
            task["command"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        running.append((task, process, log))
    failures = []
    for task, process, log in running:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append((task["name"], return_code, task["log_path"]))
        else:
            print(f"[done] {task['name']}", flush=True)
    if failures:
        details = ", ".join(f"{name}: exit={code}, log={log}" for name, code, log in failures)
        raise RuntimeError(f"Evaluation task failed: {details}")


def comparison(output_root: Path) -> None:
    rows = []
    for method, directory in (("OnePass", "onepass7b_eval"), ("STAMP", "stamp7b_eval")):
        for split in ("val", "test"):
            path = output_root / f"{directory}_{split}" / "eval_summary.json"
            if not path.exists():
                continue
            values = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "samples": values.get("samples"),
                    "gIoU": values.get("gIoU", values.get("mean_iou")),
                    "cIoU": values.get("cIoU"),
                    "seconds_per_sample": values.get("end_to_end_seconds_per_sample"),
                    "no_seg_rate": values.get("no_seg_rate", 0.0),
                }
            )
    (output_root / "fair_7b_eval_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    lines = [
        "| Method | Split | Samples | gIoU | cIoU | sec/sample | no-SEG rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['split']} | {row['samples']} | "
            f"{float(row['gIoU']):.6f} | {float(row['cIoU']):.6f} | "
            f"{float(row['seconds_per_sample']):.6f} | {float(row['no_seg_rate']):.6f} |"
        )
    (output_root / "fair_7b_eval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2:
        raise ValueError("--gpus must contain exactly two GPU IDs, for example 0,1.")
    if min(args.onepass_batch_size, args.stamp_batch_size, args.num_workers, args.max_new_tokens) <= 0:
        raise ValueError("Batch sizes, workers and max-new-tokens must be positive.")
    for checkpoint in (args.onepass_checkpoint, args.stamp_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    common = [
        "--stamp-code-dir", str(args.stamp_code_dir),
        "--base-model", str(args.base_model),
        "--data-root", str(args.data_root) if args.data_root is not None else ".",
        "--num-workers", str(args.num_workers),
        "--device", "cuda",
        "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
        "--min-pixels", str(args.min_pixels),
        "--max-pixels", str(args.max_pixels),
        "--limit", str(args.limit),
    ]
    splits = (("val", args.val_json, gpus[0]), ("test", args.test_json, gpus[1]))
    onepass_tasks = []
    stamp_tasks = []
    for split, json_path, gpu in splits:
        onepass_output = args.output_root / f"onepass7b_eval_{split}"
        onepass_tasks.append(
            {
                "name": f"OnePass-{split}",
                "gpu": gpu,
                "output_dir": onepass_output,
                "log_path": args.output_root / f"onepass7b_eval_{split}.log",
                "command": [
                    sys.executable, "-m", "onepass_qwen7b.eval", *common,
                    "--eval-json", str(json_path),
                    "--checkpoint", str(args.onepass_checkpoint),
                    "--output-dir", str(onepass_output),
                    "--batch-size", str(args.onepass_batch_size),
                ],
            }
        )
        stamp_output = args.output_root / f"stamp7b_eval_{split}"
        stamp_tasks.append(
            {
                "name": f"STAMP-{split}",
                "gpu": gpu,
                "output_dir": stamp_output,
                "log_path": args.output_root / f"stamp7b_eval_{split}.log",
                "command": [
                    sys.executable, "-m", "fair_stamp7b.eval", *common,
                    "--eval-json", str(json_path),
                    "--checkpoint", str(args.stamp_checkpoint),
                    "--output-dir", str(stamp_output),
                    "--batch-size", str(args.stamp_batch_size),
                    "--max-new-tokens", str(args.max_new_tokens),
                ],
            }
        )
    print("=== Phase 1: OnePass val/test in parallel ===", flush=True)
    run_pair(onepass_tasks, args.skip_existing)
    print("=== Phase 2: native STAMP val/test in parallel ===", flush=True)
    run_pair(stamp_tasks, args.skip_existing)
    comparison(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
