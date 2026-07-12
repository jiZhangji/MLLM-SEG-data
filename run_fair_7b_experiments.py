from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially train fair native STAMP-7B and OnePass-Qwen2-VL-7B."
    )
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--train-json", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1, help="Fallback per-GPU batch size.")
    parser.add_argument("--onepass-batch-size", type=int, help="Per-GPU OnePass micro batch size.")
    parser.add_argument("--stamp-batch-size", type=int, help="Per-GPU native STAMP micro batch size.")
    parser.add_argument("--target-global-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default for high-memory H200 runs.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume-stamp", type=Path)
    parser.add_argument("--resume-onepass", type=Path)
    parser.add_argument("--skip-stamp", action="store_true")
    parser.add_argument("--skip-onepass", action="store_true")
    return parser.parse_args()


def launch(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== Launching ===", flush=True)
    print(" ".join(command), flush=True)
    print(f"log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> int:
    args = parse_args()
    onepass_batch_size = args.onepass_batch_size or args.batch_size
    stamp_batch_size = args.stamp_batch_size or args.batch_size
    if min(
        args.nproc_per_node,
        onepass_batch_size,
        stamp_batch_size,
        args.target_global_batch_size,
    ) <= 0:
        raise ValueError("Process count, batch sizes and target global batch size must be positive.")

    def gradient_accumulation(batch_size: int, name: str) -> int:
        micro_global = args.nproc_per_node * batch_size
        if args.target_global_batch_size % micro_global:
            raise ValueError(
                f"target-global-batch-size={args.target_global_batch_size} must be divisible by "
                f"{name} micro global batch={micro_global}."
            )
        return args.target_global_batch_size // micro_global

    onepass_grad_accum = gradient_accumulation(onepass_batch_size, "OnePass")
    stamp_grad_accum = gradient_accumulation(stamp_batch_size, "STAMP")
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
    ]
    common = [
        "--stamp-code-dir", str(args.stamp_code_dir),
        "--base-model", str(args.base_model),
        "--train-json", *[str(path) for path in args.train_json],
        "--epochs", str(args.epochs),
        "--num-workers", str(args.num_workers),
        "--attn-implementation", str(args.attn_implementation),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--min-pixels", str(args.min_pixels),
        "--max-pixels", str(args.max_pixels),
        "--save-steps", str(args.save_steps),
        "--logging-steps", str(args.logging_steps),
        "--limit", str(args.limit),
    ]
    if args.data_root is not None:
        common += ["--data-root", str(args.data_root)]
    common.append(
        "--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing"
    )

    def training_args(output: Path, batch_size: int, grad_accum: int) -> list[str]:
        return [
            *common,
            "--output-dir", str(output),
            "--batch-size", str(batch_size),
            "--gradient-accumulation", str(grad_accum),
        ]

    manifest = {
        "base_model": str(args.base_model),
        "train_json": [str(path) for path in args.train_json],
        "epochs": args.epochs,
        "execution_order": ["onepass7b", "stamp7b_native"],
        "onepass_per_gpu_batch_size": onepass_batch_size,
        "stamp_per_gpu_batch_size": stamp_batch_size,
        "nproc_per_node": args.nproc_per_node,
        "onepass_gradient_accumulation": onepass_grad_accum,
        "stamp_gradient_accumulation": stamp_grad_accum,
        "effective_global_batch_size": args.target_global_batch_size,
        "learning_rate": args.learning_rate,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "use_rslora": True,
        "initialization": "plain_qwen2_vl_7b_for_both_models",
        "stamp_7b": "autoregressive SEG followed by a MASK-query classification pass",
        "onepass_7b": "preinserted SEG and spatial MASK queries in one hybrid-attention pass",
    }
    (args.output_root / "fair_experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    if not args.skip_onepass:
        output = args.output_root / "onepass7b"
        onepass_args = training_args(output, onepass_batch_size, onepass_grad_accum)
        onepass_args += [
            "--use-rslora",
            "--query-learning-rate", str(args.learning_rate),
            "--head-learning-rate", str(args.learning_rate),
            "--lora-learning-rate", str(args.learning_rate),
        ]
        if args.resume_onepass is not None:
            onepass_args += ["--resume", str(args.resume_onepass)]
        launch(launcher + ["-m", "onepass_qwen7b.train", *onepass_args], output / "train.log", environment)
    if not args.skip_stamp:
        output = args.output_root / "stamp7b_native"
        stamp_args = training_args(output, stamp_batch_size, stamp_grad_accum)
        stamp_args += [
            "--use-rslora",
            "--token-learning-rate", str(args.learning_rate),
            "--head-learning-rate", str(args.learning_rate),
            "--lora-learning-rate", str(args.learning_rate),
        ]
        if args.resume_stamp is not None:
            stamp_args += ["--resume", str(args.resume_stamp)]
        launch(launcher + ["-m", "fair_stamp7b.train", *stamp_args], output / "train.log", environment)
    print(f"\nBoth requested experiments completed under: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
