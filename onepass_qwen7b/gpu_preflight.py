from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate two CUDA devices and NCCL before training.")
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--probe-mb", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        if free_gb < args.min_free_gb:
            raise RuntimeError(
                f"GPU {local_rank} has only {free_gb:.1f}GB free; "
                f"at least {args.min_free_gb:.1f}GB is required."
            )
        probe = torch.empty(args.probe_mb * 1024**2, dtype=torch.uint8, device=local_rank)
        probe.fill_(local_rank + 1)
        torch.cuda.synchronize(local_rank)
        del probe
        value = torch.tensor([local_rank + 1.0], device=local_rank)
        dist.all_reduce(value)
        if float(value.item()) != 3.0:
            raise RuntimeError(f"Unexpected two-rank NCCL all-reduce result: {value.item()}.")
        print(
            f"GPU preflight rank={local_rank} name={torch.cuda.get_device_name(local_rank)!r} "
            f"free={free_gb:.1f}GB total={total_gb:.1f}GB nccl_sum={value.item():.1f}",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
