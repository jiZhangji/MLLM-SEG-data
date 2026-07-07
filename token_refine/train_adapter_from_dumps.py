from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from token_refine.data import TokenDumpDataset, collate_token_dumps
from token_refine.metrics import binary_iou, logits_to_mask
from token_refine.model import MaskTokenRefinementAdapter

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for server environments.
    tqdm = None


def find_dump_paths(input_dir: Path, limit: int) -> list[Path]:
    paths = sorted(input_dir.glob("*.pt"))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .pt dumps found in {input_dir}")
    return paths


def split_paths(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def read_grid_hw(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(payload["grid_hw"])


def setup_distributed(device_arg: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() and device_arg != "cpu" else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available() and device_arg != "cpu":
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(device_arg)
    else:
        device = torch.device("cpu")
    return device, rank, local_rank, world_size


def is_rank0(rank: int) -> bool:
    return rank == 0


def maybe_barrier(world_size: int, device: torch.device | None = None) -> None:
    if world_size > 1:
        if device is not None and device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def build_grid_cache(paths: list[Path], cache_path: Path) -> dict[str, list[int]]:
    return {str(path.resolve()): list(read_grid_hw(path)) for path in paths}


def cache_ready_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".ready")


def wait_for_grid_cache(cache_path: Path, started_at: float, timeout_seconds: int = 7200) -> None:
    ready_path = cache_ready_path(cache_path)
    while True:
        if ready_path.exists() and cache_path.exists() and ready_path.stat().st_mtime >= started_at:
            return
        if time.time() - started_at > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for grid cache: {cache_path}")
        time.sleep(5)


def load_or_build_grid_cache(
    paths: list[Path],
    cache_path: Path,
    rank: int,
    world_size: int,
    started_at: float,
) -> dict[str, tuple[int, int]]:
    resolved_keys = {str(path.resolve()) for path in paths}
    if is_rank0(rank):
        rebuild = True
        ready_path = cache_ready_path(cache_path)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                grids = cached.get("grids", {})
                rebuild = not resolved_keys.issubset(set(grids))
            except Exception:
                rebuild = True
        if rebuild:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            grids = build_grid_cache(paths, cache_path)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps({"grids": grids}, indent=2), encoding="utf-8")
            tmp_path.replace(cache_path)
            ready_path.write_text("ready\n", encoding="utf-8")
            print(f"[INFO] Wrote grid cache: {cache_path}", flush=True)
        else:
            ready_path.write_text("ready\n", encoding="utf-8")
            print(f"[INFO] Using grid cache: {cache_path}", flush=True)
    elif world_size > 1:
        wait_for_grid_cache(cache_path, started_at)
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    return {key: tuple(value) for key, value in cached["grids"].items() if key in resolved_keys}


def grouped_batch_indices(
    paths: list[Path],
    batch_size: int,
    shuffle: bool,
    seed: int,
    grid_cache: dict[str, tuple[int, int]],
    rank: int = 0,
    world_size: int = 1,
) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, path in enumerate(paths):
        groups.setdefault(grid_cache[str(path.resolve())], []).append(index)

    rng = random.Random(seed)
    batches = []
    for indices in groups.values():
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    if shuffle:
        rng.shuffle(batches)
    if world_size > 1 and batches:
        target_len = math.ceil(len(batches) / world_size) * world_size
        while len(batches) < target_len:
            batches.append(list(batches[len(batches) % len(batches)]))
        batches = batches[rank::world_size]
    return batches


def token_loss(
    outputs: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
    uncertainty_loss_weight: float,
    delta_reg_weight: float,
) -> torch.Tensor:
    refined_logits = outputs["refined_logits"]
    uncertainty = outputs["uncertainty"].squeeze(-1).detach()
    ce = F.cross_entropy(
        refined_logits.reshape(-1, 2),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape_as(target_tokens).float()
    weights = 1.0 + uncertainty_loss_weight * uncertainty
    loss_ce = (ce * weights).mean()
    delta_reg = ((1.0 - uncertainty) * outputs["delta_logits"].pow(2).sum(dim=-1)).mean()
    return loss_ce + delta_reg_weight * delta_reg


def progress_iter(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    image_size: int,
    world_size: int = 1,
    show_progress: bool = False,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    totals = {"coarse_iou": 0.0, "refined_iou": 0.0, "token_acc": 0.0}
    count = 0
    iterator = progress_iter(loader, show_progress, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
    for batch in iterator:
        mask_logits = batch["mask_logits"].to(device)
        mask_hidden = batch["mask_hidden"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        gt_mask = batch["gt_mask"].to(device)
        grid_hw = batch["grid_hw"]
        outputs = model(mask_hidden, mask_logits)

        coarse_mask = logits_to_mask(mask_logits, grid_hw, image_size)
        refined_mask = logits_to_mask(outputs["refined_logits"], grid_hw, image_size)
        coarse_iou = binary_iou(coarse_mask >= 0.5, gt_mask >= 0.5).mean()
        refined_iou = binary_iou(refined_mask >= 0.5, gt_mask >= 0.5).mean()
        token_acc = (outputs["refined_logits"].argmax(dim=-1) == target_tokens).float().mean()

        batch_size = mask_logits.shape[0]
        totals["coarse_iou"] += float(coarse_iou.item()) * batch_size
        totals["refined_iou"] += float(refined_iou.item()) * batch_size
        totals["token_acc"] += float(token_acc.item()) * batch_size
        count += batch_size
        if show_progress and tqdm is not None:
            iterator.set_postfix(refined_iou=f"{totals['refined_iou'] / max(count, 1):.4f}")
    if world_size > 1:
        packed = torch.tensor(
            [totals["coarse_iou"], totals["refined_iou"], totals["token_acc"], float(count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        totals = {
            "coarse_iou": float(packed[0].item()),
            "refined_iou": float(packed[1].item()),
            "token_acc": float(packed[2].item()),
        }
        count = int(packed[3].item())
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=2.0)
    parser.add_argument("--delta-reg-weight", type=float, default=0.01)
    parser.add_argument("--use-uncertainty-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-cache", type=Path, default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    started_at = time.time()
    paths = find_dump_paths(args.input_dir, args.limit)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    sample = torch.load(paths[0], map_location="cpu", weights_only=False)
    token_dim = int(sample["mask_hidden"].shape[-1])

    device, rank, _local_rank, world_size = setup_distributed(args.device)
    cache_path = args.grid_cache or (args.input_dir / "token_refine_grid_cache.json")
    grid_cache = load_or_build_grid_cache(paths, cache_path, rank, world_size, started_at)
    print(f"[INFO] rank={rank} world_size={world_size} device={device}", flush=True)
    if args.build_cache_only:
        if is_rank0(rank):
            print(f"[INFO] Built grid cache for {len(grid_cache)} dumps: {cache_path}", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return 0

    model = MaskTokenRefinementAdapter(
        token_dim=token_dim,
        hidden_size=args.hidden_size,
        use_uncertainty_gate=args.use_uncertainty_gate,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TokenDumpDataset(train_paths, args.image_size),
        batch_sampler=grouped_batch_indices(train_paths, args.batch_size, True, args.seed, grid_cache, rank, world_size),
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TokenDumpDataset(val_paths or train_paths, args.image_size),
        batch_sampler=grouped_batch_indices(val_paths or train_paths, args.batch_size, False, args.seed, grid_cache, rank, world_size),
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )

    if is_rank0(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] rank={rank} world_size={world_size} train_batches={len(train_loader)} val_batches={len(val_loader)}",
            flush=True,
        )
    maybe_barrier(world_size, device)
    best_refined_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        show_train_progress = args.progress and is_rank0(rank)
        train_iter = progress_iter(
            train_loader,
            show_train_progress,
            desc=f"epoch {epoch}/{args.epochs} train",
            total=len(train_loader),
            dynamic_ncols=True,
            leave=True,
        )
        for batch in train_iter:
            mask_logits = batch["mask_logits"].to(device)
            mask_hidden = batch["mask_hidden"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            outputs = model(mask_hidden, mask_logits)
            loss = token_loss(outputs, target_tokens, args.uncertainty_loss_weight, args.delta_reg_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * mask_logits.shape[0]
            total_count += mask_logits.shape[0]
            if show_train_progress and tqdm is not None:
                train_iter.set_postfix(loss=f"{total_loss / max(total_count, 1):.5f}")

        if world_size > 1:
            packed_loss = torch.tensor([total_loss, float(total_count)], device=device, dtype=torch.float64)
            dist.all_reduce(packed_loss, op=dist.ReduceOp.SUM)
            total_loss = float(packed_loss[0].item())
            total_count = int(packed_loss[1].item())

        eval_model = model.module if hasattr(model, "module") else model
        val_metrics = evaluate(
            eval_model,
            val_loader,
            device,
            args.image_size,
            world_size,
            show_progress=args.progress and is_rank0(rank),
            desc=f"epoch {epoch}/{args.epochs} val",
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val": val_metrics,
            "val_delta": val_metrics["refined_iou"] - val_metrics["coarse_iou"],
        }
        if is_rank0(rank):
            history.append(row)
            print(json.dumps(row), flush=True)
        if is_rank0(rank) and val_metrics["refined_iou"] > best_refined_iou:
            best_refined_iou = val_metrics["refined_iou"]
            safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save(
                {
                    "model": eval_model.state_dict(),
                    "config": safe_config,
                    "token_dim": token_dim,
                    "history": history,
                },
                args.output_dir / "adapter.pt",
            )
        maybe_barrier(world_size, device)

    if is_rank0(rank):
        report = {
            "train_count": len(train_paths),
            "val_count": len(val_paths),
            "best_refined_iou": best_refined_iou,
            "history": history,
            "world_size": world_size,
        }
        (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
