"""torchrun DDP entry point for the 4090 data plane.

One process per GPU. Only rank 0 writes checkpoints and metrics.
Benchmark mode measures the training pipeline on the synthetic fixture and
labels results explicitly as pipeline benchmark, not model results.

No MUJOCO_EGL_DEVICE_ID here: training never touches EGL rendering, and
CUDA_VISIBLE_DEVICES renumbers ranks from 0.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from embodiedflow.dataplane.config import DatasetConfig, ModelConfig, TrainConfig, save_json
from embodiedflow.dataplane.dataset import CanonicalEpisodeDataset, make_dataloader, split_episodes
from embodiedflow.dataplane.model import ChunkedActionTransformerMinimal, masked_chunk_mse
from embodiedflow.dataplane.stats import compute_stats

ARTIFACTS_ROOT = Path(__file__).resolve().parents[3] / "artifacts"
REPORTS_ROOT = Path(__file__).resolve().parents[3] / "reports"


def _setup_distributed() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _seed(seed: int, rank: int) -> None:
    import random

    import numpy as np

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def build_distributed_datasets(
    fixture_dir: Path,
    dataset_cfg: DatasetConfig,
    rank: int,
    world_size: int,
):
    episode_dirs = sorted(
        p for p in Path(fixture_dir).iterdir() if (p / "episode.json").is_file()
    )
    train_dirs, val_dirs = split_episodes(
        episode_dirs, dataset_cfg.val_ratio, dataset_cfg.split_seed
    )
    train_ds = CanonicalEpisodeDataset(train_dirs, dataset_cfg)
    norm = compute_stats(
        (train_ds[i] for i in range(len(train_ds))),
        state_dim=train_ds.state_dim,
        action_dim=train_ds.action_dim,
        image_normalization=dataset_cfg.image_normalization,
    )
    train_ds.normalize = norm
    val_ds = CanonicalEpisodeDataset(val_dirs, dataset_cfg, normalize=norm)
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=dataset_cfg.deterministic_seed if dataset_cfg.deterministic_seed is not None else 0,
        drop_last=True,
    )
    train_loader = make_dataloader(
        train_ds,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=dataset_cfg.persistent_workers,
        prefetch_factor=dataset_cfg.prefetch_factor,
        seed=dataset_cfg.deterministic_seed,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=dataset_cfg.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    return train_ds, val_ds, norm, train_sampler, train_loader, val_loader


def run_benchmark(
    model: torch.nn.Module,
    loader,
    sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    amp: bool,
    grad_clip_norm: float,
    profile: bool = False,
) -> dict:
    step_times: list[float] = []
    data_waits: list[float] = []
    losses: list[float] = []
    utils: list[float] = []

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=min(6, measure_steps)),
        )
        profiler.start()

    def one_step(batch: dict) -> float:
        rgb = batch["rgb"].to(device, non_blocking=True)
        state = batch["robot_state"].to(device, non_blocking=True)
        target = batch["action_chunk"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            prediction = model(rgb, state, mask)
            loss = masked_chunk_mse(prediction, target, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        return float(loss)

    loader_iter = iter(loader)
    total = warmup_steps + measure_steps

    def next_batch():
        nonlocal loader_iter
        while True:
            try:
                return next(loader_iter)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                loader_iter = iter(loader)

    for step in range(total):
        t0 = time.perf_counter()
        batch = next_batch()
        t_data = time.perf_counter() - t0
        loss = one_step(batch)
        t_end = time.perf_counter()
        if step >= warmup_steps:
            step_times.append((t_end - t0) * 1e3)
            data_waits.append(t_data * 1e3)
            losses.append(loss)
            utils.append(float(torch.cuda.utilization(device)))
        if profiler is not None:
            profiler.step()

    profile_summary = None
    if profiler is not None:
        profiler.stop()
        profile_summary = _profile_summary(profiler)

    elapsed = sum(step_times) / 1e3
    samples = measure_steps * loader.batch_size
    result = {
        "measure_steps": measure_steps,
        "warmup_steps": warmup_steps,
        "batch_size_per_rank": loader.batch_size,
        "samples_per_s": samples / elapsed if elapsed > 0 else 0.0,
        "steps_per_s": measure_steps / elapsed if elapsed > 0 else 0.0,
        "step_time_ms_mean": sum(step_times) / len(step_times),
        "step_time_ms_std": _std(step_times),
        "data_wait_ms_mean": sum(data_waits) / len(data_waits),
        "data_wait_ms_std": _std(data_waits),
        "loss_mean": sum(losses) / len(losses),
        "gpu_utilization_mean": sum(utils) / len(utils),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6,
    }
    if profile_summary is not None:
        result["profiler"] = profile_summary
    return result


def _profile_summary(profiler) -> dict:
    """CUDA time split: nccl kernels (communication) vs everything else
    (compute). Data-loading share is derived from wall-clock timing."""
    nccl_ms = 0.0
    other_ms = 0.0
    for event in profiler.key_averages():
        cuda_ms = sum(event.self_device_time_total) / 1e3
        if not cuda_ms:
            continue
        if "nccl" in event.key.lower():
            nccl_ms += cuda_ms
        else:
            other_ms += cuda_ms
    total = nccl_ms + other_ms
    return {
        "total_cuda_ms": round(total, 3),
        "nccl_kernel_ms": round(nccl_ms, 3),
        "other_kernel_ms": round(other_ms, 3),
        "comm_fraction": round(nccl_ms / total, 4) if total else None,
        "compute_fraction": round(other_ms / total, 4) if total else None,
    }


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/overfit"))
    parser.add_argument("--mode", choices=["train", "benchmark"], default="benchmark")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpus-physical", type=str, default="",
                        help="comma-separated physical GPU ids, e.g. 2,4,6")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="adaptive-avg-pool CNN features to pool_size x pool_size image tokens")
    parser.add_argument("--profile", action="store_true",
                        help="run torch.profiler over measured steps; adds nccl/compute CUDA time split")
    parser.add_argument("--label", type=str, default="pipeline-benchmark")
    parser.add_argument("--report-root", type=Path, default=REPORTS_ROOT,
                        help="parent dir of <label>/ddp-world-<N>.json")
    args = parser.parse_args()

    rank, world_size, local_rank = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    _seed(args.seed, rank)

    physical_ids = (
        [int(x) for x in args.gpus_physical.split(",") if x.strip()]
        if args.gpus_physical
        else [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    )
    physical_gpu = physical_ids[local_rank] if len(physical_ids) == world_size else None

    dataset_cfg = DatasetConfig(deterministic_seed=args.seed, num_workers=args.num_workers)
    if args.batch_size is not None:
        dataset_cfg.batch_size = args.batch_size
    if args.chunk_len is not None:
        dataset_cfg.chunk_len = args.chunk_len
    model_cfg = ModelConfig()
    if args.chunk_len is not None:
        model_cfg.chunk_len = args.chunk_len
    if args.pool_size is not None:
        model_cfg.pool_size = args.pool_size
    train_cfg = TrainConfig()
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.amp is not None:
        train_cfg.amp = args.amp

    train_ds, val_ds, norm, sampler, train_loader, val_loader = build_distributed_datasets(
        args.fixture, dataset_cfg, rank, world_size
    )
    model_cfg.state_dim = train_ds.state_dim
    model_cfg.action_dim = train_ds.action_dim
    # image shape follows the data; probes the first sample so the stem's
    # token count matches (e.g. 256x256 real episodes).
    model_cfg.rgb_shape = tuple(int(d) for d in train_ds[0]["rgb"].shape)
    model = ChunkedActionTransformerMinimal(model_cfg).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp)

    result = run_benchmark(
        model, train_loader, sampler, optimizer, scaler, device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps if args.mode == "benchmark" else args.steps,
        amp=train_cfg.amp,
        grad_clip_norm=train_cfg.grad_clip_norm,
        profile=args.profile,
    )
    result.update(
        {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "physical_gpu": physical_gpu,
            "mode": args.mode,
            "benchmark_label": args.label,
            "fixture": str(args.fixture),
            "model_type": model_cfg.model_type,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "amp": train_cfg.amp,
            "chunk_len": dataset_cfg.chunk_len,
            "rgb_shape": list(model_cfg.rgb_shape),
            "pool_size": model_cfg.pool_size,
            "steps": args.steps,
        }
    )
    if "profiler" in result and "data_wait_ms_mean" in result:
        result["profiler"]["data_fraction"] = round(
            result["data_wait_ms_mean"] / max(result["step_time_ms_mean"], 1e-9), 4
        )
    if rank == 0:
        os.makedirs(args.report_root / args.label, exist_ok=True)
    dist.barrier()
    gathered = [None] * world_size
    dist.gather_object(result, gathered if rank == 0 else None, dst=0)
    if rank == 0:
        out_path = args.report_root / args.label / f"ddp-world-{world_size}.json"
        summary = {
            "label": args.label,
            "world_size": world_size,
            "per_rank": gathered,
            "note": (
                "pipeline benchmark on synthetic fixture; NOT final model results"
                if str(args.fixture) == "fixtures/overfit"
                else "pipeline benchmark"
            ),
            "physical_gpus": physical_ids,
            "no_p2p": True,
        }
        save_json(summary, out_path)
        print(json.dumps(summary, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
