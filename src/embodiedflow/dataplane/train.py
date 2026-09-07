"""Single-GPU training entry point for the 4090 data plane.

Pipeline benchmark results (synthetic fixture) are explicitly labeled; they
are never final model results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from embodiedflow.dataplane.checkpoint import (
    capture_rng,
    export_checkpoint,
    load_checkpoint,
    restore_rng,
)
from embodiedflow.dataplane.config import DatasetConfig, ModelConfig, TrainConfig
from embodiedflow.dataplane.dataset import (
    CanonicalEpisodeDataset,
    make_dataloader,
    split_episodes,
)
from embodiedflow.dataplane.model import (
    ChunkedActionTransformerMinimal,
    cosine_schedule,
    masked_chunk_mse,
)
from embodiedflow.dataplane.stats import NormStats, compute_stats

ARTIFACTS_ROOT = Path(__file__).resolve().parents[3] / "artifacts"


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(
    fixture_dir: Path,
    dataset_cfg: DatasetConfig,
    val_dirs: list[Path] | None = None,
):
    episode_dirs = sorted(
        p for p in Path(fixture_dir).iterdir() if (p / "episode.json").is_file()
    )
    train_dirs, val_dirs_computed = split_episodes(
        episode_dirs, dataset_cfg.val_ratio, dataset_cfg.split_seed
    )
    if val_dirs is not None:
        val_dirs_computed = list(val_dirs)
    train_ds = CanonicalEpisodeDataset(train_dirs, dataset_cfg)
    norm = compute_stats(
        (train_ds[i] for i in range(len(train_ds))),
        state_dim=train_ds.state_dim,
        action_dim=train_ds.action_dim,
        image_normalization=dataset_cfg.image_normalization,
    )
    train_ds.normalize = norm
    val_ds = CanonicalEpisodeDataset(val_dirs_computed, dataset_cfg, normalize=norm)
    return train_ds, val_ds, norm, train_dirs, val_dirs_computed


def evaluate(model: torch.nn.Module, loader, device: torch.device, amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            state = batch["robot_state"].to(device, non_blocking=True)
            target = batch["action_chunk"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                prediction = model(rgb, state, mask)
                loss = masked_chunk_mse(prediction, target, mask)
            total_loss += float(loss) * rgb.shape[0]
            count += rgb.shape[0]
    model.train()
    return total_loss / max(1, count)


def train(
    fixture_dir: Path,
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    run_dir: Path | None = None,
    resume_dir: Path | None = None,
    benchmark_label: str | None = None,
) -> dict:
    seed_everything(train_cfg.seed)
    train_ds, val_ds, norm, train_dirs, val_dirs = build_datasets(fixture_dir, dataset_cfg)

    model_cfg.state_dim = train_ds.state_dim
    model_cfg.action_dim = train_ds.action_dim
    model = ChunkedActionTransformerMinimal(model_cfg).to(device)

    train_loader = make_dataloader(
        train_ds,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=dataset_cfg.persistent_workers,
        prefetch_factor=dataset_cfg.prefetch_factor,
        seed=dataset_cfg.deterministic_seed,
        drop_last=dataset_cfg.drop_last,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=dataset_cfg.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler_fn = cosine_schedule(train_cfg.warmup_steps, train_cfg.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp)

    start_step = 0
    resume_batches = 0
    if resume_dir is not None:
        checkpoint = load_checkpoint(
            resume_dir, model, optimizer, scheduler, scaler
        )
        start_step = int(checkpoint["global_step"])
        rng = checkpoint.get("rng") or {}
        resume_batches = int((rng.get("dataloader") or {}).get("batches_consumed", start_step))
        restore_rng(rng, dataloader=train_loader)

    metrics: dict = {
        "run_id": train_cfg.run_id,
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "step_time_ms": [],
        "data_wait_ms": [],
        "benchmark_label": benchmark_label,
        "fixture": str(fixture_dir),
        "train_episodes": [d.name for d in train_dirs],
        "val_episodes": [d.name for d in val_dirs],
    }

    steps_per_epoch = len(train_loader)
    loader_iter = iter(train_loader)
    # Replay-skip to the exact batch position of the resumed run so the
    # shuffle permutation stream continues identically.
    for _ in range(resume_batches):
        try:
            next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            next(loader_iter)
    for step in range(start_step, train_cfg.steps):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        t_data = time.perf_counter() - t0

        rgb = batch["rgb"].to(device, non_blocking=True)
        state = batch["robot_state"].to(device, non_blocking=True)
        target = batch["action_chunk"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=train_cfg.amp):
            prediction = model(rgb, state, mask)
            loss = masked_chunk_mse(prediction, target, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        t_step = time.perf_counter() - t0

        metrics["steps"].append(step)
        metrics["train_loss"].append(float(loss))
        metrics["lr"].append(scheduler.get_last_lr()[0])
        metrics["step_time_ms"].append(t_step * 1e3)
        metrics["data_wait_ms"].append(t_data * 1e3)

        if run_dir is not None and (step + 1) % train_cfg.save_every == 0:
            rng = capture_rng(dataloader=train_loader, batches_consumed=step + 1)
            export_checkpoint(
                run_dir / "checkpoints" / f"step-{step + 1:07d}",
                f"{train_cfg.run_id}-step-{step + 1}",
                model, optimizer, scheduler, scaler, step + 1, (step + 1) // steps_per_epoch,
                model_cfg, train_cfg, norm, dict(metrics), rng,
                observation_keys=["rgb", "joint_position", "joint_velocity", "gripper_position"],
                robot_state_keys=["joint_position", "joint_velocity", "gripper_position"],
            )
        if (step + 1) % train_cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, train_cfg.amp)
            metrics["val_loss"].append((step, float(val_loss)))
        else:
            metrics["val_loss"].append(None)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    metrics["peak_gpu_memory_mb"] = peak_mb
    metrics["num_parameters"] = model.num_parameters()
    metrics["num_trainable_parameters"] = model.num_trainable_parameters()

    if run_dir is not None:
        rng = capture_rng(dataloader=train_loader, batches_consumed=train_cfg.steps)
        export_checkpoint(
            run_dir,
            train_cfg.run_id,
            model, optimizer, scheduler, scaler, train_cfg.steps,
            train_cfg.steps // steps_per_epoch,
            model_cfg, train_cfg, norm, metrics, rng,
            observation_keys=["rgb", "joint_position", "joint_velocity", "gripper_position"],
            robot_state_keys=["joint_position", "joint_velocity", "gripper_position"],
            extra_manifest={"benchmark_label": benchmark_label},
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/overfit"))
    parser.add_argument("--gpu", type=str, default=None,
                        help="physical GPU id or 'auto' to pick a free one (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="run dir with model.pt")
    parser.add_argument("--benchmark-label", type=str, default=None)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    args = parser.parse_args()

    if args.gpu is not None:
        if args.gpu == "auto":
            from embodiedflow.dataplane.gpu import pick_free_gpus

            args.gpu = pick_free_gpus(1)[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_cfg = DatasetConfig()
    if args.batch_size is not None:
        dataset_cfg.batch_size = args.batch_size
    if args.chunk_len is not None:
        dataset_cfg.chunk_len = args.chunk_len
    if args.seed is not None:
        dataset_cfg.deterministic_seed = args.seed
    model_cfg = ModelConfig()
    if args.chunk_len is not None:
        model_cfg.chunk_len = args.chunk_len
    train_cfg = TrainConfig()
    if args.steps is not None:
        train_cfg.steps = args.steps
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.amp is not None:
        train_cfg.amp = args.amp
    if args.run_id is not None:
        train_cfg.run_id = args.run_id

    run_dir = args.artifacts_root / "checkpoints" / train_cfg.run_id if args.artifacts_root else None
    resume_dir = args.resume
    metrics = train(
        fixture_dir=args.fixture,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=device,
        run_dir=run_dir,
        resume_dir=resume_dir,
        benchmark_label=args.benchmark_label,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("steps", "train_loss")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
