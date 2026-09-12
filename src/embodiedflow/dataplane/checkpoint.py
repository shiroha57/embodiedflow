"""Checkpoint delivery protocol for the 5090 ActCheckpointPolicy.

Run directory layout (artifacts/checkpoints/<run_id>/):
  model.pt            torch.save dict, see SAVED_KEYS
  model_manifest.json machine-readable description (incl. checkpoint sha256)
  normalization.json  NormStats.to_dict()
  config.yaml         merged model / train / dataset config
  metrics.json        loss curves, peak memory, timing

The episode data format is the frozen contract; this checkpoint layout is
the data plane's delivery interface and is fully described by
model_manifest.json so the 5090 side never guesses.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from embodiedflow.dataplane.config import ModelConfig, TrainConfig, save_json, save_yaml, to_dict

if TYPE_CHECKING:
    # annotation-only: keeping stats/numpy out of the runtime import graph so
    # the 5090 load bundle needs only model.py + config.py
    from embodiedflow.dataplane.stats import NormStats

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_VERSION = "0.1.0"

SAVED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "global_step",
    "epoch",
    "model_config",
    "train_config",
    "normalization",
    "rng",
)


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def contract_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "contract-v0.1^{commit}"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_model_pt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    global_step: int,
    epoch: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    normalization: NormStats,
    rng: dict,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "model_config": to_dict(model_cfg),
        "train_config": to_dict(train_cfg),
        "normalization": normalization.to_dict(),
        "rng": rng,
    }


def build_manifest(
    run_id: str,
    model_cfg: ModelConfig,
    normalization: NormStats,
    observation_keys: list[str],
    robot_state_keys: list[str],
    model_pt_path: Path,
    extra: dict | None = None,
) -> dict:
    manifest: dict = {
        "run_id": run_id,
        "model_type": model_cfg.model_type,
        "instruction_conditioning": model_cfg.instruction_conditioning,
        "git_commit": git_commit(),
        "contract_version": CONTRACT_VERSION,
        "contract_commit": contract_commit(),
        "observation_keys": observation_keys,
        "robot_state_keys": robot_state_keys,
        "inputs": {
            "rgb": {
                "shape": list(model_cfg.rgb_shape),
                "dtype": "float32",
                "range": [0.0, 1.0],
                "order": "CHW",
            },
            "robot_state": {
                "shape": [model_cfg.state_dim],
                "dtype": "float32",
                "composition": "joint_position | joint_velocity | gripper_position",
            },
        },
        "action": {
            "shape": [model_cfg.chunk_len, model_cfg.action_dim],
            "dtype": "float32",
            "unit": "rad,m",
            "representation": "joint_position_targets_then_gripper_width",
            "chunk_len": model_cfg.chunk_len,
        },
        "image_preprocessing": normalization.image_normalization,
        "normalization": {
            "method": "z-score per dimension",
            "applied_to": ["robot_state", "action"],
            "eps": 1e-8,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoint_format": "torch.save dict with keys: " + ", ".join(SAVED_KEYS),
        "checkpoint_sha256": hashlib.sha256(model_pt_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if extra:
        manifest.update(extra)
    return manifest


def export_checkpoint(
    run_dir: Path,
    run_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    global_step: int,
    epoch: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    normalization: NormStats,
    metrics: dict,
    rng: dict,
    observation_keys: list[str],
    robot_state_keys: list[str],
    extra_manifest: dict | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_pt = run_dir / "model.pt"
    torch.save(
        build_model_pt(
            model, optimizer, scheduler, scaler, global_step, epoch,
            model_cfg, train_cfg, normalization, rng,
        ),
        model_pt,
    )
    normalization.to_json(run_dir / "normalization.json")
    merged_config = {
        "model": to_dict(model_cfg),
        "train": to_dict(train_cfg),
    }
    save_yaml(merged_config, run_dir / "config.yaml")
    save_json(metrics, run_dir / "metrics.json")
    manifest = build_manifest(
        run_id, model_cfg, normalization, observation_keys, robot_state_keys,
        model_pt, extra=extra_manifest,
    )
    save_json(manifest, run_dir / "model_manifest.json")
    return model_pt


def load_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict:
    """Restore everything recorded in SAVED_KEYS; returns the checkpoint dict."""
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def capture_rng(dataloader=None, batches_consumed: int = 0) -> dict:
    """Capture all rng state. The dataloader is captured as (seed,
    batches_consumed) rather than raw generator state: RandomSampler
    materializes a fresh permutation on each iter(), so a raw state restore
    would resume with a different batch stream."""
    rng: dict = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": None,
        "python": None,
        "dataloader": None,
    }
    import numpy as np
    import random

    rng["numpy"] = np.random.get_state()
    rng["python"] = random.getstate()
    if dataloader is not None and dataloader.generator is not None:
        rng["dataloader"] = {
            "seed": dataloader.generator.initial_seed(),
            "batches_consumed": batches_consumed,
        }
    return rng


def restore_rng(rng: dict, dataloader=None) -> None:
    import numpy as np
    import random

    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if dataloader is not None and rng.get("dataloader") is not None:
        dataloader.generator.manual_seed(rng["dataloader"]["seed"])
