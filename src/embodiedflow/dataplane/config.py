"""Config dataclasses for the 4090 data plane.

Honesty rule: every capability must be declared here and exported into
config.yaml / model_manifest.json. No silent assumptions.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    chunk_len: int = 20
    window_stride: int = 1
    validate_before_load: bool = True
    skip_invalid_episodes: bool = True
    val_ratio: float = 0.2
    split_seed: int = 0
    deterministic_seed: int | None = None
    enforce_gripper_range: bool = False
    gripper_range_tol: float = 1e-3
    image_normalization: str = "uint8_to_float32_divide_255"
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    batch_size: int = 8
    drop_last: bool = True


@dataclass
class ModelConfig:
    model_type: str = "chunked_action_transformer_minimal_v0"
    # honest declaration: no language conditioning in this version
    instruction_conditioning: str = "disabled"
    rgb_shape: tuple[int, int, int] = (3, 32, 32)
    state_dim: int = 13
    action_dim: int = 7
    chunk_len: int = 20
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    cnn_stem: list[list[int]] = field(
        default_factory=lambda: [[16, 4, 2, 1], [32, 3, 2, 1], [64, 3, 2, 1]]
    )
    # adaptive avg pool of the CNN feature map to pool_size x pool_size image
    # tokens; required for large inputs (e.g. 256x256) to keep attention O(n^2)
    # tractable. None = no pooling (fixture 32x32).
    pool_size: int | None = None


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True
    seed: int = 0
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 200
    warmup_steps: int = 100
    scheduler: str = "cosine"
    run_id: str = "overfit-fixture-v0"


def to_dict(obj: Any) -> dict:
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def save_yaml(config: Any, path: Path) -> None:
    path.write_text(yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8")


def load_yaml(path: Path, cls: type) -> Any:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cls(**data)


def save_json(data: Any, path: Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
