"""Checkpoint export protocol tests: file set, manifest fields, sha256,
load-and-forward compatibility."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml

from embodiedflow.dataplane.checkpoint import (
    REPO_ROOT,
    contract_commit,
    export_checkpoint,
    load_checkpoint,
)
from embodiedflow.dataplane.config import ModelConfig, TrainConfig
from embodiedflow.dataplane.model import ChunkedActionTransformerMinimal
from embodiedflow.dataplane.stats import NormStats

import numpy as np

REQUIRED_FILES = ("model.pt", "model_manifest.json", "normalization.json", "config.yaml", "metrics.json")


def _stats() -> NormStats:
    return NormStats(
        robot_state_mean=np.zeros(13, dtype=np.float32),
        robot_state_std=np.ones(13, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
        num_samples=10,
    )


def test_export_file_set_and_manifest(tmp_path: Path):
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    model = ChunkedActionTransformerMinimal(model_cfg)
    metrics = {"run_id": train_cfg.run_id, "steps": [0], "train_loss": [1.0]}
    export_checkpoint(
        tmp_path, "untrained-v0", model, None, None, None, 0, 0,
        model_cfg, train_cfg, _stats(), metrics, {"torch": None, "cuda": None},
        observation_keys=["rgb", "joint_position", "joint_velocity", "gripper_position"],
        robot_state_keys=["joint_position", "joint_velocity", "gripper_position"],
    )
    for name in REQUIRED_FILES:
        assert (tmp_path / name).is_file(), f"missing {name}"

    manifest = json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_type"] == "chunked_action_transformer_minimal_v0"
    assert manifest["instruction_conditioning"] == "disabled"
    assert manifest["contract_version"] == "0.1.0"
    assert manifest["contract_commit"] == contract_commit()
    assert manifest["observation_keys"] == ["rgb", "joint_position", "joint_velocity", "gripper_position"]
    assert manifest["action"]["chunk_len"] == 20
    assert manifest["action"]["unit"] == "rad,m"
    assert manifest["action"]["representation"] == "joint_position_targets_then_gripper_width"
    assert manifest["image_preprocessing"] == "uint8_to_float32_divide_255"
    assert manifest["normalization"]["method"] == "z-score per dimension"
    assert manifest["torch_version"] == torch.__version__
    assert manifest["numpy_version"] == np.__version__
    assert manifest["inputs"]["rgb"]["range"] == [0.0, 1.0]

    expected_sha = hashlib.sha256((tmp_path / "model.pt").read_bytes()).hexdigest()
    assert manifest["checkpoint_sha256"] == expected_sha

    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert config["model"]["model_type"].startswith("chunked_action_transformer")


def test_checkpoint_loads_and_forwards(tmp_path: Path):
    model_cfg = ModelConfig()
    model = ChunkedActionTransformerMinimal(model_cfg)
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 32, 32)
    state = torch.randn(2, 13)
    expected = model(rgb, state)

    export_checkpoint(
        tmp_path, "untrained-v0", model, None, None, None, 0, 0,
        model_cfg, TrainConfig(), _stats(), {}, {"torch": None, "cuda": None},
        observation_keys=["rgb"],
        robot_state_keys=["joint_position", "joint_velocity", "gripper_position"],
    )
    restored = ChunkedActionTransformerMinimal(model_cfg)
    checkpoint = load_checkpoint(tmp_path, restored)
    assert checkpoint["global_step"] == 0
    actual = restored(rgb, state)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_git_commit_matches_repo_head():
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert contract_commit() == "f526981d239ce6a0488b14b05bb28218665e121f"
    assert len(head) == 40
