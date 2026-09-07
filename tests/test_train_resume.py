"""Resume verification: global step continuity, optimizer/scheduler recovery,
normalization consistency, and bitwise-close step loss after resume."""

from __future__ import annotations

from pathlib import Path

import torch

from embodiedflow.dataplane.config import DatasetConfig, ModelConfig, TrainConfig
from embodiedflow.dataplane.train import train

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "overfit"


def _cfg(steps: int, seed: int = 7) -> tuple[DatasetConfig, ModelConfig, TrainConfig]:
    dataset_cfg = DatasetConfig(
        deterministic_seed=11,
        num_workers=0,
        pin_memory=False,
        batch_size=8,
        drop_last=True,
    )
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        steps=steps,
        amp=False,
        seed=seed,
        save_every=5,
        eval_every=10_000,
        warmup_steps=0,
    )
    return dataset_cfg, model_cfg, train_cfg


def test_resume_continues_exactly(tmp_path: Path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    dataset_cfg, model_cfg, train_cfg = _cfg(steps=15)

    metrics1 = train(FIXTURE, dataset_cfg, model_cfg, train_cfg, torch.device("cpu"), run_dir=run1)
    step10_dir = run1 / "checkpoints" / "step-0000010"
    assert step10_dir.is_dir()

    # resume from the step-10 checkpoint into a fresh output dir
    dataset_cfg2, model_cfg2, train_cfg2 = _cfg(steps=15)
    metrics2 = train(
        FIXTURE, dataset_cfg2, model_cfg2, train_cfg2,
        torch.device("cpu"), run_dir=run2, resume_dir=step10_dir,
    )

    # 1. global step continuity: resumed run steps start at 10
    assert metrics2["steps"] == list(range(10, 15))

    # 2. same-batch loss at resumed step 10 matches the original run
    for step in range(10, 15):
        loss1 = metrics1["train_loss"][step]
        loss2 = metrics2["train_loss"][step - 10]
        assert abs(loss1 - loss2) < 1e-6, f"step {step}: {loss1} vs {loss2}"

    # 3. scheduler recovery: lr sequence identical
    assert metrics2["lr"] == metrics1["lr"][10:15]

    # 4. optimizer recovery: final optimizer state identical to the uninterrupted run
    ckpt1 = torch.load(run1 / "model.pt", map_location="cpu", weights_only=False)
    ckpt2 = torch.load(run2 / "model.pt", map_location="cpu", weights_only=False)
    assert ckpt1["global_step"] == 15 and ckpt2["global_step"] == 15
    for key in ckpt1["optimizer_state_dict"]["state"]:
        for subkey in ckpt1["optimizer_state_dict"]["state"][key]:
            assert torch.equal(
                ckpt1["optimizer_state_dict"]["state"][key][subkey],
                ckpt2["optimizer_state_dict"]["state"][key][subkey],
            ), f"optimizer state diverged at {key}.{subkey}"

    # 5. normalization statistics identical
    assert ckpt1["normalization"] == ckpt2["normalization"]

    # 6. final model weights identical
    for (name1, p1), (name2, p2) in zip(
        ckpt1["model_state_dict"].items(), ckpt2["model_state_dict"].items()
    ):
        assert name1 == name2
        assert torch.equal(p1, p2), f"weights diverged at {name1}"


def test_overfit_on_tiny_fixture(tmp_path: Path):
    """The model must overfit the synthetic fixture: train loss far below
    the untrained loss."""
    dataset_cfg, model_cfg, train_cfg = _cfg(steps=400)
    train_cfg.eval_every = 200
    run_dir = tmp_path / "overfit"
    metrics = train(FIXTURE, dataset_cfg, model_cfg, train_cfg, torch.device("cpu"), run_dir=run_dir)

    first = metrics["train_loss"][0]
    last = metrics["train_loss"][-1]
    assert last < 0.05 * first, f"expected overfit, got first={first:.4f} last={last:.4f}"
    # val episodes are unseen trajectories; require improvement below the
    # untrained baseline (full generalization is not expected in 400 steps)
    val_entries = [v for v in metrics["val_loss"] if v is not None]
    assert val_entries, "eval never ran"
    assert val_entries[-1][1] < first
    assert val_entries[-1][1] <= val_entries[0][1]
