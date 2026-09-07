"""CanonicalEpisodeDataset tests: validation, quarantine, split, chunking,
normalization, determinism, and the one-batch report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from embodiedflow.dataplane.config import DatasetConfig
from embodiedflow.dataplane.dataset import (
    CanonicalEpisodeDataset,
    DatasetValidationError,
    describe_batch,
    make_dataloader,
    split_episodes,
)
from embodiedflow.dataplane.stats import compute_stats

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "episode_v0"
FIXTURE = ROOT / "fixtures" / "overfit"

FIXTURE_DIRS = sorted(p for p in FIXTURE.iterdir() if (p / "episode.json").is_file())


def _copy_episode(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_sample_episode_loads_and_validates():
    ds = CanonicalEpisodeDataset([SAMPLE], DatasetConfig(chunk_len=5))
    assert len(ds.episodes) == 1
    assert ds.bad_episodes == []
    assert len(ds) == 3  # num_steps anchors with stride 1
    assert ds.state_dim == 13  # 6 joints * 2 + gripper
    assert ds.action_dim == 7
    assert ds.joint_names[0] == "shoulder_pan_joint"


def test_chunk_padding_mask():
    ds = CanonicalEpisodeDataset([SAMPLE], DatasetConfig(chunk_len=5, window_stride=1))
    sample = ds[2]  # last step: only 1 valid action remains
    assert sample["padding_mask"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert np.all(sample["action_chunk"][1:] == 0.0)
    sample0 = ds[0]  # sample has 3 steps: mask covers them, rest padded
    assert sample0["padding_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert sample0["rgb"].shape == (3, 4, 4)
    assert sample0["robot_state"].shape == (13,)
    assert sample0["action_chunk"].shape == (5, 7)


def test_split_no_frame_leakage():
    train, val = split_episodes(FIXTURE_DIRS, val_ratio=0.2, seed=0)
    train_ids = {p.name for p in train}
    val_ids = {p.name for p in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {p.name for p in FIXTURE_DIRS}
    assert len(val) == 2


def test_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        split_episodes(FIXTURE_DIRS, val_ratio=1.0)


def test_missing_frame_quarantined(tmp_path: Path):
    bad = _copy_episode(SAMPLE, tmp_path / "bad")
    (bad / "data" / "rgb.npy").unlink()
    ds = CanonicalEpisodeDataset([SAMPLE, bad], DatasetConfig())
    assert len(ds.episodes) == 1
    assert len(ds.bad_episodes) == 1
    assert "missing file" in str(ds.bad_episodes[0][1][0])


def test_missing_frame_raises_when_not_skipping(tmp_path: Path):
    bad = _copy_episode(SAMPLE, tmp_path / "bad")
    (bad / "data" / "rgb.npy").unlink()
    with pytest.raises(DatasetValidationError):
        CanonicalEpisodeDataset([bad], DatasetConfig(skip_invalid_episodes=False))


def test_non_monotonic_timestamps_quarantined(tmp_path: Path):
    bad = _copy_episode(SAMPLE, tmp_path / "bad")
    np.save(bad / "data" / "timestamps_ns.npy", np.array([0, 40, 20], dtype=np.int64))
    ds = CanonicalEpisodeDataset([SAMPLE, bad], DatasetConfig())
    assert len(ds.episodes) == 1
    assert len(ds.bad_episodes) == 1
    assert any("strictly increasing" in str(i) for i in ds.bad_episodes[0][1])


def test_length_mismatch_quarantined(tmp_path: Path):
    bad = _copy_episode(SAMPLE, tmp_path / "bad")
    manifest = json.loads((bad / "episode.json").read_text(encoding="utf-8"))
    manifest["outcome"]["num_steps"] = 7  # arrays have 3 rows
    (bad / "episode.json").write_text(json.dumps(manifest), encoding="utf-8")
    ds = CanonicalEpisodeDataset([SAMPLE, bad], DatasetConfig())
    assert len(ds.episodes) == 1
    assert len(ds.bad_episodes) == 1
    assert any("num_steps" in str(i) for i in ds.bad_episodes[0][1])


def test_out_of_range_gripper_action_quarantined(tmp_path: Path):
    bad = _copy_episode(SAMPLE, tmp_path / "bad")
    actions = np.load(bad / "data" / "actions.npy")
    actions[-1, -1] = 0.5  # way above open_position 0.085
    np.save(bad / "data" / "actions.npy", actions)
    ds = CanonicalEpisodeDataset([bad], DatasetConfig(enforce_gripper_range=False))
    assert len(ds.bad_episodes) == 1
    assert any("gripper" in str(i) for i in ds.bad_episodes[0][1])
    with pytest.raises(DatasetValidationError):
        CanonicalEpisodeDataset([bad], DatasetConfig(enforce_gripper_range=True))


def test_normalization_stats_roundtrip(tmp_path: Path):
    ds = CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig())
    stats = compute_stats(
        (ds[i] for i in range(len(ds))),
        state_dim=ds.state_dim,
        action_dim=ds.action_dim,
    )
    stats.to_json(tmp_path / "normalization.json")
    loaded = type(stats).from_json(tmp_path / "normalization.json")
    assert np.allclose(stats.robot_state_mean, loaded.robot_state_mean)
    assert np.allclose(stats.action_std, loaded.action_std)
    assert loaded.image_normalization == "uint8_to_float32_divide_255"
    ds.normalize = loaded
    normalized = np.stack([ds[i]["robot_state"] for i in range(len(ds))])
    assert abs(normalized.mean()) < 1e-5
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-4)


def test_normalization_keeps_padding_at_zero():
    ds = CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig())
    stats = compute_stats(
        (ds[i] for i in range(len(ds))),
        state_dim=ds.state_dim,
        action_dim=ds.action_dim,
    )
    ds.normalize = stats
    sample = ds[-1]  # last anchor of a 40-step episode: 1 valid entry, 19 padded
    padded = sample["action_chunk"][sample["padding_mask"] == 0]
    assert padded.shape[0] == 19
    assert np.all(padded == 0.0)


def test_dataloader_deterministic_same_seed():
    ds1 = CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig(deterministic_seed=0))
    ds2 = CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig(deterministic_seed=0))
    loader1 = make_dataloader(ds1, batch_size=8, num_workers=0, pin_memory=False, seed=0, shuffle=True)
    loader2 = make_dataloader(ds2, batch_size=8, num_workers=0, pin_memory=False, seed=0, shuffle=True)
    for batch1, batch2 in zip(loader1, loader2):
        assert torch.equal(batch1["rgb"], batch2["rgb"])
        assert torch.equal(batch1["action_chunk"], batch2["action_chunk"])
        assert batch1["episode_id"] == batch2["episode_id"]
    assert len(list(loader1)) == len(list(loader2))


def test_dataloader_different_seed_differs():
    loader1 = make_dataloader(
        CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig(deterministic_seed=0)),
        batch_size=8, num_workers=0, pin_memory=False, seed=0, shuffle=True,
    )
    loader2 = make_dataloader(
        CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig(deterministic_seed=1)),
        batch_size=8, num_workers=0, pin_memory=False, seed=1, shuffle=True,
    )
    b1, b2 = next(iter(loader1)), next(iter(loader2))
    assert not torch.equal(b1["rgb"], b2["rgb"])


def test_one_batch_report(tmp_path: Path):
    ds = CanonicalEpisodeDataset(FIXTURE_DIRS, DatasetConfig(deterministic_seed=0))
    loader = make_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False, seed=0, shuffle=False)
    batch = next(iter(loader))
    report = describe_batch(batch)
    assert "rgb" in report and "shape=(4, 3, 32, 32)" in report
    assert "robot_state" in report and "shape=(4, 13)" in report
    assert "action_chunk" in report and "shape=(4, 20, 7)" in report
    assert "padding_mask" in report and "shape=(4, 20)" in report
    assert "episode_id" in report and "step_idx" in report and "timestamps_ns" in report
    assert report.count("dtype=torch.float32") >= 4
    # dataset converts uint8 rgb to [0, 1] float32
    rgb = batch["rgb"]
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert rgb.dtype == torch.float32


def test_dataset_contains_only_given_episodes():
    train_ds = CanonicalEpisodeDataset(FIXTURE_DIRS[:10], DatasetConfig())
    ids = {s["episode_id"] for s in (train_ds[i] for i in range(len(train_ds)))}
    assert ids == {p.name for p in FIXTURE_DIRS[:10]}
