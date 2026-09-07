"""CanonicalEpisodeDataset over contract-v0.1 episode directories.

Data source is the frozen contract only: each episode is a directory with an
episode.json manifest and relative .npy artifacts. Episodes are validated
with the frozen validator before entering the dataset; invalid episodes are
quarantined (skipped and recorded) unless skip_invalid_episodes=False.

Sample layout (numpy, then converted by collate_fn):
  rgb:          float32 [3, H, W] (CHW), uint8/255
  robot_state:  float32 [S] = joint_position | joint_velocity | gripper_position
  action_chunk: float32 [L, A], A = J + 1 (joint targets then gripper width)
  padding_mask: float32 [L], 1.0 = valid entry
  episode_id:   str
  step_idx:     int (anchor step within episode)
  timestamps_ns:int64 (anchor timestamp)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from embodiedflow.contracts.validator import ValidationIssue, validate_episode
from embodiedflow.dataplane.config import DatasetConfig
from embodiedflow.dataplane.stats import NormStats


class DatasetValidationError(RuntimeError):
    def __init__(self, episode_dir: Path, issues: list[ValidationIssue]):
        detail = "; ".join(str(issue) for issue in issues[:5])
        super().__init__(f"{episode_dir}: {len(issues)} validation issue(s): {detail}")
        self.episode_dir = episode_dir
        self.issues = issues


def episode_ids_from_dirs(episode_dirs: list[Path]) -> list[str]:
    """Episode id = manifest episode_id (fallback: directory name)."""
    ids = []
    for episode_dir in episode_dirs:
        manifest_path = episode_dir / "episode.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ids.append(str(manifest.get("episode_id", episode_dir.name)))
        else:
            ids.append(episode_dir.name)
    return ids


def split_episodes(
    episode_dirs: list[Path],
    val_ratio: float,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """Episode-level split; whole episodes go to one side only (no frame leakage)."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    dirs = sorted(episode_dirs, key=lambda d: str(d))
    rng = np.random.default_rng(seed)
    n_val = int(round(len(dirs) * val_ratio))
    if n_val == 0 and val_ratio > 0 and dirs:
        n_val = 1
    val_idx = set(rng.choice(len(dirs), size=n_val, replace=False).tolist())
    val = [d for i, d in enumerate(dirs) if i in val_idx]
    train = [d for i, d in enumerate(dirs) if i not in val_idx]
    return train, val


def _gripper_range_issues(actions: np.ndarray, manifest: dict, tolerance: float) -> list[ValidationIssue]:
    robot = manifest.get("robot", {})
    gripper = robot.get("gripper", {})
    if not isinstance(gripper, dict):
        return []
    closed = gripper.get("closed_position")
    opened = gripper.get("open_position")
    if not (isinstance(closed, (int, float)) and isinstance(opened, (int, float))):
        return []
    low, high = sorted((float(closed), float(opened)))
    gripper_col = actions[:, -1]
    bad = int(((gripper_col < low - tolerance) | (gripper_col > high + tolerance)).sum())
    if bad:
        return [
            ValidationIssue(
                "artifacts.actions",
                f"{bad} gripper action(s) outside declared range [{low}, {high}] "
                f"(tolerance {tolerance})",
            )
        ]
    return []


@dataclass
class _Episode:
    directory: Path
    episode_id: str
    manifest: dict
    num_steps: int
    joint_names: list[str]
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, directory: Path) -> "_Episode":
        manifest_path = directory / "episode.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arrays = {}
        for descriptor in _iter_artifact_descriptors(manifest):
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                continue
            arrays[relative] = np.load(directory / relative, mmap_mode="r", allow_pickle=False)
        num_steps = int(manifest.get("outcome", {}).get("num_steps", 0))
        if num_steps <= 0:
            raise ValueError(f"{directory}: outcome.num_steps missing or <= 0")
        return cls(
            directory=directory,
            episode_id=str(manifest.get("episode_id", directory.name)),
            manifest=manifest,
            num_steps=num_steps,
            joint_names=list(manifest.get("robot", {}).get("joint_names", ())),
            arrays=arrays,
        )

    def array(self, relative: str) -> np.ndarray:
        try:
            return self.arrays[relative]
        except KeyError as exc:
            raise KeyError(f"{self.directory}: artifact {relative!r} not present") from exc

    def rgb(self) -> np.ndarray:
        return self.array(self.manifest["artifacts"]["observations"]["rgb"]["path"])

    def joint_position(self) -> np.ndarray:
        return self.array(self.manifest["artifacts"]["observations"]["joint_position"]["path"])

    def joint_velocity(self) -> np.ndarray:
        return self.array(self.manifest["artifacts"]["observations"]["joint_velocity"]["path"])

    def gripper_position(self) -> np.ndarray:
        return self.array(self.manifest["artifacts"]["observations"]["gripper_position"]["path"])

    def actions(self) -> np.ndarray:
        return self.array(self.manifest["artifacts"]["actions"]["path"])

    def timestamps(self) -> np.ndarray:
        return self.array(self.manifest["artifacts"]["timestamps"]["path"])

    def robot_state(self, step: int) -> np.ndarray:
        return np.concatenate(
            [
                self.joint_position()[step],
                self.joint_velocity()[step],
                self.gripper_position()[step],
            ]
        ).astype(np.float32)


def _iter_artifact_descriptors(manifest: dict):
    artifacts = manifest.get("artifacts", {})
    timestamps = artifacts.get("timestamps")
    if isinstance(timestamps, dict):
        yield timestamps
    for name, descriptor in (artifacts.get("observations") or {}).items():
        if isinstance(descriptor, dict):
            yield descriptor
    actions = artifacts.get("actions")
    if isinstance(actions, dict):
        yield actions


class CanonicalEpisodeDataset(Dataset):
    def __init__(
        self,
        episode_dirs: list[Path],
        cfg: DatasetConfig,
        normalize: NormStats | None = None,
        schema_path: Path | None = None,
    ):
        self.cfg = cfg
        self.normalize = normalize
        self.episodes: list[_Episode] = []
        self.bad_episodes: list[tuple[Path, list[ValidationIssue]]] = []
        self.validated: dict[Path, list[ValidationIssue]] = {}

        for directory in episode_dirs:
            directory = Path(directory)
            manifest_path = directory / "episode.json"
            if not manifest_path.is_file():
                self.bad_episodes.append(
                    (directory, [ValidationIssue("$", "episode.json not found")])
                )
                continue
            issues = validate_episode(manifest_path, schema_path) if cfg.validate_before_load else []
            if issues:
                self.bad_episodes.append((directory, issues))
                if not cfg.skip_invalid_episodes:
                    raise DatasetValidationError(directory, issues)
                continue
            episode = _Episode.load(directory)
            gripper_issues = _gripper_range_issues(
                np.asarray(episode.actions()), episode.manifest, cfg.gripper_range_tol
            )
            if gripper_issues:
                self.bad_episodes.append((directory, gripper_issues))
                if cfg.enforce_gripper_range:
                    raise DatasetValidationError(directory, gripper_issues)
            self.validated[directory] = issues
            self.episodes.append(episode)

        if not self.episodes:
            raise ValueError("no valid episodes: check dataset paths and validation logs")

        self._build_index()

    def _build_index(self) -> None:
        self.index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            for step in range(0, episode.num_steps, self.cfg.window_stride):
                self.index.append((episode_idx, step))

    def __len__(self) -> int:
        return len(self.index)

    def _chunk(self, episode: _Episode, start: int) -> tuple[np.ndarray, np.ndarray]:
        chunk_len = self.cfg.chunk_len
        actions = episode.actions()
        available = max(0, min(chunk_len, episode.num_steps - start))
        chunk = np.zeros((chunk_len, actions.shape[1]), dtype=np.float32)
        chunk[:available] = actions[start : start + available]
        mask = np.zeros(chunk_len, dtype=np.float32)
        mask[:available] = 1.0
        return chunk, mask

    def __getitem__(self, index: int) -> dict:
        episode_idx, step = self.index[index]
        episode = self.episodes[episode_idx]
        rgb = np.ascontiguousarray(episode.rgb()[step].astype(np.float32) / 255.0).transpose(2, 0, 1)
        state = episode.robot_state(step)
        action_chunk, padding_mask = self._chunk(episode, step)
        if self.normalize is not None:
            state = self.normalize.normalize_state(state)
            action_chunk = self.normalize.normalize_action(action_chunk, padding_mask)
        timestamps = episode.timestamps()
        return {
            "rgb": rgb,
            "robot_state": state,
            "action_chunk": action_chunk,
            "padding_mask": padding_mask,
            "episode_id": episode.episode_id,
            "step_idx": int(step),
            "timestamps_ns": int(timestamps[step]) if timestamps.size else 0,
        }

    @property
    def state_dim(self) -> int:
        return len(self.episodes[0].joint_names) * 2 + 1

    @property
    def action_dim(self) -> int:
        return self.episodes[0].actions().shape[1]

    @property
    def joint_names(self) -> list[str]:
        return self.episodes[0].joint_names


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for key in ("rgb", "robot_state", "action_chunk", "padding_mask"):
        out[key] = torch.stack([torch.as_tensor(item[key]) for item in batch])
    out["episode_id"] = [item["episode_id"] for item in batch]
    out["step_idx"] = torch.as_tensor([item["step_idx"] for item in batch], dtype=torch.long)
    out["timestamps_ns"] = torch.as_tensor(
        [item["timestamps_ns"] for item in batch], dtype=torch.int64
    )
    return out


def make_dataloader(
    dataset: CanonicalEpisodeDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    seed: int | None = None,
    drop_last: bool = True,
    shuffle: bool = True,
    sampler=None,
) -> DataLoader:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32) if seed is not None else None
        if worker_seed is not None:
            np.random.seed(worker_seed)

    kwargs = {}
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        generator=generator,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        **kwargs,
    )


def describe_batch(batch: dict, name: str = "batch") -> str:
    """Human-readable one-batch report: shape, dtype, range, device."""
    lines = [f"== {name} =="]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value
            lines.append(
                f"{key:>16}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} min={tensor.min().item():.4f} "
                f"max={tensor.max().item():.4f}"
            )
        else:
            lines.append(f"{key:>16}: {value}")
    return "\n".join(lines)
