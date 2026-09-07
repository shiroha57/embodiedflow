"""Normalization statistics: compute, apply, save, load.

robot_state and action are z-scored per dimension. Images use the method
declared in DatasetConfig.image_normalization (no learned stats).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

EPS = 1e-8


@dataclass
class NormStats:
    robot_state_mean: np.ndarray
    robot_state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    num_samples: int = 0
    image_normalization: str = "uint8_to_float32_divide_255"
    computed_on: str = "train_split"

    def to_dict(self) -> dict:
        return {
            "robot_state": {
                "mean": self.robot_state_mean.tolist(),
                "std": self.robot_state_std.tolist(),
            },
            "action": {
                "mean": self.action_mean.tolist(),
                "std": self.action_std.tolist(),
            },
            "image": {"method": self.image_normalization},
            "computed_on": self.computed_on,
            "num_samples": self.num_samples,
        }

    def to_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "NormStats":
        return cls(
            robot_state_mean=np.asarray(data["robot_state"]["mean"], dtype=np.float32),
            robot_state_std=np.asarray(data["robot_state"]["std"], dtype=np.float32),
            action_mean=np.asarray(data["action"]["mean"], dtype=np.float32),
            action_std=np.asarray(data["action"]["std"], dtype=np.float32),
            num_samples=int(data.get("num_samples", 0)),
            image_normalization=data.get("image", {}).get("method", "uint8_to_float32_divide_255"),
            computed_on=data.get("computed_on", "train_split"),
        )

    @classmethod
    def from_json(cls, path: Path) -> "NormStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.robot_state_mean) / (self.robot_state_std + EPS)

    def normalize_action(
        self, action: np.ndarray, padding_mask: np.ndarray | None = None
    ) -> np.ndarray:
        normalized = (action - self.action_mean) / (self.action_std + EPS)
        if padding_mask is not None:
            # keep zero-padded chunk entries at exactly 0; normalizing them
            # would map padding to huge values for near-constant dims
            normalized = normalized * np.asarray(padding_mask, dtype=np.float32)[:, None]
        return normalized

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * (self.action_std + EPS) + self.action_mean


def compute_stats(
    samples: Iterator[dict],
    state_dim: int,
    action_dim: int,
    image_normalization: str = "uint8_to_float32_divide_255",
) -> NormStats:
    """Welford's online mean/std over robot_state and valid action entries."""
    count = 0
    state_mean = np.zeros(state_dim, dtype=np.float64)
    state_m2 = np.zeros(state_dim, dtype=np.float64)
    act_mean = np.zeros(action_dim, dtype=np.float64)
    act_m2 = np.zeros(action_dim, dtype=np.float64)
    act_count = 0

    for sample in samples:
        state = np.asarray(sample["robot_state"], dtype=np.float64)
        count += 1
        delta = state - state_mean
        state_mean += delta / count
        state_m2 += delta * (state - state_mean)

        action = np.asarray(sample["action_chunk"], dtype=np.float64)
        mask = np.asarray(sample["padding_mask"], dtype=bool)
        valid = action[mask]
        for value in valid:
            act_count += 1
            delta = value - act_mean
            act_mean += delta / act_count
            act_m2 += delta * (value - act_mean)

    if count == 0:
        raise ValueError("cannot compute normalization stats from an empty split")
    if act_count == 0:
        raise ValueError("cannot compute action stats: no valid action entries")

    return NormStats(
        robot_state_mean=state_mean.astype(np.float32),
        robot_state_std=np.sqrt(state_m2 / count).astype(np.float32),
        action_mean=act_mean.astype(np.float32),
        action_std=np.sqrt(act_m2 / act_count).astype(np.float32),
        num_samples=count,
        image_normalization=image_normalization,
    )
