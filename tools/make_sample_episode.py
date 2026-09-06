#!/usr/bin/env python3
"""Regenerate the tiny deterministic contract-v0.1 sample episode."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def build(output: Path) -> Path:
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    timestamps = np.array([0, 50_000_000, 100_000_000], dtype=np.int64)
    joint_position = np.array(
        [
            [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            [0.1, -1.50, 1.50, -1.55, -1.57, 0.0],
            [0.2, -1.45, 1.42, -1.52, -1.57, 0.0],
        ],
        dtype=np.float32,
    )
    joint_velocity = np.vstack((np.zeros((1, 6), np.float32), np.diff(joint_position, axis=0) / 0.05))
    gripper_position = np.array([[0.085], [0.085], [0.020]], dtype=np.float32)
    actions = np.concatenate((joint_position, gripper_position), axis=1).astype(np.float32)
    rgb = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 32
    rgb[:, 1:3, 1:3, 0] = np.array([100, 150, 200], dtype=np.uint8)[:, None, None]
    depth = np.full((3, 4, 4), 1.25, dtype=np.float32)
    depth[:, 1:3, 1:3] = np.array([0.80, 0.75, 0.70], dtype=np.float32)[:, None, None]

    arrays = {
        "timestamps_ns.npy": timestamps,
        "rgb.npy": rgb,
        "depth.npy": depth,
        "joint_position.npy": joint_position,
        "joint_velocity.npy": joint_velocity,
        "gripper_position.npy": gripper_position,
        "actions.npy": actions,
    }
    for name, array in arrays.items():
        np.save(data / name, array, allow_pickle=False)

    descriptor = lambda name, array, semantic, **extra: {
        "path": f"data/{name}",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "semantic": semantic,
        **extra,
    }
    manifest = {
        "schema_version": "0.1.0",
        "episode_id": "sample-direct-000000",
        "created_at": datetime(2026, 9, 6, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "direct_writer",
        "robot": {
            "name": "ur5e_robotiq_2f85",
            "model": "Universal Robots UR5e + Robotiq 2F-85",
            "joint_names": JOINT_NAMES,
            "gripper": {
                "name": "robotiq_2f85",
                "command_semantics": "position",
                "open_position": 0.085,
                "closed_position": 0.0,
            },
        },
        "task": {"name": "pick_and_place", "seed": 7, "object_id": "cube", "target_id": "target_bin"},
        "timebase": {"clock": "simulation", "unit": "nanoseconds", "start_ns": 0},
        "camera": {
            "name": "wrist_rgbd",
            "frame_id": "camera_optical_frame",
            "width": 4,
            "height": 4,
            "intrinsics": {
                "matrix": [4.0, 0.0, 1.5, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0],
                "distortion_model": "none",
                "distortion_coefficients": [],
            },
            "extrinsics": {
                "parent_frame": "world",
                "child_frame": "camera_optical_frame",
                "matrix": [1.0, 0.0, 0.0, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 0.0, 0.0, 0.0, 1.0],
            },
        },
        "artifacts": {
            "timestamps": descriptor("timestamps_ns.npy", timestamps, "aligned_sample_timestamp", unit="ns"),
            "observations": {
                "rgb": descriptor("rgb.npy", rgb, "camera_rgb", encoding="rgb8", frame_id="camera_optical_frame"),
                "depth": descriptor("depth.npy", depth, "camera_depth", encoding="32FC1", unit="m", frame_id="camera_optical_frame"),
                "joint_position": descriptor("joint_position.npy", joint_position, "joint_position", unit="rad"),
                "joint_velocity": descriptor("joint_velocity.npy", joint_velocity, "joint_velocity", unit="rad/s"),
                "gripper_position": descriptor("gripper_position.npy", gripper_position, "gripper_width", unit="m"),
            },
            "actions": descriptor("actions.npy", actions, "joint_position_plus_gripper_width", unit="rad,m"),
        },
        "outcome": {"success": True, "num_steps": 3, "duration_ns": 100_000_000, "termination": "success"},
        "provenance": {
            "producer": "tools/make_sample_episode.py",
            "producer_version": "0.1.0",
            "git_revision": None,
            "input_uri": None,
        },
    }
    manifest_path = output / "episode.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples/episode_v0"))
    args = parser.parse_args()
    path = build(args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

