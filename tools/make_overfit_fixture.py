#!/usr/bin/env python3
"""Generate the tiny synthetic overfit fixture (contract-v0.1 episodes).

Purpose: validate the training pipeline end to end on minimal data. This is
SYNTHETIC data, never a model-quality signal. The generator is deterministic
per seed and validates every episode with the frozen contract validator
before writing.

Layout: fixtures/overfit/<episode_id>/{episode.json, data/*.npy}
Default: 10 train + 2 val episodes, 40 steps each, 32x32 RGB, 30 Hz,
UR5e-like 6-DoF + Robotiq gripper (13-dim state, 7-dim action).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from embodiedflow.contracts.validator import validate_episode

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
FPS = 30.0
DT_NS = 33_333_333  # 30 Hz integer nanoseconds
H = W = 32
GRIPPER_OPEN = 0.085


def _trajectory(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth joint positions, finite-difference velocities, gripper widths."""
    t = np.arange(steps, dtype=np.float64) / FPS
    joint_position = np.zeros((steps, len(JOINT_NAMES)), dtype=np.float64)
    for j in range(len(JOINT_NAMES)):
        amplitude = rng.uniform(0.25, 0.6)
        freq = rng.uniform(0.10, 0.25)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        drift = rng.uniform(-0.02, 0.02)
        joint_position[:, j] = (
            amplitude * np.sin(2.0 * np.pi * freq * t + phase)
            + drift * t
            + rng.uniform(-0.3, 0.3)
        )
    joint_velocity = np.vstack(
        (np.zeros((1, len(JOINT_NAMES))), np.diff(joint_position, axis=0) * FPS)
    )
    cycles = rng.uniform(0.5, 1.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    gripper = 0.02 + 0.065 * (0.5 + 0.5 * np.sin(2.0 * np.pi * cycles * t / max(t[-1], 1e-9) + phase))
    return (
        joint_position.astype(np.float32),
        joint_velocity.astype(np.float32),
        gripper.reshape(-1, 1).astype(np.float32),
    )


def _actions(joint_position: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Deterministic nonlinear mapping from state: learnable but not a copy."""
    target = joint_position + 0.05 * np.sin(3.0 * joint_position)
    return np.concatenate((target, gripper), axis=1).astype(np.float32)


def _render_frame(joint_position: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic camera image: red square position/size/brightness encode state."""
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 1] = 32  # greenish background
    q = joint_position[step]
    cx = int(np.clip((q[0] + 1.6) / 3.2 * (W - 8) + 4, 4, W - 5))
    cy = int(np.clip((q[1] + 1.6) / 3.2 * (H - 8) + 4, 4, H - 5))
    half = int(np.clip(1 + (q[2] + 1.6) / 3.2 * 4, 1, 5))
    brightness = int(np.clip(120 + 80 * (q[3] + 1.6) / 3.2, 0, 255))
    rgb[cy - half : cy + half, cx - half : cx + half, 0] = brightness
    depth = np.full((H, W), 1.25, dtype=np.float32)
    depth[cy - half : cy + half, cx - half : cx + half] = 0.75
    return rgb, depth


def build_episode(output: Path, episode_id: str, seed: int, steps: int = 40) -> Path:
    rng = np.random.default_rng(seed)
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)

    timestamps = (np.arange(steps, dtype=np.int64) * DT_NS).astype(np.int64)
    joint_position, joint_velocity, gripper = _trajectory(rng, steps)
    actions = _actions(joint_position, gripper)
    rgb = np.zeros((steps, H, W, 3), dtype=np.uint8)
    depth = np.full((steps, H, W), 1.25, dtype=np.float32)
    for step in range(steps):
        rgb[step], depth[step] = _render_frame(joint_position, step)

    arrays = {
        "timestamps_ns.npy": timestamps,
        "rgb.npy": rgb,
        "depth.npy": depth,
        "joint_position.npy": joint_position,
        "joint_velocity.npy": joint_velocity,
        "gripper_position.npy": gripper,
        "actions.npy": actions,
    }
    for name, array in arrays.items():
        np.save(data / name, array, allow_pickle=False)

    def descriptor(name: str, array: np.ndarray, semantic: str, **extra) -> dict:
        return {
            "path": f"data/{name}",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "semantic": semantic,
            **extra,
        }

    manifest = {
        "schema_version": "0.1.0",
        "episode_id": episode_id,
        "created_at": datetime(2026, 9, 7, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "direct_writer",
        "robot": {
            "name": "ur5e_robotiq_2f85",
            "model": "Universal Robots UR5e + Robotiq 2F-85",
            "joint_names": JOINT_NAMES,
            "gripper": {
                "name": "robotiq_2f85",
                "command_semantics": "position",
                "open_position": GRIPPER_OPEN,
                "closed_position": 0.0,
            },
        },
        "task": {"name": "pick_and_place", "seed": seed, "object_id": "cube", "target_id": "target_bin"},
        "timebase": {"clock": "simulation", "unit": "nanoseconds", "start_ns": 0},
        "camera": {
            "name": "wrist_rgbd",
            "frame_id": "camera_optical_frame",
            "width": W,
            "height": H,
            "intrinsics": {
                "matrix": [16.0, 0.0, 15.5, 0.0, 16.0, 15.5, 0.0, 0.0, 1.0],
                "distortion_model": "none",
                "distortion_coefficients": [],
            },
            "extrinsics": {
                "parent_frame": "world",
                "child_frame": "camera_optical_frame",
                "matrix": [
                    1.0, 0.0, 0.0, 0.7,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 1.1,
                    0.0, 0.0, 0.0, 1.0,
                ],
            },
        },
        "artifacts": {
            "timestamps": descriptor("timestamps_ns.npy", timestamps, "aligned_sample_timestamp", unit="ns"),
            "observations": {
                "rgb": descriptor("rgb.npy", rgb, "camera_rgb", encoding="rgb8", frame_id="camera_optical_frame"),
                "depth": descriptor("depth.npy", depth, "camera_depth", encoding="32FC1", unit="m", frame_id="camera_optical_frame"),
                "joint_position": descriptor("joint_position.npy", joint_position, "joint_position", unit="rad"),
                "joint_velocity": descriptor("joint_velocity.npy", joint_velocity, "joint_velocity", unit="rad/s"),
                "gripper_position": descriptor("gripper_position.npy", gripper, "gripper_width", unit="m"),
            },
            "actions": descriptor("actions.npy", actions, "joint_position_plus_gripper_width", unit="rad,m"),
        },
        "outcome": {
            "success": True,
            "num_steps": steps,
            "duration_ns": int((steps - 1) * DT_NS),
            "termination": "success",
        },
        "provenance": {
            "producer": "tools/make_overfit_fixture.py",
            "producer_version": "0.1.0",
            "git_revision": None,
            "input_uri": None,
        },
    }
    manifest_path = output / "episode.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    issues = validate_episode(manifest_path)
    if issues:
        raise RuntimeError(f"{episode_id}: generated episode failed frozen validator:\n" +
                           "\n".join(str(i) for i in issues))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fixtures/overfit"))
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    generated = []
    # Episode ids are split-neutral: the pipeline owns the train/val split.
    for i in range(args.train + args.val):
        episode_id = f"overfit-{i:06d}"
        seed = args.base_seed + i
        build_episode(args.output / episode_id, episode_id, seed, steps=args.steps)
        generated.append(episode_id)
        print(f"wrote {args.output / episode_id}")
    print(f"done: {len(generated)} episodes, all passed frozen validator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
