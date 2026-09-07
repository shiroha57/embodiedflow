#!/usr/bin/env python3
"""Headless camera smoke test: render a contract-v0.1 episode via MuJoCo EGL.

HARDWARE ENVIRONMENT PROOF ONLY. This is NOT a formal project data source:
it proves the 4090 host can render headless camera frames through the
EGL -> OpenGL -> FBO -> MuJoCo stack, and that the output passes the frozen
contract validator. It is never described as Isaac Sim acceptance, and
episodes produced here must not enter model training.

The scene is a minimal MuJoCo world rendered with the DEFAULT camera
(custom xyaxes can silently produce all-black frames on this stack; see
docs/4090/env.md). EGL device i == physical GPU i on this host, so the
renderer is pinned to a free physical GPU via MUJOCO_EGL_DEVICE_ID (EGL
ignores CUDA_VISIBLE_DEVICES).

Output: artifacts/egl-proof/<episode_id>/{episode.json, data/*.npy},
validated with the frozen contract validator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.make_overfit_fixture import (  # noqa: E402
    DT_NS,
    FPS,
    GRIPPER_OPEN,
    JOINT_NAMES,
    _actions,
    _trajectory,
)

PROOF_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "egl-proof"
H, W = 480, 640

SCENE_XML = """
<mujoco model="egl_proof">
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35"/>
  </visual>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.15 0.25 0.35" rgb2="0.85 0.85 0.85" width="512" height="512"/>
    <material name="matplane" texture="texplane" texrepeat="4 4" reflectance="0.15"/>
    <texture name="texlink" type="2d" builtin="flat" rgb1="0.75 0.25 0.2" width="64" height="64"/>
    <material name="matlink" texture="texlink"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="matplane"/>
    <body name="arm" pos="0 0 0.4">
      <joint name="hinge" type="hinge" axis="0 0 1" limited="true" range="-180 180"/>
      <geom name="link" type="box" size="0.4 0.05 0.05" pos="0.2 0 0" material="matlink"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from embodiedflow.dataplane.gpu import pick_free_gpus

    physical_gpu = pick_free_gpus(1)[0]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical_gpu)

    import mujoco
    from embodiedflow.contracts.validator import validate_episode

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)

    rng = np.random.default_rng(args.seed)
    joint_position, joint_velocity, gripper = _trajectory(rng, args.steps)
    actions = _actions(joint_position, gripper)
    timestamps = (np.arange(args.steps, dtype=np.int64) * DT_NS).astype(np.int64)
    rgb = np.zeros((args.steps, H, W, 3), dtype=np.uint8)
    depth = np.zeros((args.steps, H, W), dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(args.steps):
        data.qpos[0] = joint_position[step, 0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        rgb[step] = renderer.render()
        renderer.enable_depth_rendering()
        depth[step] = renderer.render()
        renderer.disable_depth_rendering()
    elapsed = time.perf_counter() - t0
    fps = args.steps / elapsed

    episode_id = args.episode_id or f"egl-proof-{physical_gpu:02d}-{args.seed:04d}"
    out = PROOF_ROOT / episode_id
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

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
        np.save(data_dir / name, array, allow_pickle=False)

    def descriptor(name: str, array: np.ndarray, semantic: str, **extra) -> dict:
        return {"path": f"data/{name}", "dtype": str(array.dtype),
                "shape": list(array.shape), "semantic": semantic, **extra}

    # Default (free) camera at fovy 45 deg, 640x480: fy = (H/2) / tan(fovy/2)
    fy = (H / 2.0) / np.tan(np.deg2rad(45.0 / 2.0))
    manifest = {
        "schema_version": "0.1.0",
        "episode_id": episode_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
        "task": {"name": "pick_and_place", "seed": args.seed, "object_id": "cube", "target_id": "target_bin"},
        "timebase": {"clock": "simulation", "unit": "nanoseconds", "start_ns": 0},
        "camera": {
            "name": "wrist_rgbd",
            "frame_id": "camera_optical_frame",
            "width": W,
            "height": H,
            "intrinsics": {
                "matrix": [fy, 0.0, W / 2.0, 0.0, fy, H / 2.0, 0.0, 0.0, 1.0],
                "distortion_model": "none",
                "distortion_coefficients": [],
            },
            "extrinsics": {
                "parent_frame": "world",
                "child_frame": "camera_optical_frame",
                "matrix": [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5,
                           0.0, 0.0, 0.0, 1.0],
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
            "num_steps": args.steps,
            "duration_ns": int((args.steps - 1) * DT_NS),
            "termination": "success",
        },
        "provenance": {
            "producer": "tools/headless_camera_smoke.py",
            "producer_version": "0.1.0",
            "git_revision": None,
            "input_uri": "mujoco-egl-headless://minimal-scene?note=hardware_environment_proof_not_a_data_source",
        },
    }
    manifest_path = out / "episode.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    issues = validate_episode(manifest_path)
    rgb_mean = float(rgb.mean())
    print(f"episode: {out}")
    print(f"physical GPU (EGL device): {physical_gpu}")
    print(f"rendered {args.steps} frames ({H}x{W}) in {elapsed:.2f}s = {fps:.1f} fps")
    print(f"rgb mean={rgb_mean:.1f} (all-black check: {rgb_mean < 1.0})")
    print(f"validator: {len(issues)} issue(s) -> {'PASS' if not issues else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  {issue}")
        return 1
    print("HARDWARE PROOF OK: headless EGL camera render passed the frozen validator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
