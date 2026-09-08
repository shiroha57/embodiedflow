#!/usr/bin/env python3
"""Single-episode quality report against the frozen contract.

Usage: episode_quality_report.py <episode_dir> [--json out.json] [--schema PATH]

Runs the frozen validator, verifies the schema.sha256 declared in the
manifest against the frozen schema file (default contracts/v0/schema.json),
then reports per-artifact shape/dtype/min/max/NaN counts, timestamp
monotonicity/fps, and gripper action range vs the declared gripper limits.

Exit code 0 = episode passes all checks (schema hash, validator, ranges).
This is the per-episode gate used before real Isaac episodes enter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from embodiedflow.contracts.validator import validate_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "contracts" / "v0" / "schema.json"

OBSERVATION_KEYS = ("rgb", "depth", "joint_position", "joint_velocity", "gripper_position")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_range(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    return f"[{arr.min():.4g}, {arr.max():.4g}]"


def quality_report(episode_dir: Path, schema_path: Path) -> dict:
    report: dict = {
        "episode_dir": str(episode_dir),
        "schema_path": str(schema_path),
        "passed": False,
        "issues": [],
        "warnings": [],
        "artifacts": {},
        "meta": {},
    }
    manifest_path = episode_dir / "episode.json"
    if not manifest_path.is_file():
        report["issues"].append("episode.json not found")
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["meta"]["episode_id"] = manifest.get("episode_id")
    report["meta"]["task_name"] = (manifest.get("task") or {}).get("name")
    report["meta"]["robot"] = (manifest.get("robot") or {}).get("name")
    report["meta"]["joint_names"] = list(
        (manifest.get("robot") or {}).get("joint_names", ())
    )

    # schema hash gate: if the manifest declares a schema.sha256 it must match
    # the frozen schema file; absence is a warning (the frozen validator itself
    # is the real gate and always runs against the frozen schema).
    declared = (manifest.get("schema") or {}).get("sha256")
    actual = sha256_file(schema_path)
    report["meta"]["declared_schema_sha256"] = declared
    report["meta"]["frozen_schema_sha256"] = actual
    if declared is None:
        report["warnings"].append(
            "manifest declares no schema.sha256; validated against frozen schema file"
        )
    elif declared != actual:
        report["issues"].append(
            f"schema.sha256 mismatch: declared {declared} != frozen {actual}"
        )

    issues = validate_episode(manifest_path, schema_path)
    report["validator_issues"] = [str(i) for i in issues]
    if issues:
        report["issues"].extend(str(i) for i in issues)

    num_steps = int((manifest.get("outcome") or {}).get("num_steps", 0))
    report["meta"]["num_steps"] = num_steps

    # per-artifact statistics (mmap: no big memory cost for 256x256 episodes)
    artifacts = manifest.get("artifacts", {})
    timestamps_desc = artifacts.get("timestamps")
    if isinstance(timestamps_desc, dict):
        ts_path = episode_dir / timestamps_desc["path"]
        ts = np.load(ts_path, mmap_mode="r", allow_pickle=False)
        report["artifacts"]["timestamps"] = {
            "shape": list(ts.shape),
            "dtype": str(ts.dtype),
            "range": _fmt_range(ts),
        }
        diffs = np.diff(np.asarray(ts))
        strictly_increasing = bool((diffs > 0).all())
        report["meta"]["timestamps_strictly_increasing"] = strictly_increasing
        if ts.size >= 2:
            report["meta"]["duration_ns"] = int(ts[-1] - ts[0])
            report["meta"]["fps_estimate"] = round(
                (ts.size - 1) / max(ts[-1] - ts[0], 1) * 1e9, 2
            )
        if not strictly_increasing:
            report["issues"].append("timestamps not strictly increasing")

    observations = artifacts.get("observations") or {}
    for key in OBSERVATION_KEYS:
        descriptor = observations.get(key)
        if not isinstance(descriptor, dict):
            continue
        arr = np.load(episode_dir / descriptor["path"], mmap_mode="r", allow_pickle=False)
        nan_count = int(np.isnan(np.asarray(arr)).sum()) if arr.dtype.kind == "f" else 0
        entry = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "range": _fmt_range(arr),
            "nan_count": nan_count,
        }
        if arr.dtype.kind == "f" and nan_count:
            report["issues"].append(f"{key}: {nan_count} NaN value(s)")
        if key == "rgb":
            report["meta"]["resolution_hw"] = [int(arr.shape[1]), int(arr.shape[2])]
            if arr.dtype != np.uint8:
                report["issues"].append(f"rgb dtype {arr.dtype} != uint8")
        if key == "joint_position":
            expected = len(report["meta"]["joint_names"])
            if arr.shape[1] != expected:
                report["issues"].append(
                    f"joint_position width {arr.shape[1]} != joint_names count {expected}"
                )
        report["artifacts"][key] = entry

    actions_desc = artifacts.get("actions")
    if isinstance(actions_desc, dict):
        actions = np.load(episode_dir / actions_desc["path"], mmap_mode="r", allow_pickle=False)
        report["artifacts"]["actions"] = {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "range": _fmt_range(actions),
            "nan_count": int(np.isnan(np.asarray(actions)).sum()),
        }
        robot = manifest.get("robot") or {}
        gripper = robot.get("gripper") or {}
        closed, opened = gripper.get("closed_position"), gripper.get("open_position")
        if isinstance(closed, (int, float)) and isinstance(opened, (int, float)):
            low, high = sorted((float(closed), float(opened)))
            gripper_col = np.asarray(actions)[:, -1]
            tol = 1e-3
            out = int(((gripper_col < low - tol) | (gripper_col > high + tol)).sum())
            report["meta"]["gripper_action_range"] = [low, high]
            report["meta"]["gripper_action_out_of_range"] = out
            if out:
                report["issues"].append(
                    f"{out} gripper action(s) outside declared range [{low}, {high}]"
                )

    report["passed"] = not report["issues"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = quality_report(args.episode_dir, args.schema)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"== episode quality report: {args.episode_dir} -> {verdict} ==")
    for key, value in report["meta"].items():
        print(f"  {key}: {value}")
    for key, value in report["artifacts"].items():
        extra = f" nan={value['nan_count']}" if "nan_count" in value else ""
        print(f"  artifact {key}: shape={value['shape']} dtype={value['dtype']} "
              f"range={value['range']}{extra}")
    for issue in report["issues"]:
        print(f"  ISSUE: {issue}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
