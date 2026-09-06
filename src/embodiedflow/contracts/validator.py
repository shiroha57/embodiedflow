"""Schema and artifact validator for canonical EmbodiedFlow episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _default_schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "contracts" / "v0" / "schema.json"


def _walk_artifacts(manifest: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    artifacts = manifest.get("artifacts", {})
    timestamps = artifacts.get("timestamps")
    if isinstance(timestamps, Mapping):
        yield "artifacts.timestamps", timestamps
    observations = artifacts.get("observations", {})
    if isinstance(observations, Mapping):
        for name, descriptor in observations.items():
            if isinstance(descriptor, Mapping):
                yield f"artifacts.observations.{name}", descriptor
    actions = artifacts.get("actions")
    if isinstance(actions, Mapping):
        yield "artifacts.actions", actions


def _safe_artifact_path(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _semantic_issues(manifest: Mapping[str, Any], root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    loaded: dict[str, np.ndarray] = {}
    num_steps = manifest.get("outcome", {}).get("num_steps")

    for logical_path, descriptor in _walk_artifacts(manifest):
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            continue
        path = _safe_artifact_path(root, relative)
        if path is None:
            issues.append(ValidationIssue(f"{logical_path}.path", "escapes the episode directory"))
            continue
        if not path.is_file():
            issues.append(ValidationIssue(f"{logical_path}.path", f"missing file: {relative}"))
            continue
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:  # malformed inputs are reported, not raised
            issues.append(ValidationIssue(logical_path, f"cannot load NumPy artifact: {exc}"))
            continue
        loaded[logical_path] = array
        declared_dtype = descriptor.get("dtype")
        if str(array.dtype) != declared_dtype:
            issues.append(
                ValidationIssue(f"{logical_path}.dtype", f"declares {declared_dtype}, file is {array.dtype}")
            )
        declared_shape = tuple(descriptor.get("shape", ()))
        if array.shape != declared_shape:
            issues.append(
                ValidationIssue(f"{logical_path}.shape", f"declares {declared_shape}, file is {array.shape}")
            )
        if isinstance(num_steps, int) and array.ndim and array.shape[0] != num_steps:
            issues.append(
                ValidationIssue(logical_path, f"first dimension {array.shape[0]} != num_steps {num_steps}")
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            issues.append(ValidationIssue(logical_path, "contains NaN or infinity"))

    timestamps = loaded.get("artifacts.timestamps")
    if timestamps is not None:
        flat = np.asarray(timestamps).reshape(-1)
        if flat.size > 1 and not np.all(np.diff(flat) > 0):
            issues.append(ValidationIssue("artifacts.timestamps", "timestamps must be strictly increasing"))
        if flat.size:
            duration = manifest.get("outcome", {}).get("duration_ns")
            actual_duration = int(flat[-1]) - int(flat[0])
            if isinstance(duration, int) and duration != actual_duration:
                issues.append(
                    ValidationIssue("outcome.duration_ns", f"declares {duration}, timestamps imply {actual_duration}")
                )

    camera = manifest.get("camera", {})
    rgb = loaded.get("artifacts.observations.rgb")
    depth = loaded.get("artifacts.observations.depth")
    expected_hw = (camera.get("height"), camera.get("width"))
    if rgb is not None and (rgb.ndim != 4 or rgb.shape[1:3] != expected_hw or rgb.shape[-1] != 3):
        issues.append(ValidationIssue("artifacts.observations.rgb", "must have shape [T,height,width,3]"))
    if depth is not None and (depth.ndim != 3 or depth.shape[1:3] != expected_hw):
        issues.append(ValidationIssue("artifacts.observations.depth", "must have shape [T,height,width]"))

    joint_count = len(manifest.get("robot", {}).get("joint_names", ()))
    joint_position = loaded.get("artifacts.observations.joint_position")
    joint_velocity = loaded.get("artifacts.observations.joint_velocity")
    gripper = loaded.get("artifacts.observations.gripper_position")
    actions = loaded.get("artifacts.actions")
    if joint_position is not None and (joint_position.ndim != 2 or joint_position.shape[1] != joint_count):
        issues.append(ValidationIssue("artifacts.observations.joint_position", "width must equal joint_names"))
    if joint_velocity is not None and (joint_velocity.ndim != 2 or joint_velocity.shape[1] != joint_count):
        issues.append(ValidationIssue("artifacts.observations.joint_velocity", "width must equal joint_names"))
    if gripper is not None and (gripper.ndim != 2 or gripper.shape[1] != 1):
        issues.append(ValidationIssue("artifacts.observations.gripper_position", "must have shape [T,1]"))
    if actions is not None and (actions.ndim != 2 or actions.shape[1] != joint_count + 1):
        issues.append(ValidationIssue("artifacts.actions", "width must equal joint_names plus one gripper command"))

    video = manifest.get("artifacts", {}).get("video")
    if isinstance(video, Mapping):
        video_path = _safe_artifact_path(root, str(video.get("path", "")))
        if video_path is None or not video_path.is_file():
            issues.append(ValidationIssue("artifacts.video.path", "missing or unsafe video path"))
        if video.get("frame_count") != num_steps:
            issues.append(ValidationIssue("artifacts.video.frame_count", "must equal outcome.num_steps"))
    return issues


def validate_episode(
    manifest_path: str | Path,
    schema_path: str | Path | None = None,
) -> list[ValidationIssue]:
    """Return every schema and semantic issue; an empty list means valid."""

    manifest_path = Path(manifest_path).resolve()
    schema_path = Path(schema_path).resolve() if schema_path else _default_schema_path()
    with schema_path.open("r", encoding="utf-8") as stream:
        schema = json.load(stream)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    issues = [
        ValidationIssue(
            ".".join(str(part) for part in error.absolute_path) or "$",
            error.message,
        )
        for error in sorted(validator.iter_errors(manifest), key=lambda item: list(item.absolute_path))
    ]
    if not issues:
        issues.extend(_semantic_issues(manifest, manifest_path.parent))
    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="episode.json files to validate")
    parser.add_argument("--schema", type=Path, default=None, help="override schema path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results: dict[str, list[dict[str, str]]] = {}
    for episode in args.episodes:
        try:
            issues = validate_episode(episode, args.schema)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues = [ValidationIssue("$", str(exc))]
        results[str(episode)] = [{"path": item.path, "message": item.message} for item in issues]

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for episode, issues in results.items():
            if not issues:
                print(f"PASS {episode}")
            else:
                print(f"FAIL {episode}")
                for issue in issues:
                    print(f"  {issue['path']}: {issue['message']}")
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

