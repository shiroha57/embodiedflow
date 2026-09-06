"""Typed Python representation of contract-v0.1.

The JSON Schema is normative. These dataclasses provide a dependency-light API
for producers and consumers and deliberately preserve the schema field names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


Source = Literal["direct_writer", "rosbag2"]
Clock = Literal["simulation", "ros", "system"]
Termination = Literal["success", "timeout", "safety_stop", "error", "user"]


@dataclass(frozen=True)
class Artifact:
    path: str
    dtype: str
    shape: tuple[int, ...]
    semantic: str
    encoding: str | None = None
    unit: str | None = None
    frame_id: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Artifact":
        return cls(
            path=str(value["path"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            semantic=str(value["semantic"]),
            encoding=value.get("encoding"),
            unit=value.get("unit"),
            frame_id=value.get("frame_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class CameraIntrinsics:
    matrix: tuple[float, ...]
    distortion_model: Literal["none", "plumb_bob", "rational_polynomial"]
    distortion_coefficients: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraIntrinsics":
        return cls(
            matrix=tuple(float(item) for item in value["matrix"]),
            distortion_model=value["distortion_model"],
            distortion_coefficients=tuple(float(item) for item in value["distortion_coefficients"]),
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    parent_frame: str
    child_frame: str
    matrix: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraExtrinsics":
        return cls(
            parent_frame=str(value["parent_frame"]),
            child_frame=str(value["child_frame"]),
            matrix=tuple(float(item) for item in value["matrix"]),
        )


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    frame_id: str
    width: int
    height: int
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraCalibration":
        return cls(
            name=str(value["name"]),
            frame_id=str(value["frame_id"]),
            width=int(value["width"]),
            height=int(value["height"]),
            intrinsics=CameraIntrinsics.from_dict(value["intrinsics"]),
            extrinsics=CameraExtrinsics.from_dict(value["extrinsics"]),
        )


@dataclass(frozen=True)
class EpisodeManifest:
    schema_version: Literal["0.1.0"]
    episode_id: str
    created_at: str
    source: Source
    robot: Mapping[str, Any]
    task: Mapping[str, Any]
    timebase: Mapping[str, Any]
    camera: CameraCalibration
    artifacts: Mapping[str, Any]
    outcome: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeManifest":
        return cls(
            schema_version=value["schema_version"],
            episode_id=str(value["episode_id"]),
            created_at=str(value["created_at"]),
            source=value["source"],
            robot=value["robot"],
            task=value["task"],
            timebase=value["timebase"],
            camera=CameraCalibration.from_dict(value["camera"]),
            artifacts=value["artifacts"],
            outcome=value["outcome"],
            provenance=value["provenance"],
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result

