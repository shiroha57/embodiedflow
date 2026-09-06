from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from embodiedflow.contracts.types import EpisodeManifest
from embodiedflow.contracts.validator import validate_episode


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "episode_v0" / "episode.json"
SCHEMA = ROOT / "contracts" / "v0" / "schema.json"


def test_frozen_schema_digest() -> None:
    expected = (SCHEMA.parent / "schema.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(SCHEMA.read_bytes()).hexdigest() == expected


def test_sample_episode_is_valid() -> None:
    assert validate_episode(SAMPLE, SCHEMA) == []


def test_python_type_loads_sample() -> None:
    manifest = EpisodeManifest.from_dict(json.loads(SAMPLE.read_text(encoding="utf-8")))
    assert manifest.schema_version == "0.1.0"
    assert manifest.camera.width == 4
    assert manifest.robot["joint_names"][0] == "shoulder_pan_joint"


def test_validator_catches_non_monotonic_timestamps(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    data = episode / "data"
    data.mkdir(parents=True)
    manifest = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for artifact in (ROOT / "samples" / "episode_v0" / "data").iterdir():
        target = data / artifact.name
        target.write_bytes(artifact.read_bytes())
    np.save(data / "timestamps_ns.npy", np.array([0, 50_000_000, 40_000_000], dtype=np.int64))
    (episode / "episode.json").write_text(json.dumps(manifest), encoding="utf-8")
    issues = validate_episode(episode / "episode.json", SCHEMA)
    assert any("strictly increasing" in issue.message for issue in issues)

