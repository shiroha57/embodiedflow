"""Real-data entry tools: single-episode quality report and episode
manifest + train/val split."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "episode_v0"
FIXTURE = ROOT / "fixtures" / "overfit"
PYTHON = sys.executable


def _run(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def test_quality_report_passes_on_sample():
    proc = _run(PYTHON, str(ROOT / "tools" / "episode_quality_report.py"), str(SAMPLE))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "-> PASS" in proc.stdout
    assert "artifact rgb:" in proc.stdout
    assert "timestamps_strictly_increasing: True" in proc.stdout


def test_quality_report_fails_on_corrupted(tmp_path: Path):
    bad = tmp_path / "bad"
    shutil.copytree(SAMPLE, bad)
    np.save(bad / "data" / "timestamps_ns.npy", np.array([0, 40, 20], dtype=np.int64))
    proc = _run(PYTHON, str(ROOT / "tools" / "episode_quality_report.py"), str(bad))
    assert proc.returncode != 0
    assert "-> FAIL" in proc.stdout
    assert "strictly increasing" in proc.stdout


def test_quality_report_json_output(tmp_path: Path):
    out = tmp_path / "report.json"
    proc = _run(
        PYTHON, str(ROOT / "tools" / "episode_quality_report.py"),
        str(SAMPLE), "--json", str(out),
    )
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["meta"]["frozen_schema_sha256"] == (
        "fd304c7fa99a32f685bd7b60ff44a81b2b7d8797187f20fe5169b71294696125"
    )


def test_episode_manifest_and_split(tmp_path: Path):
    out = tmp_path / "manifest"
    proc = _run(
        PYTHON, str(ROOT / "tools" / "episode_manifest.py"),
        str(FIXTURE), "--out", str(out),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((out / "episode_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert manifest["num_episodes_total"] == 12
    assert manifest["num_episodes_accepted"] == 12
    assert manifest["num_episodes_rejected"] == 0
    assert len(split["train"]) == 10
    assert len(split["val"]) == 2
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]) | set(split["val"]) == {
        p.name for p in FIXTURE.iterdir() if (p / "episode.json").is_file()
    }


def test_episode_manifest_rejects_bad_episode(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copytree(SAMPLE, data / "episode_a")
    shutil.copytree(SAMPLE, data / "episode_b")
    np.save(
        data / "episode_b" / "data" / "timestamps_ns.npy",
        np.array([0, 40, 20], dtype=np.int64),
    )
    out = tmp_path / "manifest"
    proc = _run(
        PYTHON, str(ROOT / "tools" / "episode_manifest.py"),
        str(data), "--out", str(out),
    )
    assert proc.returncode == 0
    manifest = json.loads((out / "episode_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_episodes_total"] == 2
    assert manifest["num_episodes_accepted"] == 1
    assert manifest["rejected"] == ["episode_b"]
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["train"] + split["val"] == ["episode_a"]
