#!/usr/bin/env python3
"""Real-data entry: scan a directory of contract episodes and write an
episode manifest + episode-level train/val split.

Usage: episode_manifest.py <data_dir> [--val-ratio 0.2] [--split-seed 0]
                            [--out DIR] [--no-validate]

Writes <out>/episode_manifest.json and <out>/split.json (default <out> =
<data_dir>/manifest). This is 4090-side bookkeeping only: contract episode
directories are read, never modified, and no new data format is defined.
Episodes failing the frozen validator are listed under "rejected" and
excluded from the split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodiedflow.contracts.validator import validate_episode
from embodiedflow.dataplane.dataset import split_episodes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "contracts" / "v0" / "schema.json"


def episode_summary(episode_dir: Path, schema_path: Path, validate: bool) -> dict:
    manifest_path = episode_dir / "episode.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = validate_episode(manifest_path, schema_path) if validate else []
    artifacts = manifest.get("artifacts", {})
    shapes: dict[str, list[int]] = {}
    for key, descriptor in (artifacts.get("observations") or {}).items():
        if isinstance(descriptor, dict):
            arr = np.load(episode_dir / descriptor["path"], mmap_mode="r", allow_pickle=False)
            shapes[key] = list(arr.shape)
    actions = artifacts.get("actions")
    if isinstance(actions, dict):
        arr = np.load(episode_dir / actions["path"], mmap_mode="r", allow_pickle=False)
        shapes["actions"] = list(arr.shape)
    return {
        "episode_id": str(manifest.get("episode_id", episode_dir.name)),
        "directory": episode_dir.name,
        "num_steps": int((manifest.get("outcome") or {}).get("num_steps", 0)),
        "shapes": shapes,
        "validation_issue_count": len(issues),
        "validation_issues": [str(i) for i in issues],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    episode_dirs = sorted(
        p for p in data_dir.iterdir() if (p / "episode.json").is_file()
    )
    if not episode_dirs:
        print(f"[manifest] no episode dirs found under {data_dir}", flush=True)
        return 2

    entries = [
        episode_summary(d, args.schema, validate=not args.no_validate)
        for d in episode_dirs
    ]
    rejected = [e for e in entries if e["validation_issue_count"]]
    accepted = [e for e in entries if not e["validation_issue_count"]]
    accepted_dirs = [data_dir / e["directory"] for e in accepted]

    train_dirs, val_dirs = split_episodes(
        accepted_dirs, val_ratio=args.val_ratio, seed=args.split_seed
    )
    split = {
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "train": [d.name for d in train_dirs],
        "val": [d.name for d in val_dirs],
    }
    manifest = {
        "data_dir": str(data_dir),
        "num_episodes_total": len(entries),
        "num_episodes_accepted": len(accepted),
        "num_episodes_rejected": len(rejected),
        "episodes": entries,
        "rejected": [e["directory"] for e in rejected],
        "split": split,
    }

    out_dir = args.out or (data_dir / "manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[manifest] {len(entries)} episodes: {len(accepted)} accepted, "
        f"{len(rejected)} rejected -> {out_dir}/episode_manifest.json + split.json "
        f"(train {len(train_dirs)} / val {len(val_dirs)})",
        flush=True,
    )
    for episode in rejected:
        print(f"  rejected {episode['directory']}: {episode['validation_issues'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
