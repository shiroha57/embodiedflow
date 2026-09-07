"""Dataset smoke test: load the sample episode, validate, build one batch,
print the full shape/dtype/range/device report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from embodiedflow.dataplane.config import DatasetConfig
from embodiedflow.dataplane.dataset import (
    CanonicalEpisodeDataset,
    describe_batch,
    make_dataloader,
    split_episodes,
)
from embodiedflow.dataplane.stats import compute_stats

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=REPO_ROOT / "samples" / "episode_v0")
    parser.add_argument("--chunk-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    episode_dirs = [args.episodes] if (args.episodes / "episode.json").is_file() else sorted(
        p for p in args.episodes.iterdir() if (p / "episode.json").is_file()
    )
    cfg = DatasetConfig(chunk_len=args.chunk_len, deterministic_seed=0)
    dataset = CanonicalEpisodeDataset(episode_dirs, cfg)
    print(f"episodes loaded: {len(dataset.episodes)}")
    print(f"bad episodes quarantined: {len(dataset.bad_episodes)}")
    for directory, issues in dataset.bad_episodes:
        print(f"  quarantined {directory}: {[str(i) for i in issues]}")
    print(f"num samples (anchors): {len(dataset)}")
    print(f"state_dim={dataset.state_dim} action_dim={dataset.action_dim} "
          f"joint_names={dataset.joint_names}")

    norm = compute_stats(
        (dataset[i] for i in range(len(dataset))),
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        image_normalization=cfg.image_normalization,
    )
    dataset.normalize = norm
    print(f"normalization stats computed over {norm.num_samples} samples")

    loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        seed=0,
        drop_last=False,
        shuffle=False,
    )
    batch = next(iter(loader))
    device = torch.device(f"cuda:0" if args.gpu is not None else "cpu")
    if args.gpu is not None:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    print(describe_batch(batch, name=f"sample batch (chunk_len={args.chunk_len})"))

    valid_fraction = batch["padding_mask"].mean().item()
    print(f"padding_mask valid fraction: {valid_fraction:.3f}")
    if not dataset.bad_episodes:
        print("SMOKE OK: all episodes passed the frozen validator")
    return 1 if not dataset.episodes else 0


if __name__ == "__main__":
    sys.exit(main())
