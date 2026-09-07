#!/usr/bin/env python3
"""1/2/4-GPU pipeline benchmark driver.

Runs the DDP entry point with world_size 1, 2, 4 on freshly picked free
GPUs, collects the raw per-rank JSONs, and writes a summary with speedup
and scaling efficiency. All results are labeled as pipeline benchmarks on
the synthetic fixture — NOT final model results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from embodiedflow.dataplane.gpu import pick_free_gpus_with_retry

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "benchmark"


def run_world(n_gpus: int, label: str, warmup: int, measure: int) -> list[int]:
    gpus = pick_free_gpus_with_retry(n_gpus)
    env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)}
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(n_gpus),
        "-m",
        "embodiedflow.dataplane.ddp_train",
        "--gpus-physical",
        ",".join(str(g) for g in gpus),
        "--label",
        label,
        "--report-root",
        str(REPORTS),
        "--warmup-steps",
        str(warmup),
        "--measure-steps",
        str(measure),
    ]
    print(f"[benchmark] world_size={n_gpus} physical GPUs={gpus}", flush=True)
    subprocess.run(cmd, env={**__import__("os").environ, **env}, check=True)
    return gpus


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--worlds", type=str, default="1,2,4")
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    worlds = [int(w) for w in args.worlds.split(",")]
    label = args.label or datetime.now(timezone.utc).strftime("bench-%Y%m%d-%H%M%S")
    out_dir = REPORTS / label
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for n in worlds:
        gpus = run_world(n, label, args.warmup_steps, args.measure_steps)
        raw = json.loads((out_dir / f"ddp-world-{n}.json").read_text(encoding="utf-8"))
        results[n] = {"raw": raw, "physical_gpus": gpus}

    base_world = min(results)
    base = results[base_world]["raw"]
    base_samples = sum(r["samples_per_s"] for r in base["per_rank"])
    base_steps = base["per_rank"][0]["steps_per_s"]

    summary = {
        "label": label,
        "note": "pipeline benchmark on synthetic fixture; NOT final model results",
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "base_world": base_world,
        "per_world": {},
    }
    for n in worlds:
        raw = results[n]["raw"]
        global_samples = sum(r["samples_per_s"] for r in raw["per_rank"])
        peak_mb = max(r["peak_memory_mb"] for r in raw["per_rank"])
        speedup = global_samples / base_samples if base_samples else 0.0
        summary["per_world"][str(n)] = {
            "physical_gpus": results[n]["physical_gpus"],
            "global_samples_per_s": global_samples,
            "steps_per_s_per_rank": raw["per_rank"][0]["steps_per_s"],
            "step_time_ms_mean": raw["per_rank"][0]["step_time_ms_mean"],
            "data_wait_ms_mean": raw["per_rank"][0]["data_wait_ms_mean"],
            "gpu_utilization_mean": [r["gpu_utilization_mean"] for r in raw["per_rank"]],
            "peak_memory_mb_per_rank": [r["peak_memory_mb"] for r in raw["per_rank"]],
            "speedup_vs_1gpu": speedup,
            "scaling_efficiency": speedup / n,
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["per_world"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
