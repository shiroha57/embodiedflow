"""GPU selection helpers: never touch GPUs other people are using.

Every run re-queries nvidia-smi. A GPU is considered free when its used
memory and utilization are both below the thresholds.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int


def query_gpus() -> list[GpuInfo]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    infos = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total, used, util = parts
        infos.append(GpuInfo(int(index), name, int(total), int(used), int(util)))
    return infos


def is_free(info: GpuInfo, max_used_mb: int = 6000, max_util_pct: int = 25) -> bool:
    return info.memory_used_mb <= max_used_mb and info.utilization_pct <= max_util_pct


def pick_free_gpus(
    count: int,
    prefer: list[int] | None = None,
    max_used_mb: int = 6000,
    max_util_pct: int = 25,
) -> list[int]:
    """Pick `count` free GPU ids; `prefer` order is honored when possible."""
    infos = query_gpus()
    free = [info.index for info in infos if is_free(info, max_used_mb, max_util_pct)]
    if prefer:
        ordered = [gpu for gpu in prefer if gpu in free] + [gpu for gpu in free if gpu not in prefer]
        free = ordered
    if len(free) < count:
        raise RuntimeError(
            f"only {len(free)} free GPU(s) found ({free}), need {count}; "
            "re-check nvidia-smi or lower thresholds"
        )
    return free[:count]


def pick_free_gpus_with_retry(
    count: int,
    attempts: int = 30,
    delay_s: float = 10.0,
    **kwargs,
) -> list[int]:
    """pick_free_gpus with bounded retry; utilization samples are noisy on a
    shared box, so a transient spike should not abort a run."""
    last: RuntimeError | None = None
    for attempt in range(attempts):
        try:
            return pick_free_gpus(count, **kwargs)
        except RuntimeError as exc:
            last = exc
            print(f"[gpu] attempt {attempt + 1}/{attempts}: {exc}", flush=True)
            if attempt + 1 < attempts:
                time.sleep(delay_s)
    assert last is not None
    raise last


def format_gpu_table(infos: list[GpuInfo] | None = None) -> str:
    infos = infos if infos is not None else query_gpus()
    lines = ["GPU  memory_used  util  free?"]
    for info in infos:
        lines.append(
            f"{info.index:>3}  {info.memory_used_mb:>8} MB  {info.utilization_pct:>3}%   "
            f"{'yes' if is_free(info) else 'NO '}"
        )
    return "\n".join(lines)
