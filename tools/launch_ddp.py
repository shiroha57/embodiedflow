#!/usr/bin/env python3
"""Launch the DDP entry point on N free GPUs.

Usage: launch_ddp.py <n_gpus> [extra args for ddp_train ...]

Re-queries nvidia-smi, picks N free GPUs, sets CUDA_VISIBLE_DEVICES, and
execs torch.distributed.run with one process per GPU. Physical GPU ids are
recorded via --gpus-physical.
"""

from __future__ import annotations

import os
import sys

from embodiedflow.dataplane.gpu import pick_free_gpus_with_retry


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    n_gpus = int(sys.argv[1])
    rest = sys.argv[2:]
    gpus = pick_free_gpus_with_retry(n_gpus)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    physical = ",".join(str(g) for g in gpus)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(n_gpus),
        "-m",
        "embodiedflow.dataplane.ddp_train",
        "--gpus-physical",
        physical,
        *rest,
    ]
    print(f"[launch_ddp] free GPUs={gpus} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
          file=sys.stderr)
    os.execvpe(sys.executable, cmd, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
