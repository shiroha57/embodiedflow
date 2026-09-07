#!/usr/bin/env python3
"""Map EGL device indices to physical nvidia-smi GPU indices on 4090 hosts.

Why this exists: MuJoCo EGL rendering ignores CUDA_VISIBLE_DEVICES. A render
worker pinned to CUDA device k must also set MUJOCO_EGL_DEVICE_ID=<physical
GPU index>, otherwise all workers render on the first EGL device.

Uses EGL_NV_device_cuda (eglQueryDeviceAttribEXT, 0x323A) which NVIDIA
drivers expose per device. Read-only; no context creation needed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from ctypes import POINTER, c_char_p, c_int, c_void_p

os.environ.setdefault("MUJOCO_GL", "egl")

from mujoco.egl import egl_ext as EGL  # noqa: E402
from OpenGL import error as GLerror  # noqa: E402


EGL_CUDA_DEVICE_NV = 0x323A


def _query_device_attrib(device: c_void_p, attrib: int) -> int:
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [c_char_p]
    egl.eglGetProcAddress.restype = c_void_p
    proc = egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not proc:
        raise RuntimeError("eglQueryDeviceAttribEXT unavailable")
    query = ctypes.CFUNCTYPE(c_int, c_void_p, c_int, POINTER(c_int))(proc)
    value = c_int(-1)
    if not query(device, attrib, ctypes.byref(value)):
        raise RuntimeError(f"eglQueryDeviceAttribEXT failed for attrib {attrib:#x}")
    return value.value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable mapping")
    args = parser.parse_args()

    devices = EGL.eglQueryDevicesEXT()
    mapping: list[dict] = []
    for index, device in enumerate(devices):
        display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
        try:
            ok = EGL.eglInitialize(display, None, None)
        except GLerror.GLError:
            ok = False
        if not ok or EGL.eglGetError() != EGL.EGL_SUCCESS:
            mapping.append({"egl_device": index, "gpu": None, "note": "non-NVIDIA or init failed"})
            continue
        cuda_device = _query_device_attrib(device, EGL_CUDA_DEVICE_NV)
        mapping.append({"egl_device": index, "gpu": cuda_device})

    if args.json:
        print(json.dumps(mapping, indent=2))
    else:
        print("EGL device -> nvidia-smi GPU (physical):")
        for entry in mapping:
            if entry["gpu"] is None:
                print(f"  egl {entry['egl_device']:>2}: {entry['note']}")
            else:
                print(f"  egl {entry['egl_device']:>2}: GPU {entry['gpu']}  (set MUJOCO_EGL_DEVICE_ID={entry['gpu']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
