# RTX 5090 environment audit

Audit timestamp: `2026-09-06T16:48:57+08:00`  
Host: `Baiyang-5090-1`  
Scope: read-only host inspection. No driver, CUDA, Vulkan, Docker, ROS, or OS
package was installed, removed, or modified during this audit.

## Summary

| Component | Observed version/state | Evidence |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble), x86_64 | `/etc/os-release`, `uname -a` |
| Kernel | `6.14.0-37-generic` | `uname -a` |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 | `nvidia-smi --query-gpu=...` |
| NVIDIA driver | `590.48.01`, open kernel module | `nvidia-smi`, `/proc/driver/nvidia/version`, `modinfo nvidia` |
| CUDA driver compatibility | CUDA `13.1` reported by `nvidia-smi` | This is the driver's maximum compatible CUDA level, not a Toolkit install |
| CUDA Toolkit | Not found on host (`nvcc` absent; no `/usr/local/cuda*`) | `command -v nvcc`, filesystem check |
| Vulkan loader/tools | loader `1.3.275`; `vulkan-tools` `1.3.275.0+dfsg1-1` | `vulkaninfo --summary`, `dpkg-query` |
| NVIDIA Vulkan device | API `1.4.325`, driver `590.48.01`, RTX 5090 | `vulkaninfo --summary` with `DISPLAY` unset |
| Docker Engine | client/server `29.6.1`, API `1.55` | `docker version` |
| containerd / runc | containerd `2.2.6`; runc `1.3.6` | `docker version` |
| NVIDIA Container Toolkit | `1.19.0`; Docker `nvidia` runtime registered | `nvidia-container-cli`, `nvidia-ctk`, `docker info` |
| Isaac Sim | Host package absent; local image `nvcr.io/nvidia/isaac-sim:6.0.1` exists | host `pip show`, filesystem search, `docker image inspect` |
| Isaac Lab | Not found on host or in the inspected Isaac Sim base image | host `pip show`, filesystem/image inspection |
| ROS 2 | Jazzy installed, but the audit shell was initially unsourced | `/opt/ros/jazzy`, `dpkg-query`; CLI works after sourcing setup |
| rclpy | `7.1.11` | `ros2 pkg xml rclpy`, Debian package version |
| rosbag2 | `0.26.11` | `ros2 pkg xml rosbag2_py`, Debian package version |
| Python | `3.12.3` | `python3 --version` |

## Headless and display observations

Both `DISPLAY` and `WAYLAND_DISPLAY` were unset in the audit shell. Vulkan still
enumerated the discrete NVIDIA GPU and advertised `VK_EXT_headless_surface`,
which is the relevant baseline for headless off-screen rendering.

The machine is not currently free of display processes: `nvidia-smi` reported
`Disp.A On`, an Xorg process using 128 MiB, and GNOME Shell using 17 MiB. This
does not prevent an explicit `--headless` Isaac launch, but acceptance tests
must keep `DISPLAY` unset and must not depend on X11. No attempt was made to
stop or reconfigure these host services.

At audit time another Python process (`.venv-server/bin/python`) used about
8730 MiB of GPU memory. Long stability runs should record concurrent GPU use
and fail clearly on out-of-memory rather than attributing it to the renderer.

## GPU and driver detail

```text
GPU 0: NVIDIA GeForce RTX 5090
UUID: GPU-a81289d7-32e0-f05e-2420-c4ac18e12991
PCI bus: 00000000:01:00.0
VRAM: 32607 MiB
Compute capability: 12.0
Driver: 590.48.01
Kernel module: NVIDIA Open Kernel Module 590.48.01
```

The host has `libcuda.so` for both x86-64 and i386. It does not have a CUDA
compiler/toolkit in the standard path. CUDA user-space libraries should
therefore come from the pinned container image; no host Toolkit is required for
the planned containerized run.

## Vulkan detail

The loader reports instance API `1.3.275`. With no display environment it
enumerates:

- GPU 0: RTX 5090, NVIDIA proprietary driver, Vulkan API `1.4.325`, conformance `1.4.3.0`.
- GPU 1: llvmpipe software renderer, Mesa `25.2.8`.

`/usr/share/vulkan/icd.d/nvidia_icd.json` points to
`libGLX_nvidia.so.0` and declares API `1.4.325`. Container launches must retain
the image's `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json` so Isaac does
not accidentally select llvmpipe.

## Docker and Isaac

Docker uses `overlayfs`; the default runtime is `runc`, and registered runtimes
include `nvidia`. The existing Isaac image was created on 2026-06-16 and sets:

```text
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
MIN_DRIVER_VERSION=570.169
```

The installed driver `590.48.01` exceeds the image-declared minimum. This is
only a static compatibility check; a GPU-enabled container render test is still
required before claiming Isaac acceptance.

Isaac Lab is absent and must be added in the project container layer at a
pinned revision compatible with Isaac Sim 6.0.1. That is an application image
change, not a host-driver change.

## ROS 2

ROS 2 Jazzy packages are under `/opt/ros/jazzy`. A fresh non-interactive shell
does not expose `ros2` until it runs:

```bash
source /opt/ros/jazzy/setup.bash
```

Audited package versions:

```text
ros-jazzy-ros-base  0.11.0-1noble.20260616.084325
ros-jazzy-rclpy     7.1.11-1noble.20260615.133206
ros-jazzy-rosbag2   0.26.11-1noble.20260616.084050
```

## Readiness verdict

- GPU, driver, Vulkan device enumeration, Docker, and the NVIDIA container
  runtime are present.
- The driver is newer than the minimum declared by the local Isaac Sim 6.0.1 image.
- Host CUDA Toolkit absence is expected for a container-first design.
- Isaac Lab is a confirmed missing dependency and will be pinned in the
  project image rather than installed on the host.
- No Isaac off-screen frame or WebRTC path has yet been accepted at this audit
  stage. Those results must come from subsequent scripts and logs.

## Commands used

The audit used only read-only queries: `date`, `uname`, `cat /etc/os-release`,
`nvidia-smi`, `cat /proc/driver/nvidia/version`, `lsmod`, `modinfo`, `nvcc
--version`, `vulkaninfo --summary`, `ldconfig -p`, `docker version`, `docker
info`, `docker image inspect`, `docker images`, `nvidia-container-cli
--version`, `nvidia-ctk --version`, `dpkg-query`, `ros2 pkg xml`, host `pip
show`, and bounded filesystem searches for Isaac installations.

