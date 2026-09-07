# 4090 计算与数据平面环境审计

审计时间：2026-09-06 17:15–17:25（Asia/Shanghai）
审计人：Claude（EmbodiedFlow 4090 平面）
状态：**contract-v0.1 尚未到达**，本文仅覆盖硬件/软件环境，不含任何数据 schema 推断。

---

## 1. GPU

| 项 | 值 |
|---|---|
| 型号 | NVIDIA GeForce RTX 4090 D × 8（24 GB GDDR6X，sm_89） |
| 驱动 | 580.173.02（支持 CUDA 13.0 runtime） |
| nvcc 默认 | 12.4.99（/usr/local/cuda → 12.4） |
| 已装 CUDA | 11.3 / 11.6 / 11.8 / 12.4 / 12.8 / 12.9 / 13.0（/usr/local/） |
| Persistence Mode | Disabled（无 root 改不了，冷启动略慢，不影响正确性） |
| MIG | 不支持（消费卡） |
| ECC | 无（消费卡） |

### 占用快照（审计时）

| GPU | 显存 | 利用率 | 备注 |
|---|---|---|---|
| 0 | 3.9 GB | 15% | 两个小进程（1.9G + 1.9G），可用 |
| 1 | 22.6 GB | 45% | 他人在跑大任务，**避免使用** |
| 2 | 0.9 GB | 5% | 基本空闲（866MB 为共享监控进程 PID 52372） |
| 3 | 4.7 GB | 16% | 一个 3.7G 进程，可用 |
| 4 | 0.9 GB | 3% | 基本空闲 |
| 5 | 4.7 GB | 4% | 一个 3.7G 进程，可用 |
| 6 | 0.9 GB | 4% | 基本空闲 |
| 7 | 23.0 GB | 19% | 他人在跑大任务，**避免使用** |

建议四卡组：**{2, 4, 6} + {0 或 3 或 5}**（每次运行前用 nvidia-smi 复核）。
GPU 选择必须做成 CLI 参数，不做硬编码。

## 2. CPU / 内存 / NUMA

| 项 | 值 |
|---|---|
| CPU | 2× Intel Xeon Silver 4310 @ 2.10GHz，12 核/颗，48 线程 |
| NUMA | 2 节点：node0 = CPU 0-11,24-35（挂 GPU0-3）；node1 = CPU 12-23,36-47（挂 GPU4-7） |
| 内存 | 125 GB 总量，审计时 81 GB 已用 / 38 GB 可用 |
| Swap | **1 GB 已 100% 用满** —— 内存压力大，避免超大内存分配 |
| 负载 | 审计时 load average ≈ 56（48 线程）—— **CPU 严重超卖**，物理仿真密集的任务吞吐会受影响 |
| /dev/shm | 63 GB，充足 |

## 3. GPU 互联

- 4090 无 NVLink；`torch.cuda.can_device_access_peer = False`（所有 GPU 对）。
- 实测 P2P 带宽（256 MB tensor copy，经 host staging）：
  - 跨 NUMA（GPU2↔6，SYS）：9.4–11.4 GB/s
  - 同 NUMA（GPU2↔3，PIX）：7.6 GB/s
- 拓扑：GPU0-3 在 NUMA0 经 PCIe 桥（PXB/PIX），GPU4-7 在 NUMA1；跨组走 SYS。
- **对 DDP 的启示**：all-reduce 走 host 内存，通信带宽有限；训练扩展效率不要乐观预期，
  模型较小时瓶颈可能在通信。报告扩展效率时需附真实曲线（任务 6 要求）。
- PCIe 链路代数/宽度：`lspci -vvv` LnkSta 需 root，**未验证**（待 sudo 补测）。

## 4. 渲染栈（headless 相机相关）

| 项 | 值 |
|---|---|
| Vulkan | 1.3.204，NVIDIA ICD 存在（/usr/share/vulkan/icd.d/nvidia_icd.json），无 DISPLAY 可跑（surface 查询被跳过属正常） |
| EGL | **实测可用**：NVIDIA vendor、EGL 1.5、8 configs，pbuffer context+surface 创建成功（ctypes 探测，eglGetError=EGL_SUCCESS） |
| Mesa | 23.2.1（libEGL_mesa 与 libEGL_nvidia 并存，NVIDIA 为默认 ICD） |
| X11 | 无 DISPLAY（纯 headless 服务器），一切渲染走 EGL/Vulkan offscreen |

### 实测渲染验证（2026-09-06，GPU 2）

- `MUJOCO_GL=egl` + MuJoCo 2.3.7 `Renderer` offscreen：默认相机 mean=167.6（正常），
  **~1304 fps @ 480×640**（含 mj_step）。
- 原始 GL 路径验证：GL 4.6 NVIDIA 580.173.02，FBO complete，清屏/读回像素正确。
- `MUJOCO_GL=osmesa` 也可用（CPU 后备）。
- 坑：自定义命名相机的 `xyaxes` 若数值不当会渲染全黑（无报错）；smoke test 一律用
  默认相机或经校验的相机轴。该问题属相机几何，非平台问题。

### EGL 设备 ↔ 物理 GPU 映射（关键）

通过 `EGL_NV_device_cuda`（eglQueryDeviceAttribEXT, 0x323A）实测：
**EGL 设备索引 i = CUDA 设备 i = nvidia-smi GPU i**（8 张 NVIDIA 卡一一对应，
另有 2 个非 NVIDIA 设备枚举失败属正常）。

**对一 GPU 一进程生成器的硬性要求**：EGL 枚举**不受 CUDA_VISIBLE_DEVICES 影响**。
worker 若只设 `CUDA_VISIBLE_DEVICES=k`，渲染会全部落在物理 GPU 0。
必须同时设置 `MUJOCO_EGL_DEVICE_ID=<物理卡号>`，其中
物理卡号 = worker 可见设备列表的第 local_rank 项。

结论：headless 相机 smoke test 具备渲染条件（EGL pbuffer 已实测）。

## 5. Docker

| 项 | 值 |
|---|---|
| 客户端 | 28.5.1（含 buildx、compose v2.40.1） |
| nvidia-container-toolkit | 1.19.0 已装 |
| daemon 权限 | **ljt 不在 docker 组**，socket 连接 permission denied；sudo 需要密码 |

影响：如需容器化，需管理员把 ljt 加入 docker 组，或全程 sudo 交互。当前建议以
conda 环境为主，容器方案作为备选。

## 6. 磁盘

| 挂载点 | 设备/FS | 容量 | 可用 | 备注 |
|---|---|---|---|---|
| /data5 | nvme1n1 ext4 | 7.0 T | **703 G（90% 满）** | 本项目工作盘 |
| / (root) | nvme3n1p2 ext4 | 7.0 T | 392 G（95% 满） | 别往里写大文件 |
| /dev/shm | tmpfs | 63 G | 63 G | dataloader 可用 |

- 实测顺序写（dd direct，2 GB）：**1.8 GB/s**。
- 随机 IOPS 未测（无 fio；后续可用 hdparm/aio 补测）。
- **磁盘紧张**：数据集（预计 TB 级）需先做容量预算；考虑把生成数据放到
  /data5 与 /data1 中更空闲的位置，并保留 100G 以上余量。落盘前先 df 复核。

## 7. Python / 深度学习环境

| 环境 | Python | torch |
|---|---|---|
| 系统 /usr/bin/python3 | 3.10.12 | 2.7.1+cu118 |
| conda: awarevln-eval（/data5/ljt_data/conda_envs/） | — | 2.7.1+cu118 |
| conda: dct（/data5/ljt_data/conda_envs/） | — | 2.7.1+cu118 |

- torch 2.7.1+cu118 与驱动 580 兼容（已实测 CUDA 可用）。
- 建议为本项目新建独立 conda env（`work/envs/` 或 conda_envs/ 下），避免污染他人环境。

## 8. 可复用资产（不属本仓库，只读参考）

- ACT 参考实现：/data5/ljt_data/act（github.com/tonyzhaozh/act，torch 版，含
  imitate_episodes.py / policy.py / sim_env.py）
- 机器人仿真栈（/data1/ljt_data/）：LIBERO、SimplerEnv、ManiSkill2_real2sim、
  robomimic、robosuite、bridge_data_v2
- LeRobot：未安装（任务 8 需要时 pip 安装，网络可达 pypi/github）

## 9. 网络

- github.com、pypi.org 可达（HTTP 200）。
- 未发现 5090 机器的挂载/SSH 配置；contract 交付方式待与 5090 侧确认。

## 10. 风险清单

1. **CPU 超卖（load≈56/48）+ swap 满**：数据生成基准测试须记录 CPU 侧扰动；必要时与
   其他用户协调时段，或减小并发。
2. **磁盘仅 703G 可用**：数据集容量预算必须先行。
3. **P2P 关闭 + 无 NVLink**：DDP 通信走 host，扩展效率需实测曲线支撑。
4. **Docker 无权限**：容器化路径受阻，默认走 conda。
5. **1/7 号卡被他人占用**：GPU 集合每次运行前动态确认。
6. contract-v0.1 未到：**禁止猜测 schema**，数据生成/训练代码不得先行定义格式。
