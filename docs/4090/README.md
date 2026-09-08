# EmbodiedFlow 4090 数据与训练平面（dev/4090-data-plane）

本分支在 4×4090 主机上实现 contract-v0.1 的数据与训练平面。
**诚实声明**：

- 已完成的所有训练/基准均基于 **32×32 合成 fixture**，属于
  **synthetic pipeline benchmark**（管线吞吐验收），**不是模型性能结果**；
- **real Isaac training benchmark**（真实模型性能）在真实 Isaac episode
  到达并完成 10-20 episode 过拟合验收后才会运行，指标要求见下文；
- headless EGL 相机渲染仅为硬件环境证明，不是正式数据源，也不是 Isaac Sim 验收；
- 4090 只做离线验证，闭环成功率由 5090 测量。

## 环境

- conda env: `/data5/ljt_data/conda_envs/embodiedflow`（torch 2.7.1+cu118, mujoco 2.3.7, pynvml）
- `export PYTHONPATH=src`，`export PYTHONNOUSERSITE=1`（避免 ~/.local 的 glfw 泄漏）
- 新依赖 `pynvml`（`torch.cuda.utilization` 需要）：
  `/data5/ljt_data/conda_envs/embodiedflow/bin/pip install pynvml`
- GPU 选择：每次运行重新查询 nvidia-smi，只使用空闲卡（占用≤6GB 且利用率≤25%），
  多卡入口带重试等待（`pick_free_gpus_with_retry`，30 次 × 10s）。
- EGL 渲染（仅硬件证明工具）：EGL 忽略 CUDA_VISIBLE_DEVICES，且本机
  **EGL device i == 物理 GPU i**，渲染 worker 必须设 `MUJOCO_EGL_DEVICE_ID=<物理 GPU id>`；
  训练不需要设置它。自定义 camera xyaxes 可能静默全黑，用默认相机。

## 代码阅读顺序

1. [contracts/v0/schema.json](../contracts/v0/schema.json) — 冻结数据契约（勿改）
2. [src/embodiedflow/contracts/validator.py](../../src/embodiedflow/contracts/validator.py) — 冻结 validator
3. [src/embodiedflow/dataplane/config.py](../../src/embodiedflow/dataplane/config.py) — Dataset/Model/Train 配置
4. [src/embodiedflow/dataplane/dataset.py](../../src/embodiedflow/dataplane/dataset.py) — CanonicalEpisodeDataset：validate-before-load、episode 级 split、action chunk + padding mask、normalize、DataLoader
5. [src/embodiedflow/dataplane/stats.py](../../src/embodiedflow/dataplane/stats.py) — z-score NormStats（padding 保持 0）
6. [src/embodiedflow/dataplane/model.py](../../src/embodiedflow/dataplane/model.py) — ChunkedActionTransformerMinimal（**不是 ACT CVAE**，instruction_conditioning 声明为 disabled；pool_size 支持 256×256 大图）
7. [src/embodiedflow/dataplane/train.py](../../src/embodiedflow/dataplane/train.py) — 单卡训练：AMP/clip/checkpoint/resume（batch 级重放保证逐位一致）；rgb_shape 自动跟随数据
8. [src/embodiedflow/dataplane/checkpoint.py](../../src/embodiedflow/dataplane/checkpoint.py) — 交付协议：model.pt + model_manifest.json + normalization.json + config.yaml + metrics.json
9. [src/embodiedflow/dataplane/ddp_train.py](../../src/embodiedflow/dataplane/ddp_train.py) — torchrun 1 卡 1 进程，DistributedSampler + set_epoch，rank0 独占写，--profile 输出数据/计算/通信占比
10. [src/embodiedflow/dataplane/gpu.py](../../src/embodiedflow/dataplane/gpu.py) — nvidia-smi 查询与空闲卡选择
11. [tests/](../../tests/) — 数据集异常隔离、chunk 跨 episode 边界、split 无泄漏、确定性、resume 逐位一致、overfit、checkpoint 协议、工具入口
12. [tools/](../../tools/) — make_overfit_fixture.py（合成 fixture）、episode_quality_report.py（单 episode 质量报告）、episode_manifest.py（真实数据 manifest+split）、headless_camera_smoke.py（EGL 硬件证明）、launch_ddp.py、run_benchmark.py

## 一键命令（仓库根目录）

| 命令 | 内容 |
|---|---|
| `make test` | 全量 pytest（CPU 约 1-2 分钟；主机负载高时更久） |
| `make dataset-smoke` | 样例 episode → 冻结 validator → 一个 batch 的 shape/dtype/range 报告 |
| `make overfit` | 单卡（自动选空闲卡）400 步 AMP 过拟合合成 fixture → artifacts/checkpoints/overfit-fixture-v0/ |
| `make ddp-smoke` | 2 空闲卡 torchrun pipeline 冒烟（5 warmup + 15 measure）→ reports/ddp-smoke/ |
| `make benchmark` | 1/2/4 卡 **synthetic** pipeline benchmark（20 warmup + 100 measure）→ reports/benchmark/<label>/summary.json |
| `make quality-report EPISODE=<dir>` | 单 episode 质量报告（schema hash + 冻结 validator + shape/dtype/range/NaN） |
| `make manifest DATA=<dir>` | 真实数据入口：扫描 episode → 质量摘要 → episode 级 train/val split JSON |
| `make real-overfit DATA=<dir>` | 真实数据 10-20 episode 过拟合（见下） |

等价手工命令（当前目录为 work/contract，PYTHONPATH=src）：

```bash
/data5/ljt_data/conda_envs/embodiedflow/bin/python -m pytest -q
/data5/ljt_data/conda_envs/embodiedflow/bin/python -m embodiedflow.dataplane.smoke --chunk-len 20 --batch-size 4
/data5/ljt_data/conda_envs/embodiedflow/bin/python -m embodiedflow.dataplane.train --gpu auto --run-id overfit-fixture-v0 --steps 400
/data5/ljt_data/conda_envs/embodiedflow/bin/python tools/launch_ddp.py 2 --warmup-steps 5 --measure-steps 15 --label ddp-smoke
/data5/ljt_data/conda_envs/embodiedflow/bin/python tools/run_benchmark.py --warmup-steps 20 --measure-steps 100 --worlds 1,2,4
# 未训练 checkpoint（5090 加载兼容性测试用）
/data5/ljt_data/conda_envs/embodiedflow/bin/python -m embodiedflow.dataplane.train --fixture fixtures/overfit --steps 0 --run-id untrained-fixture-v0
# headless 相机硬件证明（需要 1 张空闲卡做 EGL 渲染）
/data5/ljt_data/conda_envs/embodiedflow/bin/python tools/headless_camera_smoke.py
```

## 两类 benchmark 的严格区分

### A. Synthetic pipeline benchmark（已完成，管线验收）

- 数据：`fixtures/overfit` 32×32 合成 fixture（validator 通过）
- 结论用途：验证 Dataset/DDP/checkpoint 管线吞吐，**不代表任何模型性能**
- 结果：reports/benchmark/bench-20260908-032851/summary.json
  （1 卡 404.8 samples/s → 2 卡 626.9（1.55×）→ 4 卡 992.3（2.45×，效率 61%））
- 报告内 `note` 字段固定标注 "pipeline benchmark on synthetic fixture;
  NOT final model results"；**不得把 2.45× / 61% 写成真实模型性能**

### B. Real Isaac training benchmark（未开始，待真实数据）

触发条件：真实 Isaac episode 到达并完成下述「真实数据入口」全流程，
**真实数据 overfit 通过后**才允许运行。必须记录：

- 实际分辨率、模型参数量、batch size
- samples/s、step time、data wait
- 每 rank GPU 利用率和峰值显存
- profiler 中数据、计算、通信占比（ddp_train `--profile`）
- 1/2/4 卡 speedup 和 scaling efficiency

命令（示例，label 区分真实数据）：

```bash
/data5/ljt_data/conda_envs/embodiedflow/bin/python tools/launch_ddp.py 4 \
  --fixture <real_data_dir> --pool-size 8 --profile \
  --warmup-steps 20 --measure-steps 100 --label real-bench-v1
```

## 归一化规范

| 对象 | 方法 | 说明 |
|---|---|---|
| image (rgb) | `uint8 → float32 / 255` | 无 per-channel 统计，范围 [0,1]，CHW |
| robot_state | z-score per dimension | 13 维 = joint_position(6) \| joint_velocity(6) \| gripper_position(1)；均值/方差仅由 train 集统计，写入 normalization.json |
| action | z-score per dimension | [chunk_len, 7]；**仅对 mask==1 的有效条目累计统计**，padding 条目归一化后保持精确 0（乘以 mask），否则近常量维度会把 padding 放大成巨大值 |

checkpoint 内 `normalization` 随模型一起交付（normalization.json），
推理端用同一组均值/方差还原 action。

## 真实 Isaac 数据入口（待 5090 数据到达）

Dataset 对任意 H×W uint8 RGB（含 256×256）与 contract 真实 state/action
天然支持（mmap 读取，无 32×32 硬编码）；模型侧 `rgb_shape` 在 train/ddp
入口自动跟随首个样本，大图需 `--pool-size`（如 256×256 建议 8）控制
transformer token 数。真实数据到达后按序执行：

1. **单 episode 质量报告**：`make quality-report EPISODE=<episode_dir>`
   （schema sha256 比对 + 冻结 validator + 各 artifact shape/dtype/range/NaN
   + 时间戳单调性/fps + 夹爪动作范围；任一失败退出码非 0）
2. **数据集 manifest + split**：`make manifest DATA=<real_data_dir>`
   → `<out>/episode_manifest.json`（逐 episode 质量摘要，rejected 列表）
   + `split.json`（episode 级 train/val，无帧泄漏）
3. **1 条 episode Dataset smoke**：
   `python -m embodiedflow.dataplane.smoke --fixture <real_data_dir> --chunk-len 20 --batch-size 4`
4. **10-20 episode 真实数据 overfit**：
   `make real-overfit DATA=<real_data_dir>`（`--pool-size 8`，400-1000 步，
   过拟合判定：train loss 显著低于初始，见 test_overfit_on_tiny_fixture 同款标准）
5. **导出 checkpoint 交 5090 加载**（协议见 docs/4090/checkpoint_handoff.md）
6. 通过后 → 真实 1/2/4 卡 benchmark（上节 B 的指标清单）

**不生成 MuJoCo 正式训练数据**：本机 MuJoCo EGL 输出只用于硬件环境证明。

## 关键设计决定

- **resume 逐位一致**：checkpoint 记录 dataloader 的 (seed, batches_consumed)，
  resume 时重播种并从头部重放跳过，避免 RandomSampler 每次 iter() 重新物化
  排列导致的 batch 流分叉（CPU 测试验证 step/loss/lr/optimizer/权重逐位一致）。
- **chunk 不跨 episode**：锚点按 episode 内步进生成，尾部零填充 + mask=0，
  valid 前缀与 episode actions 逐元素相等（有专门测试覆盖全部锚点）。
- **cosine 调度**：返回 LambdaLR 乘子（0..1），不是绝对学习率。
- **诚实命名**：模型为 ChunkedActionTransformerMinimal，无 VAE、无语言通路，
  manifest 中 instruction_conditioning="disabled"。

## 已知限制与待办

- 4 卡 benchmark 依赖 4 张空闲卡；空闲不足时命令会重试等待（最多约 5 分钟）后报错。
- `gpu_utilization_mean` 来自 NVML 瞬时采样，小模型 kernel 时间远小于 step
  时间，读数偏低属预期；真实数据 benchmark 以 `--profile` 的 CUDA 时间占比为准。
- resume 重放需重读已消费 batch（时间线性），长训练可后续优化为快进寻址。
- 单卡 step_time ~28ms（8×1025 tokens, 480 anchors/epoch）：数值受共享主机
  CPU/内存带宽影响，仅供同机横向对比。
- 待 5090 真实 Isaac episode：按「真实数据入口」流程执行。闭环成功率由
  5090 测量，4090 只做离线验证。
