# EmbodiedFlow 4090 数据与训练平面（dev/4090-data-plane）

本分支在 4×4090 主机上实现 contract-v0.1 的数据与训练平面。
**诚实声明**：目前所有训练/基准均基于合成 fixture，标注为 pipeline
benchmark，不是最终模型结果；headless EGL 相机渲染仅为硬件环境证明，
不是正式数据源，也不是 Isaac Sim 验收。

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
6. [src/embodiedflow/dataplane/model.py](../../src/embodiedflow/dataplane/model.py) — ChunkedActionTransformerMinimal（**不是 ACT CVAE**，instruction_conditioning 声明为 disabled）
7. [src/embodiedflow/dataplane/train.py](../../src/embodiedflow/dataplane/train.py) — 单卡训练：AMP/clip/checkpoint/resume（batch 级重放保证逐位一致）
8. [src/embodiedflow/dataplane/checkpoint.py](../../src/embodiedflow/dataplane/checkpoint.py) — 交付协议：model.pt + model_manifest.json + normalization.json + config.yaml + metrics.json
9. [src/embodiedflow/dataplane/ddp_train.py](../../src/embodiedflow/dataplane/ddp_train.py) — torchrun 1 卡 1 进程，DistributedSampler + set_epoch，rank0 独占写
10. [src/embodiedflow/dataplane/gpu.py](../../src/embodiedflow/dataplane/gpu.py) — nvidia-smi 查询与空闲卡选择
11. [tests/](../../tests/) — 24 个测试：数据集异常隔离、chunk 语义、split 无泄漏、确定性、resume 逐位一致、overfit、checkpoint 协议
12. [tools/](../../tools/) — make_overfit_fixture.py（合成 fixture，validator 通过）、headless_camera_smoke.py（EGL 硬件证明）、launch_ddp.py、run_benchmark.py

## 一键命令（仓库根目录）

| 命令 | 内容 |
|---|---|
| `make test` | 全量 pytest（24 个测试，CPU 约 6 分钟） |
| `make dataset-smoke` | 样例 episode → 冻结 validator → 一个 batch 的 shape/dtype/range 报告 |
| `make overfit` | 单卡（自动选空闲卡）400 步 AMP 过拟合 → artifacts/checkpoints/overfit-fixture-v0/ |
| `make ddp-smoke` | 2 空闲卡 torchrun pipeline 冒烟（5 warmup + 15 measure）→ reports/ddp-smoke/ |
| `make benchmark` | 1/2/4 卡 pipeline benchmark（20 warmup + 100 measure）→ reports/benchmark/<label>/summary.json |

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

## 关键设计决定

- **resume 逐位一致**：checkpoint 记录 dataloader 的 (seed, batches_consumed)，
  resume 时重播种并从头部重放跳过，避免 RandomSampler 每次 iter() 重新物化
  排列导致的 batch 流分叉（CPU 测试验证 step/loss/lr/optimizer/权重逐位一致）。
- **padding 归一化**：action chunk 尾部零填充在 z-score 后保持精确 0，
  否则近常量维度的 padding 会被放大成巨大值。
- **cosine 调度**：返回 LambdaLR 乘子（0..1），不是绝对学习率。
- **诚实命名**：模型为 ChunkedActionTransformerMinimal，无 VAE、无语言通路，
  manifest 中 instruction_conditioning="disabled"。

## 已知限制与待办

- 4 卡 benchmark 依赖 4 张空闲卡；当前机器其他用户占用较多，空闲不足时命令会重试
  等待（最多约 5 分钟）后报错，稍后重跑即可。
- `gpu_utilization_mean` 来自 NVML 瞬时采样，本合成模型 kernel 时间远小于 step
  时间，读数接近 0 属预期；不作为利用率结论。
- resume 重放需重读已消费 batch（时间线性），长训练可后续优化为快进寻址。
- 单卡 step_time ~28ms（8×1025 tokens, 480 anchors/epoch）：数值受共享主机
  CPU/内存带宽影响，仅供同机横向对比。
- 待 5090 真实 Isaac episode：冻结 validator → 质量报告 → 10-20 episode overfit
  → 全量训练。闭环成功率由 5090 测量，4090 只做离线验证。
