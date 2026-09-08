# Checkpoint 交付说明（5090 加载协议测试用）

**用途声明**：本交付仅用于验证 5090 侧 `ActCheckpointPolicy` 的加载协议
（文件结构、字段、sha256 校验、模型重建）。**不声称该策略有效** ——
所选 checkpoint 是 32×32 合成 fixture 上的过拟合样例，不是真实模型。

## 交付内容

路径：`artifacts/checkpoints/overfit-fixture-v0/`（已提交到
dev/4090-data-plane 分支；顶层 5 个文件为协议交付物，`checkpoints/`
子目录为训练过程中的周期备份，可忽略）

| 文件 | 大小 | sha256 |
|---|---|---|
| model.pt | 1.7 MB | `656fadd46dfeb87d326638ace7de1d7db99693e5ea04f961fb2dcaed2e0904de` |
| model_manifest.json | 1.7 KB | `e173be23587e5cfc5e6517a39d380d5a7c2365ac57e301e81587a1021543e6c1` |
| normalization.json | 1.3 KB | `39853f952b0cef24047b86f459c9ce72ebcbc1a751f88e1077e3e1d0c07b7f6d` |
| config.yaml | 0.6 KB | `6534d6852c5600e9ac49d010c24c84e3d97f47a5b38f8680f79592b0c6f1f355` |
| metrics.json | 47 KB | `809a2fea55c4fda8f8d37de8e11b12e70d7d86d37e222a9fd7b44eee7d5b80fa` |

## model.pt 结构

`torch.save` 的 dict，键：

```
model_state_dict, optimizer_state_dict, scheduler_state_dict,
scaler_state_dict, global_step, epoch, model_config, train_config,
normalization, rng
```

- `model_config` 为 dataclass dict，包含 `model_type`, `rgb_shape`,
  `state_dim`, `action_dim`, `chunk_len`, `d_model`, `n_heads`, `n_layers`,
  `cnn_stem`, `pool_size`, `instruction_conditioning="disabled"` 等
- `model_manifest.json` 中 `checkpoint_sha256` 必须等于 model.pt 的 sha256，
  加载前先校验
- action 语义（contract-v0.1）：`[chunk_len, J+1]`，前 J 列关节目标(rad)，
  末列夹爪宽度(m)，z-score 归一化，`normalization.json` 中给均值和标准差

## 5090 最小加载命令（协议测试）

```python
import hashlib, json, torch

MANIFEST = json.load(open("model_manifest.json", encoding="utf-8"))

# 1) sha256 校验
assert hashlib.sha256(open("model.pt", "rb").read()).hexdigest() == \
    MANIFEST["checkpoint_sha256"], "checkpoint sha256 mismatch"

# 2) 加载与重建（torch>=2.1，weights_only=False 因为内嵌 dataclass 配置）
ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
assert ckpt["global_step"] == 400

# 3) 按 model_config 重建模型结构（或由 5090 侧等价实现按同样字段构造）
from embodiedflow.dataplane.model import ChunkedActionTransformerMinimal
from embodiedflow.dataplane.config import ModelConfig
cfg = ModelConfig(**ckpt["model_config"])
model = ChunkedActionTransformerMinimal(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 4) 推理样例：rgb float32 [B,3,H,W] ∈ [0,1]（uint8/255），
#    robot_state float32 [B,13] = joint_position|joint_velocity|gripper_position
#    （z-score 归一化后输入），输出 action chunk [B,20,7]（z-score 空间，
#    需乘 std 加 mean 还原）
rgb = torch.rand(1, 3, 32, 32)
state = torch.randn(1, 13)
with torch.no_grad():
    chunk = model(rgb, state)          # [1, 20, 7]
assert chunk.shape == (1, 20, 7)
```

5090 侧只需要 model.pt + model_manifest.json 即可完成协议验证；
normalization.json / config.yaml / metrics.json 供人工核对。

## 注意

- manifest 中 `git_commit: a818620...` 是导出时刻的代码提交，
  当前分支头为 33d6882 及之后；加载兼容性由内嵌 `model_config` 保证，
  与训练代码版本无关
- 本模型无语言通路（`instruction_conditioning="disabled"`），
  输入接口只有 rgb + robot_state
- 若 5090 加载遇到 `weights_only` 报错或 dataclass 兼容问题，请反馈具体
  报错；协议以本文件为唯一交付契约
