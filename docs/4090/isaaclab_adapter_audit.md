# Isaac Lab（5090 官方任务）适配审计

**分支**：`dev/4090-isaaclab-adapter`（基线 `dev/4090-data-plane` @ 787401d，含已验收的加载修复 ff529e7）
**日期**：2026-09-13
**范围**：静态审计——硬编码位置、已配置化/需修改清单、等待 5090 的输入、数据到达后的入口命令。
**非目标（本分支当前不做）**：不猜 Franka 动作定义；不预造 v0.2 数据或假 v0.1 adapter；
不安装 Isaac；不跑 synthetic benchmark / DDP 调优；**contract-v0.1 全程冻结**（schema、validator、
样例 episode、旧测试原样保留）。

## 1. 硬编码审计（搜索项 → 结论）

| # | 搜索项 | 位置 | 结论 | 分类 |
|---|---|---|---|---|
| 1 | state_dim=13 | config.py:44 默认；dataset.py:259 `= len(joint_names)*2+1`；train.py:106 / ddp_train.py:280 运行时覆盖 | 运行时**从数据推导**，但推导公式写死「pos\|vel\|gripper」布局 | 部分配置化，布局需改 |
| 2 | action_dim=7 | config.py:45 默认；dataset.py:263 `= actions.shape[1]` | 数据驱动 | **已配置化** |
| 3 | 6 机械臂关节 | tools/make_sample_episode.py:14、make_overfit_fixture.py:25；tests/test_dataset.py:41-42 | 仅存在于合成生成器与旧样例断言 | 无需迁移 |
| 4 | UR5e 关节限位 | 全仓库检索无此常量 | 4090 代码从未持有关节限位；episode.json 只声明夹爪 open/closed | 无硬编码；**缺 Franka 限位来源**（见 §3） |
| 5 | 夹爪范围 | dataset.py:73-93、tools/episode_quality_report.py:152-165 | 范围值读自 episode 声明（数据驱动），但「动作最后一列 = 夹爪」写死 | 半配置化 |
| 6 | 输出 [1,20,7] | chunk_len 默认 20（config.py:20,46）可被 `--chunk-len` 覆盖；字面量仅在 test_package_layout.py:80,88、test_checkpoint.py:59、checkpoint_handoff.md | 形状已配置化；字面量属**旧接口回归 / v0.1 交付文档**，保留 | 已配置化（旧测试保留） |
| 7 | normalization 维度 | stats.py:83-125 全泛型（维度是参数）；train.py:63-70 在 train 集拟合、val 复用同一统计量 | 维度配置化；「逐维标量 z-score」语义对四元数等非标量维不成立 | 已配置化；**语义需按表示扩展** |
| 8 | robot_state 布局 | dataset.py:136-158 按 v0.1 artifact 名读取并固定拼接顺序 | 写死 | **需改** |
| 9 | manifest/ROS2 假设 | checkpoint.py:98-150：composition 字符串(:127)、unit "rad,m"(:133)、representation(:134)；train.py:214-215,236-237 写死 observation_keys/robot_state_keys | 训练代码无 ROS2 耦合；manifest 字段语义写死 | **需改** |
| 10 | 校验版本选择 | validator.py 默认 contracts/v0/schema.json；dataset.py:196、episode_quality_report.py:86 直接调用 | 单版本 v0.1 | **需改**（按版本选择） |
| 11 | 模型输出约束 | model.py:77,107 线性 head 无激活 | 无输出约束 | 视动作表示新增（可配置） |
| 12 | 交付包结构 | checkpoint.py:72-95 单包内嵌 optimizer/RNG（numpy 2.x pickle）；六要求 inference/resume 拆分 | 未拆分 | **需改** |

补充发现：

- README「真实数据入口」step 3 的 smoke flag 写错（`--fixture`，实际为 `--episodes`，见 smoke.py:26）——本分支已修正。
- DDP 入口（ddp_train.py）只写 metrics，无 checkpoint 导出/resume；单卡 train.py 有。真实训练若用 DDP 需补（后置项）。
- manifest 相对新交付要求（§六）缺失字段：robot_id、joint_names、state layout、state 单位、
  坐标系、姿态约定、action scale/offset、控制周期。

## 2. 已配置化清单（直接复用，不动）

- **Dataset 框架**：validate-before-load、episode 级 split（无帧泄漏）、chunk + padding mask
  （不跨 episode 边界）、window_stride、mmap 任意 H×W
- **维度跟随数据**：state_dim / action_dim / chunk_len 由 Dataset 与 CLI 推导
  （train.py:106-107、ddp_train.py:280-281、smoke.py:42-48）；rgb_shape 自动跟随首个样本
- **训练栈**：AMP、grad clip、cosine 调度、bit-exact resume（(seed, batches_consumed) 重放）
- **归一化流程**：train 集拟合 → val 复用同一统计量（train.py:63-70，满足「验证集复用统计量」）；
  padding 条目精确 0
- **夹爪范围校验**：按 episode 声明值 + 可配 tolerance（enforce 开关）
- **质量报告框架**：shape/dtype/range/NaN/时间戳单调/fps/夹爪/分辨率，mmap 低内存
- **checkpoint 协议骨架**：model.pt + manifest（sha256）+ normalization.json + config.yaml + metrics.json
- **大图支持**：pool_size（256×256 建议 8）
- **旧接口回归**：samples/episode_v0、fixtures/overfit、全部旧测试原样保留

## 3. 需要 5090 提供的输入（缺失字段）

- [ ] 官方 task ID、Isaac Lab 上游 tag/commit
- [ ] 1 条原始 episode（原始落盘格式即可）
- [ ] joint_names 及顺序（Franka 7 关节 + 夹爪）
- [ ] observation / action 逐维定义（含 state 布局）
- [ ] 单位、坐标系、绝对/相对、姿态约定（四元数顺序）
- [ ] action scale/offset；夹爪含义（宽度 / 开合度 / 二值；力控？）
- [ ] physics / control / camera 周期（控制频率、decimation）
- [ ] 终止及成功判定
- [ ] （建议）关节限位（用于动作裁剪/越界检查）

## 4. 首条原始 episode 到达后的入口命令

```bash
cd /data5/ljt_data/work/contract && make quality-report EPISODE=<原始 episode 目录>
```

等价：`/data5/ljt_data/conda_envs/embodiedflow/bin/python tools/episode_quality_report.py <dir> --json artifacts/quality/<name>.json`

输出：字段 shape/dtype/min/max/NaN、时间戳单调性/fps、夹爪范围、schema 校验。
说明：若 episode 为 v0.2 结构，**冻结 v0.1 validator 报差异属预期** —— 这些差异正是
v0.2 联合校验（5090 主导）的输入；不修改 v0.1 去迁就。v0.2 定义就绪后接
`episode_manifest.py` → Dataset smoke（`--episodes`）→ real-overfit。

## 5. 需修改清单（数据到达并联合确认 v0.2 后实施，当前不改）

1. **版本选择校验/解析**：按 manifest `schema_version` 分派 v0.1（冻结）/ v0.2 schema 与解析器；
   contracts/v0 与旧测试不动（dataset.py:196、quality report、validator 调用点加选择层）。
2. **robot_state 布局配置化**：dataset.py:151-158 拼接与 :259 维度公式改为由 episode 定义驱动
   （state 字段列表 + 各自宽度），不再假设 pos|vel|gripper。
3. **动作语义配置化**：dataset.py:73-93 与 quality_report.py:152-165 的「最后一列 = 夹爪」改为按定义；
   action_dim 继续数据驱动。
4. **manifest 扩展**：新增 robot_id、joint_names、state layout、单位、坐标系、姿态约定、
   控制周期、action scale/offset（checkpoint.py:98-150 参数化，调用方从 episode 元数据取）。
   不改 v0.1 已交付包与旧测试。
5. **交付包拆分**：inference 包（weights + model_config + normalization，避免内嵌 numpy/RNG 对象，
   目标 `weights_only=True` 可载）/ resume 包（optimizer、scheduler、scaler、RNG、dataloader）；
   旧 v0.1 包仅作旧接口回归。
6. **归一化按动作表示**：四元数等需明确预处理与反变换；必要时加可配置输出约束头
   （model.py:77,107 目前无激活）；拟合仍仅用训练集（现有流程已满足）。
7. **后置**：DDP 训练补齐导出/resume；真实 1/2/4 卡 benchmark（指标口径见 README 节 B）。

**硬约束（四.6/四.7/四.10）**：不截断 Franka 关节；不把末端 IK 动作当关节目标；
无法无损转换时明确声明不兼容，不做假 v0.1 adapter；旧 checkpoint 只用于旧接口回归，新机器人重新训练。

## 6. 验证顺序（数据到达后）

首条 episode 质量报告 → v0.2 validator + Dataset smoke → 少量成功轨迹 real-overfit
（记录 loss 与**反归一化后动作误差**）→ padding/episode 边界/观测-动作时间对齐检查 →
10-20 条成功轨迹扩大 overfit → 导出 checkpoint 交 5090 同任务闭环 → 闭环通过后才做正式训练与
1/2/4 卡 benchmark。失败 episode 可用于管线验证，但不得标为成功演示；训练 loss 下降不代表任务成功率。
