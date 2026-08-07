---
schema_version: 1
stage: 4A-R1
task_type: semantics_repair
status: completed
branch: main
code_commit: 9b19c8d5fc2b396b91ea3360c8af1ee99ea1f93c
code_push_status: pushed
report_commit: self
---

# Stage 4A-R1 MAPPO Semantics Repair Report

## 1. 本轮任务

- 阶段：4A-R1，修正 4A MAPPO 的概率比与 entropy 语义；未启动 4B 正式训练。
- 起始基线：`main`，包含 `34ca5b4`；唯一预期未提交项是用户授权的 `AGENTS.md`。
- 结束代码：`9b19c8d5fc2b396b91ea3360c8af1ee99ea1f93c`。

## 2. PPO ratio 语义修正

旧实现先对所有中继的 log probability 求和，再计算一个 joint-action ratio。该语义会把各中继的策略变化耦合为单一乘积，不能实现标准的 per-relay PPO surrogate。

新实现：

- rollout 保存 `old_per_relay_log_probabilities`，完整 shape 为 `(rollout_steps, K, 1)`；采集时不再先求和。
- Actor 对 requested action 重新计算 `(batch, K, 1)` 的 new per-relay log probability。
- `ratio = exp(new_per_relay_log_probability - old_per_relay_log_probability)`，每个 relay 独立计算。
- 团队 advantage `(batch, 1)` 广播为 `(batch, K, 1)`；surrogate、clip fraction、approx KL 和 policy loss 均在 batch×relay 全部元素上求平均。
- applied action 仍只用于 mismatch 诊断，未用于 ratio、policy loss 或 critic/value loss。

## 3. Entropy 语义修正

- 旧实现把 `-log_probability_of_rollout_action` 作为 entropy；该值依赖具体 rollout action，不是策略分布 entropy。
- 新实现使用 `Normal(mean, std).entropy().sum(dim=-1, keepdim=True)`，shape 为 `(batch, K, 1)`。
- 该值是 **pre-tanh Gaussian entropy approximation**，不是精确的 squashed-action entropy；此限制已在 Actor docstring 和实现中明确记录。
- `evaluate_actions()` 保留安全 atanh 裁剪和与采样一致的 tanh Jacobian log-prob 修正；joint log probability 只作为诊断返回值。

## 4. Checkpoint 与 partial rollout

- MAPPO checkpoint 格式仍为独立版本 1；保存 Actor、Value Critic、两个 optimizer、`MAPPOConfig`、维度和 metadata。
- 测试已从“加载后与加载后自身比较”修正为：记录保存前原 Agent 的 deterministic action、value、Actor/Value 参数、两个 optimizer state、config 和 metadata，并与加载后 Agent 逐项比较。
- 未满 rollout 仍不更新；训练 summary 与实验 `summary.json` 新增 `discarded_partial_rollout_steps`，用于显示未参与更新的 on-policy 样本数。

## 5. 新增和修正测试

- 随机 sampled requested action 的 per-relay sample/evaluate log-prob 逐元素一致。
- 两 relay 的 `1.1` 与 `0.9` ratio 保持独立，不折叠为 joint `0.99`；K=1 与标准单智能体 PPO ratio 等价。
- relay 置换不改变核心 Actor loss/entropy；仅改变 applied action 时，参数、policy loss、ratio 相关指标一致，而 mismatch 指标可变。
- entropy 验证 shape、有限性、随 log_std 增大而增大，以及在策略分布不变时不随 rollout action 改变。
- 覆盖真实 checkpoint 往返、损坏拒绝、恢复后继续更新和 partial rollout 丢弃统计。

## 6. 验证

- 完整测试：`python -m pytest`，`191 passed in 58.14s`，无新增 Pytest 警告。
- MAPPO 专项测试：`17 passed in 4.18s`。
- 编译：`python -m compileall -q src tests scripts` 成功。
- 未删除或弱化既有 186 项测试；总数增加至 191。

## 7. 1,000 步烟雾实验

使用固定命令配置：steps `1000`、rollout `250`、max_steps `50`、radius `90`、epochs `2`、mini-batch `125`、evaluation/checkpoint interval `500`、2 episodes、seed `0`、evaluation seed `10000`、CPU 和奖励权重 `1,1,1,0.1,0.1,1`。

- 输出目录：`outputs/stage4a_r1_mappo_semantics_smoke`，未提交 Git。
- 已完成 1,000 环境步和 4 次 PPO 更新。
- `discarded_partial_rollout_steps = 0`。
- 训练日志 4 行、评估日志 2 行；entropy、approx KL、clip fraction、Actor/Critic 梯度均有限。
- checkpoint：`step_000000.pt`、`step_000500.pt`、`step_001000.pt`、best 和 final 均存在。
- final checkpoint 加载后的 deterministic action 和 value 与保存前记录一致。
- 同路径重跑被非空输出目录保护拒绝；未覆盖、删除或篡改已完成产物。
- 本实验不是性能比较，未作 MAPPO 优于任何基线的声明。

## 8. 不变项和下一步

- 未修改环境、状态转移、安全过滤、奖励、通信/运动模型、MASAC、现有基线或 MASAC checkpoint。
- 代码提交：`9b19c8d5fc2b396b91ea3360c8af1ee99ea1f93c`（`fix: use per-relay MAPPO probability ratios`），已推送至 `origin/main`。
- 报告提交：self；包含本报告的提交已推送至 `origin/main`。
- 下一建议任务：阶段 4B——MAPPO 固定配置训练与公平比较。
