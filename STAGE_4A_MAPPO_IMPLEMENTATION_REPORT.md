---
schema_version: 1
stage: 4A
task_type: implementation
status: completed
branch: main
code_commit: 34ca5b4e913206f05e6b9e775f596d7e7bfac407
code_push_status: pushed
report_commit: self
---

# Stage 4A MAPPO Implementation Report

## 1. 本轮任务

- 阶段：4A——参数共享 MAPPO 实现。
- 起始基线：`main`，包含 `71eb014` 和 `704c675`；唯一预期未提交改动为用户授权的 `AGENTS.md` 计划。
- 范围：完成独立的 MAPPO 网络、on-policy rollout、GAE、PPO、训练、评估、checkpoint、CLI、测试和短烟雾实验。
- 未执行：20,000 步正式训练、七策略比较或任何 MAPPO 性能优于基线的声明。

## 2. 动作语义

MAPPO 的 Actor 采样并保存 **requested normalized action**。环境仍按既有链路执行：requested action → 安全过滤 → applied action → 状态转移。

- PPO ratio 使用 `exp(new_joint_log_probability - old_joint_log_probability)`，两者均基于 requested action。
- `applied_actions` 被 rollout 保存，但仅用于 requested/applied mismatch 与安全干预诊断，绝不参与 PPO ratio。
- 现有 MASAC Replay Buffer 仍保存 applied action，语义未变；两种算法的动作语义差异将在后续公平性消融中处理。

## 3. 新增实现

- `learning/mappo.py`：`MAPPOAgent`、默认 PPO 配置、固定长度 rollout、GAE、clipped objective、value loss、熵项、mini-batch、多 epoch 和双网络梯度裁剪。
- `learning/networks.py`：`SharedGaussianActor.evaluate_actions()`（tanh Jacobian 与采样一致、边界安全反变换）及 `CentralizedValueCritic`。
- `training/mappo_trainer.py`：可复现 on-policy 收集、完整 rollout 后更新、更新后清空 rollout、terminated/truncated 处理与训练统计。
- `training/mappo_evaluator.py`：无参数更新的 deterministic Actor mean 评估。
- `training/mappo_checkpoints.py`：独立 MAPPO checkpoint 格式，原子写入、CPU 加载、Actor/Value/optimizers/config/metadata 完整保存。
- `training/mappo_experiment.py` 和 `scripts/run_mappo_experiment.py`：日志、评估、best/final/周期 checkpoint 与空输出目录保护。

## 4. GAE 和 PPO 语义

- 默认 `gamma=0.99`、`gae_lambda=0.95`。
- 真实终止：bootstrap mask 为 `1 - terminated`，因此不 bootstrap。
- 时间截断：允许其下一观测 value bootstrap，但 trace continuation mask 为 `1 - terminated - truncated`，不会跨 reset 串联 advantage。
- PPO Actor loss：`-min(ratio * advantage, clamp(ratio) * advantage) - entropy_coefficient * entropy`。
- Value Critic 使用集中式 global state，输出 `V(s)`；未复用 MASAC twin-Q critic。

## 5. 测试与编译

- 完整命令：`python -m pytest`。
- 结果：`186 passed in 57.98s`，无新增 Pytest 警告（原 174 项测试全部保留）。
- MAPPO 相关命令：`python -m pytest -q tests/test_mappo.py tests/test_mappo_training.py tests/test_mappo_experiment.py`。
- MAPPO 结果：`12 passed in 4.04s`。
- 编译：`python -m compileall -q src tests scripts` 成功。
- 测试覆盖：Actor sample/evaluate/boundary、Value Critic、普通/terminated/truncated/多 episode GAE、rollout 完整性、PPO 参数更新与有限指标、requested-vs-applied ratio 隔离、训练更新、checkpoint 往返/损坏拒绝/恢复更新、实验文件和非空输出目录拒绝。

## 6. 512 步烟雾实验

命令使用：`steps=512`、`rollout_steps=128`、`max_steps=50`、radius `90`、`update_epochs=2`、`mini_batch_size=64`、evaluation/checkpoint interval `256`、2 evaluation episodes、seed `0`、evaluation seed `10000`、CPU，以及奖励权重 `1,1,1,0.1,0.1,1`。

- 输出目录：`outputs/stage4a_mappo_smoke`（未提交）。
- 完成 512 环境步和 4 次 PPO 更新。
- 训练日志 4 行、评估日志 2 行；全部数值有限且 JSON/JSONL 可读。
- checkpoint：`step_000000.pt`、`step_000256.pt`、`step_000512.pt`、best、final 均存在。
- final checkpoint metadata 的 environment steps 为 512；CPU 加载后的 deterministic action 完全一致。
- best mean return 为 `214.0077881934801`；该 smoke 结果不构成性能判断。

## 7. 不变性

- 未修改环境状态转移、安全过滤器、运动学、通信模型、奖励公式或奖励默认权重。
- 未修改 MASAC loss、MASAC trainer、MASAC Replay Buffer、现有基线策略或 MASAC checkpoint 格式。

## 8. Git 状态与下一步

- 代码提交：`34ca5b4e913206f05e6b9e775f596d7e7bfac407`（`feat: implement parameter-sharing MAPPO`），已推送至 `origin/main`。
- 报告提交：self；包含本报告的提交已推送至 `origin/main`。
- 运行产物处于 `outputs/` 且未加入 Git。
- 下一建议任务：阶段 4B——MAPPO 固定配置训练与 MASAC 公平比较。
