# Codex 执行计划：阶段 4A——参数共享 MAPPO 实现

## 1. 任务性质

当前任务属于正式阶段：

```text
阶段 4A
```

任务：

```text
实现参数共享 MAPPO 的完整训练基础设施
```

阶段 3已正式关闭：

```text
实现完成
性能验收失败
```

当前 MASAC 保留为失败基线，不再继续进行阶段 3修复。

本轮只完成 MAPPO 的：

1. 网络与算法更新；
2. On-policy rollout；
3. GAE；
4. PPO clipped objective；
5. 训练与确定性评估；
6. checkpoint；
7. 单元和集成测试；
8. 短程冒烟实验。

本轮不运行 20,000 步正式训练，不进行七策略最终比较。

本轮结果文档必须命名为：

```text
STAGE_4A_MAPPO_IMPLEMENTATION_REPORT.md
```

------

## 2. 开始前检查

首先运行：

```bash
git status --short
git branch --show-current
git log -3 --oneline
```

要求：

```text
当前分支为 main
包含提交 71eb014
包含报告提交 704c675
工作区除 M AGENTS.md 外无其他未提交修改
```

`M AGENTS.md` 是本轮授权的执行计划，可以继续。

如果存在其他未提交文件：

1. 立即停止；
2. 列出文件；
3. 不恢复；
4. 不提交；
5. 不开始实现。

------

## 3. MAPPO 动作语义

MAPPO 的策略动作必须定义为：

```text
Actor 实际采样的 requested normalized action
```

安全过滤器仍属于环境状态转移的一部分：

```text
requested action
→ 环境安全过滤
→ applied action
→ 状态转移
```

因此 MAPPO rollout 必须保存：

```text
requested_actions
old_log_probabilities
applied_actions（仅用于诊断）
```

PPO 概率比必须基于：

```text
requested_actions
```

不得使用 applied action 反算策略概率，因为 applied action 不是直接从策略分布采样的。

本规则只适用于新实现的 MAPPO。

本轮不得修改：

```text
MASAC Replay Buffer 的 applied action 语义
环境安全过滤器
环境动作接口
```

结果文档必须明确记录 MAPPO 与当前 MASAC 的动作语义差异，后续公平性问题留到阶段 4消融中处理。

------

## 4. 文件结构

建议新增：

```text
src/uav_multi_relay/learning/mappo.py
src/uav_multi_relay/training/mappo_trainer.py
src/uav_multi_relay/training/mappo_experiment.py
src/uav_multi_relay/training/mappo_checkpoints.py
src/uav_multi_relay/training/mappo_evaluator.py
scripts/run_mappo_experiment.py
tests/test_mappo.py
tests/test_mappo_training.py
tests/test_mappo_experiment.py
```

允许小幅修改：

```text
src/uav_multi_relay/learning/networks.py
src/uav_multi_relay/learning/__init__.py
src/uav_multi_relay/training/__init__.py
README.md
AGENTS.md
```

不得修改：

```text
src/uav_multi_relay/environment.py
src/uav_multi_relay/safety.py
src/uav_multi_relay/kinematics.py
通信模型
奖励公式和权重
MASAC loss
MASAC Replay Buffer
现有基线策略
```

------

## 5. 网络实现

### 5.1 共享 Actor

复用或扩展现有 `SharedGaussianActor`。

必须支持：

```python
sample(local_observations, deterministic=False)
evaluate_actions(local_observations, actions)
```

`evaluate_actions()` 至少返回：

```text
joint_log_probability
per_relay_log_probability
entropy_estimate
```

输入 action 必须位于：

```text
[-1, 1]
```

对 tanh 反变换使用安全裁剪，避免 `atanh(±1)`。

概率计算必须与 `sample()` 中的 tanh Jacobian 修正一致。

### 5.2 集中式 Value Critic

新增：

```python
CentralizedValueCritic
```

输入：

```text
global_state
```

输出：

```text
V(s)
```

要求：

```text
输入 shape = (batch, global_state_dim)
输出 shape = (batch, 1)
输出必须有限
```

MAPPO 不得复用 MASAC 的 Q Critic 代替 Value Critic。

------

## 6. Rollout 数据结构

实现固定长度 on-policy rollout。

至少保存：

```text
local_observations
global_states
requested_actions
applied_actions
old_joint_log_probabilities
rewards
values
next_values
terminated
truncated
advantages
returns
```

要求：

- 数组 shape 明确；
- 全部数值有限；
- requested/applied action 均位于 `[-1, 1]`；
- rollout 满后才允许执行 PPO 更新；
- 更新完成后必须清空旧 rollout；
- 不得跨更新周期重复使用旧样本。

`applied_actions` 本轮只用于：

```text
安全干预率
requested/applied mismatch
诊断日志
```

不得用于 PPO ratio。

------

## 7. GAE 与终止语义

固定默认参数：

```text
gamma = 0.99
gae_lambda = 0.95
```

必须正确区分：

### 真实终止

```text
terminated = True
```

不 bootstrap：

```text
next value contribution = 0
```

### 时间截断

```text
truncated = True
```

允许使用截断前 next observation 的 value bootstrap，但优势递推不得跨 reset 连接到下一个 episode。

实现时区分：

```text
bootstrap mask = 1 - terminated
trace continuation mask = 1 - terminated - truncated
```

必须增加精确数值测试，覆盖：

```text
普通连续轨迹
真实终止
时间截断
一个 rollout 中包含多个 episode
```

------

## 8. PPO 更新

实现参数共享 MAPPO 更新，至少包含：

```text
clipped policy objective
centralized value loss
entropy bonus
advantage normalization
多 epoch 更新
mini-batch
Actor 梯度裁剪
Value Critic 梯度裁剪
```

默认配置：

```text
clip_ratio = 0.2
update_epochs = 10
mini_batch_size = 256
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
value_loss_coefficient = 0.5
entropy_coefficient = 0.01
max_gradient_norm = 0.5
```

策略比率使用：

```text
ratio = exp(new_joint_log_probability - old_joint_log_probability)
```

Actor loss 使用：

```text
-min(
    ratio * advantage,
    clipped_ratio * advantage
)
```

不得：

```text
把 applied action 用于 PPO ratio
跨 rollout 重复训练旧数据
对 terminated transition bootstrap
把 truncated 错当成 terminated
```

------

## 9. MAPPO 指标

每次更新至少输出：

```text
policy_loss
value_loss
entropy
approx_kl
clip_fraction
actor_gradient_norm
critic_gradient_norm
value_mean
return_mean
advantage_mean
advantage_std
requested_applied_mismatch_mean
requested_applied_mismatch_rate
```

所有指标必须：

```text
有限
可 JSON 序列化
不得使用 nan_to_num 隐藏异常
```

检测到 NaN 或 Infinity 时立即失败。

------

## 10. 训练循环

新增 MAPPO trainer。

训练流程：

```text
采集 rollout
→ 计算 GAE 和 returns
→ 执行 PPO 多 epoch 更新
→ 清空 rollout
→ 继续采集
```

必须记录：

```text
environment steps
completed episodes
episode return
episode length
mean rate
termination rate
intervention rate
requested/applied mismatch
PPO update metrics
```

训练 reset seed 规则必须确定且可复现。

不得修改现有 MASAC trainer。

------

## 11. 评估

新增确定性 MAPPO 评估。

要求：

```text
使用 Actor mean 对应的 deterministic action
不更新参数
不改变训练 RNG
支持固定 episode seeds
```

至少输出：

```text
mean return
return std
mean rate_e2e_bps
minimum rate_e2e_bps
termination rate
mean episode length
intervention rate
requested/applied mismatch rate
```

------

## 12. Checkpoint

MAPPO checkpoint 至少保存：

```text
Actor state
Value Critic state
Actor optimizer
Critic optimizer
算法配置
网络维度
训练 environment steps
更新次数
completed episodes
```

要求：

- 原子写入；
- 可指定 CPU 加载；
- 保存加载后 deterministic action 一致；
- 恢复后 update 可继续执行；
- 不与 MASAC checkpoint 格式混淆。

MAPPO 使用独立函数和 metadata 类型。

------

## 13. 命令行入口

新增：

```text
scripts/run_mappo_experiment.py
```

至少支持：

```text
--output-dir
--steps
--rollout-steps
--max-steps
--waypoint-radius
--update-epochs
--mini-batch-size
--evaluation-interval
--evaluation-episodes
--checkpoint-interval
--seed
--evaluation-seed
--device
--reward-rate
--reward-link
--reward-separation
--reward-intervention
--reward-motion
--reward-failure
```

输出目录必须为空。

输出至少包含：

```text
run_config.json
training_metrics.jsonl
evaluation_metrics.jsonl
summary.json
best_checkpoint.pt
final_checkpoint.pt
checkpoints/
```

不得覆盖已有输出目录。

------

## 14. 必须新增的测试

至少覆盖：

### 网络

- Actor sample shape 和有限性；
- deterministic action 一致；
- `evaluate_actions()` 与 sample log probability 一致；
- 边界 action 不产生 NaN；
- Value Critic shape 和有限性。

### GAE

- 普通轨迹的精确结果；
- terminated 不 bootstrap；
- truncated bootstrap 但不跨 reset；
- 多 episode rollout。

### PPO 更新

- 参数实际更新；
- clipped objective 数值正确；
- advantage normalization；
- mini-batch 覆盖完整；
- 所有指标有限；
- requested action 用于 ratio；
- applied action 只影响诊断，不改变 ratio。

### Checkpoint

- 保存加载；
- deterministic action 一致；
- metadata 一致；
- 训练可继续；
- 损坏文件被拒绝。

### 集成

- 短 rollout 完成一次更新；
- 训练和评估均可运行；
- 输出文件完整；
- 非空输出目录被拒绝；
- 环境终止与截断均可处理。

不得删除或弱化现有 174 项测试。

------

## 15. 验证命令

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

再运行：

```bash
python -m pytest -q \
  tests/test_mappo.py \
  tests/test_mappo_training.py \
  tests/test_mappo_experiment.py
```

要求：

```text
所有原测试继续通过
新增测试全部通过
测试总数高于 174
编译成功
无新增 Pytest 警告
```

------

## 16. 冒烟实验

使用新的空目录：

```text
outputs/stage4a_mappo_smoke
```

运行一个短实验：

```bash
python scripts/run_mappo_experiment.py \
  --output-dir outputs/stage4a_mappo_smoke \
  --steps 512 \
  --rollout-steps 128 \
  --max-steps 50 \
  --waypoint-radius 90 \
  --update-epochs 2 \
  --mini-batch-size 64 \
  --evaluation-interval 256 \
  --evaluation-episodes 2 \
  --checkpoint-interval 256 \
  --reward-rate 1.0 \
  --reward-link 1.0 \
  --reward-separation 1.0 \
  --reward-intervention 0.1 \
  --reward-motion 0.1 \
  --reward-failure 1.0 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

验收：

```text
完成 512 环境步
至少完成一次 PPO 更新
训练指标全部有限
评估完成
best/final checkpoint 可加载
checkpoint deterministic action 一致
输出 JSON/JSONL 可读取
```

冒烟实验不用于判断 MAPPO 性能。

`outputs/` 不得提交 Git。

------

## 17. 本轮验收标准

本轮通过必须同时满足：

1. MAPPO Actor 和 Value Critic 完成；
2. requested action 概率语义明确；
3. rollout 完成；
4. GAE 正确区分 terminated/truncated；
5. PPO clipped objective 完成；
6. checkpoint 完成；
7. 确定性评估完成；
8. 原有测试全部通过；
9. 新增测试全部通过；
10. 冒烟实验完成；
11. 未修改环境、奖励和安全过滤；
12. 未修改 MASAC 行为；
13. 未进行正式性能宣称；
14. 结果文档名称正确；
15. 最终工作区干净。

本轮通过后，下一任务为：

```text
阶段 4B——MAPPO 固定配置训练与 MASAC 公平比较
```

------

## 18. 结果文档

将当前结果文档改名为：

```bash
git mv STAGE_3G_R4_FINAL_VALIDATION_REPORT.md \
        STAGE_4A_MAPPO_IMPLEMENTATION_REPORT.md
```

结果文档至少记录：

```text
MAPPO 动作语义
新增文件
网络结构
GAE 终止语义
PPO 更新公式
checkpoint 格式
测试结果
冒烟实验结果
是否修改环境或 MASAC
代码 Commit
push 状态
下一建议任务
```

仓库根目录最终只保留：

```text
STAGE_4A_MAPPO_IMPLEMENTATION_REPORT.md
```

------

## 19. Git 提交

代码、测试和文档规则完成后：

```bash
git status --short
git add src scripts tests AGENTS.md README.md
git commit -m "feat: implement parameter-sharing MAPPO"
git push
```

记录真实代码 Commit SHA。

结果文档完成后：

```bash
git add -A
git status --short
git commit -m "docs: record stage 4A MAPPO implementation"
git push
git status --short
```

提交前确认没有加入：

```text
outputs/
checkpoint
JSON/JSONL 运行产物
缓存
临时日志
```

------

## 20. Codex CLI 最终输出

任务结束后必须打印：

```text
========================================
阶段 4A MAPPO 实现结果
========================================

本轮结果文档：
STAGE_4A_MAPPO_IMPLEMENTATION_REPORT.md

代码 Commit SHA：
<真实 SHA>

结果文档 Commit SHA：
<真实 SHA>

代码 push 结果：
<真实结果>

报告 push 结果：
<真实结果>

完整测试：
<真实 passed 数量和耗时>

MAPPO 相关测试：
<真实结果>

编译检查：
<真实结果>

冒烟实验：
<真实结果>

是否修改环境或 MASAC：
<真实结果>

最终 git status --short：
<真实输出；干净时写 clean>

下一建议任务：
阶段 4B——MAPPO 固定配置训练与 MASAC 公平比较

========================================
```

必须明确显示：

```text
STAGE_4A_MAPPO_IMPLEMENTATION_REPORT.md
```

------

## 21. 禁止事项

本轮禁止：

```text
重新修改 MASAC
修改环境状态转移
修改安全过滤器
修改奖励公式或权重默认值
使用 applied action 计算 PPO ratio
混用 MASAC 与 MAPPO checkpoint
运行 20,000 步正式训练
宣称 MAPPO 已优于任何基线
删除现有测试
提交 outputs/
伪造测试、Commit 或 push 结果
```
