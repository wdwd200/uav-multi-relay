# Codex 执行计划：阶段 4A-R1——MAPPO 概率比语义与 Checkpoint 验收修复

## 1. 任务性质

当前任务属于：

```text
阶段 4A-R1
```

性质：

```text
阶段 4A 的算法语义验收修复
不新增正式阶段
不开始 4B 正式训练
```

本轮只修复：

1. 将 joint-action PPO ratio 改为标准的 per-relay ratio；
2. 将旧动作负 log probability 改为明确的策略 entropy 近似；
3. 修正无效的 checkpoint 一致性测试；
4. 增加 MAPPO 概率比和 entropy 的精确测试；
5. 整理本轮直接修改文件的可读性；
6. 完成短冒烟验证。

本轮通过后直接进入：

```text
阶段 4B——MAPPO 固定配置训练与公平比较
```

本轮结果文档必须命名为：

```text
STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md
```

------

## 2. 启动检查

运行：

```bash
git status --short
git branch --show-current
git log -3 --oneline
```

允许且只允许：

```text
M AGENTS.md
```

该文件是本轮授权计划。

如果存在其他未提交文件，立即停止并列出，不自动恢复。

确认当前代码包含：

```text
34ca5b4 feat: implement parameter-sharing MAPPO
```

------

## 3. 修正 Actor 概率接口

修改：

```text
src/uav_multi_relay/learning/networks.py
```

`SharedGaussianActor.evaluate_actions()` 必须返回：

```python
per_relay_log_probability
per_relay_entropy
```

形状均为：

```text
(batch, num_relays, 1)
```

可以额外返回 joint log probability 用于诊断，但 PPO loss 不得依赖 joint ratio。

### 3.1 per-relay log probability

保持 tanh 反变换和 Jacobian 修正：

```python
bounded = actions.clamp(-1 + epsilon, 1 - epsilon)
pre_tanh = atanh(bounded)

per_relay_log_probability = (
    Normal(mean, std).log_prob(pre_tanh)
    - log(1 - bounded**2 + epsilon)
).sum(dim=-1, keepdim=True)
```

### 3.2 entropy

不得继续使用：

```python
entropy = -log_probability_of_rollout_action
```

改为明确的 pre-tanh Gaussian entropy approximation：

```python
per_relay_entropy = Normal(mean, std).entropy().sum(
    dim=-1,
    keepdim=True,
)
```

必须在 docstring 和报告中说明：

```text
这是 pre-tanh Gaussian entropy approximation，
不是精确的 squashed-action entropy。
```

所有输出必须有限。

------

## 4. Rollout 保存 per-relay old log probability

修改：

```text
src/uav_multi_relay/learning/mappo.py
src/uav_multi_relay/training/mappo_trainer.py
```

将 rollout 中的：

```text
old_joint_log_probabilities
```

改为：

```text
old_per_relay_log_probabilities
```

每一步 shape：

```text
(K, 1)
```

完整 rollout shape：

```text
(rollout_steps, K, 1)
```

`MAPPOAgent.act_with_stats()` 应返回：

```text
requested action
per-relay log probabilities
value
```

不得在采集时先求和并丢失 per-relay 数据。

如需要 joint log probability 做诊断，可以从 per-relay 值临时求和，不得作为 PPO ratio 输入。

------

## 5. 修正 PPO Actor loss

当前团队 advantage shape 为：

```text
(batch, 1)
```

归一化后广播为：

```text
(batch, K, 1)
```

每个中继分别计算：

```python
ratio = exp(
    new_per_relay_log_probability
    - old_per_relay_log_probability
)
```

再计算：

```python
unclipped = ratio * normalized_advantage
clipped = clamp(
    ratio,
    1 - clip_ratio,
    1 + clip_ratio,
) * normalized_advantage
```

Actor surrogate loss：

```python
surrogate_loss = -minimum(
    unclipped,
    clipped,
).mean()
```

Entropy bonus：

```python
entropy_bonus = per_relay_entropy.mean()
```

最终：

```python
policy_loss = (
    surrogate_loss
    - entropy_coefficient * entropy_bonus
)
```

平均范围必须同时覆盖：

```text
mini-batch 时间样本
全部 K 个中继
```

不得先将 log probability 在中继维求和后再计算 ratio。

Value Critic、GAE、梯度裁剪和 optimizer 顺序保持不变。

------

## 6. 更新指标

继续保留：

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

定义修正为：

### approx_kl

对 per-relay 值求平均：

```python
mean(
    old_per_relay_log_probability
    - new_per_relay_log_probability
)
```

### clip_fraction

对所有：

```text
batch × relay
```

元素计算被 clip 的比例。

### entropy

记录：

```text
per-relay pre-tanh Gaussian entropy 的全体均值
```

不得把旧动作的负 log probability 命名为 entropy。

------

## 7. Checkpoint 兼容性

MAPPO checkpoint 网络结构没有变化，因此格式版本可以保持 `1`。

必须验证：

1. 保存前 Agent 的确定性 action；
2. 加载后 Agent 的确定性 action；
3. 保存前 Value Critic 输出；
4. 加载后 Value Critic 输出；
5. Actor 参数；
6. Value Critic 参数；
7. Actor optimizer state；
8. Critic optimizer state；
9. MAPPOConfig；
10. metadata。

全部一致。

不得继续使用：

```python
loaded.act(...) == loaded.act(...)
```

这种自比较断言。

------

## 8. 必须新增和修正的测试

主要修改：

```text
tests/test_mappo.py
tests/test_mappo_training.py
tests/test_mappo_experiment.py
```

### 8.1 采样与评估一致

从 Actor 随机采样 requested action，验证：

```text
sample() 返回的 per-relay log probability
与
evaluate_actions() 对同一 action 的 per-relay log probability
```

逐元素近似相等。

不得只测试 deterministic mean action。

### 8.2 per-relay ratio 精确测试

构造两个中继的固定 old/new log probability，验证 ratio 逐中继计算。

例如：

```text
relay 1 ratio = 1.1
relay 2 ratio = 0.9
```

必须保留两个独立 ratio，不得变成：

```text
joint ratio = 0.99
```

### 8.3 `K=1` 等价性

`num_relays=1` 时，per-relay PPO loss 应等于对应单智能体 PPO loss。

### 8.4 中继置换不变性

对 rollout 的 relay 维做相同置换：

```text
observations
requested actions
applied actions
old log probabilities
```

在关闭随机 mini-batch 差异或固定 RNG 后，Actor loss 和核心指标应一致。

### 8.5 applied action 隔离

保持 requested action、old log probability 和 RNG 相同，只改变 applied action。

验证：

```text
Actor 参数更新一致
Value Critic 参数更新一致
policy loss 一致
ratio/clip fraction/approx KL 一致
```

只允许 mismatch 诊断指标变化。

### 8.6 entropy 测试

验证 entropy：

- shape 为 `(batch, K, 1)`；
- 全部有限；
- 增大 `log_std` 时 entropy 增大；
- 改变 rollout action 但保持策略分布不变时，entropy 不变。

### 8.7 checkpoint 真正往返

保存前记录原 Agent 的 action、value、参数和 optimizer state。

加载后与原 Agent 比较，不得加载后自比较。

### 8.8 partial rollout 行为

保留当前“未满 rollout 不更新”的语义，但必须：

- 在训练 summary 中记录 `discarded_partial_rollout_steps`；
- 正式实验时能够看出有多少 on-policy 样本未用于更新。

本轮不实现 partial rollout 更新。

------

## 9. 可读性要求

本轮直接修改的 MAPPO 文件不得继续新增一行多个语句的写法。

至少整理：

```text
networks.py 中新增/修改方法
mappo.py 中 rollout 与 update
mappo_trainer.py 中直接相关逻辑
三个 MAPPO 测试文件
```

要求：

- 每条主要语句独立一行；
- 公共类和复杂函数有简短 docstring；
- 不进行全仓库无关格式化；
- 不改变未涉及代码的行为。

顺便修正 README 标题结构：

```text
## MAPPO Training
## MASAC Training
```

两部分内容不得混在同一标题下。

------

## 10. 验证

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

- 原有 186 项测试全部保留；
- 新增测试全部通过；
- 测试总数高于 186；
- 无新增 Pytest 警告；
- 编译成功。

------

## 11. 冒烟实验

使用新的空目录：

```text
outputs/stage4a_r1_mappo_semantics_smoke
```

运行：

```bash
python scripts/run_mappo_experiment.py \
  --output-dir outputs/stage4a_r1_mappo_semantics_smoke \
  --steps 1000 \
  --rollout-steps 250 \
  --max-steps 50 \
  --waypoint-radius 90 \
  --update-epochs 2 \
  --mini-batch-size 125 \
  --evaluation-interval 500 \
  --evaluation-episodes 2 \
  --checkpoint-interval 500 \
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

1. 完成 1000 环境步；
2. 完成 4 次 PPO 更新；
3. `discarded_partial_rollout_steps = 0`；
4. entropy、KL、clip fraction、梯度全部有限；
5. best/final checkpoint 可加载；
6. 保存前后 action 和 value 一致；
7. JSON/JSONL 全部可读取；
8. 未修改环境或 MASAC。

冒烟结果不用于性能判断。

------

## 12. 本轮结果文档

将当前结果文档改名：

```bash
git mv STAGE_4A_MAPPO_IMPLEMENTATION_REPORT.md \
  STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md
```

结果文档至少记录：

```text
旧 joint-ratio 语义
新 per-relay-ratio 语义
为什么修正
old log probability 的新 shape
entropy 的新定义
新增测试
checkpoint 测试修正
partial rollout 统计
完整测试结果
冒烟结果
是否修改环境或 MASAC
代码 Commit 和 push
```

仓库根目录最终只保留：

```text
STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md
```

------

## 13. 验收标准

必须同时满足：

1. PPO ratio 按中继分别计算；
2. 团队 advantage 正确广播；
3. Actor loss 对 batch 和 relay 维共同平均；
4. applied action 不参与 PPO ratio；
5. entropy 不再是旧动作负 log probability；
6. entropy 定义明确为 pre-tanh Gaussian approximation；
7. checkpoint 测试比较原 Agent 与加载 Agent；
8. GAE 语义未改变；
9. Value Critic 语义未改变；
10. 完整测试通过；
11. MAPPO 专项测试通过；
12. 冒烟实验通过；
13. 未修改环境、MASAC、奖励和安全过滤；
14. 最终工作区干净；
15. CLI 显示结果文档完整名称。

通过后下一任务固定为：

```text
阶段 4B——MAPPO 固定配置训练与公平比较
```

------

## 14. Git 提交

代码与测试：

```bash
git status --short
git add \
  src/uav_multi_relay/learning/networks.py \
  src/uav_multi_relay/learning/mappo.py \
  src/uav_multi_relay/training/mappo_trainer.py \
  tests/test_mappo.py \
  tests/test_mappo_training.py \
  tests/test_mappo_experiment.py \
  README.md \
  AGENTS.md
git commit -m "fix: use per-relay MAPPO probability ratios"
git push
```

只添加实际修改文件。

报告：

```bash
git add -A
git status --short
git commit -m "docs: record stage 4A-R1 MAPPO semantics repair"
git push
git status --short
```

提交前确认未加入：

```text
outputs/
checkpoint
JSON/JSONL 运行产物
缓存
临时日志
```

------

## 15. Codex CLI 最终输出

```text
========================================
阶段 4A-R1 MAPPO 语义修复结果
========================================

本轮结果文档：
STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md

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

MAPPO 专项测试：
<真实结果>

PPO ratio 语义：
per-relay / 未通过

Entropy 定义：
<真实结果>

Checkpoint 往返：
<真实结果>

冒烟实验：
<真实结果>

是否修改环境或 MASAC：
<真实结果>

最终 git status --short：
<真实输出；干净时写 clean>

下一建议任务：
阶段 4B——MAPPO 固定配置训练与公平比较

========================================
```

------

## 16. 禁止事项

本轮禁止：

```text
开始 20,000 步正式训练
使用 joint-action ratio
使用 applied action 计算 PPO ratio
把旧动作负 log probability 当作 entropy
修改环境
修改安全过滤器
修改奖励
修改 MASAC
删除现有测试
提交 outputs/
为普通格式问题继续新增修复阶段
伪造测试、Commit 或 push 结果
```
