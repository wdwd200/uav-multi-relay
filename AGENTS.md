# Codex 执行计划：阶段 4D——确定性算法正式训练、十策略比较与通信模型敏感性

## 1. 本轮目标

当前任务属于正式阶段：

```text
阶段 4D
```

本轮一次完成：

1. 修复 MATD3/MADDPG Replay Buffer 未设 seed 的可复现性问题；
2. 补强确定性算法关键语义测试；
3. 从头完成 MATD3 20,000 步训练；
4. 从头完成 MADDPG 20,000 步训练；
5. 完成四种学习算法加六种基线的十策略统一比较；
6. 完成 TDMA 和天线模型的冻结策略敏感性实验；
7. 输出统一结果报告。

不得把上述事项再拆成单独修复轮次。

本轮结果文档：

```text
STAGE_4D_TEN_POLICY_COMPARISON_SENSITIVITY_REPORT.md
```

本轮完成后进入：

```text
阶段 4E——核心结构与动态场景消融
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

确认当前代码包含：

```text
1c9bbfb feat: implement parameter-sharing MATD3 and MADDPG
0642944 docs: record stage 4C deterministic MARL implementation
```

确认以下 checkpoint 存在：

```text
outputs/stage3g_r4_final_seed0/best_checkpoint.pt
outputs/stage4b_mappo_seed0/best_checkpoint.pt
```

缺失任一 checkpoint 时立即停止，不重新训练 MASAC 或 MAPPO。

------

## 3. 正式训练前的阻断修复

### 3.1 Replay Buffer seed

修改：

```text
scripts/_run_deterministic_experiment.py
```

创建 Replay Buffer 时必须传入：

```python
seed=args.seed
```

即：

```python
buffer = MultiAgentReplayBuffer(
    capacity=args.replay_capacity,
    num_relays=config.num_relays,
    local_observation_dim=local_dim,
    global_state_dim=global_dim,
    action_dim=agent.action_dim,
    seed=args.seed,
)
```

不得使用额外随机 seed。

### 3.2 可复现性测试

使用相同：

```text
算法
环境配置
训练 seed
Replay Buffer seed
Torch seed
NumPy seed
```

运行两次短训练，验证除输出路径和计时外：

```text
训练日志一致
评估日志一致
summary 核心数值一致
final Actor 参数一致
final Critic 参数一致
```

再使用不同 seed，验证至少一个模型参数或训练指标不同。

### 3.3 训练指标命名

将训练器当前的：

```text
termination_rate
```

改为更准确的：

```text
termination_event_rate_per_step
```

同时新增：

```text
terminated_episode_rate
mean_episode_length
mean_episode_return
```

不得改变环境终止逻辑。

旧日志字段如需兼容，可以保留并标记 deprecated，但新报告不得把每步事件率解释成 episode 终止率。

------

## 4. 补强关键测试

不单独增加修复阶段，直接在本轮补齐。

至少增加：

### MADDPG

- `terminated=True` 时 target 只有 reward；
- `truncated=True, terminated=False` 时仍 bootstrap；
- Actor、Critic和两个 target 网络都按预期更新；
- Polyak update 的精确数值测试。

### MATD3

- target Q 使用 `min(target_q1, target_q2)`；
- target noise 被 `noise_clip` 限制；
- target action 被限制到 `[-1,1]`；
- 非 delayed step 不更新 Actor 和 target 网络；
- delayed step 更新 Actor 和全部 target 网络；
- checkpoint 恢复后 `update_count` 和 delay 节奏保持一致。

### Checkpoint

保存前与加载后比较：

```text
Actor
Target Actor
Critic/Twin Critic
Target Critic
全部 optimizer state
确定性 action
Q 输出
配置
metadata
```

不得只比较 Actor action。

------

## 5. 代码可读性整理

仅整理本轮直接涉及的文件：

```text
src/uav_multi_relay/learning/deterministic.py
src/uav_multi_relay/training/deterministic_trainer.py
src/uav_multi_relay/training/deterministic_experiment.py
src/uav_multi_relay/training/deterministic_checkpoints.py
scripts/_run_deterministic_experiment.py
相关测试
```

要求：

- 不再新增一行多个主要语句；
- 复杂方法拆分为清晰步骤；
- 公共类和函数保留简短 docstring；
- 不做全仓库无关格式化；
- 不改变算法公式。

------

## 6. 完整验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

要求：

```text
现有 200 项测试全部保留
新增测试全部通过
无新增 Pytest 警告
编译成功
```

正式训练开始前，必须先完成两次同 seed 的短训练复现测试。

------

## 7. MATD3 正式训练

输出目录：

```text
outputs/stage4d_matd3_seed0
```

运行：

```bash
python scripts/run_matd3_experiment.py \
  --output-dir outputs/stage4d_matd3_seed0 \
  --steps 20000 \
  --max-steps 250 \
  --waypoint-radius 90 \
  --batch-size 256 \
  --replay-capacity 100000 \
  --random-action-steps 2000 \
  --update-after-steps 2000 \
  --updates-per-step 1 \
  --exploration-noise-std 0.1 \
  --log-interval 1000 \
  --evaluation-interval 2500 \
  --evaluation-episodes 5 \
  --checkpoint-interval 2500 \
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

算法配置固定使用：

```text
gamma = 0.99
tau = 0.005
actor learning rate = 3e-4
critic learning rate = 3e-4
policy noise std = 0.2
noise clip = 0.5
policy delay = 2
```

不得根据结果修改参数。

------

## 8. MADDPG 正式训练

输出目录：

```text
outputs/stage4d_maddpg_seed0
```

运行：

```bash
python scripts/run_maddpg_experiment.py \
  --output-dir outputs/stage4d_maddpg_seed0 \
  --steps 20000 \
  --max-steps 250 \
  --waypoint-radius 90 \
  --batch-size 256 \
  --replay-capacity 100000 \
  --random-action-steps 2000 \
  --update-after-steps 2000 \
  --updates-per-step 1 \
  --exploration-noise-std 0.1 \
  --log-interval 1000 \
  --evaluation-interval 2500 \
  --evaluation-episodes 5 \
  --checkpoint-interval 2500 \
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

算法配置固定使用：

```text
gamma = 0.99
tau = 0.005
actor learning rate = 3e-4
critic learning rate = 3e-4
```

不得根据结果修改参数。

------

## 9. 正式训练验收

两种算法都必须完成：

```text
20,000 environment steps
18,001 updates
周期 checkpoint
best checkpoint
final checkpoint
全部日志有限
```

分别报告：

```text
各 checkpoint mean return
mean rate
terminated episode rate
mean episode length
intervention rate
requested/applied mismatch rate
Actor loss
Critic loss
Q mean
TD error
Actor/Critic gradient
```

不得因为结果差而提前停止。

------

## 10. 十策略统一比较

checkpoint：

```text
MAPPO:
outputs/stage4b_mappo_seed0/best_checkpoint.pt

MASAC:
outputs/stage3g_r4_final_seed0/best_checkpoint.pt

MATD3:
outputs/stage4d_matd3_seed0/best_checkpoint.pt

MADDPG:
outputs/stage4d_maddpg_seed0/best_checkpoint.pt
```

比较策略：

```text
mappo
masac
matd3
maddpg
random
stationary
equal_spacing
weighted_spacing
greedy
mpc
```

固定：

```text
episodes = 10
seeds = 20000–20009
max_steps = 250
waypoint radius = 90
greedy sweeps = 1
MPC horizon = 2
MPC population = 8
MPC iterations = 2
```

输出目录：

```text
outputs/stage4d_ten_policy_comparison
```

必须分别报告：

```text
mean return
return std
mean return per step
mean rate
minimum rate
terminated episode rate
mean episode length
intervention rate
requested/applied mismatch rate
mean action computation time
```

不得将 intervention 与 mismatch 合并成一个字段。

------

## 11. TDMA 与天线冻结策略敏感性

本轮只做：

```text
冻结策略敏感性
```

不是重新训练后的正式消融。

### 11.1 配置支持

为环境增加保持向后兼容的配置：

```text
tdma_mode = "optimal" | "equal"
antenna_mode = "dipole" | "isotropic"
```

默认必须保持：

```text
tdma_mode = "optimal"
antenna_mode = "dipole"
```

默认配置下的环境结果必须与修改前一致。

实现：

- `equal` 使用现有 `equal_tdma_rate()`；
- `isotropic` 使用收发天线增益 `1.0`；
- 不修改其他信道参数；
- 配置必须经过校验；
- run config 和比较输出必须记录两个模式。

### 11.2 敏感性策略

使用冻结 checkpoint 和同一 seeds，比较：

```text
mappo
masac
matd3
maddpg
stationary
equal_spacing
```

场景：

```text
A：optimal TDMA + dipole antenna
B：equal TDMA + dipole antenna
C：optimal TDMA + isotropic antenna
```

每个场景：

```text
10 episodes
seeds 20000–20009
```

不得重新训练策略。

报告必须使用准确名称：

```text
冻结策略通信模型敏感性
```

不得称为完整的“重新训练消融实验”。

------

## 12. 客观结果分类

不设定某个学习算法必须获胜。

按结果分类：

### A

至少一种学习算法超过 Stationary，并降低终止率。

### B

学习算法优于 Random，但仍低于规则基线。

### C

所有学习算法仍明显低于规则基线。

### D

训练或比较因程序错误未完成。

性能差不属于代码错误，不得在 4D 内继续调参。

------

## 13. 结果文档

将当前报告改名：

```bash
git mv STAGE_4C_MATD3_MADDPG_IMPLEMENTATION_REPORT.md \
  STAGE_4D_TEN_POLICY_COMPARISON_SENSITIVITY_REPORT.md
```

报告至少记录：

```text
Replay Buffer seed 修复
同 seed 复现验证
新增和恢复测试
MATD3 完整训练
MADDPG 完整训练
十策略完整结果
TDMA 敏感性
天线敏感性
冻结策略实验限制
四种学习算法动作语义
结果分类
代码和报告 Commit
最终工作区状态
```

------

## 14. 验收标准

必须同时满足：

1. Replay Buffer 使用训练 seed；
2. 同 seed 短训练可重复；
3. 不同 seed 产生不同结果；
4. 完整测试通过；
5. MATD3 20,000 步完成；
6. MADDPG 20,000 步完成；
7. 两算法 checkpoint 完整；
8. 十策略全部完成 10 episode；
9. 三个通信敏感性场景完成；
10. 默认环境行为未改变；
11. 未修改 MASAC 或 MAPPO 算法；
12. 未筛选失败 episode；
13. `outputs/` 未提交；
14. 最终工作区干净。

通过后进入：

```text
阶段 4E——核心结构与动态场景消融
```

------

## 15. Git 与 CLI 输出

建议代码提交：

```text
fix: make deterministic MARL experiments reproducible
feat: add deterministic training comparison and communication sensitivity
```

结果报告提交：

```text
docs: record stage 4D ten-policy comparison
```

Codex CLI 最终必须输出：

```text
========================================
阶段 4D 正式训练与比较结果
========================================

本轮结果文档：
STAGE_4D_TEN_POLICY_COMPARISON_SENSITIVITY_REPORT.md

代码 Commit SHA：
<真实 SHA>

结果文档 Commit SHA：
<真实 SHA>

完整测试：
<真实结果>

可复现性验证：
<同 seed 是否一致>

MATD3 训练：
<完整结果>

MADDPG 训练：
<完整结果>

十策略比较：
<是否完整及四个学习算法核心指标>

TDMA 敏感性：
<核心结果>

天线敏感性：
<核心结果>

性能分类：
<A / B / C / D>

代码 push：
<真实结果>

报告 push：
<真实结果>

最终 git status --short：
<真实输出；干净时写 clean>

下一建议任务：
阶段 4E——核心结构与动态场景消融

========================================
```
