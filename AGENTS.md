# Codex 执行计划：阶段 4C——参数共享 MATD3 与 MADDPG 统一实现

## 1. 本轮目标

当前任务属于正式阶段：

```text
阶段 4C
```

本轮一次性完成：

1. 参数共享 MATD3；
2. 参数共享 MADDPG；
3. 两算法共用的确定性 Actor、训练器、实验运行器和 checkpoint 基础设施；
4. 统一比较器对 MATD3、MADDPG 的支持；
5. 完整单元测试、集成测试和两个短程冒烟实验。

本轮不进行 20,000 步正式训练，不判断两算法性能。

本轮结果文档：

```text
STAGE_4C_MATD3_MADDPG_IMPLEMENTATION_REPORT.md
```

本轮通过后直接进入：

```text
阶段 4D——确定性算法正式训练、十策略比较与核心消融
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

确认当前代码至少包含：

```text
87bd320 feat: compare MAPPO with MASAC and baselines
ec4d7f1 docs: record stage 4B MAPPO comparison
```

出现其他未提交文件时立即停止，不自动恢复。

------

## 3. 统一动作与 Replay 语义

MATD3 和 MADDPG 的 Actor 输出：

```text
requested normalized action ∈ [-1, 1]
```

环境仍执行：

```text
requested action
→ safety filter
→ applied action
→ state transition
```

两算法必须继续使用现有 `MultiAgentReplayBuffer`：

```text
Replay Buffer 保存 applied normalized action
```

不得修改现有 Replay Buffer 数据语义。

Critic 的真实 transition 训练使用：

```text
replay_batch.applied_actions
```

Actor 更新和 target action 使用 Actor 直接产生的 requested action。

报告必须明确：

> 安全过滤器不可微，因此 Actor 更新不是精确的约束策略梯度；这是当前 MASAC、MATD3 和 MADDPG 共享的 off-policy 动作语义限制。

requested/applied mismatch 只用于诊断。

本轮不得通过修改安全过滤器解决该问题。

------

## 4. 文件结构

建议新增：

```text
src/uav_multi_relay/learning/deterministic.py
src/uav_multi_relay/learning/matd3.py
src/uav_multi_relay/learning/maddpg.py

src/uav_multi_relay/training/deterministic_trainer.py
src/uav_multi_relay/training/deterministic_experiment.py
src/uav_multi_relay/training/deterministic_checkpoints.py
src/uav_multi_relay/training/deterministic_evaluator.py

scripts/run_matd3_experiment.py
scripts/run_maddpg_experiment.py

tests/test_deterministic_learning.py
tests/test_deterministic_training.py
tests/test_deterministic_experiment.py
```

允许修改：

```text
src/uav_multi_relay/learning/networks.py
src/uav_multi_relay/learning/__init__.py
src/uav_multi_relay/training/__init__.py
src/uav_multi_relay/analysis/comparison.py
scripts/compare_baselines.py
tests/test_comparison.py
README.md
AGENTS.md
```

禁止修改：

```text
environment.py
safety.py
kinematics.py
奖励公式和默认权重
通信模型
TDMA
MASAC 算法
MAPPO 算法
规则基线
现有 checkpoint 格式
```

------

## 5. 共用网络

### 5.1 参数共享确定性 Actor

新增：

```python
SharedDeterministicActor
```

输入：

```text
(batch, K, local_observation_dim)
```

输出：

```text
(batch, K, action_dim)
```

输出必须通过 `tanh` 限制到：

```text
[-1, 1]
```

所有中继共享 Actor 参数，角色差异由现有局部观测中的角色编码表达。

### 5.2 集中式 Critic

MADDPG 使用：

```python
CentralizedCritic
```

输入：

```text
global state + flattened joint action
```

输出：

```text
Q(s, a1, ..., aK)
```

MATD3 继续使用或复用：

```python
CentralizedTwinCritic
```

不得重复实现相同 MLP 拼接逻辑。

------

## 6. 参数共享 MADDPG

新增：

```python
ParameterSharingMADDPG
```

固定语义：

- 一个共享 Actor；
- 一个集中式 Critic；
- Actor target；
- Critic target；
- 每次 update 都更新 Actor；
- 每次 update 后软更新 Actor target 和 Critic target；
- target Q 使用 target Actor 的联合动作；
- 真实终止不 bootstrap；
- truncated transition 允许 bootstrap。

默认参数：

```text
gamma = 0.99
tau = 0.005
actor learning rate = 3e-4
critic learning rate = 3e-4
```

Actor loss：

```text
-Q(s, actor(local_observations)).mean()
```

Critic loss：

```text
MSE(current Q, target Q)
```

------

## 7. 参数共享 MATD3

新增：

```python
ParameterSharingMATD3
```

必须包含 TD3 的三个核心特征：

1. Twin Critic；
2. Target policy smoothing；
3. Delayed Actor update。

默认参数：

```text
gamma = 0.99
tau = 0.005
actor learning rate = 3e-4
critic learning rate = 3e-4

policy noise std = 0.2
noise clip = 0.5
policy delay = 2
```

Target action：

```text
target_actor(next_observation)
+ clipped Gaussian noise
```

最终裁剪到：

```text
[-1, 1]
```

Target Q：

```text
reward
+ gamma * (1 - terminated)
  * min(target_q1, target_q2)
```

Actor update时使用：

```text
-Q1(s, actor(local_observations)).mean()
```

Actor 和全部 target 网络只在 delayed update 时更新。

不得把 MATD3 实现成仅有 Twin Critic 的 MADDPG。

------

## 8. 训练和探索

两算法共用确定性训练器。

训练流程：

```text
随机动作预热
→ Actor requested action
→ 添加探索高斯噪声
→ clip 到 [-1, 1]
→ 环境安全过滤
→ 保存 applied action
→ 采样 ReplayBatch
→ 算法 update
```

默认正式训练配置留到 4D，本轮冒烟配置可缩小。

训练日志至少包括：

```text
environment steps
total updates
completed episodes
episode return
episode length
mean rate
termination rate
intervention rate
requested/applied mismatch rate

critic loss
actor loss
current Q mean
target Q mean
TD error mean
Actor gradient norm
Critic gradient norm
Actor 是否更新
```

MATD3 额外记录：

```text
policy delay counter
actor update rate
target smoothing noise mean/std/max
```

所有指标必须有限；不得使用 `nan_to_num()` 隐藏异常。

------

## 9. Checkpoint 和实验入口

使用共用 checkpoint 容器，但 metadata 必须包含：

```text
algorithm = matd3 或 maddpg
```

保存：

```text
Actor
Critic 或 Twin Critic
全部 target 网络
全部 optimizer
算法配置
网络维度
environment steps
total updates
completed episodes
```

加载错误算法类型时必须拒绝。

新增两个薄 CLI：

```text
scripts/run_matd3_experiment.py
scripts/run_maddpg_experiment.py
```

两者复用相同实验基础设施，不得复制整套 trainer。

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

输出目录必须为空。

------

## 10. 统一比较支持

将比较器扩展为：

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

CLI 新增：

```text
--matd3-checkpoint
--maddpg-checkpoint
```

要求：

- 仅请求对应算法时才加载 checkpoint；
- 四个学习算法可在同一次比较中运行；
- 所有策略使用同一 episode seeds；
- 所有学习算法使用 deterministic action；
- 比较不得修改模型参数；
- intervention rate 与 requested/applied mismatch rate 在报告中分开显示。

同时修正文字表述：

> 4B 只能证明当前 MAPPO 实现优于当前 MASAC 实现，不能单独证明动作语义是性能差异的原因。

不得修改 4B 历史报告；在本轮报告中写明修正解释即可。

------

## 11. 必须测试

### 网络

- 确定性 Actor shape、范围和有限性；
- 相同输入产生相同输出；
- 单 Critic 和 Twin Critic shape、有限性；
- 动态支持 `K=1` 和 `K=4`。

### MADDPG

- target Q 精确数值；
- terminated 不 bootstrap；
- truncated 可 bootstrap；
- Actor、Critic 和 target 网络实际更新；
- soft update 数值正确；
- 所有指标有限。

### MATD3

- Twin Critic 使用较小 target Q；
- target noise 被正确 clip；
- target action 保持在 `[-1, 1]`；
- policy delay 生效；
- 非 delayed step 不更新 Actor 和 target；
- delayed step 更新 Actor 和 target；
- Q1 用于 Actor loss。

### Replay 和动作语义

- Critic 使用 replay 中 applied action；
- 改变 requested 诊断值不改变 Critic batch；
- Actor update 使用 Actor requested action；
- requested/applied mismatch 仅改变诊断指标。

### Checkpoint

- MATD3 完整往返；
- MADDPG 完整往返；
- 保存前后 deterministic action 和 Q 一致；
- optimizer state 一致；
- 算法类型不匹配时拒绝；
- 恢复后可继续 update。

### 比较器

- MATD3、MADDPG 按需加载；
- 四个学习算法同次运行；
- seeds 一致；
- 模型参数不变；
- 缺失 checkpoint 报清晰错误；
- intervention 和 mismatch 分别输出。

不得删除或弱化现有 194 项测试。

------

## 12. 验证和冒烟

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

要求：

```text
现有 194 项测试全部保留
新增测试全部通过
无新增 Pytest 警告
编译成功
```

### MATD3 冒烟

```bash
python scripts/run_matd3_experiment.py \
  --output-dir outputs/stage4c_matd3_smoke \
  --steps 1000 \
  --max-steps 50 \
  --waypoint-radius 90 \
  --batch-size 64 \
  --random-action-steps 200 \
  --update-after-steps 200 \
  --updates-per-step 1 \
  --evaluation-interval 500 \
  --evaluation-episodes 2 \
  --checkpoint-interval 500 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

### MADDPG 冒烟

```bash
python scripts/run_maddpg_experiment.py \
  --output-dir outputs/stage4c_maddpg_smoke \
  --steps 1000 \
  --max-steps 50 \
  --waypoint-radius 90 \
  --batch-size 64 \
  --random-action-steps 200 \
  --update-after-steps 200 \
  --updates-per-step 1 \
  --evaluation-interval 500 \
  --evaluation-episodes 2 \
  --checkpoint-interval 500 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

两个冒烟实验必须：

- 完成 1000 环境步；
- 至少完成一次参数更新；
- 所有指标有限；
- best/final/周期 checkpoint 完整；
- checkpoint 加载后 deterministic action 一致；
- JSON/JSONL 可读取；
- 输出目录不提交 Git。

冒烟结果不用于性能判断。

------

## 13. 验收、报告和 Git

本轮通过条件：

1. MATD3 和 MADDPG 均完整实现；
2. 两算法共享公共基础设施；
3. MATD3 三项核心机制均正确；
4. MADDPG 单 Critic 语义正确；
5. terminated/truncated 处理正确；
6. Replay 继续保存 applied action；
7. checkpoint 完整；
8. 比较器支持十策略；
9. 完整测试通过；
10. 两个冒烟实验通过；
11. 未修改环境、奖励、MASAC 或 MAPPO；
12. 最终工作区干净。

将当前结果文档改名为：

```bash
git mv STAGE_4B_MAPPO_TRAINING_COMPARISON_REPORT.md \
  STAGE_4C_MATD3_MADDPG_IMPLEMENTATION_REPORT.md
```

代码提交建议：

```bash
git commit -m "feat: implement parameter-sharing MATD3 and MADDPG"
git push
```

报告提交建议：

```bash
git commit -m "docs: record stage 4C deterministic MARL implementation"
git push
```

提交前不得加入：

```text
outputs/
checkpoint
JSON/JSONL 运行产物
缓存
临时日志
```

------

## 14. Codex CLI 最终输出

```text
========================================
阶段 4C MATD3 与 MADDPG 实现结果
========================================

本轮结果文档：
STAGE_4C_MATD3_MADDPG_IMPLEMENTATION_REPORT.md

代码 Commit SHA：
<真实 SHA>

结果文档 Commit SHA：
<真实 SHA>

完整测试：
<真实 passed 数量和耗时>

MATD3 专项测试：
<真实结果>

MADDPG 专项测试：
<真实结果>

MATD3 冒烟：
<真实结果>

MADDPG 冒烟：
<真实结果>

十策略比较支持：
<完成或未完成>

是否修改环境、MAPPO 或 MASAC：
<真实结果>

代码 push：
<真实结果>

报告 push：
<真实结果>

最终 git status --short：
<真实输出；干净时写 clean>

下一建议任务：
阶段 4D——确定性算法正式训练、十策略比较与核心消融

========================================
```
