# Codex 执行计划：阶段 3D——MASAC 环境采集与训练循环

## 1. 目标

实现参数共享 MASAC 的真实环境采集和训练闭环：

```text
环境 reset
→ 选择请求动作
→ 环境安全过滤并执行
→ 保存 applied_action
→ Replay Buffer 采样
→ MASAC 更新
→ episode 结束后 reset
```

本次不实现 checkpoint、独立评估器、图表或多随机种子实验。

开始工作后，用本计划覆盖根目录 `AGENTS.md`。

## 2. 文件范围

新增：

```text
src/uav_multi_relay/training/__init__.py
src/uav_multi_relay/training/trainer.py
scripts/train.py
tests/test_training.py
```

修改：

```text
README.md
AGENTS.md
aaa.md
```

不得修改环境、通信、奖励、MPC、Replay Buffer 或 MASAC 更新公式，除非测试发现确定性错误。

## 3. 训练配置与返回结果

在 `trainer.py` 实现：

```python
@dataclass(frozen=True)
class MASACTrainingConfig:
    total_environment_steps: int = 10_000
    replay_capacity: int = 100_000
    batch_size: int = 256
    random_action_steps: int = 1_000
    update_after_steps: int = 1_000
    updates_per_step: int = 1
    seed: int = 0
```

要求：

- `total_environment_steps`、`replay_capacity`、`batch_size`、`updates_per_step` 为正整数；
- `random_action_steps` 和 `update_after_steps` 为非负整数；
- `batch_size <= replay_capacity`；
- 布尔值不得作为整数；
- 非法配置抛出 `ValueError`。

实现：

```python
@dataclass(frozen=True)
class MASACTrainingSummary:
    total_environment_steps: int
    total_updates: int
    completed_episodes: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    mean_rate_e2e_bps: float
    intervention_rate: float
    last_update_metrics: MASACUpdateMetrics | None
```

所有统计值必须有限。

## 4. 训练函数

实现：

```python
def train_masac(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    replay_buffer: MultiAgentReplayBuffer,
    config: MASACTrainingConfig,
) -> MASACTrainingSummary:
```

### 4.1 开始时验证

调用：

```python
observation, info = env.reset(seed=config.seed)
```

根据实际 observation 验证：

- 中继数量一致；
- Actor 局部观测维度一致；
- Critic 全局状态维度一致；
- Replay Buffer 的中继数、观测维度和动作维度一致；
- 动作维度必须为 3。

不得写死局部或全局观测维度。

### 4.2 动作选择

使用独立随机数生成器：

```python
rng = np.random.default_rng(config.seed)
```

当已采集步数小于 `random_action_steps` 时：

```python
rng.uniform(-1.0, 1.0, size=(K, 3))
```

之后调用：

```python
agent.act(observation["local"], deterministic=False)
```

### 4.3 Transition 保存

调用环境：

```python
next_observation, reward, terminated, truncated, info = env.step(requested_actions)
```

Replay Buffer 必须保存：

```python
info["applied_relay_actions"]
```

不得保存请求动作。

同时保存：

- 当前 local observation；
- 当前 global state；
- reward；
- 下一 local observation；
- 下一 global state；
- terminated；
- truncated。

### 4.4 参数更新

Transition 写入后，只有同时满足以下条件才更新：

```text
已采集步数 >= update_after_steps
buffer.size >= batch_size
```

每个满足条件的环境步执行：

```text
updates_per_step
```

次更新。

每次均重新从 Replay Buffer 采样，不得重复使用同一个 batch。

### 4.5 Episode 处理

遇到：

```text
terminated or truncated
```

时：

1. 记录 episode return 和 episode length；

2. episode 计数加一；

3. 使用以下 seed reset：

   ```python
   config.seed + completed_episodes
   ```

4. 不得对已经结束的环境继续调用 `step()`。

训练必须精确执行 `total_environment_steps` 个环境步。

### 4.6 统计

跨全部环境步统计：

- 平均 `info["rate_e2e_bps"]`；
- intervention rate。

一次环境步中，只要任一：

```python
info["intervention_norms"] > 1e-9
```

该步就记为发生安全干预。

不在训练器中重新计算通信或奖励。

## 5. 训练脚本

新增：

```text
scripts/train.py
```

至少支持：

```text
--steps
--seed
--num-relays
--batch-size
--random-action-steps
--update-after-steps
--updates-per-step
--device
```

脚本必须：

1. 在创建网络前设置 NumPy和 PyTorch seed；
2. 从环境 reset 的 observation 自动得到观测维度；
3. 创建 MASAC、Replay Buffer 和训练配置；
4. 调用 `train_masac()`；
5. 最后打印一份简洁 JSON 摘要。

本次不写 checkpoint 和训练日志文件。

## 6. 测试

新增 `tests/test_training.py`，至少覆盖：

1. 配置参数验证；
2. 精确执行指定环境步数；
3. warm-up 期间使用随机动作；
4. warm-up 后调用 Actor；
5. Replay Buffer 保存的是 `applied_relay_actions`，不是请求动作；
6. Buffer 或步数不足时不更新；
7. 满足条件后更新次数准确；
8. 截断后自动 reset，不会继续 step 已结束环境；
9. 相同 seed 的纯随机采集结果可复现；
10. 支持 `K=1` 和 `K=4`；
11. 使用小网络和短流程完成真实 MASAC 更新，所有指标有限。

测试应使用短流程和较小隐藏层，不得运行正式训练规模。

## 7. README

增加：

- 当前已经具备环境采集和 MASAC 训练循环；
- 训练脚本最小示例；
- 明确 checkpoint、独立评估、多随机种子和完整实验仍未实现。

## 8. 验证与 Git

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
python scripts/train.py \
  --steps 30 \
  --batch-size 4 \
  --random-action-steps 4 \
  --update-after-steps 4 \
  --updates-per-step 1 \
  --seed 0
```

所有命令必须成功，训练摘要中的数值必须有限。

提交代码：

```bash
git add AGENTS.md README.md scripts/train.py \
  src/uav_multi_relay/training tests/test_training.py
git commit -m "stage-3: add MASAC training loop"
git push
```

随后覆盖写入 `aaa.md`，必须记录最终实际测试数量，不得沿用旧数字：

```markdown
# 本次执行结果

- 阶段：3D
- 任务：MASAC 环境采集与训练循环
- 完成状态：
- 修改和新增文件：
- 训练流程：
- 默认训练配置：
- 测试结果：
- 训练脚本冒烟结果：
- 编译验证：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3E——Checkpoint 与独立评估
```

再提交并推送：

```bash
git add aaa.md
git commit -m "docs: record MASAC training loop result"
git push
git status --short
```

若任何 Git 命令弹出 `git.exe` 内存读取错误：

- 立即停止，不自动重试；
- 不运行 `git reset --hard`、`git gc` 或 `git prune`；
- 记录发生错误的完整命令；
- 在 `aaa.md` 中如实记录提交和推送实际状态；
- 不虚构 Commit ID 或推送成功。

最终工作区必须干净。
