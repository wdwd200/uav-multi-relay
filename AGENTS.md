# Codex 执行计划：阶段 3E——Checkpoint 与独立评估

## 1. 本次目标

完成两项功能：

1. 保存和加载 MASAC Agent checkpoint；
2. 使用确定性 Actor 独立评估已保存的模型。

本阶段的 checkpoint 包含：

- Actor；
- Critic；
- Target Critic；
- Alpha；
- 三个优化器；
- Agent 架构和超参数；
- 训练步数等元数据。

本阶段不保存：

- Replay Buffer；
- 环境当前 episode 状态；
- 完整环境配置快照。

因此 checkpoint 可用于模型评估和继续优化器状态，但不宣称能够逐位恢复中断时的完整训练轨迹。

开始后使用本计划覆盖 `AGENTS.md`。

------

## 2. 文件范围

新增：

```text
src/uav_multi_relay/training/checkpoints.py
src/uav_multi_relay/training/evaluator.py
scripts/evaluate.py
tests/test_checkpoints.py
tests/test_evaluation.py
```

修改：

```text
src/uav_multi_relay/learning/masac.py
src/uav_multi_relay/training/__init__.py
src/uav_multi_relay/training/trainer.py
scripts/train.py
README.md
AGENTS.md
aaa.md
```

不得修改环境、通信、奖励、安全过滤、MPC 和 MASAC 更新公式。

------

## 3. Checkpoint

### 3.1 Agent 配置记录

在 `ParameterSharingMASAC` 中保存：

```python
self.hidden_dims
```

必须是经过验证后的不可变整数元组。

Checkpoint 中记录：

```text
local_observation_dim
global_state_dim
num_relays
action_dim
hidden_dims
gamma
tau
actor_learning_rate
critic_learning_rate
alpha_learning_rate
initial_alpha
target_entropy
```

### 3.2 接口

在 `training/checkpoints.py` 中实现：

```python
@dataclass(frozen=True)
class MASACCheckpointMetadata:
    environment_steps: int
    updates: int
    completed_episodes: int
```

实现：

```python
def save_masac_checkpoint(
    path: str | Path,
    agent: ParameterSharingMASAC,
    metadata: MASACCheckpointMetadata,
) -> Path:
```

以及：

```python
def load_masac_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[ParameterSharingMASAC, MASACCheckpointMetadata]:
```

Checkpoint 格式必须包含：

```text
format_version = 1
agent_config
actor_state_dict
critic_state_dict
target_critic_state_dict
actor_optimizer_state_dict
critic_optimizer_state_dict
alpha_optimizer_state_dict
log_alpha
metadata
```

要求：

- 使用 `torch.save()`；
- 使用同目录临时文件并通过 `os.replace()` 原子替换；
- 保存失败时清理临时文件；
- 加载时使用 `map_location`；
- 优先使用 `torch.load(..., weights_only=True)`；
- 缺少字段、版本错误、非法元数据或架构不一致时抛出 `ValueError`；
- 加载后 Target Critic 参数仍为 `requires_grad=False`；
- 不接受来源不可信的 checkpoint。

------

## 4. 独立评估器

在 `training/evaluator.py` 中实现：

```python
@dataclass(frozen=True)
class MASACEvaluationConfig:
    episodes: int = 10
    seed: int = 0
```

实现单 episode 结果和汇总结果数据类，至少包含：

```text
episode return
episode length
平均端到端速率
最低端到端速率
安全过滤介入率
terminated
truncated
```

汇总至少包含：

```text
episode 数量
平均 return
return 标准差
平均端到端速率
全部 episode 中的最低速率
平均介入率
terminated episode 比例
每个 episode 的详细结果
```

实现：

```python
def evaluate_masac(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    config: MASACEvaluationConfig,
) -> MASACEvaluationSummary:
```

要求：

1. 深拷贝传入环境，不修改原环境；

2. 每个 episode 使用：

   ```python
   seed = config.seed + episode_index
   ```

3. 始终调用：

   ```python
   agent.act(local_observation, deterministic=True)
   ```

4. 不写 Replay Buffer；

5. 不执行 Agent 更新；

6. 不修改模型参数或优化器状态；

7. 运行至 `terminated` 或 `truncated`；

8. 直接使用环境 reward 和 `info` 统计指标；

9. 所有返回统计必须有限。

------

## 5. 脚本集成

### 5.1 train.py

增加可选参数：

```text
--checkpoint-out
```

训练完成后，若提供该参数，则保存 Agent checkpoint。

元数据使用训练摘要中的：

```text
total_environment_steps
total_updates
completed_episodes
```

JSON 摘要增加：

```text
checkpoint_path
```

未指定时不得创建 checkpoint。

同时在 `train_masac()` 开始时增加一致性检查：

```python
replay_buffer.capacity == config.replay_capacity
```

不一致时抛出 `ValueError`。

### 5.2 evaluate.py

支持：

```text
--checkpoint
--episodes
--seed
--device
```

流程：

1. 加载 checkpoint；
2. 根据 checkpoint 的 `num_relays` 创建环境；
3. reset 后核对 observation 维度；
4. 调用 `evaluate_masac()`；
5. 输出有限 JSON 汇总。

不得在评估脚本中进行训练或参数更新。

------

## 6. 测试

### Checkpoint 测试

至少覆盖：

- 保存后文件存在且没有残留临时文件；
- 完成至少一次更新后保存和加载；
- 加载前后确定性动作完全一致；
- Actor、Critic、Target Critic 和 Alpha 状态一致；
- 优化器状态已恢复；
- 原 Agent 与加载 Agent 在相同随机种子、相同 batch 下继续更新后结果一致；
- 元数据正确恢复；
- 错误版本、缺少字段和损坏文件被拒绝；
- Target Critic 加载后仍冻结。

### 评估测试

至少覆盖：

- 精确执行配置的 episode 数量；
- episode seed 按顺序递增；
- 始终使用确定性动作；
- 不修改传入环境；
- 不修改 Agent 参数和优化器状态；
- 所有指标有限；
- 支持 `K=1` 和 `K=4`；
- 提前终止和正常截断均能统计；
- 从 checkpoint 加载后可以完成真实环境评估。

测试使用小网络和短 episode。

------

## 7. README 与验证

README 增加：

- 保存 checkpoint 示例；
- 加载并评估示例；
- checkpoint 包含和不包含的内容；
- 明确当前尚不能精确恢复 Replay Buffer 和 episode 中间状态。

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

训练并保存 checkpoint：

```bash
python scripts/train.py \
  --steps 20 \
  --batch-size 4 \
  --random-action-steps 4 \
  --update-after-steps 4 \
  --seed 0 \
  --checkpoint-out masac_smoke.pt
```

评估：

```bash
python scripts/evaluate.py \
  --checkpoint masac_smoke.pt \
  --episodes 2 \
  --seed 100 \
  --device cpu
```

两个脚本输出的 JSON 必须有限。

测试完成后删除根目录中的 `masac_smoke.pt`，不得提交测试生成物。

------

## 8. Git 与结果记录

提交代码：

```bash
git add AGENTS.md README.md scripts/train.py scripts/evaluate.py \
  src/uav_multi_relay/learning/masac.py \
  src/uav_multi_relay/training \
  tests/test_checkpoints.py tests/test_evaluation.py
git commit -m "stage-3: add MASAC checkpoints and evaluation"
git push
```

随后覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3E
- 任务：MASAC Checkpoint 与独立评估
- 完成状态：
- 修改和新增文件：
- Checkpoint 内容：
- Checkpoint 限制：
- 评估指标：
- 测试结果：
- 训练保存冒烟结果：
- 加载评估冒烟结果：
- 编译验证：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3F——训练日志、周期评估与实验运行器
```

提交结果文档：

```bash
git add aaa.md
git commit -m "docs: record MASAC checkpoint and evaluation result"
git push
git status --short
```

如果任何 Git 命令触发 `git.exe` 内存读取错误：

- 立即停止且不自动重试；
- 不运行 `git reset --hard`、`git gc` 或 `git prune`；
- 记录触发错误的完整命令；
- 在 `aaa.md` 中如实记录实际提交和推送状态。

最终工作区必须干净。
