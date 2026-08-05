# Codex 执行计划：阶段 3F——训练日志、周期评估与实验运行器

## 1. 目标

在现有 MASAC 训练、Checkpoint 和独立评估基础上，实现：

```text
单次实验配置
→ 连续训练
→ 周期记录训练指标
→ 周期确定性评估
→ 保存最佳与最终 Checkpoint
→ 输出完整实验目录
```

本次不实现：

- 多随机种子批量运行；
- 规则基线和 MPC 对比；
- MAPPO、MATD3、MADDPG；
- 图表和统计显著性分析；
- 中断训练的精确恢复。

开始工作后，用本计划覆盖根目录 `AGENTS.md`。

------

## 2. 文件范围

新增：

```text
src/uav_multi_relay/training/experiment.py
scripts/run_experiment.py
tests/test_experiment.py
```

修改：

```text
src/uav_multi_relay/training/trainer.py
src/uav_multi_relay/training/__init__.py
README.md
AGENTS.md
aaa.md
```

不得修改环境、通信、安全过滤、奖励、Replay Buffer、MPC 或 MASAC 更新公式。

不得增加新依赖。

------

## 3. 训练进度回调

在 `trainer.py` 中新增：

```python
@dataclass(frozen=True)
class MASACTrainingProgress:
    environment_steps: int
    total_updates: int
    completed_episodes: int
    replay_size: int
    mean_rate_e2e_bps: float
    intervention_rate: float
    last_update_metrics: MASACUpdateMetrics | None
```

扩展：

```python
def train_masac(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    replay_buffer: MultiAgentReplayBuffer,
    config: MASACTrainingConfig,
    *,
    progress_interval_steps: int | None = None,
    progress_callback: Callable[[MASACTrainingProgress], None] | None = None,
) -> MASACTrainingSummary:
```

要求：

1. 两个参数均不提供时，保持现有行为不变；
   2.提供 callback 时必须同时提供正整数 interval；
2. 在达到 interval 整数倍时调用 callback；
3. 最后一个训练步始终调用一次，若已在该步调用则不得重复；
4. 回调发生在该环境步的 transition、更新和 episode 处理完成之后；
5. 进度中的统计为从训练开始到当前步的累计值；
6. 回调异常正常向外传播，不得静默忽略。

------

## 4. 实验运行器

在 `training/experiment.py` 中实现：

```python
@dataclass(frozen=True)
class MASACExperimentConfig:
    output_directory: str | Path
    log_interval_steps: int = 1_000
    evaluation_interval_steps: int = 5_000
    evaluation_episodes: int = 10
    evaluation_seed: int = 10_000
```

所有 interval 和 episode 数必须为正整数，布尔值无效。

实现：

```python
@dataclass(frozen=True)
class MASACExperimentResult:
    output_directory: Path
    final_checkpoint: Path
    best_checkpoint: Path
    training_log: Path
    evaluation_log: Path
    summary_file: Path
    best_mean_return: float
```

以及：

```python
def run_masac_experiment(
    training_env: MultiRelayEnvironment,
    evaluation_env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    replay_buffer: MultiAgentReplayBuffer,
    training_config: MASACTrainingConfig,
    experiment_config: MASACExperimentConfig,
) -> MASACExperimentResult:
```

### 4.1 输出目录

输出目录必须是：

- 不存在；或
- 已存在但为空。

非空目录必须抛出 `ValueError`，不得覆盖旧实验。

创建以下文件：

```text
run_config.json
training_metrics.jsonl
evaluation_metrics.jsonl
best_checkpoint.pt
final_checkpoint.pt
summary.json
```

不得生成 CSV、图表或 TensorBoard 文件。

### 4.2 配置记录

`run_config.json` 至少记录：

- 训练配置；
- 实验配置；
- Agent 结构和学习率；
- 中继数量；
- local/global observation 维度；
- action dimension；
- Python、NumPy 和 PyTorch 版本。

所有 JSON 必须使用：

```python
allow_nan=False
```

### 4.3 训练日志

每次训练进度回调向 `training_metrics.jsonl` 写入一行，至少包含：

```text
environment_steps
total_updates
completed_episodes
replay_size
mean_rate_e2e_bps
intervention_rate
critic_loss
actor_loss
alpha_loss
alpha
```

尚未发生更新时，损失字段写 `null`，不得写 NaN。

每写一行后立即 `flush()`。

### 4.4 周期评估

在以下训练步执行确定性评估：

```text
evaluation_interval_steps 的整数倍
最终训练步
```

若最终步已经完成周期评估，不得重复。

调用现有：

```python
evaluate_masac(...)
```

评估种子固定使用：

```python
experiment_config.evaluation_seed
```

每次评估使用相同轨迹集合，便于纵向比较。

向 `evaluation_metrics.jsonl` 写入一行，至少包含：

```text
environment_steps
mean_return
return_std
mean_rate_e2e_bps
minimum_rate_e2e_bps
mean_intervention_rate
terminated_episode_rate
```

### 4.5 Checkpoint

- 每次评估的 `mean_return` 严格高于历史最佳值时，覆盖保存 `best_checkpoint.pt`；
- 训练完成后保存 `final_checkpoint.pt`；
- Checkpoint 元数据使用当时真实的环境步数、更新次数和完成 episode 数；
- 最终步必须至少产生一次评估，因此 `best_checkpoint.pt` 必须存在；
- 不保存 Replay Buffer。

### 4.6 最终汇总

`summary.json` 至少包含：

```text
训练总步数
更新总次数
完成 episode 数
训练平均速率
训练安全干预率
最终评估指标
最佳 mean return
最佳 Checkpoint 路径
最终 Checkpoint 路径
日志路径
```

返回的 `MASACExperimentResult` 必须与文件内容一致。

------

## 5. 命令行脚本

新增：

```text
scripts/run_experiment.py
```

支持：

```text
--output-dir
--steps
--seed
--num-relays
--batch-size
--random-action-steps
--update-after-steps
--updates-per-step
--log-interval
--evaluation-interval
--evaluation-episodes
--evaluation-seed
--device
```

要求：

1. 设置 NumPy 和 PyTorch seed；
2. 自动推断 observation 维度；
3. 创建独立训练环境和评估环境；
4. 创建 Agent、Replay Buffer 和配置；
5. 调用 `run_masac_experiment()`；
6. 最后向终端输出一份有限 JSON 摘要；
7. 不在脚本中复制训练或评估逻辑。

保留现有 `train.py` 和 `evaluate.py`，不得删除。

------

## 6. 测试

新增 `tests/test_experiment.py`，并补充必要的 trainer 测试，至少覆盖：

1. 训练回调在正确步数触发，最终步不重复；
2. 未提供 callback 时原训练行为不变；
3. 非法 interval 被拒绝；
4. 非空输出目录被拒绝；
5. 所有规定文件均生成；
6. JSON 和 JSONL 每一行都可解析且不含 NaN；
7. 训练日志步数严格递增；
8. 周期评估发生在预期步数和最终步；
9. 相同评估轨迹集合被重复使用；
10. 最佳 Checkpoint 只在 `mean_return` 改善时更新；
11. 最佳和最终 Checkpoint 均可加载；
12. Checkpoint 元数据与对应训练步一致；
13. 使用小网络完成真实短实验，结果全部有限；
14. 支持至少 `K=1` 和 `K=4`。

测试必须使用临时目录和短训练流程，不得在仓库根目录遗留运行产物。

------

## 7. README 与验证

README 增加：

- 单次正式实验命令；
- 输出目录文件说明；
- 最佳与最终 Checkpoint 的区别；
- 周期评估使用固定轨迹集合；
- 尚未实现多随机种子和对比算法批量实验。

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

执行冒烟实验：

```bash
python scripts/run_experiment.py \
  --output-dir masac_experiment_smoke \
  --steps 20 \
  --batch-size 4 \
  --random-action-steps 4 \
  --update-after-steps 4 \
  --log-interval 5 \
  --evaluation-interval 10 \
  --evaluation-episodes 2 \
  --seed 0 \
  --evaluation-seed 100 \
  --device cpu
```

确认所有输出文件存在且 JSON 有限后，删除：

```text
masac_experiment_smoke/
```

不得提交实验产物或缓存文件。

------

## 8. Git 与结果记录

提交代码并推送：

```bash
git add AGENTS.md README.md scripts/run_experiment.py \
  src/uav_multi_relay/training tests/test_experiment.py
git commit -m "stage-3: add MASAC experiment runner"
git push
```

随后使用真实最终结果覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3F
- 任务：训练日志、周期评估与实验运行器
- 完成状态：
- 修改和新增文件：
- 实验输出文件：
- 日志间隔：
- 评估间隔与轨迹种子：
- 最佳 Checkpoint 规则：
- 测试结果：
- 实验冒烟结果：
- 编译验证：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3G——MASAC 正式训练与基础基线比较
```

提交并推送：

```bash
git add aaa.md
git commit -m "docs: record MASAC experiment runner result"
git push
git status --short
```

若 Git 命令触发 `git.exe` 内存读取错误，立即停止，不自动重试，不执行 `git reset --hard`、`git gc` 或 `git prune`，并在 `aaa.md` 中记录真实状态。

最终工作区必须干净。
