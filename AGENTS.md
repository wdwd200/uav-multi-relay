# Codex 执行计划：阶段 2D（补全）——有限时域 MPC 基线

## 1. 当前状态与本次目标

当前项目已完成阶段 3C，包括：

- 完整环境与现有规则基线；
- Replay Buffer；
- 参数共享 Actor；
- 集中式双 Critic；
- MASAC 单批次更新核心。

总方案的阶段 2 仍缺少：

```text
有限时域 MPC 或直接轨迹优化基线
```

本次只实现一个基于环境预测的有限时域 MPC：

```text
随机射击 + CEM + 滚动时域控制
```

每个环境步搜索联合动作序列，但只执行最佳序列的第一个动作，下一步重新规划。

开始工作后，使用本计划全文覆盖根目录：

```text
AGENTS.md
```

Codex 不负责规划后续阶段。

------

## 2. 文件范围

### 新增

```text
src/uav_multi_relay/policies/__init__.py
src/uav_multi_relay/policies/mpc.py
tests/test_mpc.py
```

### 修改

```text
README.md
AGENTS.md
aaa.md
```

除发现会阻断本任务的确定性错误外，不修改现有环境、物理模型、奖励、MASAC 和规则基线代码。

不得增加新依赖。

------

## 3. 公开接口

在 `src/uav_multi_relay/policies/mpc.py` 中实现并从 `policies/__init__.py` 导出：

```python
MPCConfig
MPCSequenceEvaluation
MPCPlan
evaluate_action_sequence
plan_mpc
mpc_actions
```

不要修改项目根包 `uav_multi_relay/__init__.py`。

### 3.1 MPCConfig

```python
@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 3
    population_size: int = 64
    iterations: int = 3
    elite_fraction: float = 0.2
    discount: float = 0.99
    initial_standard_deviation: float = 0.6
    minimum_standard_deviation: float = 0.05
```

至少验证：

```text
horizon >= 1
population_size >= 3
iterations >= 1
0 < elite_fraction <= 1
0 <= discount <= 1
0 < minimum_standard_deviation <= initial_standard_deviation
```

数值必须有限；布尔值不能作为整数参数；非法配置抛出 `ValueError`。

### 3.2 MPCSequenceEvaluation

```python
@dataclass(frozen=True)
class MPCSequenceEvaluation:
    discounted_return: float
    steps_evaluated: int
    terminated: bool
    truncated: bool
    mean_rate_e2e_bps: float
```

### 3.3 MPCPlan

```python
@dataclass(frozen=True)
class MPCPlan:
    first_action: np.ndarray
    action_sequence: np.ndarray
    predicted_return: float
    evaluation: MPCSequenceEvaluation
```

要求：

```text
first_action.shape == (K, 3)
action_sequence.shape == (horizon, K, 3)
```

返回数组必须为独立副本，数值有限且位于 `[-1, 1]`。

------

## 4. 动作序列评估

实现：

```python
def evaluate_action_sequence(
    env: MultiRelayEnvironment,
    action_sequence: object,
    discount: float = 0.99,
) -> MPCSequenceEvaluation:
```

要求：

1. `action_sequence` 形状必须为：

   ```text
   (horizon, env.config.num_relays, 3)
   ```

2. 动作必须有限并位于 `[-1, 1]`。

3. 使用：

   ```python
   copy.deepcopy(env)
   ```

   创建预测环境，不得推进或修改传入的原环境。

4. 在预测环境中依次调用 `step()`，累计：

   ```python
   discounted_return += discount**t * reward
   ```

5. 优化目标必须使用环境返回的完整团队奖励，不得重新实现奖励函数，也不得只优化通信速率。

6. 从 `info["rate_e2e_bps"]` 计算预测期间的平均端到端服务速率。

7. 遇到 `terminated` 或 `truncated` 后立即停止预测。

不得通过直接修改 UAV 状态绕过环境和安全过滤器。

------

## 5. CEM MPC

实现：

```python
def plan_mpc(
    env: MultiRelayEnvironment,
    config: MPCConfig | None = None,
    seed: int | None = 0,
) -> MPCPlan:
```

随机数必须使用：

```python
rng = np.random.default_rng(seed)
```

不得使用全局随机状态。

### 5.1 初始分布

使用现有动态等距基线初始化均值：

```python
equal_action = equal_spacing_actions(env)
mean_sequence = np.repeat(
    equal_action[np.newaxis, :, :],
    config.horizon,
    axis=0,
)
```

初始标准差全部设为：

```python
config.initial_standard_deviation
```

### 5.2 每轮候选

候选总体形状：

```text
(population_size, horizon, K, 3)
```

普通候选从当前正态分布采样并裁剪至 `[-1, 1]`。

每轮必须包含三个确定性锚点：

```text
候选 0：当前均值序列
候选 1：全零动作序列
候选 2：动态等距动作序列
```

全零动作使用现有：

```python
stationary_actions(env)
```

### 5.3 精英更新

精英数量：

```python
elite_count = max(
    1,
    int(np.ceil(config.population_size * config.elite_fraction)),
)
```

按预测累计奖励从高到低稳定排序。

使用精英序列更新均值和标准差：

```python
mean_sequence = elite_sequences.mean(axis=0)
standard_deviation = elite_sequences.std(axis=0)
standard_deviation = np.maximum(
    standard_deviation,
    config.minimum_standard_deviation,
)
```

均值裁剪至 `[-1, 1]`。

### 5.4 最终选择

所有 CEM 迭代完成后，从最后一轮已经评估的候选中选择最高分序列。

不得再次随机采样。

返回：

- 最佳动作序列；
- 序列第一步动作；
- 预测累计奖励；
- 对应评估结果。

实现简化接口：

```python
def mpc_actions(
    env: MultiRelayEnvironment,
    config: MPCConfig | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    return plan_mpc(env, config=config, seed=seed).first_action.copy()
```

不得写死 `K=4`。

------

## 6. 测试要求

新增 `tests/test_mpc.py`，至少覆盖：

1. `MPCConfig` 合法与非法参数；
2. 错误动作形状、NaN、无穷值和越界动作被拒绝；
3. `evaluate_action_sequence()` 不修改原环境，包括：
   - `step_index`
   - UAV 位置与速度
   - H/L 轨迹内部进度
4. 折扣回报和平均速率为有限值；
5. 截断或终止后不再继续预测；
6. MPC 输出形状和范围正确；
7. 相同环境状态、配置和 seed 得到相同结果；
8. 支持至少 `K=1` 和 `K=4`；
9. 最终预测回报不低于零动作和动态等距动作两个锚点；
10. 使用小型配置连续运行至少 10 个真实环境步，不出现 NaN 或异常。

测试中使用小型配置，例如：

```python
MPCConfig(
    horizon=2,
    population_size=6,
    iterations=2,
    elite_fraction=0.5,
    initial_standard_deviation=0.4,
    minimum_standard_deviation=0.05,
)
```

不得用默认完整规模拖慢测试。

------

## 7. README 更新

README 必须与实际状态一致：

- 将有限时域 CEM MPC 加入已实现功能；
- 删除“MPC 尚未实现”的描述；
- 明确完整训练、评估和 checkpoint 流程仍未完成；
- 增加 `MPCConfig` 和 `mpc_actions()` 的最小使用示例；
- 说明 MPC 使用深拷贝环境预测，并优化折扣团队奖励。

------

## 8. 验证与 Git

依次运行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
python -m pytest
python -m compileall -q src tests scripts
```

若普通安装仅因软件源无法获得构建依赖而失败，可以补充运行：

```bash
python -m pip install -e ".[dev]" --no-build-isolation
```

必须在 `aaa.md` 中如实记录两种安装结果。

开始前检查：

```bash
git status --short
git branch --show-current
git log -1 --oneline
```

若存在非本任务产生的未提交修改，停止并报告，不得清除。

完成代码后提交并推送：

```bash
git add AGENTS.md README.md src/uav_multi_relay/policies tests/test_mpc.py
git commit -m "stage-2: add finite-horizon MPC baseline"
git push
```

随后使用以下结构**覆盖写入** `aaa.md`：

```markdown
# 本次执行结果

- 阶段：2D（补全）
- 任务：有限时域 MPC 基线
- 完成状态：
- 修改和新增文件：
- MPC 方法：
- 默认配置：
- 测试结果：
- 普通安装结果：
- 无构建隔离安装结果：
- 编译验证：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3D——MASAC 环境采集与训练循环
```

提交并推送结果文档：

```bash
git add aaa.md
git commit -m "docs: record stage-2 MPC result"
git push
git status --short
```

最终工作区必须干净。

------

## 9. 完成标准

只有以下条件全部满足才算完成：

- MPC 搜索多步联合动作序列，而不是单步贪心；
- 预测不修改原环境；
- 使用环境团队奖励作为目标；
- 支持动态中继数量；
- 结果可复现；
- 新旧测试全部通过；
- README 和 `aaa.md` 已更新；
- 两次提交均已推送；
- 最终 Git 工作区干净。

Codex 最终只回复：

```text
测试结果：
普通安装结果：
无构建隔离安装结果：
代码 Commit ID：
aaa.md Commit ID：
当前分支：
GitHub 推送结果：
Git 工作区状态：
计划偏差：
遗留问题：
```
