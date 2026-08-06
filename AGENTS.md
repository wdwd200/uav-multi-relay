# Codex 执行计划：阶段 3H-A——奖励一致性与动态场景诊断

## 1. 目标

本次完成：

1. 修复运动代价与总方案不一致的问题；
2. 将奖励权重正式配置化；
3. 增加场景难度和奖励分量诊断工具；
4. 生成诊断结果，为下一次训练配置提供依据。

本次不进行 MASAC 调参或长时间训练。

开始后使用本计划覆盖根目录 `AGENTS.md`。

------

## 2. 文件范围

新增：

```text
src/uav_multi_relay/analysis/diagnostics.py
scripts/diagnose_scenarios.py
tests/test_diagnostics.py
```

允许修改：

```text
src/uav_multi_relay/config.py
src/uav_multi_relay/environment.py
src/uav_multi_relay/analysis/__init__.py
tests/test_environment.py
README.md
AGENTS.md
aaa.md
```

不得修改 MASAC、Replay Buffer、Checkpoint、MPC 或现有策略算法。

------

## 3. 奖励权重

在 `config.py` 中增加不可变数据类：

```python
@dataclass(frozen=True)
class RewardWeights:
    rate: float = 1.0
    link: float = 1.0
    separation: float = 1.0
    intervention: float = 1.0
    motion: float = 1.0
    failure: float = 1.0
```

要求：

- 所有值必须有限且非负；
- `rate > 0`；
- 非法值抛出 `ValueError`。

在 `EnvironmentConfig` 中增加：

```python
reward_weights: RewardWeights
```

默认配置使用上述默认权重，保持原有奖励尺度。

------

## 4. 奖励公式修复

### 4.1 运动代价只计算中继

当前实现计算了 H、L 和所有中继。

修改为只使用：

```python
new_states[1:-1]
old_states[1:-1]
```

即仅计算受策略控制的中继速度和加速度。

不得把 H/L 运动写入中继运动代价。

### 4.2 应用显式权重

正常步骤奖励改为：

```python
reward = (
    weights.rate * rate_reward
    - weights.link * link_cost
    - weights.separation * separation_cost
    - weights.intervention * intervention_cost
    - weights.motion * motion_cost
)
```

失败终止奖励改为：

```python
-weights.failure
```

`info["reward_terms"]` 继续保存未加权的原始分量，并新增：

```text
weighted_reward
```

其值必须与环境返回的 reward 一致。

------

## 5. 场景诊断接口

在 `analysis/diagnostics.py` 中实现：

```python
@dataclass(frozen=True)
class ScenarioDiagnosticConfig:
    waypoint_radii_m: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0)
    max_steps_values: tuple[int, ...] = (100, 250)
    episodes: int = 5
    seed: int = 30_000
    policies: tuple[str, ...] = ("stationary", "equal_spacing")
```

合法策略仅限：

```text
stationary
equal_spacing
```

实现：

```python
def diagnose_scenarios(
    base_config: EnvironmentConfig,
    config: ScenarioDiagnosticConfig,
) -> ScenarioDiagnosticResult:
```

对每个：

```text
waypoint radius × max_steps × policy × episode seed
```

运行完整 episode。

H 和 L 使用相同的候选航点半径，其他物理参数保持不变。

------

## 6. 诊断指标

每个策略和场景至少统计：

```text
完成 episode 数
termination rate
平均 return
平均端到端速率
最低端到端速率
平均 rate_reward
平均 link_cost
平均 separation_cost
平均 intervention_cost
平均 motion_cost
平均安全干预率
H 平均位移与最大位移
L 平均位移与最大位移
中继平均路径长度
平均最小单跳容量
平均最大单跳距离
平均 episode 长度
```

位移均相对于 episode 初始位置。

终止 episode 必须记录：

```text
failure_reason
```

所有汇总值必须有限。

------

## 7. 诊断脚本

新增：

```text
scripts/diagnose_scenarios.py
```

支持：

```text
--output
--radii
--max-steps
--episodes
--seed
--policies
--num-relays
```

默认运行：

```text
radii = 30 60 90 120
max steps = 100 250
episodes = 5
policies = stationary equal_spacing
```

输出一个 JSON 文件，包含：

```text
resolved configuration
每个 episode 的原始结果
每个场景和策略的汇总结果
```

必须使用：

```python
allow_nan=False
```

输出文件已存在时拒绝覆盖。

------

## 8. 测试

至少覆盖：

1. `RewardWeights` 参数验证；
2. 默认奖励权重保持现有公式；
3. H/L 以相同速度运动、所有中继静止时，中继 `motion_cost == 0`；
4. 中继运动时 `motion_cost > 0`；
5. 返回 reward 等于各加权分量组合；
6. 失败奖励等于 `-weights.failure`；
7. 诊断使用相同 episode seed 比较策略；
8. 场景半径和 `max_steps` 正确应用；
9. 位移、路径长度、容量和奖励分量统计正确；
10. 终止原因被保存；
11. JSON 可解析且不含 NaN；
12. 输出文件存在时拒绝覆盖；
13. 支持 `K=1` 和 `K=4`。

测试使用短 episode，不运行完整默认诊断矩阵。

------

## 9. 实际诊断

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

随后运行：

```bash
python scripts/diagnose_scenarios.py \
  --output scenario_diagnostics.json \
  --radii 30 60 90 120 \
  --max-steps 100 250 \
  --episodes 5 \
  --seed 30000 \
  --policies stationary equal_spacing \
  --num-relays 4
```

诊断完成后，从 JSON 中如实汇总：

- 哪些场景静止策略仍接近最优；
- 哪些场景中等距策略速率明显更高；
- 哪些场景出现硬约束终止；
- 各奖励分量对 return 的平均贡献。

`scenario_diagnostics.json` 不提交 Git。

不得在本任务中自行选择最终奖励权重或训练场景。

------

## 10. README、Git 与 aaa.md

README 增加：

- 奖励权重配置说明；
- 运动代价只统计受控中继；
- 场景诊断命令；
- 诊断结果不等于正式实验结论。

提交代码：

```bash
git add AGENTS.md README.md \
  src/uav_multi_relay/config.py \
  src/uav_multi_relay/environment.py \
  src/uav_multi_relay/analysis \
  scripts/diagnose_scenarios.py \
  tests/test_environment.py tests/test_diagnostics.py
git commit -m "stage-3: add reward and scenario diagnostics"
git push
```

随后覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3H-A
- 任务：奖励一致性与动态场景诊断
- 完成状态：
- 修改和新增文件：
- 运动代价修复：
- 奖励权重实现：
- 完整测试结果：
- 编译验证：
- 诊断矩阵：
- 各场景终止率：
- 静止与等距速率比较：
- 奖励分量诊断：
- 诊断输出位置：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3H-B——确定训练场景与奖励权重后重新训练 MASAC
```

提交结果：

```bash
git add aaa.md
git commit -m "docs: record reward and scenario diagnostics"
git push
git status --short
```

若发生 `git.exe` 内存读取错误，立即停止，不自动重试，不运行 `git reset --hard`、`git gc` 或 `git prune`，并如实记录实际状态。

最终工作区必须干净。
