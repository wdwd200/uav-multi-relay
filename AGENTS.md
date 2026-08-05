# Codex 执行计划：阶段 3G——统一基线评估与 MASAC 初步性能验证

## 1. 目标

完成统一的策略比较工具，在相同环境配置和相同 episode seed 下比较：

```text
MASAC
随机动作
静止中继
动态等距链
单步贪心
有限时域 MPC
```

随后进行一次开发级 MASAC 训练和基础基线比较。

本次结果只用于确认训练与比较流程有效，不作为论文正式多随机种子结果。

开始后使用本计划覆盖根目录 `AGENTS.md`。

------

## 2. 文件范围

新增：

```text
src/uav_multi_relay/analysis/__init__.py
src/uav_multi_relay/analysis/comparison.py
scripts/compare_baselines.py
tests/test_comparison.py
```

允许修改：

```text
scripts/run_experiment.py
README.md
AGENTS.md
aaa.md
```

不得修改环境、通信、安全过滤、奖励、Replay Buffer、MASAC 更新公式或现有基线算法。

不得增加新依赖。

------

## 3. 统一比较接口

在 `analysis/comparison.py` 中实现：

```python
@dataclass(frozen=True)
class PolicyComparisonConfig:
    episodes: int = 5
    seed: int = 20_000
    policies: tuple[str, ...] = (
        "masac",
        "random",
        "stationary",
        "equal_spacing",
        "greedy",
        "mpc",
    )
    greedy_sweeps: int = 1
    mpc_config: MPCConfig = MPCConfig(
        horizon=2,
        population_size=8,
        iterations=2,
        elite_fraction=0.5,
    )
```

合法策略名称仅限：

```text
masac
random
stationary
equal_spacing
greedy
mpc
```

策略名称不得重复。

实现单 episode 结果、单策略汇总和总比较结果数据类，至少统计：

```text
episode return
episode length
平均端到端速率
最低端到端速率
安全过滤介入率
terminated
truncated
每步平均动作计算时间
```

单策略汇总至少包含：

```text
平均 return 与标准差
平均端到端速率
全部 episode 的最低速率
平均安全干预率
平均 episode 长度
terminated episode 比例
平均动作计算时间
```

实现：

```python
def compare_policies(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    config: PolicyComparisonConfig,
) -> PolicyComparisonResult:
```

------

## 4. 比较规则

### 4.1 公平性

每种策略的第 `episode_index` 个 episode 都必须使用：

```python
episode_seed = config.seed + episode_index
```

因此所有策略面对相同的 H/L 轨迹和初始状态。

比较过程必须深拷贝传入环境，不得修改原环境。

不得训练或修改 MASAC Agent。

### 4.2 策略动作

- `masac`：调用 `agent.act(local_observation, deterministic=True)`；
- `random`：每个 episode 使用独立 `np.random.default_rng(episode_seed)`；
- `stationary`：调用现有 `stationary_actions()`；
- `equal_spacing`：调用现有 `equal_spacing_actions()`；
- `greedy`：调用现有 `greedy_one_step_actions()`；
- `mpc`：调用现有 `mpc_actions()`。

MPC 每一步使用确定性 seed，例如：

```python
config.seed + episode_index * env.config.max_steps + step_index
```

不得写死中继数量。

### 4.3 计算时间

使用 `time.perf_counter()` 只测量动作生成时间，不包含：

```text
env.step()
日志写入
结果汇总
```

动作计算时间必须非负且有限。

------

## 5. 比较脚本

新增：

```text
scripts/compare_baselines.py
```

至少支持：

```text
--checkpoint
--output-dir
--episodes
--seed
--max-steps
--policies
--greedy-sweeps
--mpc-horizon
--mpc-population-size
--mpc-iterations
--device
```

流程：

1. 加载 MASAC checkpoint；
2. 根据 checkpoint 的中继数量创建环境；
3. 使用 `--max-steps` 替换 episode 最大步数；
4. 核对环境 observation 与 Agent 维度；
5. 调用 `compare_policies()`；
6. 输出以下文件：

```text
comparison_config.json
comparison_episodes.jsonl
comparison_summary.json
```

输出目录必须不存在或为空，不得覆盖旧结果。

所有 JSON 使用：

```python
allow_nan=False
```

终端最后输出简洁 JSON 摘要。

------

## 6. run_experiment.py

增加可选参数：

```text
--max-steps
```

训练环境和周期评估环境必须使用相同的 `max_steps`。

默认值保持当前环境默认值，不改变现有命令行为。

------

## 7. 测试

新增 `tests/test_comparison.py`，至少覆盖：

1. 配置非法值和重复策略名称被拒绝；
2. 各策略使用完全相同的 episode seed；
3. 随机策略在相同 seed 下可复现；
4. MASAC 始终使用确定性动作；
5. 比较过程不修改传入环境；
6. 比较过程不修改 Agent 参数和优化器状态；
7. 所有统计值有限，动作计算时间非负；
8. 支持策略子集；
9. 支持 `K=1` 和 `K=4`；
10. 输出 JSON/JSONL 可解析且不含 NaN；
11. 输出目录非空时拒绝覆盖；
12. 使用短 episode 和小型 MPC 配置完成真实冒烟比较。

测试不得运行长 episode 或默认完整 MPC 配置。

------

## 8. 开发级训练与比较

代码测试通过后，执行一次开发级训练：

```bash
python scripts/run_experiment.py \
  --output-dir outputs/stage3g_seed0 \
  --steps 5000 \
  --max-steps 100 \
  --batch-size 256 \
  --random-action-steps 1000 \
  --update-after-steps 1000 \
  --updates-per-step 1 \
  --log-interval 500 \
  --evaluation-interval 1000 \
  --evaluation-episodes 3 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

然后比较最佳 checkpoint：

```bash
python scripts/compare_baselines.py \
  --checkpoint outputs/stage3g_seed0/best_checkpoint.pt \
  --output-dir outputs/stage3g_seed0/comparison \
  --episodes 3 \
  --seed 20000 \
  --max-steps 100 \
  --policies masac random stationary equal_spacing greedy mpc \
  --greedy-sweeps 1 \
  --mpc-horizon 2 \
  --mpc-population-size 8 \
  --mpc-iterations 2 \
  --device cpu
```

要求：

- 结果全部有限；
- 各策略完成相同的 episode 数量；
- 如实记录 MASAC 是否超过随机和静止策略；
- MASAC 未超过基线时不得修改、筛选或美化结果；
- 不在本任务中进行奖励调参或算法调参。

`outputs/` 不得提交到 Git。

------

## 9. README、验证与 Git

README 增加：

- 统一策略比较命令；
- 输出指标说明；
- 同 seed 公平比较原则；
- 本次单 seed 结果不属于论文正式结论。

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

提交代码：

```bash
git add AGENTS.md README.md scripts/run_experiment.py \
  scripts/compare_baselines.py src/uav_multi_relay/analysis \
  tests/test_comparison.py
git commit -m "stage-3: add unified policy comparison"
git push
```

随后覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3G
- 任务：统一基线评估与 MASAC 初步性能验证
- 完成状态：
- 修改和新增文件：
- 比较策略：
- 公平比较方式：
- 完整测试结果：
- 编译验证：
- 开发级训练结果：
- 基线比较结果：
- MASAC 是否超过随机策略：
- MASAC 是否超过静止策略：
- 平均安全干预率：
- 各策略平均动作计算时间：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：
```

下一建议阶段按真实结果填写：

- MASAC 明显超过随机和静止：进入阶段 4A；
- MASAC 未超过其中任一项：进入阶段 3H——训练稳定性诊断与调参。

提交结果：

```bash
git add aaa.md
git commit -m "docs: record MASAC baseline comparison result"
git push
git status --short
```

若发生 `git.exe` 内存读取错误，立即停止 Git 操作，不自动重试，不执行 `git reset --hard`、`git gc` 或 `git prune`，并如实记录实际状态。

最终工作区必须干净。
