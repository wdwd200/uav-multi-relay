# Codex 修复计划：阶段 3G-R1——诊断汇总与结果记录修复

## 1. 目标

修正场景诊断汇总语义、测试覆盖和阶段编号。

本次仍属于阶段 3G 的验收修复，不新增总计划阶段，不进行 MASAC 训练或奖励调参。

开始后用本计划覆盖 `AGENTS.md`。

## 2. 修改范围

允许修改：

```text
src/uav_multi_relay/analysis/diagnostics.py
scripts/diagnose_scenarios.py
tests/test_diagnostics.py
AGENTS.md
aaa.md
```

如需要补充说明，可修改：

```text
README.md
```

不得修改环境奖励、MASAC、MPC 或现有策略行为。

## 3. 阶段编号

将所有本次任务编号从：

```text
3H-A
```

改为：

```text
3G-R1
```

并明确：

```text
类型：阶段 3G 验收修复
不新增总计划阶段
```

下一任务写为：

```text
3G-R2——确定训练场景与奖励权重后重新训练和比较
```

## 4. 修正汇总指标

在 `ScenarioDiagnosticSummary` 中同时保存：

```python
mean_episode_min_rate_e2e_bps: float
minimum_rate_e2e_bps: float
mean_high_displacement_m: float
maximum_high_displacement_m: float
mean_low_displacement_m: float
maximum_low_displacement_m: float
```

定义：

```text
mean_episode_min_rate_e2e_bps
= 各 episode 最低速率的平均值

minimum_rate_e2e_bps
= 该场景全部 episode 中的最低速率

maximum_high_displacement_m
= 全部 episode 中 H 最大位移的最大值

maximum_low_displacement_m
= 全部 episode 中 L 最大位移的最大值
```

现有其他指标保持不变。

不得通过修改字段名称掩盖错误计算。

## 5. 代码清理

删除 `diagnostics.py` 中未使用的局部变量，例如：

```python
initial
initial_positions
```

将 `_scenario_config()` 改为使用：

```python
dataclasses.replace(...)
```

避免按位置重新构造 `EnvironmentConfig`。

## 6. 测试

补充或修正测试，至少验证：

1. 中继静止且 H/L 移动时 `motion_cost == 0`；
2. 中继实际运动时 `motion_cost > 0`；
3. `minimum_rate_e2e_bps` 是全部 episode 的真正最小值；
4. `mean_episode_min_rate_e2e_bps` 是 episode 最小值的平均值；
5. H/L 最大位移汇总正确；
6. 终止原因保存在 episode 结果中；
7. 路径长度、容量和距离汇总使用正确数据；
8. `K=1` 和 `K=4` 均能完成短诊断；
9. 实际调用 `scripts/diagnose_scenarios.py` 生成 JSON；
10. 输出文件已存在时，实际脚本返回非零并拒绝覆盖；
11. JSON 可解析且不含 NaN。

不得保留只检查测试夹具自身、却没有调用生产代码的无效测试。

## 7. 验证与诊断

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

重新运行完整诊断：

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

从当前最终代码重新计算并记录：

- 两个策略的全矩阵平均 return；
- 各场景终止率；
- 平均 episode 最低速率；
- 全局最低速率；
- H/L 最大位移；
- 平均运动成本和干预成本。

不得沿用旧数字。

验证后删除 `scenario_diagnostics.json`。

## 8. Git 与 aaa.md

提交代码：

```bash
git add AGENTS.md \
  src/uav_multi_relay/analysis/diagnostics.py \
  scripts/diagnose_scenarios.py \
  tests/test_diagnostics.py
git commit -m "fix: correct scenario diagnostic summaries"
git push
```

随后覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3G-R1
- 类型：阶段 3G 验收修复，不新增总计划阶段
- 任务：奖励一致性与动态场景诊断
- 完成状态：
- 修改文件：
- 运动代价修复状态：
- 奖励权重状态：
- 诊断汇总修复：
- 完整测试结果：
- 编译验证：
- 诊断矩阵：
- 静止策略平均 return：
- 等距策略平均 return：
- 各场景终止率：
- 平均 episode 最低速率：
- 全局最低速率：
- H/L 最大位移：
- 奖励分量诊断：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：曾修改根包 __init__.py 以导出 RewardWeights，该修改已如实记录
- 遗留问题：
- 下一建议任务：3G-R2——确定训练场景与奖励权重后重新训练和比较
```

提交结果：

```bash
git add aaa.md
git commit -m "docs: correct stage-3G diagnostic result"
git push
git status --short
```

若发生 `git.exe` 内存读取错误，立即停止 Git 操作，不自动重试，不执行破坏性 Git 命令，并如实记录实际状态。

最终工作区必须干净。
