# Codex 任务：修复阶段 2C 验收测试

## 目标

修复规则与贪心基线的两个测试缺口。

本次以测试修复为主，不新增文件，不开始强化学习。

## 允许修改

- `tests/test_environment.py`
- `AGENTS.md`
- `aaa.md`

只有在修正后的测试暴露真实实现错误时，才允许修改：

- `src/uav_multi_relay/baselines.py`

## 任务 1：修正贪心速率断言

把当前错误断言：

```python
assert info["rate_e2e_bps"] >= min(stationary_rate, equal_rate)
```

改为验证：

```python
expected_minimum = max(stationary_rate, equal_rate)
assert info["rate_e2e_bps"] >= pytest.approx(expected_minimum)
```

如果 `pytest.approx` 不能直接用于该不等式，则使用合理的小数容差，例如：

```python
assert info["rate_e2e_bps"] + 1e-9 >= expected_minimum
```

## 任务 2：完整验证环境未被贪心搜索修改

调用 `greedy_one_step_actions()` 前保存：

- 所有 UAV 的位置；
- 所有 UAV 的速度；
- `step_index`；
- 上一步实际中继速度。

调用后逐项确认完全不变。

不要只检查 H 的位置。

## 任务 3：补上贪心 50 步测试

现有测试遗漏了 `greedy_one_step_actions()`。

分别对以下四个基线运行 50 个尝试步骤：

```text
stationary_actions
equal_spacing_actions
weighted_spacing_actions
greedy_one_step_actions
```

每一步检查：

```text
positions_m 有限
velocities_mps 有限
hop_capacities_bps 有限
rate_e2e_bps 有限且非负
动作形状正确且位于 [-1, 1]
```

逐时隙贪心允许因为短视行为返回：

```text
terminated = True
```

出现 `terminated` 或 `truncated` 后立即使用下一个确定性 seed 重置，继续完成总计 50 个尝试步骤。

不得出现未处理异常。

不要为了让贪心永不终止而改变其“最大化下一步端到端速率”的定义。

## 验证

运行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
python -m pytest
```

所有测试必须通过。

## Git

先提交测试修复：

```bash
git add .
git commit -m "test: strengthen stage-2 baseline checks"
git push
```

然后覆盖 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：2C 修复
- 任务：加强规则与贪心基线验收测试
- 完成状态：
- 修改文件：
- 测试结果：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- 计划偏差：
- 遗留问题：
```

填写真实结果后提交：

```bash
git add aaa.md
git commit -m "docs: record stage-2 baseline test repair"
git push
git status
```

工作区必须干净。

## 禁止事项

不要实现：

- MPC
- replay buffer
- MASAC、SAC、MAPPO
- 神经网络
- 训练脚本
- 新测试文件

完成后只回复：

- 测试结果
- 两个 Commit ID
- 推送结果
- 遗留问题
