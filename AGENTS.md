# Codex 任务：修复阶段 1 物理核心

## 目标

修复阶段 1 代码审查发现的输入校验和测试缺口。

本次不要新增文件，不要开始环境开发。

## 允许修改

- `src/uav_multi_relay/kinematics.py`
- `src/uav_multi_relay/communication.py`
- `tests/test_physics.py`
- `AGENTS.md`
- `aaa.md`

## 任务 1：校验当前速度

在 `make_velocity_feasible()` 开始计算前，检查 `current_velocity_mps` 是否已经满足：

- 水平速度范数不超过 `max_horizontal_speed_mps`
- 垂直速度不超过 `max_climb_speed_mps`
- 垂直速度不低于 `-max_descent_speed_mps`

若当前速度非法，直接抛出 `ValueError`。

不要自动修复非法当前速度。

增加测试：

- 非法水平当前速度抛出 `ValueError`
- 非法垂直当前速度抛出 `ValueError`

## 任务 2：校验偶极子参数

在 `dipole_gain()` 中增加校验：

```text
0 <= elevation_angle_rad <= pi / 2
min_gain_linear <= max_gain_linear
```

不满足时抛出 `ValueError`。

增加对应测试。

## 任务 3：补充信道函数说明

修改 `channel_power_gain()` 的 docstring，明确写明：

```text
reference_gain_linear excludes directional antenna gains.
```

不要改变现有计算公式。

## 任务 4：加强最优 TDMA 测试

使用：

```python
capacities = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
```

验证：

```python
expected_rate = 1.0 / np.sum(1.0 / capacities)
```

并检查：

```python
assert rate == pytest.approx(expected_rate)
assert np.sum(fractions) == pytest.approx(1.0)
assert fractions * capacities == pytest.approx(
    np.full_like(capacities, rate)
)
```

## 验证

运行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
pytest
```

所有测试必须通过。

## AGENTS.md

保留 `AGENTS.md`。

本次执行指令继续存放在该文件中，不要删除。

## aaa.md

完成后覆盖写入：

```markdown
# 本次执行结果

- 阶段：1 修复
- 任务：物理核心输入校验
- 完成状态：
- 修改文件：
- 测试命令：
- 测试结果：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- 计划偏差：
- 遗留问题：
```

## Git 提交

先提交代码：

```bash
git add .
git commit -m "fix: validate physical core inputs"
git push
```

然后把真实 Commit ID 和推送结果写入 `aaa.md`，再提交：

```bash
git add aaa.md
git commit -m "docs: record stage-1 repair result"
git push
```

最后运行：

```bash
git status
```

工作区必须干净。

## 禁止事项

本次不要实现：

- Gymnasium 环境
- H/L 航点轨迹
- 多 UAV 碰撞过滤
- 奖励函数
- replay buffer
- SAC 或 MASAC
- 训练脚本
- 新测试文件

## 完成后回复

只回复：

- 测试结果
- 两个 Commit ID
- 推送结果
- 遗留问题