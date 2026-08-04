# Codex 任务：修复阶段 2 初始化与轨迹

## 目标

修复阶段 2 的一般 K 初始化、随机轨迹、航点死区和配置可行性问题。

不要开始强化学习，不要新增测试文件。

## 允许修改

- `src/uav_multi_relay/config.py`
- `src/uav_multi_relay/trajectories.py`
- `src/uav_multi_relay/environment.py`
- `tests/test_environment.py`
- `README.md`
- `AGENTS.md`
- `aaa.md`

## 1. 配置可行性

在 `EnvironmentConfig` 中要求：

```text
hard_max_link_distance_m >= hard_safety_distance_m
```

不满足时抛出 `ValueError`。

## 2. 配置驱动的 reset

删除 `reset()` 中固定的：

```text
H = [-300, 0, 150]
L = [300, 0, 150]
```

根据以下配置生成初始链路：

```text
flight_bounds
num_relays
hard_safety_distance_m
hard_max_link_distance_m
seed
```

必须保证：

- H、全部中继和 L 都在飞行边界内；
- 所有 UAV 两两距离不小于硬安全距离；
- 所有相邻逻辑链路不超过硬链路上限；
- 中继初始位置沿 H 到 L 合理分布；
- `K=1`、`K=4`、`K=8` 均可正常重置；
- 对无法构造合法链路的配置，抛出说明明确的 `ValueError`；
- 不得返回已经违反约束的初始状态。

## 3. 使用 seed 生成 H/L 轨迹

在每次 `reset(seed=...)` 时，根据环境随机数生成 H 和 L 各自的循环航点。

要求：

- 所有航点位于飞行边界内并保留合理边界余量；
- H 和 L 使用不同航点；
- 相同 seed 产生相同初始化和轨迹；
- 不同 seed 在前 20 步内产生不同的 H/L 位置轨迹；
- 轨迹仍通过 `WaypointFollower`、速度限制和加速度限制执行；
- H/L 候选位置出界时不得提交状态，应终止并在 `info` 中记录原因。

不要使用全局 `np.random`，只使用环境的 `self._rng`。

## 4. 修复航点停止死区

`WaypointFollower` 的“到达判断”和“生成速度判断”必须使用一致的距离标准。

当三维距离大于 `arrival_tolerance_m` 时，在初始速度为零且限制允许的情况下，返回速度不得因为水平和垂直分量分别小于容差而变成全零。

增加回归测试：

```python
state = UAVState("H", [0, 0, 0], [0, 0, 0])
waypoint = [1.5, 0, 1.5]
arrival_tolerance_m = 2.0
```

三维距离大于 2，因此生成速度必须非零。

## 5. 加强测试

仍然只修改：

```text
tests/test_environment.py
```

增加或加强以下测试：

1. 相同 seed 的初始化和 H/L 轨迹一致；
2. 不同 seed 的前 20 步 H/L 轨迹不同；
3. `K=1`、`K=4`、`K=8` 重置后全部硬约束成立；
4. 使用较小但可行的自定义 `FlightBounds` 时，所有节点仍在边界内；
5. 不可能构造链路的配置明确抛出 `ValueError`；
6. `hard_max_link_distance_m < hard_safety_distance_m` 被拒绝；
7. 航点停止死区回归测试；
8. 明确断言 H 和 L 在若干步内实际发生位移；
9. 等距基线运行 500 步，每个已提交状态都满足：
   - 所有 UAV 在边界内；
   - 两两安全距离；
   - 相邻链路距离；
   - 无 NaN。

不要创建新的测试文件。

## 6. README

补充说明：

- H/L 航点由 seed 可复现地随机生成；
- 环境支持一般 K，但配置必须存在可行初始链路。

## 验证

运行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
python -m pytest
```

所有测试必须通过。

## Git

先提交代码：

```bash
git add .
git commit -m "fix: make environment initialization configuration-aware"
git push
```

然后覆盖 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：2 修复
- 任务：一般 K 初始化与随机轨迹
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
git commit -m "docs: record stage-2 repair result"
git push
git status
```

工作区必须干净。

## 禁止事项

不要实现：

- Gymnasium 或 PettingZoo
- replay buffer
- MASAC、SAC、MAPPO
- 神经网络
- 训练脚本
- MPC
- 动态路由
- 新测试文件

完成后只回复：

- 测试结果
- 两个 Commit ID
- 推送结果
- 遗留问题
