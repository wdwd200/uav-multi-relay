# Codex 任务：阶段 2 第二次修复——可靠初始化与有效端点轨迹

## 目标

修复以下问题：

1. 可行配置依赖随机碰运气才能初始化；
2. H 没有保持高空源 UAV 语义；
3. 某些 seed 会让 H 或 L 永久静止。

不要开始强化学习，不要修改观测结构，不要新增测试文件。

## 允许修改

- `src/uav_multi_relay/config.py`
- `src/uav_multi_relay/environment.py`
- `src/uav_multi_relay/trajectories.py`
- `tests/test_environment.py`
- `README.md`
- `AGENTS.md`
- `aaa.md`

## 1. 增加端点轨迹配置

在 `config.py` 中增加不可变数据类：

```python
EndpointTrajectoryConfig
```

字段：

```text
altitude_min_m
altitude_max_m
waypoint_radius_m
waypoint_count
arrival_tolerance_m
```

要求：

- 数值有限；
- `altitude_min_m < altitude_max_m`；
- `waypoint_radius_m > arrival_tolerance_m > 0`；
- `waypoint_count >= 2`。

在 `EnvironmentConfig` 中增加：

```text
high_trajectory
low_trajectory
```

要求：

- 两个高度区间都位于 `flight_bounds` 的垂直范围内；
- 必须满足：

```text
high_trajectory.altitude_min_m
>
low_trajectory.altitude_max_m
```

默认配置使用：

```text
H 高度范围：170–230 m
L 高度范围：50–110 m
航点半径：30 m
航点数量：4
到达容差：2 m
```

## 2. 改为构造式初始化

重写 `_sample_initial_chain()`。

不要再通过随机采两个端点并最多重试 2048 次来碰运气。

相邻节点间距必须处于：

```text
[hard_safety_distance_m, hard_max_link_distance_m]
```

端点总距离必须处于：

```text
[(K + 1) * hard_safety_distance_m,
 (K + 1) * hard_max_link_distance_m]
```

直接在飞行区域内构造一条满足该距离区间的线段，再用 `np.linspace()` 放置中继。

要求：

- H 高度位于 H 高度区间；
- L 高度位于 L 高度区间；
- 所有节点在边界内；
- 所有两两距离满足硬安全距离；
- 所有相邻链路满足硬链路上限；
- 相同 seed 结果一致；
- 不得依赖随机数恰好落入很窄的距离区间；
- 数学上无法构造时才抛出明确的 `ValueError`。

必须支持：

```text
hard_safety_distance_m == hard_max_link_distance_m
```

## 3. 保证端点轨迹有效

生成 H/L 航点时使用各自的：

```text
高度区间
waypoint_radius_m
waypoint_count
arrival_tolerance_m
```

要求：

- 航点全部位于飞行边界和对应高度区间内；
- 只使用 `self._rng`；
- H 和 L 使用不同航点序列；
- 每个端点至少有一个航点与初始位置的距离大于到达容差；
- 不允许全部航点都被 `WaypointFollower` 当作已经到达；
- 保持现有速度和加速度可行化流程。

## 4. 测试

仍然只修改：

```text
tests/test_environment.py
```

增加或调整测试：

1. 默认配置在 `seed=0..99` 下始终满足：
   ```text
   H.z > L.z
   ```
   且两者位于对应高度区间；

2. 构造一个可行配置，其中：
   ```text
   hard_safety_distance_m = 10
   hard_max_link_distance_m = 10
   ```
   在 `seed=0..19` 下全部能够 reset；

3. 构造一个可行窄区间配置：
   ```text
   hard_safety_distance_m = 10
   hard_max_link_distance_m = 10.001
   ```
   在 `seed=0..19` 下全部能够 reset；

4. 使用 `seed=5169` 和 `seed=6397`，运行 20 步后 H 和 L 都必须发生非零位移；

5. 等距基线使用 `seed=0、1、2` 各运行完整 500 步：
   - 不得提前 terminated；
   - 最终正常 truncated；
   - 每个已提交状态满足边界、安全距离和链路限制；
   - 不出现 NaN。

不要新增测试文件。

## 5. README

补充说明：

- H 使用高空轨迹区间；
- L 使用较低任务轨迹区间；
- 初始链路采用构造式生成，不依赖随机拒绝采样；
- seed 控制初始化和航点。

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
git commit -m "fix: construct reliable endpoint trajectories"
git push
```

然后覆盖 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：2 第二次修复
- 任务：可靠初始化与有效端点轨迹
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
git commit -m "docs: record stage-2 trajectory repair"
git push
git status
```

工作区必须干净。

## 禁止事项

不要实现：

- 新观测结构
- 新奖励
- 新基线
- Gymnasium
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
