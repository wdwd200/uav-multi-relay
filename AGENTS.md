- # Codex 任务：阶段 2——多中继动态环境

  ## 目标

  在现有物理核心上，实现一个不依赖 Gymnasium 的多中继动态环境。

  本阶段完成：

  - H 和 L 航点运动；
  - 四个中继联合动作；
  - 同步状态更新；
  - 执行前安全过滤；
  - 多跳通信计算；
  - 局部观测和全局状态；
  - 团队奖励；
  - 一个等距跟随基线。

  不要实现强化学习。

  ---

  ## 新增文件

  ```text
  src/uav_multi_relay/config.py
  src/uav_multi_relay/trajectories.py
  src/uav_multi_relay/safety.py
  src/uav_multi_relay/environment.py
  src/uav_multi_relay/baselines.py
  tests/test_environment.py
  ```

  允许修改：

  ```text
  src/uav_multi_relay/__init__.py
  README.md
  AGENTS.md
  aaa.md
  ```

  不要新增其他测试文件。

  ---

  ## 1. 配置

  在 `config.py` 中实现不可变数据类：

  ```python
  FlightBounds
  ChannelConfig
  EnvironmentConfig
  ```

  ### `FlightBounds`

  字段：

  ```python
  minimum_m: np.ndarray
  maximum_m: np.ndarray
  ```

  要求：

  - 均为有限 `(3,)` 数组；
  - 每个最小值必须小于对应最大值；
  - 数组复制并设为只读；
  - 提供 `contains(position_m) -> bool`。

  ### `ChannelConfig`

  至少包含：

  ```text
  carrier_frequency_hz
  reference_distance_m
  path_loss_exponent
  bandwidth_hz
  transmit_power_w
  noise_psd_w_per_hz
  noise_figure_linear
  maximum_antenna_gain_linear
  minimum_antenna_gain_linear
  minimum_distance_m
  ```

  提供只读属性：

  ```python
  reference_gain_linear
  ```

  使用自由空间参考增益：

  ```text
  (c / (4 * pi * f * d0)) ** 2
  ```

  该参考增益不包含方向性天线增益。

  ### `EnvironmentConfig`

  至少包含：

  ```text
  num_relays
  delta_t_s
  max_steps
  relay_motion_limits
  high_motion_limits
  low_motion_limits
  flight_bounds
  hard_safety_distance_m
  soft_safety_distance_m
  hard_max_link_distance_m
  rate_reference_bps
  channel
  ```

  增加默认工厂：

  ```python
  default_environment_config() -> EnvironmentConfig
  ```

  默认：

  ```text
  num_relays = 4
  delta_t_s = 0.2
  max_steps = 500
  ```

  所有配置必须验证为有限且物理上有效。

  ---

  ## 2. H 和 L 航点轨迹

  在 `trajectories.py` 中实现：

  ```python
  WaypointFollower
  ```

  要求：

  - 接收形状为 `(N, 3)` 的有限航点；
  - 至少包含一个航点；
  - 支持循环；
  - 支持 `reset()`；
  - 根据当前位置生成期望速度；
  - 使用现有 `make_velocity_feasible()` 满足速度和加速度约束；
  - 到达航点容差范围后切换下一个航点；
  - 不直接修改 `UAVState`。

  主要方法：

  ```python
  velocity_for(
      state: UAVState,
      limits: MotionLimits,
      delta_t_s: float,
  ) -> np.ndarray
  ```

  ---

  ## 3. 执行前安全过滤

  在 `safety.py` 中实现：

  ```python
  SafetyFilterResult
  NoFeasibleActionError
  normalized_action_to_velocity()
  filter_relay_velocities()
  ```

  ### 动作格式

  环境接收：

  ```python
  actions.shape == (K, 3)
  ```

  每个动作必须有限，且每个分量位于 `[-1, 1]`。

  动作含义：

  - 前两个分量映射为水平速度；
  - 水平向量范数大于 1 时按比例缩放到单位圆；
  - 第三个分量为正时映射到最大上升速度；
  - 为负时映射到最大下降速度。

  ### 过滤顺序

  1. 动作映射为请求速度；
  2. 使用 `make_velocity_feasible()` 满足速度和加速度限制；
  3. 计算所有 UAV 的候选下一位置；
  4. 检查中继边界；
  5. 检查所有 UAV 两两硬安全距离；
  6. 检查所有相邻逻辑链路距离。

  如果完整动作不可行，在以下插值线上搜索：

  ```text
  v(lambda) = current_velocity
            + lambda * (feasible_velocity - current_velocity)
  ```

  从 `lambda = 1` 递减到 `0`，至少检查 21 个候选值。

  选择最大的可行 `lambda`。

  这种插值必须保持速度和加速度约束。

  如果没有任何可行候选，抛出：

  ```python
  NoFeasibleActionError
  ```

  不得先更新位置再裁剪位置。

  `SafetyFilterResult` 至少返回：

  ```text
  requested_velocities_mps
  applied_velocities_mps
  intervention_norms
  scale
  ```

  ---

  ## 4. 多中继环境

  在 `environment.py` 中实现：

  ```python
  MultiRelayEnvironment
  ```

  不继承 Gymnasium。

  ### `reset()`

  接口：

  ```python
  observation, info = env.reset(seed=0)
  ```

  默认初始化：

  ```text
  H
  R1
  R2
  R3
  R4
  L
  ```

  初始位置必须：

  - 位于边界内；
  - 满足安全距离；
  - 满足相邻链路距离限制；
  - 默认沿 H 到 L 方向合理分布。

  H 和 L 分别使用不同的循环航点。

  ### `step()`

  接口：

  ```python
  observation, reward, terminated, truncated, info = env.step(actions)
  ```

  执行顺序必须是：

  ```text
  读取全部旧状态
  → 计算 H/L 下一速度
  → 计算中继请求速度
  → 执行安全过滤
  → 一次性提交全部 UAV 下一状态
  → 计算通信
  → 计算奖励
  → 构造下一观测
  ```

  禁止逐架修改状态。

  若安全过滤无解：

  - 不提交候选状态；
  - `terminated = True`；
  - 返回有限负奖励；
  - 在 `info` 中记录失败原因。

  达到 `max_steps` 时：

  ```python
  truncated = True
  ```

  ### 通信计算

  使用所有旧位置与新位置的时隙中点：

  ```text
  q_mid = (q_old + q_new) / 2
  ```

  对每条相邻链路计算：

  - 三维距离；
  - 仰角；
  - 两端偶极子增益；
  - 信道增益；
  - SNR；
  - Shannon 容量。

  使用：

  ```python
  optimal_tdma_rate()
  ```

  计算端到端服务速率。

  K=4 时必须生成五条链路。

  ---

  ## 5. 观测

  返回字典：

  ```python
  {
      "local": local_observations,
      "global": global_state,
  }
  ```

  ### 局部观测

  形状：

  ```text
  (K, 23)
  ```

  每个中继包括：

  ```text
  自身归一化位置                 3
  自身归一化速度                 3
  前一节点相对位置               3
  前一节点相对速度               3
  后一节点相对位置               3
  后一节点相对速度               3
  上一步实际中继速度             3
  归一化中继序号                 1
  episode 进度                   1
  ```

  所有值必须有限。

  ### 全局状态

  包括：

  - 所有 UAV 的归一化位置；
  - 所有 UAV 的归一化速度；
  - 当前每跳归一化容量；
  - episode 进度。

  返回一维有限数组。

  不要在本阶段固定写死全局状态长度，应根据 `K` 计算。

  ---

  ## 6. 奖励

  使用共享团队奖励：

  ```text
  rate_reward
  - link_cost
  - separation_cost
  - intervention_cost
  - motion_cost
  ```

  要求：

  - 端到端速率除以 `rate_reference_bps`；
  - 链路距离接近硬上限时产生软成本；
  - UAV 距离低于软安全距离时产生软成本；
  - 请求速度与实际速度差异产生干预成本；
  - 速度平方和速度变化平方只能称为运动代价；
  - 所有奖励项均为有限值；
  - 在 `info["reward_terms"]` 中分别返回各项。

  不要加入链路平衡奖励。

  ---

  ## 7. `info`

  每一步至少返回：

  ```text
  positions_m
  velocities_mps
  requested_relay_velocities_mps
  applied_relay_velocities_mps
  intervention_norms
  safety_scale
  hop_distances_m
  hop_elevation_angles_rad
  hop_capacities_bps
  tdma_fractions
  rate_e2e_bps
  minimum_uav_distance_m
  reward_terms
  step_index
  ```

  所有数值数组必须是副本，避免外部修改环境状态。

  ---

  ## 8. 等距跟随基线

  在 `baselines.py` 中实现：

  ```python
  equal_spacing_actions(env: MultiRelayEnvironment) -> np.ndarray
  ```

  目标位置为当前 H 和 L 之间的等比例位置：

  ```text
  Rk target = H + k / (K + 1) * (L - H)
  ```

  根据目标方向产生归一化动作。

  返回：

  ```text
  (K, 3)
  ```

  动作必须有限并位于合法范围。

  ---

  ## 9. 测试

  所有新测试放在：

  ```text
  tests/test_environment.py
  ```

  至少验证：

  1. 默认重置生成 H、四个中继和 L；
  2. 局部观测形状为 `(4, 23)`；
  3. 全局状态为有限一维数组；
  4. 相同 seed 的 reset 结果一致；
  5. 非法动作形状或范围抛出 `ValueError`；
  6. 一步状态满足 `q_next = q_old + v_applied * delta_t`；
  7. H 和 L 能沿航点运动；
  8. 所有 UAV 同步更新；
  9. K=4 时输出五条链路；
  10. TDMA 比例之和为 1；
  11. 请求速度和实际速度均出现在 `info`；
  12. 安全过滤后边界、距离和链路约束满足；
  13. 等距跟随基线输出 `(4, 3)`；
  14. 使用随机动作运行至少 500 个环境步骤，必要时 reset，不得出现 NaN 或异常崩溃。

  不要创建更多测试文件。

  ---

  ## 10. 公开接口和 README

  在 `__init__.py` 中只额外导出：

  ```python
  MultiRelayEnvironment
  default_environment_config
  ```

  README 增加：

  - 当前已实现多中继动态环境；
  - 动作形状；
  - `reset()` 和 `step()` 的最小示例；
  - 当前尚未实现强化学习。

  ---

  ## 11. 验证

  运行：

  ```bash
  python -m pip install -e ".[dev]"
  python scripts/check_install.py
  python -m pytest
  ```

  所有测试必须通过。

  ---

  ## 12. Git 和执行记录

  先提交功能代码：

  ```bash
  git add .
  git commit -m "stage-2: add multi-relay dynamic environment"
  git push
  ```

  然后覆盖 `aaa.md`：

  ```markdown
  # 本次执行结果

  - 阶段：2
  - 任务：多中继动态环境
  - 完成状态：
  - 修改和新增文件：
  - 测试结果：
  - 代码 Commit ID：
  - 当前分支：
  - GitHub 推送结果：
  - 计划偏差：
  - 遗留问题：
  ```

  填写真实结果，再提交：

  ```bash
  git add aaa.md
  git commit -m "docs: record stage-2 result"
  git push
  ```

  最终确认：

  ```bash
  git status
  ```

  工作区必须干净。

  ---

  ## 禁止事项

  本阶段不要实现：

  - Gymnasium 或 PettingZoo
  - replay buffer
  - SAC、MASAC、MAPPO
  - 神经网络
  - 训练脚本
  - 精确旋翼能耗
  - 跨时隙通信队列
  - 动态路由
  - MPC
  - 绘图和实验结果导出

  完成后只回复：

  - 测试结果
  - 两个 Commit ID
  - 推送结果
  - 遗留问题
