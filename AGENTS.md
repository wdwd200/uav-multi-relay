- # Codex 任务：阶段 3C——参数共享 MASAC 更新核心

  ## 目标

  实现参数共享 MASAC 的动作选择、目标 Critic、损失计算和一次参数更新。

  本次不实现环境采集循环、训练脚本、checkpoint 或实验输出。

  ## 允许新增

  ```text
  src/uav_multi_relay/learning/masac.py
  ```

  允许修改：

  ```text
  src/uav_multi_relay/learning/__init__.py
  tests/test_learning.py
  README.md
  AGENTS.md
  aaa.md
  ```

  不要修改环境、物理模型、网络结构或 Replay Buffer。

  ## 1. MASAC 类

  在 `learning/masac.py` 中实现：

  ```python
  MASACUpdateMetrics
  ParameterSharingMASAC
  ```

  初始化参数至少包括：

  ```text
  local_observation_dim
  global_state_dim
  num_relays
  action_dim
  hidden_dims
  gamma
  tau
  actor_learning_rate
  critic_learning_rate
  alpha_learning_rate
  initial_alpha
  target_entropy
  device
  ```

  默认值：

  ```text
  action_dim = 3
  hidden_dims = (256, 256)
  gamma = 0.99
  tau = 0.005
  actor_learning_rate = 3e-4
  critic_learning_rate = 3e-4
  alpha_learning_rate = 3e-4
  initial_alpha = 0.2
  ```

  `target_entropy=None` 时使用：

  ```python
  -float(num_relays * action_dim)
  ```

  所有维度、学习率和超参数必须验证。

  ## 2. 网络和优化器

  类中创建：

  ```text
  actor
  critic
  target_critic
  actor_optimizer
  critic_optimizer
  log_alpha
  alpha_optimizer
  ```

  要求：

  - Actor 使用现有 `SharedGaussianActor`；
  - Critic 使用现有 `CentralizedTwinCritic`；
  - target Critic 初始化时与在线 Critic 完全相同；
  - 在线和目标 Critic 不共享参数对象；
  - target Critic 参数必须 `requires_grad=False`；
  - `alpha = exp(log_alpha)`，始终为有限正数；
  - `initial_alpha` 必须严格大于零。

  ## 3. 动作选择

  实现：

  ```python
  act(local_observations, deterministic=False) -> np.ndarray
  ```

  要求：

  - 输入形状为 `(K, local_observation_dim)`；
  - 接受有限 NumPy 数组；
  - 内部转换为 `float32` Tensor；
  - 使用 `torch.no_grad()`；
  - 返回形状 `(K, action_dim)` 的 NumPy `float32` 数组；
  - 输出有限并位于 `[-1, 1]`；
  - 确定性模式使用 `tanh(mean)`；
  - 不写死 `K=4`。

  ## 4. 联合策略熵

  Actor 返回每个中继的：

  ```text
  (batch, K, 1)
  ```

  联合 log probability 必须按中继求和：

  ```python
  joint_log_probability = log_probability.sum(dim=1)
  ```

  结果形状必须为：

  ```text
  (batch, 1)
  ```

  不得对中继取平均。

  ## 5. Critic 目标

  实现可测试的方法：

  ```python
  compute_critic_target(batch: ReplayBatch) -> torch.Tensor
  ```

  使用：

  ```text
  next_action, next_log_probability = actor.sample(next_local_observations)
  next_joint_log_probability = sum over relays
  next_q = min(target_q1, target_q2)

  target =
  reward
  + gamma * (1 - terminated)
    * (next_q - alpha * next_joint_log_probability)
  ```

  要求：

  - 使用 `torch.no_grad()`；
  - 输出形状 `(batch, 1)`；
  - `terminated=True` 时不 bootstrap；
  - `truncated=True` 仍然 bootstrap；
  - 不得把 `terminated` 和 `truncated` 合并；
  - 所有结果有限。

  ## 6. 一次更新

  实现：

  ```python
  update(batch: ReplayBatch) -> MASACUpdateMetrics
  ```

  更新顺序：

  ```text
  1. Critic
  2. Actor
  3. entropy temperature
  4. target Critic Polyak 更新
  ```

  ### Critic

  Critic 输入必须使用：

  ```text
  batch.applied_actions
  ```

  损失：

  ```text
  MSE(q1, target) + MSE(q2, target)
  ```

  ### Actor

  重新采样当前动作：

  ```text
  action, log_probability = actor.sample(local_observations)
  joint_log_probability = sum over relays
  q = min(q1, q2)
  ```

  损失：

  ```text
  mean(alpha.detach() * joint_log_probability - q)
  ```

  Actor 更新时不要为 Critic 参数保留或累积梯度。

  ### Alpha

  损失：

  ```text
  -mean(
      log_alpha
      * (joint_log_probability.detach() + target_entropy)
  )
  ```

  ### Polyak

  ```text
  target =
  (1 - tau) * target
  + tau * online
  ```

  不要硬复制目标网络。

  ## 7. 更新指标

  `MASACUpdateMetrics` 使用不可变数据类，至少返回：

  ```text
  critic_loss
  actor_loss
  alpha_loss
  alpha
  q1_mean
  q2_mean
  target_q_mean
  joint_log_probability_mean
  ```

  全部转换为有限 Python `float`。

  ## 8. 输入和设备

  - `update()` 必须检查 batch 形状与当前 K 和维度一致；
  - 所有 batch Tensor 移动到配置的 device；
  - 训练使用 `float32`；
  - 支持 CPU；
  - CUDA 仅在用户传入且 PyTorch 支持时使用；
  - 不修改传入的 `ReplayBatch`；
  - 不让项目根包导入 PyTorch。

  ## 9. 公开接口

  在 `learning/__init__.py` 中导出：

  ```python
  MASACUpdateMetrics
  ParameterSharingMASAC
  ```

  不要修改根：

  ```text
  src/uav_multi_relay/__init__.py
  ```

  ## 10. 测试

  只修改：

  ```text
  tests/test_learning.py
  ```

  至少验证：

  1. `act()` 返回合法动作；
  2. 确定性动作重复调用结果一致；
  3. target Critic 初始值与在线 Critic一致但参数不共享；
  4. target Critic 参数不需要梯度；
  5. 联合 log probability 对 K 求和而不是平均；
  6. Critic target 形状正确且有限；
  7. `terminated=True` 时 target 等于即时奖励；
  8. `truncated=True` 时仍包含 bootstrap；
  9. 一次 `update()` 后 Actor 和在线 Critic参数发生变化；
  10. target Critic 按 Polyak 方式移动；
  11. alpha 始终为有限正数；
  12. 所有更新指标有限；
  13. Critic 更新使用 Replay Buffer 中的实际动作；
  14. 支持 `K=1、4、8`；
  15. 非法 batch 形状抛出 `ValueError`；
  16. 根包导入仍不加载 PyTorch。

  测试使用小批量和小网络，避免明显增加测试时间。

  ## 11. README

  补充：

  ```text
  已实现参数共享 MASAC 的单批次更新核心。
  环境采集循环、训练脚本、checkpoint 和完整实验尚未实现。
  ```

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
  git commit -m "stage-3: add parameter-sharing MASAC updates"
  git push
  ```

  然后覆盖 `aaa.md`：

  ```markdown
  # 本次执行结果

  - 阶段：3C
  - 任务：参数共享 MASAC 更新核心
  - 完成状态：
  - 修改和新增文件：
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
  git commit -m "docs: record stage-3 MASAC update result"
  git push
  git status
  ```

  工作区必须干净。

  ## 禁止事项

  不要实现：

  - 环境采集循环
  - 完整训练脚本
  - checkpoint
  - 日志系统
  - TensorBoard
  - MAPPO、MATD3、MADDPG
  - Actor 共享消融
  - 修改环境、奖励或物理公式
  - 新测试文件

  完成后只回复：

  - 测试结果
  - 两个 Commit ID
  - 推送结果
  - 遗留问题
