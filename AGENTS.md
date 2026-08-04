# Codex 任务：阶段 3B——执行动作接口与经验回放

## 目标

建立 MASAC 使用的完整 transition 数据接口，并实现固定容量经验回放。

经验回放必须保存安全过滤后实际执行的归一化动作，不得保存物理速度，也不得用请求动作代替实际动作。

本次不要实现 MASAC 损失、目标网络或训练循环。

## 允许新增

```text
src/uav_multi_relay/learning/replay_buffer.py
```

允许修改：

```text
src/uav_multi_relay/safety.py
src/uav_multi_relay/environment.py
src/uav_multi_relay/learning/__init__.py
tests/test_environment.py
tests/test_learning.py
README.md
AGENTS.md
aaa.md
```

不要新增测试文件。

## 1. 物理速度反向映射

在 `safety.py` 中实现：

```python
velocity_to_normalized_action(
    velocity_mps,
    limits: MotionLimits,
) -> np.ndarray
```

要求：

- 输入必须是有限 `(3,)` 向量；
- 输入速度必须满足水平、上升和下降速度限制；
- 非法速度抛出 `ValueError`；
- 水平分量除以最大水平速度；
- 正垂直速度除以最大上升速度；
- 负垂直速度除以最大下降速度；
- 输出有限并位于 `[-1, 1]`；
- 不修改输入；
- 对合法速度满足：

```python
normalized_action_to_velocity(
    velocity_to_normalized_action(velocity, limits),
    limits,
) == pytest.approx(velocity)
```

允许仅为浮点误差执行极小范围的 `clip`，不得用裁剪掩盖非法速度。

## 2. 环境输出实际归一化动作

在环境每次 `reset()` 和 `step()` 的 `info` 中增加：

```text
requested_relay_actions
applied_relay_actions
```

要求：

- 两者形状均为 `(K, 3)`；
- `requested_relay_actions` 是传入环境的归一化动作；
- `applied_relay_actions` 由实际执行速度通过
  `velocity_to_normalized_action()` 得到；
- `applied_relay_actions` 再映射为速度时，必须等于
  `applied_relay_velocities_mps`；
- reset 时两者均为零；
- 安全过滤无解导致终止时，也必须返回真实有限值；
- 返回数组必须是副本；
- 保留现有物理速度字段。

不要改变环境动作含义、状态转移或奖励。

## 3. 经验回放

在 `learning/replay_buffer.py` 中实现：

```python
ReplayBatch
MultiAgentReplayBuffer
```

### 初始化参数

至少包括：

```text
capacity
num_relays
local_observation_dim
global_state_dim
action_dim
seed
```

所有维度和容量必须是正整数。

### 保存字段

预分配 NumPy 数组保存：

```text
local_observations
global_states
applied_actions
rewards
next_local_observations
next_global_states
terminated
truncated
```

形状：

```text
local_observations      (capacity, K, local_dim)
global_states           (capacity, global_dim)
applied_actions         (capacity, K, action_dim)
rewards                 (capacity, 1)
next_local_observations (capacity, K, local_dim)
next_global_states      (capacity, global_dim)
terminated              (capacity, 1)
truncated               (capacity, 1)
```

要求：

- 使用 `float32` 保存连续数据；
- `terminated` 和 `truncated` 分开保存；
- 不得把 `truncated` 合并为终止；
- action 必须有限且位于 `[-1, 1]`；
- 文档明确说明 action 是安全过滤后实际执行的归一化动作；
- 输入数据必须复制，之后修改调用方数组不得影响 buffer；
- 容量满后使用循环覆盖；
- `len(buffer)` 返回当前有效数量；
- 使用 buffer 自己的 `np.random.Generator`；
- 相同 seed 和相同数据产生相同采样序列。

### `sample()`

接口：

```python
batch = buffer.sample(batch_size, device=None)
```

要求：

- 有效样本不足时抛出 `ValueError`；
- 默认无放回采样；
- 返回 `ReplayBatch`；
- 所有字段转换为 PyTorch Tensor；
- 连续数据为 `torch.float32`；
- tensor 默认 `requires_grad=False`；
- 支持传入 CPU 或 CUDA device；
- 不返回内部数组的可修改视图。

不要让根包导入 PyTorch。

## 4. 公开接口

在 `learning/__init__.py` 中导出：

```python
ReplayBatch
MultiAgentReplayBuffer
velocity_to_normalized_action 不从 learning 导出
```

不要修改根：

```text
src/uav_multi_relay/__init__.py
```

## 5. 测试

### `tests/test_environment.py`

增加：

1. 合法速度反向映射后能够精确恢复；
2. 非法水平、上升和下降速度被拒绝；
3. `info` 同时返回请求动作和实际动作；
4. 实际动作重新映射后等于实际速度；
5. 第一步使用极端动作时，至少一个实际动作与请求动作不同；
6. 返回数组不能用于修改环境内部状态。

### `tests/test_learning.py`

增加：

1. buffer 初始长度为零；
2. 添加 transition 后长度正确；
3. sampled batch 所有形状正确；
4. sampled tensor 均为 `float32` 且有限；
5. applied action 保持在 `[-1, 1]`；
6. 修改原输入数组不会改变已保存内容；
7. 容量满后正确循环覆盖；
8. 相同 seed 的采样结果一致；
9. 样本不足和非法输入抛出 `ValueError`；
10. `terminated` 与 `truncated` 保持独立；
11. 使用环境 `info["applied_relay_actions"]` 能完成一次真实 transition 的保存和采样；
12. 根包导入仍不加载 `torch`。

测试保持小而明确。

## 6. README

补充说明：

- 环境现在同时报告请求动作和实际执行归一化动作；
- replay buffer 保存实际执行归一化动作；
- terminated 与时间截断分开保存；
- MASAC 更新和训练循环尚未实现。

## 验证

运行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
python -m pytest
```

所有测试必须通过。

## Git

先提交功能代码：

```bash
git add .
git commit -m "stage-3: add multi-agent replay buffer"
git push
```

然后覆盖 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3B
- 任务：执行动作接口与经验回放
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
git commit -m "docs: record stage-3 replay result"
git push
git status
```

工作区必须干净。

## 禁止事项

不要实现：

- MASAC 或 SAC 损失
- target critic
- Polyak 更新
- entropy temperature
- 优化器更新
- 训练循环
- checkpoint
- MAPPO、MATD3、MADDPG
- 新测试文件
- 奖励或通信公式修改

完成后只回复：

- 测试结果
- 两个 Commit ID
- 推送结果
- 遗留问题
