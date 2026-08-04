# Codex 任务：阶段 3A——参数共享 MASAC 网络基础

## 目标

实现参数共享随机策略网络和集中式双 Q 网络。

本次只实现神经网络基础，不实现 replay buffer、更新算法或训练循环。

## 允许新增

```text
src/uav_multi_relay/learning/__init__.py
src/uav_multi_relay/learning/networks.py
tests/test_learning.py
```

允许修改：

```text
pyproject.toml
README.md
AGENTS.md
aaa.md
```

不要修改现有环境、物理模型和基线代码。

## 1. PyTorch 依赖

在 `pyproject.toml` 中增加：

```toml
[project.optional-dependencies]
learning = ["torch>=2.2"]
dev = ["pytest", "torch>=2.2"]
```

保留现有依赖。

基础物理环境在未安装 PyTorch 时仍应能够正常导入。

## 2. 参数共享 Actor

在 `learning/networks.py` 中实现：

```python
SharedGaussianActor
```

接口至少包括：

```python
forward(local_observations)
sample(local_observations, deterministic=False)
```

要求：

- 默认局部观测维度为 23；
- 默认动作维度为 3；
- 使用两个隐藏层，默认每层 256；
- 输出高斯分布的 `mean` 和 `log_std`；
- `log_std` 限制在 `[-20, 2]`；
- 使用 `Normal.rsample()` 重参数采样；
- 使用 `tanh` 把动作限制到 `[-1, 1]`；
- 正确计算 tanh 变换后的 log probability；
- log probability 对动作维度求和并保留末尾单维；
- 支持输入形状：
  ```text
  (batch, K, local_observation_dim)
  ```
- 同一个网络必须同时处理所有中继，不能为每个中继创建独立 Actor；
- 所有输出必须有限。

不要在网络中写死 `K=4`。

## 3. 集中式双 Q 网络

实现：

```python
CentralizedTwinCritic
```

初始化参数至少包括：

```text
global_state_dim
num_relays
action_dim
hidden_dims
```

调用接口：

```python
q1, q2 = critic(global_state, joint_actions)
```

要求：

- `global_state` 形状为 `(batch, global_state_dim)`；
- `joint_actions` 形状为 `(batch, K, action_dim)`；
- 内部展平联合动作并与全局状态拼接；
- Q1 和 Q2 必须是参数独立的两个网络；
- 输出形状均为 `(batch, 1)`；
- 不写死全局状态长度或中继数量；
- 对非法输入形状抛出 `ValueError`；
- 所有输出必须有限。

## 4. 公开接口

在：

```text
src/uav_multi_relay/learning/__init__.py
```

导出：

```python
SharedGaussianActor
CentralizedTwinCritic
```

暂时不要从项目根 `uav_multi_relay.__init__` 导出学习模块。

## 5. 测试

只新增：

```text
tests/test_learning.py
```

至少验证：

1. 从默认环境动态读取局部和全局观测维度；
2. Actor 输入 `(5, 4, 23)` 时，动作形状为 `(5, 4, 3)`；
3. Actor log probability 形状为 `(5, 4, 1)`；
4. Actor 动作和 log probability 全部有限；
5. Actor 动作始终位于 `[-1, 1]`；
6. 确定性动作等于 `tanh(mean)`；
7. 随机动作反向传播后 Actor 参数获得有限梯度；
8. Critic 接收批量全局状态和联合动作；
9. Q1、Q2 形状均为 `(batch, 1)`；
10. Q1 与 Q2 不共享参数对象；
11. Critic 反向传播后两个 Q 网络均获得有限梯度；
12. 非法 Actor 或 Critic 输入形状抛出 `ValueError`。

测试应小而明确，不要测试 PyTorch 内部实现。

## 6. README

补充当前状态：

```text
已实现参数共享高斯 Actor 和集中式双 Q Critic。
MASAC 更新、经验回放和训练循环尚未实现。
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
git commit -m "stage-3: add shared actor and twin critic"
git push
```

然后覆盖 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3A
- 任务：参数共享 MASAC 网络基础
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
git commit -m "docs: record stage-3 network result"
git push
git status
```

工作区必须干净。

## 禁止事项

本次不要实现：

- replay buffer
- SAC 或 MASAC 损失更新
- target critic
- entropy temperature
- 训练循环
- checkpoint
- MAPPO、MATD3、MADDPG
- Actor 共享消融
- 实验脚本
- 修改环境或奖励函数

完成后只回复：

- 测试结果
- 两个 Commit ID
- 推送结果
- 遗留问题
