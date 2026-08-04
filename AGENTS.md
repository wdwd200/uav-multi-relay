# Codex 任务：修复阶段 3A 网络结构与测试

## 目标

修复参数共享 Actor 多出一个隐藏线性层的问题，并让测试真正从环境中动态读取维度。

不要开始 replay buffer、MASAC 更新或训练循环。

## 允许修改

- `src/uav_multi_relay/learning/networks.py`
- `tests/test_learning.py`
- `AGENTS.md`
- `aaa.md`

不要新增文件。

## 1. 修复 Actor 网络结构

当前默认 Actor backbone 实际包含三个线性层：

```text
23 -> 256 -> 256 -> 256
```

修复后，`hidden_dims=(256, 256)` 必须严格生成两个隐藏线性层：

```text
23 -> 256 -> 256
```

然后直接连接：

```text
mean_head
log_std_head
```

要求：

- backbone 中线性层数量等于 `len(hidden_dims)`；
- 默认 backbone 中恰好有两个 `nn.Linear`；
- 每个隐藏线性层后使用 `ReLU`；
- 不改变 Actor 的输入输出接口；
- 保留任意合法 `hidden_dims` 的支持；
- 不为各个中继创建独立 Actor；
- 不改变 tanh 动作和 log probability 公式。

不要继续使用会额外增加线性层的：

```python
_mlp(input_dim, hidden_dims, hidden_dims[-1])
```

## 2. 动态读取环境维度

修改 `tests/test_learning.py`。

从默认环境实际返回的观测中读取：

```python
num_relays = observation["local"].shape[0]
local_observation_dim = observation["local"].shape[-1]
global_state_dim = observation["global"].shape[-1]
```

使用这些变量构造：

```python
SharedGaussianActor(
    local_observation_dim=local_observation_dim,
)

CentralizedTwinCritic(
    global_state_dim=global_state_dim,
    num_relays=num_relays,
)
```

不要在该测试中直接写死：

```text
4
23
42
```

## 3. 加强高价值测试

仍然只修改：

```text
tests/test_learning.py
```

增加或加强以下检查：

1. 默认 Actor backbone 恰好包含两个 `nn.Linear`；
2. 自定义 `hidden_dims=(64, 32, 16)` 时，backbone 恰好包含三个 `nn.Linear`；
3. 从环境动态读取维度后，Actor 和 Critic 能完成一次前向传播；
4. 使用 `K=1` 和 `K=8` 检查网络没有写死四个中继；
5. Actor 反向传播后，所有可训练参数均有有限梯度；
6. Critic 反向传播后，Q1 和 Q2 的所有参数均有有限梯度；
7. 保留现有动作范围、确定性动作、参数独立性和非法形状测试。

测试保持小而明确。

## 4. 基础包无 PyTorch 导入

确认以下行为继续成立：

```python
import uav_multi_relay
```

项目根包不得导入：

```python
uav_multi_relay.learning
torch
```

不要修改根 `uav_multi_relay/__init__.py`。

## 验证

运行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
python -m pytest
```

所有测试必须通过。

另外运行：

```bash
python - <<'PY'
import torch
from uav_multi_relay.learning import SharedGaussianActor

actor = SharedGaussianActor()
linear_count = sum(
    isinstance(module, torch.nn.Linear)
    for module in actor.backbone
)
assert linear_count == 2
print("actor hidden layers:", linear_count)
PY
```

输出必须为：

```text
actor hidden layers: 2
```

## Git

先检查并同步现有提交：

```bash
git status
git log --oneline -5
git push
```

然后提交本次修复：

```bash
git add .
git commit -m "fix: correct shared actor architecture"
git push
```

覆盖 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3A 修复
- 任务：修复共享 Actor 网络结构与动态维度测试
- 完成状态：
- 修改文件：
- 测试结果：
- Actor 隐藏层检查：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- 计划偏差：
- 遗留问题：
```

填写真实结果后提交：

```bash
git add aaa.md
git commit -m "docs: record stage-3 network repair"
git push
git status
```

工作区必须干净。

## 禁止事项

不要实现：

- replay buffer
- target critic
- MASAC 或 SAC 损失
- entropy temperature
- 训练循环
- checkpoint
- MAPPO、MATD3、MADDPG
- 新测试文件
- 环境或奖励修改

完成后只回复：

- 测试结果
- Actor 隐藏层数量
- 两个 Commit ID
- 推送结果
- 遗留问题
