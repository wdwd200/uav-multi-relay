# Codex 修复计划：阶段 3G-R2 未通过项——速度边界数值稳定性

## 1. 目标

修复合法速度因浮点舍入略高于上限而被拒绝的问题，然后使用完全相同的场景、奖励权重、随机种子和训练参数重新执行阶段 3G-R2。

本任务不新增总计划阶段，也不修改奖励、场景或 MASAC 参数。

开始后用本计划覆盖 `AGENTS.md`。

## 2. 修改范围

允许修改：

```text
src/uav_multi_relay/kinematics.py
src/uav_multi_relay/safety.py
tests/test_physics.py
tests/test_environment.py
AGENTS.md
aaa.md
```

不得修改：

```text
奖励公式和奖励权重
MASAC 更新公式
Replay Buffer
通信模型
场景参数
训练步数
随机种子
基线算法
```

## 3. 数值修复

### 3.1 当前速度验证

在 `make_velocity_feasible()` 中，对速度上限使用尺度相关容差：

```python
horizontal_tolerance = 1e-9 * max(
    1.0,
    limits.max_horizontal_speed_mps,
)
climb_tolerance = 1e-9 * max(
    1.0,
    limits.max_climb_speed_mps,
)
descent_tolerance = 1e-9 * max(
    1.0,
    limits.max_descent_speed_mps,
)
```

只有超过：

```text
limit + tolerance
```

才抛出 `ValueError`。

明显超限的当前速度仍必须被拒绝。

### 3.2 状态规范化

对处于容差范围内、但数值略高于限制的当前速度，先规范化到精确可行范围：

- 水平速度投影到最大水平速度球；
- 垂直速度裁剪到上升/下降范围。

随后再计算加速度限制和请求速度。

不得只放宽检查却继续传播超限数值。

### 3.3 安全过滤器输出

`filter_relay_velocities()` 对插值得到的每组 applied velocity 再执行一次可行化，确保返回并写入状态的速度满足：

```text
horizontal norm <= max horizontal speed
vertical speed within climb/descent limits
acceleration change within limits
```

候选位置必须使用规范化后的 applied velocity 计算。

不得使用事后位置裁剪。

## 4. 测试

至少新增以下回归测试：

1. 精确位于速度上限的当前速度被接受；
2. 使用 `np.nextafter(limit, np.inf)` 产生的一单位末位超限被接受并规范化；
3. `30.000000000000004 m/s` 被视为浮点边界误差；
4. 超过容差的当前速度仍抛出 `ValueError`；
5. 水平、上升和下降三个方向均覆盖；
6. `make_velocity_feasible()` 返回值严格位于配置限制内；
7. 安全过滤器插值后的 applied velocity 严格位于限制内；
8. 连续多步边界运动不会积累出非法状态；
9. 使用以下固定回归场景运行 greedy 至少 30 步，不得再出现速度边界异常：

```text
K = 4
waypoint radius = 90
max_steps = 250
reward intervention = 0.1
reward motion = 0.1
episode seed = 20004
greedy sweeps = 1
```

不得将该异常简单捕获后忽略。

## 5. 验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

然后执行固定 greedy 回归，确认 seed `20004` 的前 30 步正常。

## 6. 提交数值修复

```bash
git add AGENTS.md \
  src/uav_multi_relay/kinematics.py \
  src/uav_multi_relay/safety.py \
  tests/test_physics.py tests/test_environment.py
git commit -m "fix: stabilize velocity limit boundaries"
git push
```

## 7. 原样重跑阶段 3G-R2

创建新的空目录：

```text
outputs/stage3g_r2_seed0_retry
```

不得继续使用中止训练的 Replay Buffer 或 episode 状态，因为 checkpoint 不包含这些内容。

重新训练时必须保持：

```text
steps = 20000
seed = 0
num_relays = 4
waypoint radius = 90
max_steps = 250
batch size = 256
random action steps = 2000
update after steps = 2000
updates per step = 1
evaluation interval = 2500
evaluation episodes = 5

reward:
rate = 1.0
link = 1.0
separation = 1.0
intervention = 0.1
motion = 0.1
failure = 1.0
```

训练必须从头完成 20,000 步。

随后使用最佳 checkpoint 比较：

```text
masac
random
stationary
equal_spacing
weighted_spacing
greedy
mpc
```

比较设置保持：

```text
episodes = 10
seed = 20000
max_steps = 250
greedy sweeps = 1
MPC horizon = 2
MPC population = 8
MPC iterations = 2
```

不得改变参数、筛选 episode 或提前停止。

## 8. 判定规则

阶段 3G 仍使用原判定规则：

```text
MASAC mean return > stationary mean return
MASAC mean return > random mean return
MASAC mean rate >= stationary mean rate
MASAC termination rate <= stationary termination rate
```

如果完整训练和比较执行成功但未达到性能条件，下一任务才进入：

```text
阶段 3G-R3——训练稳定性和奖励贡献诊断
```

如果再次因新的代码异常中止，则记录具体异常，不得直接进行调参。

## 9. aaa.md

完成后覆盖写入：

```markdown
# 本次执行结果

- 阶段：3G-R2
- 类型：阶段 3G 验收修复，不新增总计划阶段
- 任务：速度边界修复后重新训练与基线比较
- 完成状态：
- 数值错误根因：
- 容差规则：
- 状态规范化方式：
- 固定 greedy 回归结果：
- 完整测试结果：
- 编译验证：
- 实际训练步数：
- 最佳评估步数：
- 七策略比较完成情况：
- 各策略平均 return：
- 各策略平均端到端速率：
- 各策略终止率：
- 各策略安全干预率：
- 各策略平均动作计算时间：
- MASAC 相对 Stationary 的 return 提升：
- MASAC 相对 Stationary 的速率提升：
- 阶段 3G 是否通过：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议任务：
```

提交结果：

```bash
git add aaa.md
git commit -m "docs: record retried MASAC validation result"
git push
git status --short
```

如果 Git 再次出现内存读取错误，立即停止，不自动重试，不执行破坏性 Git 命令，并如实记录。

最终工作区必须干净，`outputs/` 不得提交。
