# Codex 执行计划：阶段 3G-R4——Critic 梯度裁剪受控修复与阶段 3 最终验收

## 1. 任务性质

当前任务属于：

```text
阶段 3G-R4
```

本轮是阶段 3最后一轮修复与性能验收。

本轮结束后只允许两种结论：

```text
通过阶段 3，进入阶段 4
```

或：

```text
阶段 3实现完成，但性能验收失败，停止继续追加 R 修复
```

不得再建议 `3G-R5`、`3G-R6` 或新的补充诊断。

本轮结果文档必须命名为：

```text
STAGE_3G_R4_FINAL_VALIDATION_REPORT.md
```

------

## 2. 开始前检查

首先运行：

```bash
git status --short
git log -1 --oneline
```

要求：

- 当前分支为 `main`；
- 当前代码包含 `f989dc1`；
- 工作区干净。

如果 `git status --short` 有任何输出：

- 立即停止；
- 列出实际文件；
- 不自动恢复；
- 不继续修改或训练。

------

## 3. 本轮唯一算法修改

当前诊断已经确认：

```text
Actor gradient 最大值约 26
Critic gradient 最大值约 39871
Critic loss 和 TD error 在训练中后期显著增大
```

本轮只增加：

```text
Critic 全局梯度范数裁剪
```

不得同时修改：

- Critic learning rate；
- Actor learning rate；
- 奖励公式或权重；
- Replay Buffer；
- Actor/applied action 语义；
- 网络结构；
- 安全过滤器；
- 终止条件。

### 3.1 配置参数

为 `ParameterSharingMASAC` 增加可选参数：

```python
critic_gradient_clip_norm: float | None = None
```

规则：

- `None` 表示完全保持现有行为；
- 非 `None` 时必须是正有限值；
- 本轮正式候选值固定为：

```text
1000.0
```

### 3.2 更新位置

Critic 更新顺序必须是：

```python
critic_optimizer.zero_grad(...)
critic_loss.backward()
计算裁剪前梯度范数
clip_grad_norm_(critic.parameters(), critic_gradient_clip_norm)
计算裁剪后梯度范数
critic_optimizer.step()
```

不得改变：

- loss 公式；
- target Q 公式；
- backward 次数；
- optimizer step 次数；
- Actor 和 alpha 更新顺序。

### 3.3 新增指标

至少记录：

```text
critic_gradient_norm_pre_clip
critic_gradient_norm_post_clip
critic_gradient_clip_applied
critic_gradient_clip_rate
```

定义：

```text
critic_gradient_clip_applied：
裁剪前范数 > 配置阈值时为 1，否则为 0

critic_gradient_clip_rate：
日志区间中发生裁剪的更新比例
```

所有值必须有限。

------

## 4. 允许修改的文件

主要允许修改：

```text
src/uav_multi_relay/learning/masac.py
src/uav_multi_relay/training/experiment.py
src/uav_multi_relay/training/trainer.py
scripts/run_experiment.py
tests/test_learning.py
tests/test_training.py
tests/test_experiment.py
AGENTS.md
README.md
STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
STAGE_3G_R4_FINAL_VALIDATION_REPORT.md
```

不需要修改的文件不要改动。

禁止修改：

```text
src/uav_multi_relay/kinematics.py
src/uav_multi_relay/safety.py
环境奖励计算
通信模型
TDMA
Replay Buffer action 语义
基线策略
比较种子
场景参数
```

------

## 5. 必须新增的测试

### 5.1 默认行为不变

使用相同初始化、ReplayBatch 和随机状态，对比：

```text
critic_gradient_clip_norm=None
```

与修改前的更新结果。

验证：

- Actor 参数一致；
- Critic 参数一致；
- Target Critic 参数一致；
- alpha 一致；
- optimizer state 一致。

### 5.2 裁剪生效

构造会产生大梯度的 ReplayBatch，设置：

```text
critic_gradient_clip_norm=1.0
```

验证：

```text
pre_clip > 1.0
post_clip <= 1.0 + 数值容差
critic_gradient_clip_applied = 1
```

### 5.3 不需要裁剪

构造小梯度更新，验证：

```text
pre_clip <= 阈值
pre_clip 与 post_clip 近似相等
critic_gradient_clip_applied = 0
```

### 5.4 参数校验

以下值必须拒绝：

```text
0
负数
NaN
Infinity
布尔值
```

### 5.5 Checkpoint

保存和加载后必须保留：

```text
critic_gradient_clip_norm
```

旧 checkpoint 没有该字段时应兼容为：

```text
None
```

------

## 6. 测试和冒烟验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

要求：

- 测试数量不得低于 `166`；
- 全部通过；
- 无新增警告；
- 编译成功。

然后运行一个短冒烟训练：

```bash
python scripts/run_experiment.py \
  --output-dir outputs/stage3g_r4_smoke \
  --steps 100 \
  --max-steps 20 \
  --waypoint-radius 90 \
  --batch-size 8 \
  --random-action-steps 8 \
  --update-after-steps 8 \
  --updates-per-step 1 \
  --reward-rate 1.0 \
  --reward-link 1.0 \
  --reward-separation 1.0 \
  --reward-intervention 0.1 \
  --reward-motion 0.1 \
  --reward-failure 1.0 \
  --critic-gradient-clip-norm 1000 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

确认：

- 所有指标有限；
- 裁剪前后梯度均被记录；
- 训练和 checkpoint 可完成。

------

## 7. 5,000 步受控筛选

分别从头运行两个实验。

### 7.1 无裁剪对照

输出目录：

```text
outputs/stage3g_r4_screen_control
```

配置：

```text
steps = 5000
critic_gradient_clip_norm = None
seed = 0
```

### 7.2 裁剪候选

输出目录：

```text
outputs/stage3g_r4_screen_clip1000
```

配置与对照完全相同，只增加：

```text
critic_gradient_clip_norm = 1000
```

固定配置：

```text
num_relays = 4
waypoint radius = 90
max_steps = 250
batch size = 256
random action steps = 2000
update after steps = 2000
updates per step = 1
evaluation interval = 2500
evaluation episodes = 5
training seed = 0
evaluation seed = 10000
```

奖励权重保持：

```text
rate = 1.0
link = 1.0
separation = 1.0
intervention = 0.1
motion = 0.1
failure = 1.0
```

------

## 8. 是否进入完整训练的门槛

只有裁剪候选同时满足以下条件，才继续完整 20,000 步训练：

1. 无 NaN 或 Infinity；
2. `critic_gradient_norm_post_clip <= 1000`；
3. 5,000 步评估 termination rate 不高于无裁剪对照；
4. 5,000 步评估 mean return 不低于对照的 90%；
5. Critic loss 或 TD error 没有比对照明显恶化；
6. Checkpoint 保存、加载和确定性评估成功。

如果任一条件失败：

```text
不运行完整 20,000 步
阶段 3性能验收判定为失败
停止继续追加修复轮次
```

报告中必须如实说明：

```text
Critic 梯度裁剪没有获得继续完整训练的证据
```

------

## 9. 完整 20,000 步训练

筛选通过后，使用新的空目录：

```text
outputs/stage3g_r4_final_seed0
```

运行：

```bash
python scripts/run_experiment.py \
  --output-dir outputs/stage3g_r4_final_seed0 \
  --steps 20000 \
  --max-steps 250 \
  --waypoint-radius 90 \
  --batch-size 256 \
  --random-action-steps 2000 \
  --update-after-steps 2000 \
  --updates-per-step 1 \
  --log-interval 1000 \
  --evaluation-interval 2500 \
  --evaluation-episodes 5 \
  --checkpoint-interval 2500 \
  --reward-rate 1.0 \
  --reward-link 1.0 \
  --reward-separation 1.0 \
  --reward-intervention 0.1 \
  --reward-motion 0.1 \
  --reward-failure 1.0 \
  --critic-gradient-clip-norm 1000 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

不得恢复旧 checkpoint。

不得中途修改参数。

------

## 10. 七策略最终比较

完整训练成功后，使用最佳 checkpoint：

```bash
python scripts/compare_baselines.py \
  --checkpoint outputs/stage3g_r4_final_seed0/best_checkpoint.pt \
  --output-dir outputs/stage3g_r4_final_seed0/comparison \
  --episodes 10 \
  --seed 20000 \
  --max-steps 250 \
  --waypoint-radius 90 \
  --reward-rate 1.0 \
  --reward-link 1.0 \
  --reward-separation 1.0 \
  --reward-intervention 0.1 \
  --reward-motion 0.1 \
  --reward-failure 1.0 \
  --policies masac random stationary equal_spacing weighted_spacing greedy mpc \
  --greedy-sweeps 1 \
  --mpc-horizon 2 \
  --mpc-population-size 8 \
  --mpc-iterations 2 \
  --device cpu
```

所有策略必须完成 seeds：

```text
20000 至 20009
```

------

## 11. 阶段 3最终判定

阶段 3通过必须同时满足：

```text
MASAC mean return > stationary mean return
MASAC mean return > random mean return
MASAC mean rate >= stationary mean rate
MASAC termination rate <= stationary termination rate
```

### 通过时

报告填写：

```text
阶段 3：通过
下一任务：阶段 4A
```

### 未通过时

报告填写：

```text
阶段 3：实现完成，但性能验收失败
不再增加阶段 3 修复编号
下一步：由用户决定进行算法大改，或将 MASAC 保留为失败基线
```

不得建议 `3G-R5`。

------

## 12. 结果文档

将当前结果文档改名为：

```bash
git mv STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md \
        STAGE_3G_R4_FINAL_VALIDATION_REPORT.md
```

报告至少包含：

```text
开始 Commit
修改文件
新增配置
测试结果
5,000 步对照结果
是否满足完整训练门槛
完整训练结果（如执行）
七策略比较（如执行）
梯度裁剪率
裁剪前后梯度
Critic loss 和 TD error
阶段 3四项条件
阶段 3最终结论
Git 和 push 状态
```

仓库根目录最终只保留：

```text
STAGE_3G_R4_FINAL_VALIDATION_REPORT.md
```

运行产物不得提交 Git。

------

## 13. Git 提交

代码和测试完成后：

```bash
git status --short
git add src/uav_multi_relay/learning/masac.py \
        src/uav_multi_relay/training/experiment.py \
        src/uav_multi_relay/training/trainer.py \
        scripts/run_experiment.py \
        tests/test_learning.py \
        tests/test_training.py \
        tests/test_experiment.py
git commit -m "fix: stabilize MASAC critic gradients"
git push
```

只添加实际修改文件。

最终报告完成后：

```bash
git add -A
git status --short
git commit -m "docs: record final stage 3 validation"
git push
git status --short
```

提交前确认没有加入：

```text
outputs/
checkpoint
JSON/JSONL
缓存
临时日志
```

------

## 14. Codex CLI 最终输出

Codex 完成后必须打印：

```text
========================================
阶段 3G-R4 最终验收结果
========================================

本轮结果文档：
STAGE_3G_R4_FINAL_VALIDATION_REPORT.md

代码 Commit SHA：
<真实 SHA>

结果文档 Commit SHA：
<真实 SHA>

代码 push 结果：
<真实结果>

报告 push 结果：
<真实结果>

完整测试：
<真实结果>

5,000 步筛选：
<通过或未通过及核心数据>

20,000 步训练：
<已完成、未执行或失败>

七策略比较：
<已完成、未执行或失败>

阶段 3最终结论：
<通过 / 实现完成但性能验收失败>

下一任务：
<阶段 4A / 等待用户决定项目路线>

最终 git status --short：
<真实输出；干净时写 clean>

========================================
```

必须显示完整文件名：

```text
STAGE_3G_R4_FINAL_VALIDATION_REPORT.md
```

------

## 15. 禁止事项

本轮禁止：

```text
修改奖励
修改安全过滤
修改终止条件
修改 Replay Buffer action 语义
同时修改多个算法方向
跳过 5,000 步对照
筛选不通过仍强行完整训练
删除失败结果
继续追加 3G-R5
进入阶段 4 前伪造通过
提交 outputs/
伪造测试、Commit 或 push 状态
```
