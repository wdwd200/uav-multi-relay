# Codex 执行计划：阶段 4B——MAPPO 固定配置训练与公平比较

## 1. 任务目标

当前任务属于正式阶段：

```text
阶段 4B
```

目标：

1. 使用固定配置从头训练参数共享 MAPPO；
2. 完成 20,000 环境步；
3. 使用最佳 MAPPO checkpoint 进行统一比较；
4. 与阶段 3最终 MASAC、Random、Stationary 和规则/优化基线使用相同场景与评估种子比较；
5. 如实记录结果，不进行事后调参。

本轮不要求 MAPPO 必须超过基线。

本轮是否通过只取决于：

```text
训练完整
评估完整
比较公平
结果可复现
```

MAPPO 性能不理想时也不得新增 `4B-R1` 调参任务；记录结果后继续阶段 4C。

本轮结果文档：

```text
STAGE_4B_MAPPO_TRAINING_COMPARISON_REPORT.md
```

------

## 2. 启动检查

运行：

```bash
git status --short
git branch --show-current
git log -3 --oneline
```

允许且只允许：

```text
M AGENTS.md
```

如果存在其他未提交文件，立即停止并列出。

确认代码包含：

```text
9b19c8d fix: use per-relay MAPPO probability ratios
```

确认以下 MASAC checkpoint 是否存在：

```text
outputs/stage3g_r4_final_seed0/best_checkpoint.pt
```

如果不存在：

- 不重新训练 MASAC；
- 立即停止比较准备；
- 在 CLI 中报告缺失路径。

------

## 3. 固定 MAPPO 配置

不得进行超参数搜索。

固定配置：

```text
num_relays = 4
waypoint radius = 90 m
max_steps = 250
training steps = 20000

rollout steps = 1000
update epochs = 10
mini-batch size = 250

gamma = 0.99
gae lambda = 0.95
clip ratio = 0.2
actor learning rate = 3e-4
critic learning rate = 3e-4
value loss coefficient = 0.5
entropy coefficient = 0.01
max gradient norm = 0.5

training seed = 0
evaluation seed = 10000
evaluation interval = 2500
evaluation episodes = 5
checkpoint interval = 2500
device = cpu
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

不得根据中间结果修改配置。

------

## 4. 统一比较支持

扩展现有统一比较入口，使其支持：

```text
mappo
masac
random
stationary
equal_spacing
weighted_spacing
greedy
mpc
```

推荐修改：

```text
src/uav_multi_relay/analysis/comparison.py
scripts/compare_baselines.py
tests/test_comparison.py
```

如现有结构更适合新增独立入口，可以新增：

```text
scripts/compare_algorithms.py
```

但不得复制整套比较逻辑。

命令行必须能够分别接收：

```text
--mappo-checkpoint
--masac-checkpoint
```

要求：

- 只有请求 `mappo` 时才加载 MAPPO checkpoint；
- 只有请求 `masac` 时才加载 MASAC checkpoint；
- 缺少所需 checkpoint 时立即报清晰错误；
- 保持旧的纯 MASAC/基线比较用法兼容；
- MAPPO 使用 deterministic Actor action；
- MASAC 使用 deterministic Actor action；
- 所有算法使用相同环境配置、奖励、episode seeds 和指标。

不得改变任何算法本身。

------

## 5. 公平性规则

所有比较策略统一使用：

```text
episodes = 10
episode seeds = 20000–20009
max_steps = 250
waypoint radius = 90 m
num_relays = 4
device = cpu
```

规则基线参数保持：

```text
greedy sweeps = 1
MPC horizon = 2
MPC population = 8
MPC iterations = 2
```

必须报告：

```text
mean return
return std
mean return per step
mean rate_e2e_bps
minimum rate_e2e_bps
termination rate
mean episode length
intervention rate
requested/applied mismatch rate
mean action computation time
```

MAPPO 与 MASAC 的内部动作训练语义不同：

```text
MAPPO ratio 使用 requested action
MASAC Replay Buffer 使用 applied action
```

这属于算法实现差异，报告中必须说明；不得因此修改其中任一算法。

------

## 6. 比较功能测试

至少增加：

1. MAPPO checkpoint 可由统一比较器加载；
2. MAPPO deterministic action 被实际调用；
3. MASAC 与 MAPPO 可在同一次比较中运行；
4. 所有算法收到相同 episode seeds；
5. MAPPO checkpoint 缺失时报错；
6. MASAC checkpoint 缺失时报错；
7. 未请求某算法时不要求对应 checkpoint；
8. 输出 JSON 包含全部统一指标；
9. 旧比较接口仍可运行；
10. 比较过程不修改模型参数。

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

要求：

```text
现有 191 项测试全部保留
新增测试全部通过
无新增 Pytest 警告
编译成功
```

------

## 7. MAPPO 正式训练

使用新的空目录：

```text
outputs/stage4b_mappo_seed0
```

运行：

```bash
python scripts/run_mappo_experiment.py \
  --output-dir outputs/stage4b_mappo_seed0 \
  --steps 20000 \
  --rollout-steps 1000 \
  --max-steps 250 \
  --waypoint-radius 90 \
  --update-epochs 10 \
  --mini-batch-size 250 \
  --evaluation-interval 2500 \
  --evaluation-episodes 5 \
  --checkpoint-interval 2500 \
  --reward-rate 1.0 \
  --reward-link 1.0 \
  --reward-separation 1.0 \
  --reward-intervention 0.1 \
  --reward-motion 0.1 \
  --reward-failure 1.0 \
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu
```

如果部分算法参数没有 CLI 参数，则确认其值与 `MAPPOConfig` 默认值完全一致，并在报告中记录；不得为了添加无必要的 CLI 参数修改算法。

训练必须完成：

```text
20,000 environment steps
20 full rollouts
discarded_partial_rollout_steps = 0
周期 checkpoint
best checkpoint
final checkpoint
```

所有日志必须有限。

------

## 8. MAPPO 训练检查

报告各评估 checkpoint 的：

```text
mean return
mean rate
termination rate
episode length
intervention rate
requested/applied mismatch rate
policy loss
value loss
entropy
approx KL
clip fraction
Actor gradient norm
Critic gradient norm
```

必须明确：

- 最佳 checkpoint；
- 最终 checkpoint；
- 是否出现训练退化；
- 是否出现 NaN/Infinity；
- 是否出现持续 100% clip fraction；
- 是否出现梯度长期触及 0.5 上限；
- 是否发生大量 requested/applied mismatch。

不得因为结果不好中断训练。

------

## 9. 七基线加两学习算法比较

使用：

```text
MAPPO：
outputs/stage4b_mappo_seed0/best_checkpoint.pt

MASAC：
outputs/stage3g_r4_final_seed0/best_checkpoint.pt
```

运行统一比较，策略顺序：

```text
mappo
masac
random
stationary
equal_spacing
weighted_spacing
greedy
mpc
```

这里实际为 8 个策略。

比较输出目录：

```text
outputs/stage4b_mappo_seed0/comparison
```

必须完成全部 10 个 episode，不得删除失败 episode。

------

## 10. 结果判读

本轮不得设定“MAPPO 必须获胜”的通过门槛。

只进行客观分类：

### 情况 A

```text
MAPPO 明显优于 MASAC，且终止率更低
```

说明 MAPPO 更适合当前 on-policy requested-action 语义。

### 情况 B

```text
MAPPO 与 MASAC 均明显低于规则基线
```

说明问题可能不只是 MASAC 特有，后续算法比较仍需保留环境硬约束和动作过滤影响分析。

### 情况 C

```text
MAPPO 数值稳定但性能一般
```

继续阶段 4C，不进行本轮调参。

### 情况 D

```text
训练或比较因代码错误未完成
```

阶段 4B 不通过，只修复直接导致中断的问题。

性能差不属于代码错误。

------

## 11. 结果文档

将当前报告改名：

```bash
git mv STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md \
  STAGE_4B_MAPPO_TRAINING_COMPARISON_REPORT.md
```

报告至少记录：

```text
固定训练配置
完整测试结果
训练步数和更新次数
discarded partial rollout steps
各 checkpoint 训练轨迹
最佳和最终 checkpoint
8 策略完整指标
MAPPO 与 MASAC 动作语义差异
公平性保证
性能分类
是否出现数值异常
代码 Commit 和 push
下一任务
```

根目录最终只保留：

```text
STAGE_4B_MAPPO_TRAINING_COMPARISON_REPORT.md
```

------

## 12. Git 提交

比较功能和测试提交：

```bash
git status --short
git add <实际修改的比较代码和测试> AGENTS.md
git commit -m "feat: compare MAPPO with MASAC and baselines"
git push
```

完成训练和报告后：

```bash
git add -A
git status --short
git commit -m "docs: record stage 4B MAPPO comparison"
git push
git status --short
```

提交前确认没有加入：

```text
outputs/
checkpoint
JSON/JSONL 运行产物
缓存
临时日志
```

------

## 13. 本轮验收标准

必须同时满足：

1. 完整测试通过；
2. 比较器支持 MAPPO 和 MASAC；
3. 20,000 步 MAPPO 训练完成；
4. 20 个 rollout 全部更新；
5. partial rollout 丢弃数为 0；
6. 周期、best、final checkpoint 完整；
7. 所有日志有限；
8. 8 个策略均完成 10 episode；
9. episode seeds 完全一致；
10. 不筛选失败结果；
11. 未修改环境、奖励、安全过滤、MASAC 或 MAPPO 算法；
12. 结果文档名称正确；
13. 最终工作区干净。

通过后下一任务固定为：

```text
阶段 4C——参数共享 MATD3 实现与短程验证
```

无论 MAPPO 性能好坏，都不在 4B 内调参。

------

## 14. Codex CLI 最终输出

```text
========================================
阶段 4B MAPPO 训练与比较结果
========================================

本轮结果文档：
STAGE_4B_MAPPO_TRAINING_COMPARISON_REPORT.md

代码 Commit SHA：
<真实 SHA>

结果文档 Commit SHA：
<真实 SHA>

完整测试：
<真实结果>

MAPPO 训练：
<步数、rollout 数、最佳评估>

8 策略比较：
<是否完整及 MAPPO/MASAC/Stationary 核心指标>

性能分类：
<A / B / C / D>

代码 push：
<真实结果>

报告 push：
<真实结果>

最终 git status --short：
<真实输出；干净时写 clean>

下一建议任务：
阶段 4C——参数共享 MATD3 实现与短程验证

========================================
```
