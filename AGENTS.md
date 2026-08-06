# Codex 执行计划：阶段 3G-R3——训练稳定性、动作过滤失配与奖励贡献诊断

## 1. 任务性质

当前任务属于：

```text
阶段 3G-R3
```

性质：

```text
阶段 3 的性能验收修复与诊断
不新增总计划正式阶段
不得进入阶段 4
```

阶段 `3G-R2` 的速度边界数值稳定性修复已经完成。

当前需要诊断：

1. MASAC 为什么几乎每一步都依赖安全过滤器；
2. requested action 与 applied action 是否存在严重分布失配；
3. Critic 是否在 Replay Buffer 的 applied action 分布之外评估 Actor action；
4. Actor 是否出现动作饱和或方差异常；
5. 熵温度、Q 值、TD error 和梯度是否稳定；
6. 奖励各组成项的实际数量级和贡献；
7. MASAC 高终止率的主要失败原因；
8. episode 长度差异是否主导了总 return 比较。

本轮只收集证据并给出明确诊断，不直接修改：

```text
奖励公式
奖励权重
MASAC 更新公式
Replay Buffer 中 applied action 的定义
安全过滤器
运动与通信模型
训练步数
场景参数
阶段 3G 的通过条件
```

只有诊断结果明确后，下一轮才制定针对性修复。

------

## 2. 当前已确认事实

上一轮固定配置结果：

```text
MASAC mean return = 274.2579599049063
Random mean return = 567.9423682978447
Stationary mean return = 1068.0028132582956

MASAC mean rate = 40886060.897320315 bps
Stationary mean rate = 42955245.356898956 bps

MASAC termination rate = 1.0
Stationary termination rate = 0.0

MASAC intervention rate = 1.0
Random intervention rate = 1.0
Stationary intervention rate = 0.0
```

阶段 3G 未通过。

还已确认固定 greedy 回归在不再触发速度边界异常后，仍可能因为：

```text
no interpolated relay velocity satisfies the hard constraints
```

而终止。

不得把“速度异常已消失”和“episode 可以连续存活”混为一谈。

------

## 3. 结果文档改名

从本任务开始，仓库根目录的报告文件为：

```text
CODEX_EXECUTION_REPORT.md
```

原因：

- 新名称明确表示它是 Codex 每轮执行报告；
- 继续采用每轮覆盖写入，不累计旧任务结果；
- 避免报告中记录无法自洽的“本报告提交待推送”状态。

README、AGENTS.md 及仓库中所有工作流文字均应引用此文件名。根目录只能保留：

```text
CODEX_EXECUTION_REPORT.md
```

------

## 4. 新报告格式

`CODEX_EXECUTION_REPORT.md` 必须使用以下结构覆盖写入：

```markdown
---
schema_version: 1
stage: 3G-R3
task_type: diagnostic_repair
status: completed
branch: main
code_commit: <代码提交 SHA>
code_push_status: pushed
report_commit: self
---

# Codex Execution Report

## 1. 本轮任务

- 阶段：
- 任务：
- 是否新增正式阶段：
- 开始代码基线：
- 结束代码基线：

## 2. 修改文件

- 新增：
- 修改：
- 删除或改名：

## 3. 验证结果

- 完整测试命令：
- 测试结果：
- 编译验证：
- 新增测试：
- 失败测试：
- 警告：

## 4. 诊断运行配置

- num_relays：
- waypoint radius：
- max_steps：
- training steps：
- training seed：
- evaluation seed：
- batch size：
- random action steps：
- update after steps：
- updates per step：
- reward weights：
- checkpoint interval：
- 输出目录：

## 5. 训练轨迹

- 各检查点 mean return：
- 各检查点 mean rate：
- 各检查点 termination rate：
- 各检查点 intervention rate：
- 最佳检查点：
- 最终检查点：
- 是否发生退化：

## 6. Requested 与 Applied Action

- requested action 绝对值均值：
- applied action 绝对值均值：
- requested action 饱和率：
- applied action 饱和率：
- requested/applied 不一致率：
- action mismatch L2 mean：
- action mismatch L2 p95：
- action mismatch L2 max：
- safety scale mean：
- safety scale minimum：
- safety scale < 1 的比例：
- 最近 5,000 步干预率：

## 7. Actor 与熵诊断

- actor mean 绝对值均值：
- actor deterministic action 饱和率：
- actor log_std mean：
- actor log_std minimum：
- actor log_std maximum：
- alpha 初始值：
- alpha 最终值：
- joint log probability：
- 是否触及 log_std 或 alpha 数值边界：

## 8. Critic 与 TD 诊断

- critic loss 轨迹：
- q1 mean/std：
- q2 mean/std：
- q1/q2 gap：
- target Q mean/std：
- TD error mean：
- TD error p95：
- TD error maximum：
- replay applied action Q：
- actor raw action Q：
- actor raw action 与 replay action 的分布差异：
- 是否发现明显 Critic 外推：

## 9. 奖励贡献

按 MASAC、Random、Stationary 分别报告：

- mean episode length：
- mean return：
- mean return per step：
- rate reward 每步均值：
- link cost 每步均值：
- separation cost 每步均值：
- intervention cost 每步均值：
- motion cost 每步均值：
- failure penalty 每 episode 均值：
- 各加权项对总 return 的累计贡献：
- 总 return 是否主要由 episode 长度决定：

## 10. 终止分析

- MASAC terminated episode 数：
- Random terminated episode 数：
- Stationary terminated episode 数：
- failure reason 计数：
- 各 failure reason 的平均发生步数：
- 终止前 safety scale：
- 终止前 action mismatch：
- 终止前 hop distances：
- 终止前 relay velocities：

## 11. 主要结论

按证据强度排序列出：

1. 已确认根因：
2. 高概率原因：
3. 尚不能确认的原因：
4. 已排除原因：

## 12. 阶段判定

- 阶段 3G 是否通过：
- 本轮诊断是否完整：
- 是否允许进入阶段 4：
- 下一建议任务：
- 下一任务必须解决的问题：

## 13. Git 状态

- 代码提交：
- 代码提交是否推送：
- 报告提交：self
- 最终工作区状态：
- Git 异常：
- 未提交输出目录：
- 计划偏差：
```

注意：

```text
report_commit: self
```

表示“包含当前报告文件的提交”。

不得在报告中写：

```text
本报告待提交
本报告待推送
报告 Commit ID 待补
```

报告无法可靠记录包含它自身的最终 SHA，也不应制造自引用提交循环。

报告提交和推送成功后的真实 SHA，应在 Codex 最终聊天输出中单独给出，不再修改报告文件。

------

## 5. 允许修改的文件

允许修改：

```text
AGENTS.md
README.md
CODEX_EXECUTION_REPORT.md

src/uav_multi_relay/learning/masac.py
src/uav_multi_relay/learning/networks.py
src/uav_multi_relay/training/trainer.py
src/uav_multi_relay/training/experiment.py
src/uav_multi_relay/training/evaluator.py
src/uav_multi_relay/analysis/comparison.py
src/uav_multi_relay/analysis/diagnostics.py

scripts/run_experiment.py
scripts/compare_baselines.py
scripts/diagnose_masac.py

tests/test_learning.py
tests/test_training.py
tests/test_experiment.py
tests/test_evaluation.py
tests/test_comparison.py
tests/test_diagnostics.py
```

不需要修改的文件不要改动。

禁止修改：

```text
src/uav_multi_relay/kinematics.py
src/uav_multi_relay/safety.py
src/uav_multi_relay/communication.py
src/uav_multi_relay/environment.py 中的奖励计算公式
src/uav_multi_relay/config.py 中的现有默认参数
Replay Buffer 的 applied action 保存语义
MASAC 的损失函数和更新顺序
```

若为了输出已经存在于 `info` 中的诊断字段，确实必须对环境做非行为性修改，必须满足：

```text
不改变状态转移
不改变奖励
不改变安全过滤结果
不改变随机数调用
不改变 episode 终止条件
```

并在报告中单独说明。

------

## 6. 诊断实现原则

### 6.1 不得改变训练随机轨迹

增加指标时不得额外调用随机 Actor sampling。

优先复用 MASAC 更新过程中已经生成的：

```text
actions
log_probability
q1
q2
target
actor_q1
actor_q2
```

Actor 分布统计应调用确定性的 `forward()` 获取：

```text
mean
log_std
```

不得为日志额外调用随机 `sample()`，避免消耗 PyTorch RNG 并改变训练轨迹。

如果某项诊断无法避免随机调用，必须保存并恢复：

```text
torch CPU RNG state
torch CUDA RNG state
NumPy RNG state
```

但应尽量不采用这种方式。

### 6.2 指标必须是观测性的

增加和删除诊断开关时，在相同：

```text
模型初始化
ReplayBatch
随机种子
```

下，单次更新后的模型参数必须一致。

不得因为记录梯度、Q 值或 Actor 分布而改变优化结果。

### 6.3 不得掩盖异常

所有新增指标必须：

```text
有限
可 JSON 序列化
不允许 NaN
不允许 Infinity
```

出现非有限值时必须立即失败并报告具体字段。

不得使用 `nan_to_num()` 把诊断异常静默替换为零。

------

## 7. 扩展 MASAC 更新指标

扩展 `MASACUpdateMetrics`，至少增加：

```text
q1_std
q2_std
q_gap_mean
target_q_std
td_error_mean
td_error_p95
td_error_max
actor_q_mean
replay_action_q_mean
actor_action_abs_mean
actor_action_saturation_rate
actor_mean_abs_mean
actor_log_std_mean
actor_log_std_min
actor_log_std_max
actor_gradient_norm
critic_gradient_norm
```

定义要求：

### Q gap

```python
q_gap_mean = mean(abs(q1 - q2))
```

### TD error

分别计算：

```python
abs(q1 - target)
abs(q2 - target)
```

合并后计算：

```text
mean
95th percentile
maximum
```

### 动作饱和

动作分量满足：

```python
abs(action) >= 0.95
```

即计为饱和。

同时报告：

```text
分量饱和率
任意中继至少一个分量饱和的 batch 比例
```

### 梯度范数

在 optimizer step 前计算所有有梯度参数的全局 L2 norm。

不得执行额外 backward。

------

## 8. 扩展训练采集指标

训练循环必须按日志区间统计，而不仅是从训练开始以来的累计均值。

至少收集：

```text
requested action mean/std/abs mean
applied action mean/std/abs mean
requested action saturation rate
applied action saturation rate
requested-applied L2 mean
requested-applied L2 p95
requested-applied L2 max
action mismatch event rate
safety scale mean
safety scale minimum
safety scale < 1 rate
intervention event rate
termination count
truncation count
episode length mean
episode return mean
return per step mean
failure reason counts
```

Action mismatch event 定义为任意中继：

```python
norm(requested_action - applied_action) > 1e-6
```

同时统计物理速度空间中的：

```text
requested velocity - applied velocity
```

不得只统计归一化动作。

------

## 9. 奖励贡献统计

使用环境现有：

```python
info["reward_terms"]
```

收集以下原始项：

```text
rate_reward
link_cost
separation_cost
intervention_cost
motion_cost
failure_penalty
weighted_reward
```

同时计算有符号加权贡献：

```text
+ weight_rate * rate_reward
- weight_link * link_cost
- weight_separation * separation_cost
- weight_intervention * intervention_cost
- weight_motion * motion_cost
- weight_failure * failure_penalty
```

必须区分：

```text
每步均值
每 episode 累积
所有 episode 总计
终止 episode
截断 episode
```

必须增加：

```text
return_per_step
```

因为当前 MASAC 与 Stationary 的 episode 长度和终止率差异很大，总 return 可能主要反映存活时间。

本轮不得修改 reward weights。

------

## 10. 失败轨迹记录

对每个 terminated episode，记录终止前最多 10 步的轻量轨迹。

输出：

```text
failure_traces.jsonl
```

每个 episode 一条记录，至少包含：

```text
episode_seed
episode_index
termination_step
failure_reason
last_10_requested_actions
last_10_applied_actions
last_10_action_mismatch_norms
last_10_safety_scales
last_10_positions
last_10_velocities
last_10_hop_distances
last_10_rates
last_10_reward_terms
```

不得保存完整训练 Replay Buffer。

确保数组转换为普通 JSON list，并禁止 NaN。

------

## 11. 周期 Checkpoint

扩展实验运行器，支持：

```text
--checkpoint-interval
```

本次固定为：

```text
2500
```

保存：

```text
checkpoints/step_000000.pt
checkpoints/step_002500.pt
checkpoints/step_005000.pt
checkpoints/step_007500.pt
checkpoints/step_010000.pt
checkpoints/step_012500.pt
checkpoints/step_015000.pt
checkpoints/step_017500.pt
checkpoints/step_020000.pt
```

`step_000000.pt` 必须在任何环境采集和参数更新前保存。

周期 checkpoint 不替代：

```text
best_checkpoint.pt
final_checkpoint.pt
```

保存 checkpoint 不得消耗随机数或改变训练轨迹。

------

## 12. 新增诊断脚本

新增：

```text
scripts/diagnose_masac.py
```

脚本接收：

```text
run directory
checkpoint directory
诊断输出目录
evaluation episodes
evaluation seed
comparison seed
device
```

至少完成：

1. 加载每个周期 checkpoint；
2. 在相同评估 seeds 上确定性评估；
3. 输出性能随训练步数变化；
4. 汇总 Actor、Alpha、Critic 和动作分布；
5. 对 best 和 final checkpoint 执行详细策略诊断；
6. 对 MASAC、Random、Stationary 执行奖励贡献比较；
7. 汇总失败原因；
8. 生成机器可读 JSON 和 Markdown 摘要。

输出目录至少包含：

```text
diagnostic_config.json
checkpoint_evolution.jsonl
policy_diagnostics.jsonl
reward_contributions.json
failure_summary.json
diagnostic_summary.json
diagnostic_summary.md
```

这些运行产物不得提交 Git。

------

## 13. 测试要求

至少新增以下测试。

### 13.1 指标有限性

验证扩展后的 `MASACUpdateMetrics`：

```text
所有字段为有限 float
TD error 非负
gradient norm 非负
saturation rate 位于 [0, 1]
log_std min <= mean <= max
```

### 13.2 诊断不改变训练更新

创建两个初始化和 RNG 完全相同的 Agent。

对同一个 ReplayBatch：

```text
一个执行启用诊断的 update
一个执行等价普通 update
```

验证：

```text
Actor 参数一致
Critic 参数一致
Target Critic 参数一致
log_alpha 一致
optimizer state 一致
```

若实现中不存在诊断开关，则使用内部辅助函数测试诊断统计不产生额外随机调用和参数修改。

### 13.3 区间统计

构造已知 requested/applied action，验证：

```text
mismatch rate
L2 mean
L2 p95
saturation rate
safety scale mean/min
intervention rate
```

计算准确。

### 13.4 奖励分解

构造固定 `reward_terms`，验证原始项和加权贡献之和等于：

```text
weighted_reward
```

允许浮点近似误差。

### 13.5 失败轨迹

验证：

```text
最多保留最后 10 步
字段完整
JSON 无 NaN
failure reason 保留
```

### 13.6 周期 Checkpoint

短实验中验证：

```text
step_000000.pt
周期 checkpoint
final checkpoint
best checkpoint
```

均存在，且 metadata 中的 environment steps 正确。

### 13.7 固定速度回归文案修正

现有固定 greedy 测试可以继续验证 30 次调用不再出现速度异常，但必须：

- 统计发生过多少次 termination/reset；
- 不得把跨 reset 的 30 次调用命名为“同一 episode 连续 30 步”；
- 测试名称和报告文字必须准确表达真实语义。

不得为了让该测试通过而改变环境终止行为。

------

## 14. 完整验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

再执行相关测试，例如：

```bash
python -m pytest -q \
  tests/test_learning.py \
  tests/test_training.py \
  tests/test_experiment.py \
  tests/test_evaluation.py \
  tests/test_comparison.py \
  tests/test_diagnostics.py
```

记录真实测试数量和警告。

------

## 15. 正式诊断训练

使用新的空目录：

```text
outputs/stage3g_r3_seed0_diagnostic
```

保持上一轮配置：

```bash
python scripts/run_experiment.py \
  --output-dir outputs/stage3g_r3_seed0_diagnostic \
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
  --seed 0 \
  --evaluation-seed 10000 \
  --device cpu \
  --diagnostics
```

不得使用上一轮 checkpoint 续训。

不得修改固定配置。

本轮重新训练的目的不是再次证明性能，而是获得完整、可解释的训练轨迹。

------

## 16. 诊断评估

训练完成后运行：

```bash
python scripts/diagnose_masac.py \
  --run-dir outputs/stage3g_r3_seed0_diagnostic \
  --output-dir outputs/stage3g_r3_seed0_diagnostic/diagnostics \
  --evaluation-episodes 5 \
  --evaluation-seed 10000 \
  --comparison-episodes 10 \
  --comparison-seed 20000 \
  --device cpu
```

详细比较至少包含：

```text
masac
random
stationary
```

使用 episode seeds：

```text
20000 至 20009
```

不得筛选失败 episode。

------

## 17. 必须回答的诊断问题

`diagnostic_summary.md` 和最终执行报告必须明确回答：

1. MASAC 的 requested action 是否长期饱和？
2. Actor 的 log_std 是否接近上下界？
3. Alpha 是否持续单调异常变化或触及数值夹紧范围？
4. Critic loss、Q 值和 TD error 是否随训练发散？
5. Critic 对 Actor raw action 的 Q 是否显著高于 replay applied action？
6. requested/applied action mismatch 是否在训练后期仍接近 100%？
7. safety scale 是否长期显著低于 1？
8. MASAC 的主要终止原因是什么？
9. 终止通常发生在哪些 step？
10. 终止前 hop distance、速度和动作 mismatch 有何共同模式？
11. 总 return 差异有多少来自 episode 长度？
12. rate reward、intervention cost、motion cost 和 failure penalty 的数量级是否匹配？
13. 当前 failure penalty 是否相对于一个完整 episode 的累计正奖励过小？
14. 当前证据是否支持“Actor/Applied Action 分布失配”为主要原因？
15. 下一轮应修复算法一致性、奖励尺度还是训练超参数？

不得只写“需要进一步调参”。

必须给出按证据强弱排序的结论。

------

## 18. 本轮完成标准

本轮 `3G-R3` 完成不要求 MASAC 超过基线。

必须同时满足：

```text
20,000 步诊断训练完整完成
周期 checkpoint 完整
所有新增指标有限
MASAC/Random/Stationary 10 episode 诊断完整
奖励贡献完整
终止原因完整
失败轨迹完整
requested/applied 分布差异完整
Actor、Alpha、Critic 和 TD 轨迹完整
诊断报告明确回答 15 个问题
```

若任一正式运行因代码错误中止：

```text
仍停留在 3G-R3
不得根据不完整结果制定调参结论
不得进入阶段 4
```

------

## 19. 下一任务判定

本轮结束后只允许以下建议。

### 情况 A：动作过滤失配得到明确支持

下一建议：

```text
阶段 3G-R4——Actor、Critic 与安全过滤动作语义一致性修复
```

### 情况 B：Critic 或熵训练明显不稳定

下一建议：

```text
阶段 3G-R4——MASAC 数值稳定性与更新尺度修复
```

### 情况 C：奖励贡献或 episode 长度明显主导结果

下一建议：

```text
阶段 3G-R4——奖励尺度和性能指标的受控修正实验
```

### 情况 D：诊断仍不足

下一建议：

```text
阶段 3G-R3 补充诊断
```

不得建议进入阶段 4。

不得在没有证据的情况下同时修改多个方向。

------

## 20. Git 提交

### 20.1 代码和测试提交

确认：

```bash
git status --short
```

添加真实修改文件，提交：

```bash
git commit -m "stage-3: add MASAC stability and action diagnostics"
git push
```

记录真实代码 Commit SHA 和 push 结果。

### 20.2 报告改名和结果提交

诊断全部完成后，覆盖写入规定格式的最终报告。

更新所有必要引用，然后：

```bash
git add AGENTS.md README.md CODEX_EXECUTION_REPORT.md
git commit -m "docs: record stage 3G-R3 diagnostic result"
git push
git status --short
```

若精确暂存命令不适用，则使用：

```bash
git add -A
```

但必须先检查 staged 文件，确保没有加入：

```text
outputs/
checkpoint
JSONL 运行产物
临时文件
缓存
```

最终 Codex 聊天输出必须给出：

```text
代码 Commit SHA
报告 Commit SHA
两次 push 的真实结果
最终 git status --short
```

不得再修改报告来记录其自身 SHA。

------

## 21. Git 异常处理

任何 Git 命令再次触发 `git.exe` 内存读取错误时：

1. 立即停止 Git 操作；
2. 不自动重试；
3. 不运行 `git reset --hard`；
4. 不运行 `git gc`；
5. 不运行 `git prune`；
6. 记录触发错误的完整命令；
7. 如实记录哪些提交已经创建；
8. 如实记录哪些提交已经推送；
9. 不虚构 Commit SHA；
10. 不虚构远程状态。

------

## 22. 禁止事项

本轮禁止：

```text
进入阶段 4
修改速度边界修复
修改安全过滤器行为
修改奖励公式
修改奖励权重
修改 MASAC loss
修改 Replay Buffer action 语义
修改固定训练配置
减少训练步数
更换训练 seed
筛选有利 episode
删除失败轨迹
额外随机采样导致训练轨迹变化
用 NaN 替换为零继续运行
提交 outputs/
保留旧报告文件
报告中写“待推送”
只写“需要进一步调参”而不给证据
```
