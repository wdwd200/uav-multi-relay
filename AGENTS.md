# Codex 执行计划：阶段 3G-R3 验收修复

## 1. 任务性质

当前任务属于：

```text
阶段 3G-R3 验收修复
```

性质：

```text
阶段 3 的诊断验收修复
不新增正式阶段
不得进入阶段 4
不得开始 3G-R4 的算法修改
```

本轮只处理以下问题：

1. 恢复上一版本中被删除的诊断回归测试；
2. 保留当前版本新增的 MASAC 诊断测试；
3. 修正评估 applied action Q 指标的错误命名；
4. 收紧关于 Critic 外推和数值稳定性的结论；
5. 从现有训练日志补充 Actor/Critic 梯度轨迹证据；
6. 按当前阶段名称重新命名结果文档；
7. Codex 执行结束后，在 CLI 最终输出中明确显示结果文档名称。

本轮不得重新训练 20,000 步。

------

## 2. 当前已确认问题

### 2.1 诊断测试发生回退

当前 `tests/test_diagnostics.py` 增加了新的 MASAC 诊断测试，但上一版本中的部分测试被删除。

必须恢复的旧测试包括：

1. `RewardWeights` 默认值检查；
2. 非法奖励权重校验；
3. 运动成本只计算受控中继；
4. 加权奖励和奖励组成项一致性；
5. failure penalty 使用 failure 权重；
6. `num_relays=1` 的场景诊断；
7. `num_relays=4` 的场景诊断；
8. 场景诊断输出文件禁止覆盖已有文件。

不得为了恢复旧测试而删除当前新增测试。

### 2.2 指标命名不准确

当前诊断脚本将评估 episode 中的 applied action Q 命名为：

```text
replay_applied_action_q_mean
```

但该动作来自当前确定性评估轨迹的安全过滤结果，不是 Replay Buffer 样本。

因此名称必须修正。

### 2.3 诊断结论存在过度推断

当前结果只能说明：

```text
在最终确定性评估状态下，
Actor raw action 和 safety-filtered applied action 的平均 Q 差异不显著。
```

不能直接得出：

```text
已经排除 Critic 外推问题。
```

同时，Critic loss 和 TD error 的增大属于高优先级异常迹象，但目前还没有证明它们是高终止率的直接根因。

### 2.4 结果文档命名不符合新规则

当前固定结果文档名：

```text
CODEX_EXECUTION_REPORT.md
```

不再使用。

本轮结果文档必须根据阶段和任务命名为：

```text
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

------

## 3. 允许修改的文件

主要允许修改：

```text
tests/test_diagnostics.py
scripts/diagnose_masac.py
AGENTS.md
README.md
CODEX_EXECUTION_REPORT.md
aaa.md
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

如诊断摘要由其他文件生成，可以修改直接相关的报告生成代码，但必须保持最小范围。

除非测试发现直接相关错误，否则不得修改：

```text
MASAC loss
MASAC 更新顺序
Actor 网络结构
Critic 网络结构
Target Critic 更新方式
Replay Buffer
Replay Buffer 中 applied action 的保存语义
奖励公式
奖励权重
安全过滤器
运动模型
通信模型
TDMA 模型
环境状态转移
终止条件
固定训练配置
```

------

## 4. 恢复并合并诊断测试

以提交 `6fc0ba8` 中原有的 `tests/test_diagnostics.py` 为参考，恢复被删除的旧测试。

同时保留当前版本中的新增测试，包括：

1. interval action 统计；
2. requested/applied action mismatch；
3. reward contribution 分解；
4. 周期 checkpoint；
5. failure trace；
6. 诊断 RNG 不改变更新结果；
7. MASACUpdateMetrics 有限性。

实现要求：

- 不得使用旧文件整体覆盖新文件；
- 必须把两组测试合并；
- 测试函数重名时应重新命名；
- 不得删除测试来获得全绿；
- 不得弱化已有断言；
- 不得把精确断言改成只验证“不抛异常”。

合并后，`tests/test_diagnostics.py` 至少应覆盖：

```text
奖励权重配置
奖励计算
失败惩罚
场景诊断
输出文件保护
动作区间统计
奖励贡献统计
失败轨迹
周期 checkpoint
诊断有限性
诊断不改变随机数和参数更新
```

------

## 5. 修正评估 Q 指标命名

在 `scripts/diagnose_masac.py` 中，将评估阶段的字段：

```text
replay_applied_action_q_mean
actor_raw_minus_replay_q_mean
```

修改为：

```text
evaluation_applied_action_q_mean
actor_raw_minus_evaluation_applied_q_mean
```

同步修改：

```text
JSON 输出字段
JSONL 输出字段
Markdown 诊断摘要
结果文档
相关测试
变量名称
文字说明
```

注意：

如果 `MASACUpdateMetrics` 中存在真正基于 ReplayBatch 计算的：

```text
replay_action_q_mean
```

该名称是正确的，不得全局替换。

修复后必须明确区分：

```text
Replay Buffer sampled applied action Q
```

和：

```text
Current evaluation trajectory applied action Q
```

------

## 6. 修正诊断结论

诊断摘要和结果文档中不得继续写：

```text
已排除 Critic 外推
```

应修改为：

```text
最终确定性评估状态下，Actor raw action 与 safety-filtered applied action
的平均 Q 差异不显著。

该结果不能排除训练全过程中，或 Replay Buffer 动作分布之外的
Critic 外推问题。
```

对于 Critic 稳定性，必须分成三个层次。

### 6.1 已确认事实

至少包括：

```text
critic loss 在训练中后期明显增大
TD error 在训练中后期增大
已记录的 Q、loss、TD error 和梯度均为有限值
Q1、Q2 和 target Q 的均值没有发生明显分离
```

### 6.2 高优先级假设

可以写：

```text
Critic 更新尺度可能不稳定
```

### 6.3 尚未确认

必须写：

```text
尚未证明 Critic 更新尺度异常是高终止率的直接根因
尚未证明动作过滤失配是 Critic 异常的主要原因
尚未证明 failure penalty 过小直接导致策略失败
```

不得在本轮提前确定 `3G-R4` 一定修改 Critic、奖励或动作语义。

------

## 7. 补充梯度和更新尺度证据

从现有目录读取：

```text
outputs/stage3g_r3_seed0_diagnostic/training_metrics.jsonl
```

至少提取以下字段：

```text
actor_gradient_norm
critic_gradient_norm
critic_loss
td_error_mean
td_error_p95
td_error_max
q1_mean
q2_mean
target_q_mean
alpha
```

每个主要字段至少报告：

```text
第一个有效值
最大值
最大值对应的 environment step
最后一个有效值
是否全部有限
是否存在持续增长
是否存在单点尖峰
```

梯度至少需要明确报告：

```text
actor_gradient_norm 初始有效值
actor_gradient_norm 最大值及 step
actor_gradient_norm 最终值

critic_gradient_norm 初始有效值
critic_gradient_norm 最大值及 step
critic_gradient_norm 最终值
```

不得只写：

```text
梯度变大
梯度异常
梯度不稳定
```

必须给出真实数值和对应步数。

如果日志字段在部分 step 中为空，应忽略尚未开始训练更新的区间，但必须说明第一个有效值出现在哪一步。

如果以下任一情况发生：

```text
训练目录不存在
training_metrics.jsonl 不存在
所需字段不存在
日志无法解析
出现 NaN
出现 Infinity
```

必须停止正式结果生成，并在结果文档中如实说明。

不得自动重新运行 20,000 步训练。

------

## 8. 使用现有训练结果重新生成诊断

不得重新训练。

使用现有目录：

```text
outputs/stage3g_r3_seed0_diagnostic
```

创建新的空输出目录：

```text
outputs/stage3g_r3_acceptance_repair_diagnostics
```

运行：

```bash
python scripts/diagnose_masac.py \
  --run-dir outputs/stage3g_r3_seed0_diagnostic \
  --output-dir outputs/stage3g_r3_acceptance_repair_diagnostics \
  --evaluation-episodes 5 \
  --evaluation-seed 10000 \
  --comparison-episodes 10 \
  --comparison-seed 20000 \
  --device cpu
```

确认：

- 所有 JSON 文件可读取；
- 所有 JSONL 文件可逐行读取；
- 不存在 NaN；
- 不存在 Infinity；
- 新字段名称正确；
- 旧错误字段名称不再出现；
- 梯度轨迹进入诊断摘要；
- 不修改原有 checkpoint；
- 不修改原始训练日志；
- 不改变原有训练结果。

新输出目录不得提交 Git。

------

## 9. 测试与编译

先运行完整测试：

```bash
python -m pytest
```

再运行编译检查：

```bash
python -m compileall -q src tests scripts
```

然后单独运行相关测试：

```bash
python -m pytest -q \
  tests/test_diagnostics.py \
  tests/test_learning.py \
  tests/test_environment.py
```

要求：

1. 所有旧诊断测试恢复；
2. 所有新诊断测试保留；
3. 测试总数应高于本轮开始时的 `150 passed`；
4. 不得删除或跳过已有测试；
5. 不得新增无理由的 `xfail`；
6. 不得通过降低断言精度获得通过；
7. 编译检查必须成功；
8. 所有警告必须记录；
9. 测试失败时不得继续生成“completed”状态报告。

------

## 10. 结果文档命名规则

本轮结果文档必须命名为：

```text
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

不得继续使用：

```text
aaa.md
CODEX_EXECUTION_REPORT.md
```

处理规则：

```bash
git rm aaa.md
git mv CODEX_EXECUTION_REPORT.md STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

如果实际文件状态与上述命令不一致，应先运行：

```bash
git status --short
```

然后根据真实状态处理。

不得伪造：

```text
文件已删除
文件已改名
git mv 成功
```

最终仓库根目录必须满足：

```text
不存在 aaa.md
不存在 CODEX_EXECUTION_REPORT.md
存在 STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

历史结果由 Git 历史保存，不要求在仓库根目录长期保留旧报告。

------

## 11. 更新长期文档规则

在 `AGENTS.md` 中写入以下永久规则：

1. 每轮结果文档必须根据当前阶段和任务命名；
2. 文件名格式：

```text
STAGE_<阶段编号>_<简短任务名>_REPORT.md
```

1. 文件名使用大写英文字母和下划线；
2. 每份计划必须提前给出本轮精确结果文档名称；
3. Codex 不得自行决定结果文档名称；
4. 仓库根目录只保留当前轮结果文档；
5. 历史报告通过 Git 历史查看；
6. Codex CLI 最终输出必须明确打印结果文档完整文件名；
7. 不再使用固定名称：

```text
aaa.md
CODEX_EXECUTION_REPORT.md
```

同步修改 README 中所有仍指向旧结果文档的说明。

------

## 12. 本轮结果文档内容

`STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md` 至少包含以下内容：

```markdown
---
schema_version: 1
stage: 3G-R3
task_type: acceptance_repair
status: completed
branch: main
code_commit: <代码提交 SHA>
code_push_status: pushed
report_commit: self
---

# Stage 3G-R3 Acceptance Repair Report

## 1. 本轮任务
- 阶段：
- 任务：
- 是否新增正式阶段：
- 是否重新训练：
- 开始代码基线：
- 结束代码基线：

## 2. 修改文件
- 新增：
- 修改：
- 删除：
- 改名：

## 3. 测试恢复
- 恢复的旧测试：
- 保留的新测试：
- 测试总数变化：
- 是否删除测试：

## 4. 指标命名修正
- 旧字段：
- 新字段：
- 修改原因：
- Replay action 和 evaluation applied action 的区别：

## 5. 梯度与更新尺度
- Actor gradient 初始值：
- Actor gradient 最大值及 step：
- Actor gradient 最终值：
- Critic gradient 初始值：
- Critic gradient 最大值及 step：
- Critic gradient 最终值：
- Critic loss 初始/最大/最终：
- TD error 初始/最大/最终：
- 是否全部有限：
- 是否存在持续增长：
- 是否存在尖峰：

## 6. 修正后的诊断结论
- 已确认事实：
- 高优先级假设：
- 尚未确认：
- 已排除：
- Critic 外推是否被排除：
- 动作过滤失配是否被确认：
- 奖励尺度是否被确认：

## 7. 验证结果
- 完整测试命令：
- 完整测试结果：
- 相关测试结果：
- 编译结果：
- JSON/JSONL 检查：
- NaN/Infinity 检查：
- 诊断输出目录：

## 8. 阶段判定
- 阶段 3G 是否通过：
- 阶段 3G-R3 是否验收通过：
- 是否允许进入阶段 4：
- 下一建议任务：
- 下一任务依据：

## 9. Git 状态
- 代码 Commit：
- 代码是否推送：
- 报告 Commit：self
- 报告是否推送：
- 最终工作区状态：
- Git 异常：
- 未提交输出目录：
- 计划偏差：
```

不得在报告中写：

```text
本报告待提交
本报告待推送
报告 Commit ID 待补
```

报告自身所在提交使用：

```text
report_commit: self
```

报告提交的真实 SHA 在 Codex CLI 最终输出中单独显示。

------

## 13. 本轮验收标准

必须同时满足：

1. 上一版本被删除的诊断测试全部恢复；
2. 当前新增诊断测试全部保留；
3. 完整测试通过；
4. 测试数量高于 `150 passed`；
5. 编译检查通过；
6. 评估 applied action Q 不再错误命名为 replay Q；
7. JSON 和 Markdown 中旧错误字段全部消失；
8. Critic 外推结论不再过度推断；
9. 结果文档包含真实 Actor/Critic 梯度数字；
10. 结果文档区分已确认事实、假设和未确认结论；
11. 未修改 MASAC 算法行为；
12. 未修改奖励；
13. 未修改安全过滤；
14. 未重新训练；
15. 未提交 `outputs/`；
16. 根目录不存在 `aaa.md`；
17. 根目录不存在 `CODEX_EXECUTION_REPORT.md`；
18. 根目录存在：

```text
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

1. Codex CLI 最终输出显示结果文档完整名称；
2. Git 提交和推送状态真实可核对。

如果任何一项未满足：

```text
仍停留在阶段 3G-R3
不得进入阶段 3G-R4
不得进入阶段 4
```

------

## 14. 下一任务判定

本轮验收通过后，根据修正后的证据选择下一任务。

### 情况 A：Critic 梯度、loss 和 TD error 明显持续增长

下一建议：

```text
阶段 3G-R4——MASAC Critic 更新尺度与数值稳定性修复
```

### 情况 B：梯度稳定，但 requested/applied action 长期失配仍显著

下一建议：

```text
阶段 3G-R4——Actor、Critic 与安全过滤动作语义一致性修复
```

### 情况 C：算法数值基本稳定，总 return 主要由终止惩罚和 episode 长度主导

下一建议：

```text
阶段 3G-R4——奖励尺度与终止指标受控实验
```

### 情况 D：证据仍不足

下一建议：

```text
阶段 3G-R3 补充诊断
```

不得在一轮中同时修改 Critic、动作语义和奖励。

不得建议进入阶段 4。

------

## 15. Git 提交

### 15.1 代码和测试提交

先运行：

```bash
git status --short
```

确认只包含本轮允许修改的代码和测试文件。

添加：

```bash
git add tests/test_diagnostics.py scripts/diagnose_masac.py
```

如实际还修改了直接相关的诊断生成代码，应按真实文件添加。

提交：

```bash
git commit -m "fix: complete stage 3G-R3 diagnostic validation"
```

推送：

```bash
git push
```

记录真实代码 Commit SHA 和推送结果。

### 15.2 报告和规则提交

完成以下内容：

```text
删除 aaa.md
改名 CODEX_EXECUTION_REPORT.md
生成 STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
更新 AGENTS.md
更新 README.md
```

然后运行：

```bash
git add -A
git status --short
```

提交前检查暂存区，确保没有加入：

```text
outputs/
checkpoint
JSON
JSONL
.pytest_cache/
__pycache__/
临时文件
日志文件
```

提交：

```bash
git commit -m "docs: record stage 3G-R3 acceptance repair"
```

推送：

```bash
git push
```

最后运行：

```bash
git status --short
```

最终工作区必须干净。

------

## 16. Git 异常处理

如果任何 Git 命令触发 `git.exe` 内存读取错误或类似系统异常：

1. 立即停止 Git 操作；
2. 不自动重试；
3. 不运行 `git reset --hard`；
4. 不运行 `git gc`；
5. 不运行 `git prune`；
6. 记录触发错误的完整命令；
7. 如实记录代码是否已提交；
8. 如实记录代码是否已推送；
9. 如实记录报告是否已提交；
10. 如实记录报告是否已推送；
11. 不虚构 Commit SHA；
12. 不虚构远程状态。

------

## 17. Codex CLI 最终输出

任务结束后，必须在 CLI 最后明确打印：

```text
========================================
阶段 3G-R3 验收修复执行完成
========================================

本轮结果文档：
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md

代码 Commit SHA：
<真实代码 Commit SHA>

结果文档 Commit SHA：
<真实报告 Commit SHA>

代码 push 结果：
<真实结果>

报告 push 结果：
<真实结果>

完整测试结果：
<真实 passed 数量和警告>

编译检查：
<真实结果>

最终 git status --short：
<真实输出；干净时明确写 clean>

下一建议任务：
<根据修正后的证据填写>

========================================
```

不得只输出：

```text
报告已生成
任务已完成
已推送
```

必须明确显示本轮结果文档的完整文件名：

```text
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

------

## 18. 禁止事项

本轮禁止：

```text
进入阶段 4
直接开始 3G-R4 算法修改
重新训练 20,000 步
修改奖励公式
修改奖励权重
修改安全过滤器
修改速度边界修复
修改通信模型
修改 TDMA 模型
修改 Replay Buffer action 语义
修改 MASAC loss
修改 MASAC 更新顺序
删除旧测试
删除新测试
弱化测试断言
把 evaluation applied action 称为 replay action
宣称已排除 Critic 外推
在没有梯度数字时宣称梯度异常
提交 outputs/
继续保留 aaa.md
继续保留 CODEX_EXECUTION_REPORT.md
结果文档使用固定名称
CLI 不显示结果文档名称
伪造测试、Commit 或 push 结果
```

---

## 长期结果文档规则

每轮计划必须预先给出唯一的结果文档名，格式为：

```text
STAGE_<阶段编号>_<简短任务名>_REPORT.md
```

文件名使用大写英文字母和下划线。仓库根目录只保留当前轮报告，历史报告由 Git 历史保存；Codex CLI 最终输出必须打印该轮报告的完整文件名。不得自行决定报告文件名，也不得继续使用固定报告文件名。
