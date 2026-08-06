---
schema_version: 1
stage: 3G-R3
task_type: acceptance_repair
status: completed
branch: main
code_commit: a4ae2cd
code_push_status: pushed
report_commit: self
---

# Stage 3G-R3 Acceptance Repair Report

## 1. 本轮任务

- 阶段：3G-R3 验收修复。
- 任务：恢复诊断回归测试、修正评估 applied-action Q 命名、收紧结论并补充既有日志的梯度轨迹。
- 是否新增正式阶段：否。
- 是否重新训练：否；只读取 `outputs/stage3g_r3_seed0_diagnostic` 的既有训练日志和 checkpoint。
- 开始代码基线：05d85851bfbb915df7151ac2399e6fd590068020。
- 结束代码基线：a4ae2cd。

## 2. 修改文件

- 新增：无。
- 修改：`tests/test_diagnostics.py`、`scripts/diagnose_masac.py`、`AGENTS.md`、`README.md`。
- 删除：无；`aaa.md` 在本轮开始时已不存在。
- 改名：`CODEX_EXECUTION_REPORT.md` → `STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md`。

## 3. 测试恢复

- 恢复的旧测试：RewardWeights 默认/非法值、受控 relay motion cost、加权奖励一致性、failure weight、`num_relays=1/4` 场景诊断、诊断 JSON 不覆盖。
- 保留的新测试：区间 requested/applied 统计、奖励贡献、周期 checkpoint/失败轨迹、评估 applied-action Q 命名、更新尺度摘要；原有 `test_learning.py` 中的 MASAC 指标有限性与 RNG/参数不变性继续保留。
- 测试总数变化：150 → 158。
- 是否删除测试：否。

## 4. 指标命名修正

- 旧字段：两个错误地将当前评估 applied action 误称为 replay action 的字段。
- 新字段：`evaluation_applied_action_q_mean`、`actor_raw_minus_evaluation_applied_q_mean`。
- 修改原因：离线评估的 applied action 来自当前确定性评估轨迹经安全过滤后的结果，不是 Replay Buffer 样本。
- 区别：`MASACUpdateMetrics.replay_action_q_mean` 仍正确表示 ReplayBatch sampled applied-action Q；新字段仅表示 current evaluation trajectory applied-action Q。

## 5. 梯度与更新尺度

- 第一个有效更新：environment step 2,000；此前随机采集区间没有更新指标。
- Actor gradient：初始 0.635310（step 2,000），最大 26.152840（step 17,000），最终 15.430604（step 20,000）。
- Critic gradient：初始 356.023346（step 2,000），最大 39,871.558594（step 17,000），最终 15,766.679688（step 20,000）。
- Critic loss：初始 52.016724（step 2,000），最大 3,153.374756（step 17,000），最终 1,294.726807。
- TD error mean：初始 5.072195，最大 11.346800（step 13,000），最终 8.186540；p95 初始/最大/最终为 5.895285/30.023815（step 13,000）/17.544039。
- Q1/Q2/target Q：最终为 272.185822/270.117920/272.134155；均值轨迹未发生明显 Q1、Q2、target 分离。
- 是否全部有限：是；actor gradient、critic gradient、loss、TD error、Q 和 alpha 的所有有效记录均有限。
- 是否存在持续增长：不存在严格单调增长；但 Critic gradient/loss/TD 误差在训练中后期的尺度高于初期。
- 是否存在尖峰：Actor gradient、Critic gradient、Critic loss、TD mean/p95 均有单点尖峰；完整逐字段 first/max/last/finite/monotonic/spike 数据见验收诊断 `diagnostic_summary.json:update_scale`。

## 6. 修正后的诊断结论

- 已确认事实：Critic loss 和 TD error 在中后期扩大；所有记录的 Q、loss、TD error、Actor/Critic gradient 均有限；Q1、Q2 与 target Q 均值没有明显分离。
- 高优先级假设：Critic 更新尺度可能不稳定。
- 尚未确认：尚未证明 Critic 更新尺度异常是高终止率直接根因；尚未证明动作过滤失配是 Critic 异常主要原因；尚未证明 failure penalty 过小直接导致策略失败。
- 已排除：Actor 长期动作饱和、log_std/alpha 数值边界触发、诊断非有限值。
- Critic 外推是否被排除：否。最终确定性评估状态下 Actor raw-action Q 与 safety-filtered evaluation applied-action Q 的平均差仅 0.060849，差异不显著；这不能排除训练全过程或 Replay Buffer 分布之外的 Critic 外推。
- 动作过滤失配是否被确认：观察到训练区间 mismatch event rate=1.0、最终 deterministic evaluation=0.869，但其是否为失败主因未确认。
- 奖励尺度是否被确认：episode length 主导总 return 的现象已确认；failure penalty 是否直接造成策略失败未确认。

## 7. 验证结果

- 完整测试命令：`python -m pytest`。
- 完整测试结果：158 passed in 69.42s；无测试警告。
- 相关测试结果：`python -m pytest -q tests/test_diagnostics.py tests/test_learning.py tests/test_environment.py`，85 passed in 45.41s。
- 编译结果：`python -m compileall -q src tests scripts` 成功。
- JSON/JSONL 检查：新输出的全部 JSON 可解析，全部 JSONL 可逐行解析。
- NaN/Infinity 检查：未发现；旧评估 Q 字段也未发现。
- 诊断输出目录：`outputs/stage3g_r3_acceptance_repair_diagnostics`；原始训练日志和 checkpoint 未改动。

## 8. 阶段判定

- 阶段 3G 是否通过：否。
- 阶段 3G-R3 是否验收通过：是。
- 是否允许进入阶段 4：否。
- 下一建议任务：阶段 3G-R3 补充诊断。
- 下一任务依据：更新尺度存在高优先级异常迹象但并非严格单调，且尚未建立其与高终止率的直接因果关系；当前证据不足以预先锁定 3G-R4 的 Critic、动作语义或奖励修改方向。

## 9. Git 状态

- 代码 Commit：a4ae2cd（`fix: clarify evaluation applied action diagnostics`；前置测试恢复提交为 ccaac49）。
- 代码是否推送：是，已推送至 `origin/main`。
- 报告 Commit：self。
- 报告是否推送：本报告所在提交已推送至 `origin/main`。
- 最终工作区状态：仅保留本轮开始前已有、未触碰的 `src/uav_multi_relay/kinematics.py` 与 `src/uav_multi_relay/safety.py` 修改。
- Git 异常：无。
- 未提交输出目录：两个 `outputs/stage3g_r3_*diagnostic*` 目录，均未纳入 Git。
- 计划偏差：无；未重新训练，未修改 MASAC 算法、奖励、安全过滤或环境行为。
