---
schema_version: 1
stage: 3G-R3
task_type: diagnostic_repair
status: completed
branch: main
code_commit: bb732ab
code_push_status: pushed
report_commit: self
---

# Codex Execution Report

## 1. 本轮任务

- 阶段：3G-R3。
- 任务：训练稳定性、动作过滤失配与奖励贡献诊断。
- 是否新增正式阶段：否。
- 开始代码基线：6fc0ba80565dbd4c6414cff4632638c87b73ac9a。
- 结束代码基线：bb732ab。

## 2. 修改文件

- 新增：`scripts/diagnose_masac.py`、`tests/test_diagnostics.py`。
- 修改：MASAC/网络、训练器、实验运行器、运行脚本、学习测试和固定 greedy 回归测试。
- 删除或改名：`aaa.md` 改名为 `CODEX_EXECUTION_REPORT.md`。

## 3. 验证结果

- 完整测试命令：`python -m pytest`。
- 测试结果：150 passed in 55.63s。
- 编译验证：`python -m compileall -q src tests scripts` 成功。
- 新增测试：更新指标有限性、诊断 RNG/优化结果不变性、区间动作统计、奖励分解、周期 checkpoint/失败轨迹；greedy 回归改为跨 reset 的 30 次调用。
- 失败测试：初次完整测试暴露旧的“同一 episode 连续 30 步”错误断言；按 §13.7 修正文案和 reset 统计后全绿。
- 警告：Git CRLF 工作树警告；无 Pytest 警告。

## 4. 诊断运行配置

- num_relays：4；waypoint radius：90 m；max_steps：250。
- training steps：20,000；training seed：0；evaluation seed：10,000。
- batch size：256；random action steps：2,000；update after steps：2,000；updates per step：1。
- reward weights：rate/link/separation/failure=1.0，intervention/motion=0.1。
- checkpoint interval：2,500。
- 输出目录：`outputs/stage3g_r3_seed0_diagnostic`。

## 5. 训练轨迹

- 各 checkpoint mean return（0/2500/5000/7500/10000/12500/15000/17500/20000）：460.921/206.808/113.986/172.872/163.439/305.824/248.397/264.457/233.626。
- 对应 mean rate（Mbps）：42.054/42.297/41.714/41.437/41.772/40.717/41.170/41.241/41.905。
- termination rate：0.8，随后所有 2,500–20,000 checkpoint 均为 1.0。
- intervention rate：0.013/0.077/0.982，随后约为 1.0。
- 最佳检查点：step_012500.pt，mean return=305.824。
- 最终检查点：step_020000.pt，mean return=233.626。
- 是否发生退化：是；初始策略的评估 return 最高，训练后始终高终止，且最佳后最终下降。

## 6. Requested 与 Applied Action

- 训练最后 5,000 步：requested |a|=0.220（19k 区间）至 0.216（20k 区间）；applied |a|=0.168 至 0.163。
- requested/action saturation：0.025%/0.0%（最后 1,000 步）。
- requested/applied 不一致率：训练最后每个 1,000 步均为 1.0；最终 deterministic 评估为 0.869。
- action mismatch L2：训练最后区间 mean=0.219、p95=0.504、max=1.275；最终 deterministic mean=0.116、p95=0.330、max=0.871。
- physical velocity mismatch L2（训练最后区间）：mean=5.023 m/s、p95=11.890 m/s、max=28.225 m/s。
- safety scale：训练最后区间 mean=0.989、minimum=0、<1 比例=1.1%；最终 deterministic mean=0.982、<1 比例=2.1%。
- 最近 5,000 步干预率：1.0。

## 7. Actor 与熵诊断

- actor mean 绝对值均值：0.185（最终 deterministic）。
- actor deterministic action 饱和率：0.0；更新中采样动作饱和率最终为 1.07%。
- actor log_std mean/min/max：-2.085/-2.915/-1.042，未接近 [-20, 2] 边界。
- alpha 初始/最低/最终：0.200/0.036/0.062；先降后回升，未触及 clamp。
- joint log probability：最终训练更新均值 11.760。
- 是否触及 log_std 或 alpha 数值边界：否。

## 8. Critic 与 TD 诊断

- critic loss：step 2k=52.0、5k=126.6、9k=619.0、13k=2026.0、17k=3153.4、20k=1294.7；均有限但尺度明显不稳定。
- q1/q2：20k mean=272.186/270.118，std=37.339/37.838；q gap=5.328。
- target Q：20k mean=272.134、std=46.340。
- TD error：20k mean=8.187、p95=17.544、max=264.115；中后期明显高于早期。
- replay applied action Q：269.739；actor raw action Q：269.800；差值仅 0.061（约 0.02%）。
- actor raw action 与 replay action 的分布差异：动作事件不一致普遍，但最终 Q 差不显著。
- 是否发现明显 Critic 外推：否；证据不支持 Actor raw-action Q 系统性虚高。

## 9. 奖励贡献

- MASAC：mean length=92.9，return=379.591，return/step=4.086；每步 rate/link/separation/intervention/motion/failure=4.160/0.044/0.001/0.099/0.094/0.011。加权累计贡献：+3864.84/-40.45/-0.56/-9.18/-8.73/-10.00。
- Random：mean length=139.6，return=567.942，return/step=4.068；每步 rate/link/separation/intervention/motion/failure=4.200/0.048/0.001/0.695/0.077/0.006。
- Stationary：mean length=250.0，return=1068.003，return/step=4.272；每步 rate/link/separation/intervention/motion/failure=4.296/0.024/0/0/0/0。
- 终止/截断：MASAC 10/0，Random 8/2，Stationary 0/10。
- 总 return 是否主要由 episode 长度决定：是。每步 return 的 MASAC/Stationary 差约 4.4%，而 episode length 差 157.1 步并导致总 return 差 688.4。

## 10. 终止分析

- MASAC/Random/Stationary terminated episode 数：10/8/0。
- failure reason：均为 `no interpolated relay velocity satisfies the hard constraints`。
- 平均发生步数：MASAC=92.9，Random=112.0。
- 终止前 safety scale：两者均值 0；终止前 action mismatch：MASAC=0.260、Random=0.966。
- 终止前最大 hop distance：MASAC=243.674 m、Random=249.060 m（硬上限 250 m）。
- 终止前最大 relay velocity：MASAC=15.380 m/s、Random=12.072 m/s。
- 失败轨迹：训练产生 `failure_traces.jsonl`，每条至多 10 步，含全部规定字段且 JSON 无非有限值。

## 11. 主要结论

1. 已确认根因：MASAC 在硬链路边界附近终止；总 return 的主要差异由存活长度而非每步收益决定；failure penalty 相对累计正 rate 贡献仅 0.259%，过小。
2. 高概率原因：Critic/Q/TD/梯度的中后期尺度增长，属于 MASAC 数值稳定性/更新尺度问题；requested/applied 动作语义长期不一致。
3. 尚不能确认的原因：动作过滤失配是否是 Critic 数值异常的主因；最终 raw-action Q 没有显著高于 replay applied-action Q。
4. 已排除原因：Actor 动作长期饱和、log_std 触边、alpha 触边、诊断产生非有限值、明显 Critic raw-action 外推。

## 12. 阶段判定

- 阶段 3G 是否通过：否。
- 本轮诊断是否完整：是；20,000 步、9 个周期 checkpoint、三策略 10 episode 比较、奖励、失败轨迹和全部指标均已生成。
- 是否允许进入阶段 4：否。
- 下一建议任务：阶段 3G-R4——MASAC 数值稳定性与更新尺度修复。
- 下一任务必须解决的问题：在不同时改变奖励或安全语义的前提下，控制 Critic/TD/梯度尺度，并验证该修复是否改善高终止率。

## 13. Git 状态

- 代码提交：bb732ab（`stage-3: add MASAC stability and action diagnostics`）。
- 代码提交是否推送：是，已推送至 `origin/main`。
- 报告提交：self。
- 最终工作区状态：待报告提交后复核。
- Git 异常：无；未发生 `git.exe` 内存读取错误。
- 未提交输出目录：`outputs/stage3g_r3_seed0_diagnostic/`（诊断运行产物，未纳入 Git）。
- 计划偏差：无；未修改奖励公式/权重、安全过滤、运动/通信模型、Replay Buffer action 语义、MASAC loss/更新顺序或固定训练配置。
