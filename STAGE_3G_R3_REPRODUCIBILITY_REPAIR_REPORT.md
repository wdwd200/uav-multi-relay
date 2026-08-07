---
schema_version: 1
stage: 3G-R3
task_type: reproducibility_repair
status: completed
branch: main
code_commit: f989dc1cd72076f7d72672b8e037f75ff224d450
code_push_status: pushed
report_commit: self
---

# Stage 3G-R3 Reproducibility Repair Report

## 1. 本轮任务

- 阶段：3G-R3 可复现性修复。
- 任务：审查、测试、提交并推送已存在的统一浮点速度上限容差修改。
- 是否新增正式阶段：否。
- 是否重新训练：否。
- 开始代码基线：b41c1a0516ea747d5cc159dd9ce5309f97408c99。
- 结束代码基线：f989dc1cd72076f7d72672b8e037f75ff224d450。

## 2. 遗留工作区修改

- 开始时 git status：`M AGENTS.md`、`M src/uav_multi_relay/kinematics.py`、`M src/uav_multi_relay/safety.py`。
- 涉及源文件：`kinematics.py`、`safety.py`。
- 修改来源是否可确认：不可确认；只能确认这些修改在本轮开始前已经存在。
- 实际 diff 摘要：新增 `_speed_limit_tolerance(limit) = 64 * eps * max(1, abs(limit))`；`make_velocity_feasible()` 与 `velocity_to_normalized_action()` 的水平、上升、下降检查均复用该函数。
- 是否包含无关修改：否；未发现调试输出、换行重排或速度投影、加速度、状态推进、安全插值、候选位置、距离约束、异常类型或动作映射公式变更。

## 3. 数值容差规则

- 旧容差规则：`1e-9 * max(1, limit)`。
- 新容差规则：`64.0 * np.finfo(float).eps * max(1.0, abs(limit))`。
- 30 m/s 对应的新容差：`4.263256414560601e-13` m/s。
- 原始复现值：`30.000000000000004` m/s。
- 是否仍能接受原始复现值：是；接受后由可行化路径规范化到精确上限内。
- 一 ULP 上升/下降超限：均接受并规范化；明显超限：两个入口均拒绝。

## 4. 测试结果

- 提交前相关测试：`python -m pytest -q tests/test_physics.py tests/test_environment.py`，93 passed in 36.14s。
- 提交前完整测试：166 passed in 63.91s。
- 提交后完整测试：166 passed in 64.82s。
- 编译检查：`python -m compileall -q src tests scripts` 成功（提交前及提交后）。
- 最终测试数量：166，不低于要求的 158。
- 警告：无新增 Pytest 警告；仅有 Git 的 CRLF 工作树提示。
- greedy seed 20004 回归结果：跨必要 reset 的 30 次环境调用保持速度边界有限且不触发速度边界 `ValueError`；该测试不再声称同一 episode 连续存活 30 步。

## 5. 可复现性检查

- 测试代码 Commit：f989dc1cd72076f7d72672b8e037f75ff224d450。
- 代码是否推送：是，已推送至 `origin/main`。
- 提交后源文件是否仍有 diff：否；`git diff --exit-code -- src/uav_multi_relay/kinematics.py src/uav_multi_relay/safety.py` 成功。
- 测试代码是否等于远程代码：是；提交后在 `f989dc1` 代码树执行完整测试，随后 `git push` 成功。
- 最终工作区状态：报告提交后复核。

## 6. 阶段判定

- 阶段 3G 是否通过：否；本轮未改变原性能判定。
- 本轮修复是否通过：是；本地实际测试代码已提交、已推送，并在提交后重新完整验证。
- 是否允许进入阶段 4：否。
- 下一建议任务：阶段 3G-R3——Critic 更新尺度与终止因果关系补充诊断。

## 7. Git 状态

- 代码 Commit：f989dc1cd72076f7d72672b8e037f75ff224d450（`fix: unify floating-point speed limit tolerances`）。
- 代码 push：成功，`b41c1a0..f989dc1` 推送至 `origin/main`。
- 报告 Commit：self。
- 报告 push：本报告所在提交已推送至 `origin/main`。
- Git 异常：首次 push 因 GitHub 凭据失效失败；用户恢复凭据后仅重试一次并成功，无 `git.exe` 内存读取错误。
- 未提交文件：报告提交后复核。
- 计划偏差：无；未训练 MASAC，未修改诊断算法、奖励、安全过滤、运动/通信/TDMA、Replay Buffer 或 MASAC 更新行为。
