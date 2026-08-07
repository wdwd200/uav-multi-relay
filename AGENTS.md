# Codex 执行计划：阶段 3G-R3——遗留数值容差改动归档与可复现性修复

## 1. 任务性质

当前任务属于：

```text
阶段 3G-R3 验收修复
```

任务性质：

```text
解决本地已测试代码与 GitHub 已推送代码不一致的问题
不新增正式阶段
不得进入阶段 3G-R4
不得进入阶段 4
```

本轮只处理以下事项：

1. 审查工作区中 `kinematics.py` 和 `safety.py` 的既有未提交修改；
2. 确认修改内容仅为统一速度容差规则；
3. 补充或确认针对新容差规则的回归测试；
4. 将经过测试的两个文件正式提交并推送；
5. 在提交后的干净代码树上重新运行完整测试；
6. 生成按本轮任务命名的结果文档；
7. Codex CLI 最终输出明确显示结果文档名称。

本轮不得重新训练 MASAC，不得运行 20,000 步实验，也不得修改诊断算法。

------

## 2. 当前已确认状态

当前工作区存在：

```text
M src/uav_multi_relay/kinematics.py
M src/uav_multi_relay/safety.py
```

这些修改在上一轮开始前已经存在，且参与了上一轮：

```text
158 passed
```

的测试，但没有包含在上一轮推送到 GitHub 的提交中。

因此目前存在：

```text
实际测试代码 != GitHub 远程代码
```

本轮必须消除这一可复现性缺口。

------

## 3. 预期的既有修改

### 3.1 `kinematics.py`

预期新增统一容差函数：

```python
def _speed_limit_tolerance(limit: float) -> float:
    return float(64.0 * np.finfo(float).eps * max(1.0, abs(limit)))
```

并将以下三个速度容差：

```text
horizontal tolerance
climb tolerance
descent tolerance
```

统一改为调用该函数。

### 3.2 `safety.py`

预期修改导入：

```python
from .kinematics import _speed_limit_tolerance, make_velocity_feasible
```

并在 `velocity_to_normalized_action()` 中复用同一容差函数。

### 3.3 本轮不得混入其他行为变化

两个文件中不得额外修改：

```text
速度投影算法
加速度限制
状态推进
安全过滤插值次数
候选位置计算
安全距离
链路距离
异常类型
动作映射公式
```

如果实际 diff 中存在上述范围之外的修改，立即停止，不得提交，并在 CLI 中报告实际差异。

------

## 4. 首先检查真实差异

Codex 开始后先运行：

```bash
git status --short
git diff -- src/uav_multi_relay/kinematics.py src/uav_multi_relay/safety.py
git diff --check
```

必须确认：

1. 只有预期的统一容差修改；
2. 没有调试代码；
3. 没有临时打印；
4. 没有无关格式化；
5. 没有大范围换行变化；
6. `git diff --check` 无空白错误。

将实际 diff 摘要记录到结果文档。

------

## 5. 容差规则验收要求

新规则为：

```python
64.0 * np.finfo(float).eps * max(1.0, abs(limit))
```

该规则只用于吸收浮点舍入误差。

必须满足：

1. `30.000000000000004 m/s` 在 30 m/s 上限下被接受；
2. 接受后速度规范化到不超过精确上限；
3. `np.nextafter(limit, np.inf)` 被接受；
4. 最大上升速度的一 ULP 超限被接受并裁剪；
5. 最大下降速度的一 ULP 超限被接受并裁剪；
6. 明显超过容差的速度仍抛出 `ValueError`；
7. `make_velocity_feasible()` 与 `velocity_to_normalized_action()` 使用相同容差定义。

不得使用更宽的固定容差替代该规则。

------

## 6. 测试要求

先检查现有测试是否已经完整覆盖以下情况。

至少必须覆盖：

### 6.1 水平边界

```text
精确上限
np.nextafter(max_horizontal_speed, +inf)
实际复现值 30.000000000000004
明显超过新容差
```

### 6.2 垂直边界

```text
最大上升速度的一 ULP 超限
最大下降速度的一 ULP 超限
明显超过上升容差
明显超过下降容差
```

### 6.3 两个模块的一致性

增加或确认存在一个测试，验证：

```text
make_velocity_feasible() 接受的容差内速度
velocity_to_normalized_action() 也能够接受
```

以及：

```text
超过容差的速度
两个入口均拒绝
```

### 6.4 回归场景

保留并运行：

```text
greedy
seed = 20004
30 次环境调用
```

该测试只允许表述为：

```text
跨必要 reset 的 30 次环境调用不再触发速度边界 ValueError
```

不得重新写成：

```text
同一 episode 连续成功运行 30 步
```

### 6.5 禁止事项

不得：

```text
删除已有测试
弱化明显超限测试
仅验证不抛异常
使用 mock 绕过速度函数
修改环境终止逻辑
```

------

## 7. 代码提交前验证

运行：

```bash
python -m pytest -q tests/test_physics.py tests/test_environment.py
python -m compileall -q src tests scripts
```

然后运行完整测试：

```bash
python -m pytest
```

要求：

```text
全部测试通过
测试数量不得低于 158
无新增 Pytest 警告
编译检查成功
```

记录真实测试数量和耗时。

如果测试失败：

```text
不得提交
不得修改算法绕过测试
仍停留在本任务
```

------

## 8. 提交既有代码修改

确认 diff 和测试通过后，执行：

```bash
git status --short
git add src/uav_multi_relay/kinematics.py \
        src/uav_multi_relay/safety.py
```

如果本轮新增或修改了相关测试，再添加真实测试文件，例如：

```bash
git add tests/test_physics.py tests/test_environment.py
```

提交前运行：

```bash
git diff --cached --check
git diff --cached --stat
```

提交：

```bash
git commit -m "fix: unify floating-point speed limit tolerances"
```

推送：

```bash
git push
```

记录真实代码 Commit SHA。

------

## 9. 提交后重新验证

代码提交并推送后，必须确认源文件不再有未提交差异：

```bash
git status --short
git diff --exit-code -- \
  src/uav_multi_relay/kinematics.py \
  src/uav_multi_relay/safety.py
```

随后在提交后的代码树上再次运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

目的：

```text
证明测试使用的代码与已提交、已推送代码完全一致
```

如果提交后两个源文件再次显示 `M`：

```text
立即停止
不得继续生成 completed 报告
必须查明是工具自动改写、换行变化还是其他修改
```

------

## 10. 本轮结果文档

本轮结果文档必须命名为：

```text
STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
```

将上一轮当前结果文档：

```text
STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md
```

改名为本轮结果文档：

```bash
git mv STAGE_3G_R3_ACCEPTANCE_REPAIR_REPORT.md \
        STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
```

仓库根目录最终只保留：

```text
STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
```

历史报告由 Git 历史保存。

------

## 11. 结果文档内容

结果文档至少包含：

```markdown
---
schema_version: 1
stage: 3G-R3
task_type: reproducibility_repair
status: completed
branch: main
code_commit: <真实代码 SHA>
code_push_status: pushed
report_commit: self
---

# Stage 3G-R3 Reproducibility Repair Report

## 1. 本轮任务
- 阶段：
- 任务：
- 是否新增正式阶段：
- 是否重新训练：
- 开始代码基线：
- 结束代码基线：

## 2. 遗留工作区修改
- 开始时 git status：
- 涉及文件：
- 修改来源是否可确认：
- 实际 diff 摘要：
- 是否包含无关修改：

## 3. 数值容差规则
- 旧容差规则：
- 新容差规则：
- 30 m/s 对应的新容差：
- 原始复现值：
- 是否仍能接受原始复现值：
- 明显超限是否仍被拒绝：

## 4. 测试结果
- 提交前相关测试：
- 提交前完整测试：
- 提交后完整测试：
- 编译检查：
- 最终测试数量：
- 警告：
- greedy seed 20004 回归结果：

## 5. 可复现性检查
- 测试代码 Commit：
- 代码是否推送：
- 提交后源文件是否仍有 diff：
- 测试代码是否等于远程代码：
- 最终工作区状态：

## 6. 阶段判定
- 阶段 3G 是否通过：
- 本轮修复是否通过：
- 是否允许进入阶段 4：
- 下一建议任务：

## 7. Git 状态
- 代码 Commit：
- 代码 push：
- 报告 Commit：self
- 报告 push：
- Git 异常：
- 未提交文件：
- 计划偏差：
```

报告不得声称能够确认这些遗留修改最初由谁产生。

只能写：

```text
这些修改在本轮开始前已经存在
```

除非 Git、Codex 日志或其他直接证据能够证明来源。

------

## 12. 报告提交

完成结果文档后运行：

```bash
git add STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
git status --short
```

如 README 或 AGENTS 中存在旧结果文档文件名引用，才允许同步修改并添加。

提交前确保没有加入：

```text
outputs/
checkpoint
JSON/JSONL 运行产物
.pytest_cache/
__pycache__/
临时日志
```

提交：

```bash
git commit -m "docs: record stage 3G-R3 reproducibility repair"
```

推送：

```bash
git push
git status --short
```

最终工作区必须干净。

------

## 13. 本轮验收标准

必须同时满足：

1. 两个遗留源文件的真实 diff 已核对；
2. diff 只包含统一速度容差修改；
3. 新容差仍接受原始浮点舍入值；
4. 明显超限仍被拒绝；
5. 两个模块使用同一容差规则；
6. 相关测试通过；
7. 完整测试通过；
8. 测试数量不低于 158；
9. 编译检查通过；
10. 源代码修改已提交；
11. 源代码修改已推送；
12. 提交后重新运行完整测试；
13. 提交后两个源文件不再显示 `M`；
14. 测试代码与 GitHub 远程代码一致；
15. 未重新训练；
16. 未修改 MASAC、奖励或安全过滤行为；
17. 根目录结果文档为：

```text
STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
```

1. Codex CLI 最终输出显示结果文档完整名称；
2. 最终 `git status --short` 干净。

任何一项不满足：

```text
仍停留在阶段 3G-R3 可复现性修复
不得开始补充诊断
不得进入 3G-R4
不得进入阶段 4
```

------

## 14. 下一任务

本轮验收通过后，下一建议任务应为：

```text
阶段 3G-R3——Critic 更新尺度与终止因果关系补充诊断
```

该任务只做补充诊断，用于决定后续 `3G-R4` 应优先修复：

```text
Critic 更新尺度
动作语义一致性
奖励与终止尺度
```

本轮不得提前实施上述算法修复。

------

## 15. Git 异常处理

如果任何 Git 命令触发 `git.exe` 内存读取错误：

1. 立即停止 Git 操作；
2. 不自动重试；
3. 不运行 `git reset --hard`；
4. 不运行 `git gc`；
5. 不运行 `git prune`；
6. 记录触发错误的完整命令；
7. 如实记录已创建的 Commit；
8. 如实记录已推送的 Commit；
9. 不虚构 SHA；
10. 不虚构远程状态。

------

## 16. Codex CLI 最终输出

任务结束后，必须在 CLI 最后打印：

```text
========================================
阶段 3G-R3 可复现性修复执行结果
========================================

本轮结果文档：
STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md

遗留修改检查：
<真实结论>

代码 Commit SHA：
<真实 SHA>

结果文档 Commit SHA：
<真实 SHA>

代码 push 结果：
<真实结果>

报告 push 结果：
<真实结果>

提交后完整测试：
<真实 passed 数量和耗时>

编译检查：
<真实结果>

测试代码与远程代码是否一致：
<是或否，并说明依据>

最终 git status --short：
<真实输出；干净时明确写 clean>

下一建议任务：
阶段 3G-R3——Critic 更新尺度与终止因果关系补充诊断

========================================
```

必须显示结果文档完整名称：

```text
STAGE_3G_R3_REPRODUCIBILITY_REPAIR_REPORT.md
```

------

## 17. 禁止事项

本轮禁止：

```text
重新训练 MASAC
修改 MASAC loss
修改 Actor 或 Critic
修改奖励公式或权重
修改安全过滤行为
修改速度或加速度物理限制
修改通信或 TDMA
扩大容差
删除回归测试
直接丢弃两个本地修改
提交 outputs/
在源文件仍显示 M 时报告 completed
宣称已确认遗留修改的原始作者
进入阶段 3G-R4
进入阶段 4
伪造测试、Commit 或 push 结果
```
