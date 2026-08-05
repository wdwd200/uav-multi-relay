# Codex 修复计划：完成阶段 2D 交付

## 1. 目标

不要进入阶段 3D。

本次只完成阶段 2D 的测试、代码小修、文档和 Git 交付。

## 2. 修改范围

允许修改：

```text
src/uav_multi_relay/policies/mpc.py
tests/test_mpc.py
README.md
AGENTS.md
aaa.md
```

不修改环境、物理模型、奖励、MASAC 和其他基线。

## 3. 代码修复

1. `evaluate_action_sequence()` 转换非法 `action_sequence` 时，将 `TypeError` 和 `ValueError` 统一转换为带说明的 `ValueError`。
2. `plan_mpc()` 在所有 CEM 迭代中维护全局最佳序列和评估结果，不只保留最后一轮最佳结果。
3. 三个确定性锚点每轮仍必须保留。
4. 不增加依赖，不改变 MPC 默认配置。

## 4. 补充测试

在 `tests/test_mpc.py` 中补充：

- 环境位置、速度、步数、上一步实际速度和 H/L 航点进度均不被预测修改；
- `max_steps=1` 时预测立即停止并返回 `truncated=True`；
- 最佳结果不低于零动作和动态等距动作锚点；
- `predicted_return` 等于评估结果中的折扣回报；
- 返回的 `first_action` 与 `action_sequence` 相互独立；
- 错误中继数量、错误最后一维、NaN、无穷值、越界动作和非环境对象均被拒绝；
- 使用 `dataclasses.replace()` 测试 `K=1` 和 `K=4`；
- 使用小型配置连续运行 10 个环境步；
- 配置验证覆盖 NaN、无穷值及最小标准差大于初始标准差。

## 5. README

删除相互矛盾的：

```text
Multi-agent reinforcement learning remains intentionally unimplemented.
```

改为准确说明：

```text
完整训练、评估、checkpoint 和实验流程尚未实现；
MASAC 网络、Replay Buffer 和单批次更新核心已经实现。
```

保留 MPC 使用示例。

## 6. 验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

不得删除或跳过现有测试。

## 7. Git 异常处理

执行任何 Git 命令前运行：

```bash
git status --short
git branch --show-current
git log -1 --oneline
```

任何 Git 命令若弹出 `git.exe` 内存读取错误或进程崩溃：

- 立即停止；
- 不自动重试；
- 不运行 `git reset --hard`、`git gc` 或 `git prune`；
- 记录发生错误的完整命令；
- 在 `aaa.md` 中记录代码是否已提交、是否已推送；
- 不虚构 Commit ID 或推送成功状态。

## 8. 提交与 aaa.md

代码和测试完成后提交：

```bash
git add AGENTS.md README.md src/uav_multi_relay/policies/mpc.py tests/test_mpc.py
git commit -m "fix: complete MPC baseline validation"
git push
```

随后覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：2D（补全）
- 任务：有限时域 MPC 基线
- 完成状态：
- 修改文件：
- MPC 方法：
- 测试结果：
- 编译验证：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3D——MASAC 环境采集与训练循环
```

再提交并推送：

```bash
git add aaa.md
git commit -m "docs: record completed MPC baseline"
git push
git status --short
```

最终工作区必须干净；缓存文件不得提交。
