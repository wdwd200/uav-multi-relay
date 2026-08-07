---
schema_version: 1
stage: 3G-R4
task_type: final_validation
status: completed_performance_failed
branch: main
code_commit: 71eb0140537b9eb97b8731827801a707f984c28b
code_push_status: pushed
report_commit: self
---

# Stage 3G-R4 Final Validation Report

## 1. 本轮任务与基线

- 阶段：3G-R4，阶段 3 的最终受控修复与验收；未新增正式阶段。
- 开始基线：`main`，包含 `f989dc1`；唯一预期的未提交文件为用户授权的 `AGENTS.md` 执行计划。
- 结束基线：代码提交 `71eb0140537b9eb97b8731827801a707f984c28b`。
- 允许的唯一算法改动：为 Critic 增加可选全局梯度裁剪。未修改奖励、Replay Buffer 的 applied action 语义、安全过滤、运动/通信模型、基线策略或训练配置。

## 2. 实现

- `ParameterSharingMASAC(critic_gradient_clip_norm=None)` 保持既有更新路径；非 `None` 必须为有限正数。
- Critic 的顺序为 `zero_grad → backward → pre-norm → clip_grad_norm_ → post-norm → optimizer.step`。Actor、alpha、target 更新顺序和所有 loss 未变。
- 每次更新记录 `critic_gradient_norm_pre_clip`、`critic_gradient_norm_post_clip`、`critic_gradient_clip_applied`；训练日志按日志区间记录 `critic_gradient_clip_rate`。
- checkpoint 保存该参数；旧 checkpoint 缺少该键时以 `None` 加载。

## 3. 验证

- 完整测试：`python -m pytest`，`174 passed in 58.41s`，无 Pytest 警告。
- 编译：`python -m compileall -q src tests scripts` 成功。
- 相关测试：53 passed，覆盖未裁剪等价更新、裁剪前/后范数、未触发裁剪、非法参数、checkpoint 新旧兼容和日志区间裁剪率。
- 100 步 smoke：完成；所有日志指标有限，best/final checkpoint 均存在。最后一次 pre/post 为 `2.6916/2.6916`，区间裁剪率为 `0.0`。

## 4. 5,000 步受控筛选

两组均从零开始，`num_relays=4`、radius `90 m`、`max_steps=250`、batch `256`、random/update-after `2000`、seed `0`、evaluation seed `10000`、每步一更新、相同奖励权重。对照不裁剪，候选使用 `critic_gradient_clip_norm=1000`。

| 门槛 | 对照 | clip=1000 | 结果 |
| --- | ---: | ---: | --- |
| 所有训练数值有限 | 是 | 是 | 通过 |
| 最终评估 mean return | 113.9862 | 106.3830（93.3%） | 通过（≥90%） |
| 最终评估终止率 | 1.0 | 1.0 | 通过（不高于对照） |
| 最大裁剪后梯度范数 | 457.0438 | 500.3665 | 通过（≤1000） |
| 最后区间裁剪率 | 0.0 | 0.021 | 已观测 |
| Critic loss / TD error | 126.5726 / 2.4580 | 128.3376 / 2.4686 | 无明显恶化 |

两组 final checkpoint 均可加载；在同一 5 episode、seed `10000` 确定性评估中，两次结果逐字段相同。筛选因此允许完整训练。

## 5. 20,000 步最终训练

- 目录：`outputs/stage3g_r4_final_seed0`；未从任何 checkpoint 续训。
- 固定配置：`steps=20000`，`max_steps=250`，radius `90`，batch `256`，random/update-after `2000`，updates-per-step `1`，log `1000`，evaluation/checkpoint `2500`，5 evaluation episodes，seed `0`，evaluation seed `10000`，CPU，六项 reward weights 为 `1,1,1,0.1,0.1,1`，clip norm `1000`。
- 完成：20,000 环境步、18,001 次更新、317 个完成 episode；所有日志数值有限。
- 周期 checkpoint 完整：`step_000000`、`002500`、`005000`、`007500`、`010000`、`012500`、`015000`、`017500`、`020000`，并保存 best/final checkpoint。
- 训练过程最大 pre-clip 范数 `23241.1113`，最大 post-clip 范数 `999.9977`。裁剪率由早期 0 增至第 7,000 步区间 `0.751`，从 9,000 步起为 `1.0`。
- 最终日志：critic loss `2122.7568`，TD error mean `12.1905`；均有限但随训练后期上升，裁剪未使性能恢复。

确定性评估轨迹（step: mean return / termination rate）：

- 2,500: `206.8078 / 1.0`
- 5,000: `106.3830 / 1.0`
- 7,500: `202.3880 / 1.0`
- 10,000: `151.6656 / 1.0`
- 12,500（best）: `236.5004 / 1.0`
- 15,000: `185.4050 / 1.0`
- 17,500: `187.3744 / 1.0`
- 20,000（final）: `220.0101 / 1.0`

## 6. 七策略最终比较

使用 best checkpoint，10 episodes，seeds `20000–20009`，不筛选失败 episode；策略为 MASAC、Random、Stationary、equal_spacing、weighted_spacing、greedy、MPC（sweeps=1，horizon=2，population=8，iterations=2）。

| 策略 | mean return | mean rate (bps) | episode length | termination rate |
| --- | ---: | ---: | ---: | ---: |
| MASAC | 341.2665 | 41,057,618.48 | 85.8 | 1.0 |
| Random | 567.9424 | 42,093,415.46 | 139.6 | 0.8 |
| Stationary | 1068.0028 | 42,955,245.36 | 250.0 | 0.0 |
| equal_spacing | 1073.7013 | 43,751,985.20 | 250.0 | 0.0 |
| weighted_spacing | 1073.7013 | 43,751,985.20 | 250.0 | 0.0 |
| greedy | 57.6689 | 43,774,039.10 | 14.9 | 1.0 |
| MPC | 1068.0028 | 42,955,245.36 | 250.0 | 0.0 |

## 7. 阶段 3 最终判定

阶段 3 的四项性能条件均不通过：

1. MASAC mean return `341.2665` 未超过 Stationary `1068.0028`，也未超过 Random `567.9424`。
2. MASAC mean rate `41.0576 Mbps` 低于 Stationary `42.9552 Mbps`。
3. MASAC termination rate `1.0` 高于 Stationary `0.0`。
4. 训练完整、checkpoint 完整、裁剪后梯度受限且无非有限值，但这只确认实现和数值约束有效，不能替代性能验收。

结论：**阶段 3 实现完成但性能验收失败。** 本轮没有进入阶段 4，且根据本轮计划不再建议或执行 3G-R5/R6。后续项目路线须由用户决定：接受当前失败基线，或授权一次明确范围的算法方向调整。

## 8. Git 状态

- 代码提交：`71eb0140537b9eb97b8731827801a707f984c28b`（`fix: stabilize MASAC critic gradients`），已推送。
- 报告提交：self；包含本报告的提交已推送至 `origin/main`。
- 运行产物位于 `outputs/`，未加入 Git。
- Git 异常：无 `git.exe` 内存读取错误。
