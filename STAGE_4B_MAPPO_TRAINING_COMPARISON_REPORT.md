---
schema_version: 1
stage: 4B
task_type: fixed_training_comparison
status: completed
branch: main
code_commit: 87bd32051803ab85a67d14be9c1d57d1fa1ffbc8
code_push_status: pushed
report_commit: self
---

# Stage 4B MAPPO Training and Fair Comparison Report

## 1. 本轮任务

- 阶段：4B——固定配置 MAPPO 训练与统一公平比较。
- 基线：`main`，包含 per-relay MAPPO 语义修复 `9b19c8d`；MASAC checkpoint `outputs/stage3g_r4_final_seed0/best_checkpoint.pt` 存在且未重新训练。
- 不进行了超参数搜索、事后调参、失败 episode 筛选、环境/奖励/安全过滤修改或算法修改。

## 2. 统一比较支持

比较器和 `scripts/compare_baselines.py` 扩展为按需加载 `--mappo-checkpoint` 与 `--masac-checkpoint`，并保持 legacy `--checkpoint` 的纯 MASAC 用法。

- 支持策略：mappo、masac、random、stationary、equal_spacing、weighted_spacing、greedy、mpc。
- 仅请求 MAPPO/MASAC 时才加载对应 checkpoint；缺失时给出明确错误。
- 两个学习算法均调用 deterministic Actor action。
- 统一输出新增 mean return per step 与 requested/applied mismatch rate；模型参数在比较中不变。
- 新增测试覆盖两学习算法同次运行、同 seeds、按需 checkpoint、旧入口兼容、统一指标和无参数修改。

## 3. 固定训练配置

| 项目 | 值 |
| --- | --- |
| num_relays / radius / max_steps | 4 / 90 m / 250 |
| environment steps / rollout steps | 20,000 / 1,000 |
| update epochs / mini-batch | 10 / 250 |
| gamma / GAE lambda / clip ratio | 0.99 / 0.95 / 0.2 |
| actor/critic LR | 3e-4 / 3e-4 |
| value coeff / entropy coeff / max gradient norm | 0.5 / 0.01 / 0.5 |
| seed / evaluation seed | 0 / 10,000 |
| evaluation/checkpoint interval | 2,500 / 2,500 |
| evaluation episodes / device | 5 / CPU |
| reward weights rate/link/separation/intervention/motion/failure | 1/1/1/0.1/0.1/1 |

训练从零开始于 `outputs/stage4b_mappo_seed0`，完成 20,000 environment steps、20 次完整 rollout 更新、169 个完成 episode，`discarded_partial_rollout_steps = 0`。周期 checkpoint 完整：000000、002500、005000、007500、010000、012500、015000、017500、020000，并存在 best/final checkpoint。

## 4. MAPPO 训练轨迹

所有训练日志有限，未出现 NaN 或 Infinity。训练检查点的确定性评估（mean return / mean rate / termination rate / episode length / intervention-mismatch rate）为：

| step | return | rate (bps) | term. | length | intervention / mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,500 | 477.65 | 41,703,572 | 0.8 | 116.2 | 0.0209 |
| 5,000 (best) | 850.30 | 40,681,302 | 0.4 | 213.4 | 0.0101 |
| 7,500 | 700.01 | 40,757,617 | 0.8 | 175.2 | 0.0152 |
| 10,000 | 792.01 | 40,885,786 | 0.8 | 196.6 | 0.9661 |
| 12,500 | 565.61 | 41,785,627 | 1.0 | 138.2 | 0.9546 |
| 15,000 | 752.06 | 41,661,851 | 1.0 | 184.2 | 0.5538 |
| 17,500 | 679.30 | 41,833,269 | 1.0 | 166.8 | 0.1839 |
| 20,000 (final) | 733.61 | 41,230,250 | 1.0 | 182.0 | 0.9594 |

训练在 5,000 步达到最佳评估，随后 final 低于 best，存在退化。最终日志的 policy loss/value loss/entropy/approx KL/clip fraction 为 `-0.0630 / 1562.83 / 5.3981 / 0.0174 / 0.1576`，均有限。最大记录 clip fraction 为 0.1576，未出现持续 100% clip fraction；Actor 预裁剪梯度最大为 0.4083，未触及 0.5。Critic 记录的预裁剪梯度最大为 2852.16，因而会触发既有 0.5 全局裁剪；本轮未改变梯度裁剪实现。requested/applied mismatch 后期显著增加，最终评估为 0.9594。

## 5. 八策略公平比较

使用 MAPPO best checkpoint（5,000 步）和固定 MASAC best checkpoint，episodes=10、seeds `20000–20009`、同一 4 relay/90 m/250-step/奖励权重/CPU 场景；未筛选终止 episode。MAPPO 的 PPO ratio 使用 requested action，MASAC 的 Replay Buffer 训练语义为 applied action；这是保留的算法实现差异，未为比较修改任一算法。

| 策略 | return | return/step | rate (bps) | min rate (bps) | term. | length | intervention / mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAPPO | 921.29 | 4.1433 | 41,823,224 | 28,841,106 | 0.4 | 222.6 | 0.0088 |
| MASAC | 341.27 | 3.9551 | 41,057,618 | 35,284,034 | 1.0 | 85.8 | 1.0000 |
| Random | 567.94 | 4.0392 | 42,093,415 | 37,916,458 | 0.8 | 139.6 | 1.0000 |
| Stationary | 1068.00 | 4.2720 | 42,955,245 | 40,644,386 | 0.0 | 250.0 | 0.0000 |
| equal_spacing | 1073.70 | 4.2948 | 43,751,985 | 40,859,436 | 0.0 | 250.0 | 0.9952 |
| weighted_spacing | 1073.70 | 4.2948 | 43,751,985 | 40,859,436 | 0.0 | 250.0 | 0.9952 |
| greedy | 57.67 | 3.8702 | 43,774,039 | 42,763,384 | 1.0 | 14.9 | 1.0000 |
| MPC | 1068.00 | 4.2720 | 42,955,245 | 40,644,386 | 0.0 | 250.0 | 0.0000 |

每个策略的 mean action computation time 也已写入机器可读比较 JSON；MAPPO 为 0.000697 s，MASAC 为 0.000738 s，MPC 为 0.075250 s。

## 6. 结果分类

分类为 **A**：MAPPO 相比 MASAC 明显提高 return（921.29 vs 341.27）且显著降低终止率（0.4 vs 1.0），支持当前 on-policy requested-action 语义更适合 MAPPO 的证据。

这不表示 MAPPO 已优于规则基线：其 return、return-per-step、mean rate 和终止率仍落后于 Stationary/equal_spacing/MPC。该差距按固定配置如实记录，不在 4B 内调参或新增 4B-R1。

## 7. 验证、Git 和下一步

- 完整测试：`194 passed in 66.80s`，无新增 Pytest 警告；编译 `python -m compileall -q src tests scripts` 成功。
- 代码提交：`87bd32051803ab85a67d14be9c1d57d1fa1ffbc8`（`feat: compare MAPPO with MASAC and baselines`），已推送至 `origin/main`。
- 报告提交：self；包含本报告的提交已推送至 `origin/main`。
- 输出产物均在 `outputs/`，未加入 Git。
- 下一任务：阶段 4C——参数共享 MATD3 实现与短程验证。
