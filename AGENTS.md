# Codex 执行计划：阶段 3G-R2——固定场景重新训练与基线比较

## 1. 目标

在固定动态场景和固定奖励权重下重新训练 MASAC，并与规则及优化基线公平比较。

本任务仍属于阶段 3G 的验收修复，不新增总计划阶段。

不得根据训练结果临时改变场景、奖励权重或训练步数。

开始后使用本计划覆盖根目录 `AGENTS.md`。

------

## 2. 固定实验设置

统一使用：

```text
num_relays = 4
H waypoint radius = 90 m
L waypoint radius = 90 m
max_steps = 250
dt = 现有默认值
```

奖励权重固定为：

```text
rate = 1.0
link = 1.0
separation = 1.0
intervention = 0.1
motion = 0.1
failure = 1.0
```

不得修改其他物理参数、奖励公式或安全过滤器。

------

## 3. 文件范围

允许修改：

```text
scripts/run_experiment.py
scripts/compare_baselines.py
src/uav_multi_relay/analysis/comparison.py
tests/test_comparison.py
README.md
AGENTS.md
aaa.md
```

仅在存在公共场景配置重复时，可以新增一个小型纯函数到：

```text
src/uav_multi_relay/config.py
```

不得修改：

```text
MASAC 更新公式
环境状态转移
通信模型
安全过滤器
MPC 算法
Replay Buffer
Checkpoint 格式
```

不得增加新依赖。

------

## 4. 脚本参数

### 4.1 run_experiment.py

增加：

```text
--waypoint-radius
--reward-rate
--reward-link
--reward-separation
--reward-intervention
--reward-motion
--reward-failure
```

这些参数必须同时应用于训练环境和周期评估环境。

`run_config.json` 必须保存解析后的：

- 航点半径；
- `max_steps`；
- 全部奖励权重。

### 4.2 compare_baselines.py

增加相同的场景和奖励参数。

比较环境必须与训练环境使用相同：

- 航点半径；
- Episode 长度；
- 奖励权重；
- 中继数量。

不得只根据 Checkpoint 中继数量创建默认环境。

------

## 5. 完善比较策略

在现有策略列表中增加：

```text
weighted_spacing
```

调用已有的加权等距基线，不得重新实现该算法。

最终支持：

```text
masac
random
stationary
equal_spacing
weighted_spacing
greedy
mpc
```

所有策略在第 `i` 个 episode 使用相同：

```python
episode_seed = comparison_seed + i
```

MASAC 必须使用确定性动作。

比较过程不得修改 Agent 或传入环境。

------

## 6. 测试

补充测试，至少验证：

1. 训练脚本的场景参数同时应用于训练和评估环境；
2. 比较脚本使用与训练相同的奖励权重；
3. `run_config.json` 保存完整场景和奖励配置；
4. `weighted_spacing` 可以单独或与其他策略一起比较；
5. 所有策略使用相同 episode seed；
6. 默认参数保持现有行为；
7. 非法航点半径和非法奖励权重被拒绝；
8. 输出 JSON 不含 NaN；
9. 短场景真实训练和比较可以完成。

测试不得执行正式 20,000 步训练。

------

## 7. 正式执行顺序

### 7.1 代码验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

### 7.2 小型冒烟实验

运行：

```bash
python scripts/run_experiment.py \
  --output-dir outputs/stage3g_r2_smoke \
  --steps 100 \
  --max-steps 20 \
  --waypoint-radius 90 \
  --batch-size 8 \
  --random-action-steps 8 \
  --update-after-steps 8 \
  --log-interval 20 \
  --evaluation-interval 50 \
  --evaluation-episodes 1 \
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

确认成功后删除：

```text
outputs/stage3g_r2_smoke/
```

### 7.3 开发级正式训练

只运行一个固定训练：

```bash
python scripts/run_experiment.py \
  --output-dir outputs/stage3g_r2_seed0 \
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

不得因中间结果不理想而提前停止、重启或改变参数。

### 7.4 公平基线比较

使用最佳 Checkpoint：

```bash
python scripts/compare_baselines.py \
  --checkpoint outputs/stage3g_r2_seed0/best_checkpoint.pt \
  --output-dir outputs/stage3g_r2_seed0/comparison \
  --episodes 10 \
  --seed 20000 \
  --max-steps 250 \
  --waypoint-radius 90 \
  --reward-rate 1.0 \
  --reward-link 1.0 \
  --reward-separation 1.0 \
  --reward-intervention 0.1 \
  --reward-motion 0.1 \
  --reward-failure 1.0 \
  --policies masac random stationary equal_spacing weighted_spacing greedy mpc \
  --greedy-sweeps 1 \
  --mpc-horizon 2 \
  --mpc-population-size 8 \
  --mpc-iterations 2 \
  --device cpu
```

所有策略必须完成相同的 10 个 episode。

不得筛选 episode 或删除不利结果。

------

## 8. 判定规则

阶段 3G 只有同时满足以下条件才通过：

```text
MASAC mean return > stationary mean return
MASAC mean return > random mean return
MASAC mean rate_e2e_bps >= stationary mean rate_e2e_bps
MASAC termination rate <= stationary termination rate
```

同时报告相对提升：

```python
100 * (masac_value - baseline_value) / abs(baseline_value)
```

不得只报告绝对值。

安全干预率必须如实记录，不设置人为通过阈值。

如果 MASAC 未通过：

- 不自动延长训练；
- 不自动更换随机种子；
- 不再次修改奖励权重；
- 下一任务进入 `3G-R3——训练稳定性和奖励贡献诊断`。

如果通过：

- 阶段 3正式关闭；
- 下一正式阶段进入阶段 4A。

------

## 9. README、Git 与结果记录

README 增加：

- 动态场景与奖励权重参数示例；
- 加权等距策略比较说明；
- 当前结果仅为单随机种子开发验证，不是论文正式结论。

代码提交：

```bash
git add AGENTS.md README.md \
  scripts/run_experiment.py scripts/compare_baselines.py \
  src/uav_multi_relay/analysis/comparison.py \
  tests/test_comparison.py
git commit -m "stage-3: configure dynamic MASAC baseline validation"
git push
```

随后使用实际结果覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3G-R2
- 类型：阶段 3G 验收修复，不新增总计划阶段
- 任务：固定场景重新训练与基线比较
- 完成状态：
- 固定场景：
- 奖励权重：
- 训练步数：
- 最佳评估步数：
- 完整测试结果：
- 编译验证：
- 比较 episode 数：
- MASAC 平均 return：
- Random 平均 return：
- Stationary 平均 return：
- Equal spacing 平均 return：
- Weighted spacing 平均 return：
- Greedy 平均 return：
- MPC 平均 return：
- 各策略平均端到端速率：
- 各策略终止率：
- 各策略安全干预率：
- 各策略平均动作计算时间：
- MASAC 相对 Stationary 的 return 提升：
- MASAC 相对 Stationary 的速率提升：
- 阶段 3G 是否通过：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议任务：
```

下一建议任务必须根据真实结果填写：

```text
通过：阶段 4A
未通过：阶段 3G-R3——训练稳定性和奖励贡献诊断
```

提交记录：

```bash
git add aaa.md
git commit -m "docs: record dynamic MASAC validation result"
git push
git status --short
```

若 Git 发生内存读取错误，立即停止，不自动重试，不运行破坏性 Git 命令，并如实记录提交和推送状态。

`outputs/` 不得提交 Git，最终工作区必须干净。
