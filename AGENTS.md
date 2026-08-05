# Codex 修复计划：阶段 3F 周期调度修复

## 1. 目标

修复训练日志与周期评估的调度耦合。

必须保证：

- 训练日志在 `log_interval_steps` 的整数倍和最终步产生；
- 评估在 `evaluation_interval_steps` 的整数倍和最终步产生；
- 两个间隔可以任意设置，不要求存在整倍数关系；
- 同一步同时满足两个条件时，只执行一次回调，但分别完成日志和评估。

本次不进入阶段 3G。

开始后使用本计划覆盖 `AGENTS.md`。

## 2. 修改范围

允许修改：

```text
src/uav_multi_relay/training/experiment.py
tests/test_experiment.py
README.md
AGENTS.md
aaa.md
```

如确有必要，可小范围修改：

```text
src/uav_multi_relay/training/trainer.py
tests/test_training.py
```

不得修改环境、MASAC、Replay Buffer、Checkpoint 或评估指标公式。

## 3. 调度修复

推荐在 `run_masac_experiment()` 中使用：

```python
math.gcd(
    experiment_config.log_interval_steps,
    experiment_config.evaluation_interval_steps,
)
```

作为底层训练进度回调间隔。

回调内部必须分别判断：

```python
should_log = (
    environment_steps % log_interval_steps == 0
    or environment_steps == total_environment_steps
)

should_evaluate = (
    environment_steps % evaluation_interval_steps == 0
    or environment_steps == total_environment_steps
)
```

只有 `should_log` 为真时写入：

```text
training_metrics.jsonl
```

只有 `should_evaluate` 为真时执行评估并写入：

```text
evaluation_metrics.jsonl
```

最终步不得重复记录或重复评估。

不允许通过强制要求两个间隔整除来规避问题。

## 4. 测试

新增明确测试：

```text
total steps = 5
log interval = 3
evaluation interval = 2
```

必须断言：

```text
training log steps == [3, 5]
evaluation steps == [2, 4, 5]
```

再测试相反关系：

```text
total steps = 7
log interval = 2
evaluation interval = 3
```

必须断言：

```text
training log steps == [2, 4, 6, 7]
evaluation steps == [3, 6, 7]
```

同时保留并验证：

- 最终步不重复；
- 最佳 Checkpoint 元数据对应实际评估步；
- 最终 Checkpoint 元数据对应总训练步数；
- JSONL 步数严格递增；
- 所有 JSON 数值有限。

## 5. README

修正 `train.py` 的描述：

- 默认不写训练日志；
- 提供 `--checkpoint-out` 时可以保存一个最终 checkpoint；
- 完整日志、周期评估和最佳 checkpoint 由 `run_experiment.py` 负责。

## 6. 验证

运行：

```bash
python -m pytest
python -m compileall -q src tests scripts
```

额外运行非整除间隔冒烟实验：

```bash
python scripts/run_experiment.py \
  --output-dir interval_smoke \
  --steps 7 \
  --batch-size 4 \
  --random-action-steps 4 \
  --update-after-steps 4 \
  --log-interval 2 \
  --evaluation-interval 3 \
  --evaluation-episodes 1 \
  --seed 0 \
  --evaluation-seed 100 \
  --device cpu
```

检查：

```text
training_metrics.jsonl steps = [2, 4, 6, 7]
evaluation_metrics.jsonl steps = [3, 6, 7]
```

完成后删除：

```text
interval_smoke/
```

## 7. Git 与 aaa.md

提交代码并推送：

```bash
git add AGENTS.md README.md \
  src/uav_multi_relay/training/experiment.py \
  tests/test_experiment.py
git commit -m "fix: decouple experiment log and evaluation schedules"
git push
```

覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：3F（调度修复）
- 任务：解耦训练日志与周期评估间隔
- 完成状态：
- 修改文件：
- 调度实现：
- 非整除间隔测试：
- 完整测试结果：
- 编译验证：
- 冒烟实验结果：
- 代码 Commit ID：
- 当前分支：
- GitHub 推送结果：
- Git 异常：
- 计划偏差：
- 遗留问题：
- 下一建议阶段：3G——MASAC 正式训练与基础基线比较
```

随后提交：

```bash
git add aaa.md
git commit -m "docs: record experiment schedule fix"
git push
git status --short
```

若发生 `git.exe` 内存读取错误，立即停止 Git 操作并如实记录，不自动重试。

最终工作区必须干净。
