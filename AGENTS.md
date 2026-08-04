- # Codex 任务：阶段 2C——规则与单步优化基线

  ## 目标

  补齐静止、加权间距和单步贪心基线。

  不要实现强化学习，不要实现 MPC，不要新增测试文件。

  ## 允许修改

  - `src/uav_multi_relay/baselines.py`
  - `tests/test_environment.py`
  - `README.md`
  - `AGENTS.md`
  - `aaa.md`

  ## 开始前

  先执行：

  ```bash
  git status
  git push
  ```

  确认当前代码已同步到 `origin/main`。不要修改或重写已有提交。

  ## 实现

  在 `baselines.py` 中保留现有 `equal_spacing_actions()`，新增：

  ```python
  stationary_actions(env) -> np.ndarray
  weighted_spacing_actions(env, hop_weights=None) -> np.ndarray
  greedy_one_step_actions(env, sweeps=2) -> np.ndarray
  ```

  ### stationary_actions

  返回形状 `(K, 3)` 的全零动作。

  ### weighted_spacing_actions

  - `hop_weights` 长度必须为 `K + 1`；
  - 权重必须有限且严格为正；
  - 默认使用全 1 权重，此时结果应与 `equal_spacing_actions()` 一致；
  - 按权重累计比例确定各中继在当前 H–L 线段上的目标位置；
  - 输出必须有限并位于 `[-1, 1]`。

  ### greedy_one_step_actions

  实现确定性的逐坐标网格搜索：

  - 不得修改传入环境；
  - 使用 `copy.deepcopy(env)` 评估候选动作；
  - 初始候选比较静止动作和等距动作；
  - 每个中继、每个动作维度依次测试：
    ```text
    -1.0, -0.5, 0.0, 0.5, 1.0
    ```
  - 默认执行两轮搜索；
  - 目标是最大化下一步 `info["rate_e2e_bps"]`；
  - 会导致 `terminated=True` 的候选不得选用；
  - 相同结果时保持先出现的候选，保证可复现；
  - 返回动作必须有限并位于 `[-1, 1]`。

  不要调用或复制环境的私有通信公式来估算速率，必须通过克隆环境的 `step()` 评估。

  ## 测试

  只修改 `tests/test_environment.py`，增加：

  1. 三种新基线均返回合法 `(K, 3)` 动作；
  2. 全 1 权重与等距动作一致；
  3. 非法权重抛出 `ValueError`；
  4. 贪心函数不会改变原环境状态；
  5. 贪心动作的下一步速率不低于静止和等距动作中的较优者；
  6. 三种基线各运行 50 步，无 NaN 或未处理异常。

  ## README

  列出当前已有基线：

  - stationary
  - equal spacing
  - weighted spacing
  - greedy one-step coordinate search

  明确说明 MPC 尚未实现。

  ## 验证

  ```bash
  python -m pip install -e ".[dev]"
  python scripts/check_install.py
  python -m pytest
  ```

  所有测试必须通过。

  ## Git

  先提交代码：

  ```bash
  git add .
  git commit -m "stage-2: add rule and greedy baselines"
  git push
  ```

  然后覆盖 `aaa.md`：

  ```markdown
  # 本次执行结果

  - 阶段：2C
  - 任务：规则与单步优化基线
  - 完成状态：
  - 修改文件：
  - 测试结果：
  - 代码 Commit ID：
  - 当前分支：
  - GitHub 推送结果：
  - 计划偏差：
  - 遗留问题：
  ```

  填写真实结果后提交：

  ```bash
  git add aaa.md
  git commit -m "docs: record stage-2 baseline result"
  git push
  git status
  ```

  工作区必须干净。

  ## 禁止事项

  不要实现：

  - MPC
  - replay buffer
  - MASAC、SAC、MAPPO
  - 神经网络
  - 训练脚本
  - 新测试文件

  完成后只回复：

  - 测试结果
  - 两个 Commit ID
  - 推送结果
  - 遗留问题
