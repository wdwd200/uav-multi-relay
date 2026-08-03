

------

# Codex 任务：阶段 1——运动与通信物理核心

## 目标

在现有最小项目中实现第一批可独立验证的物理核心：

```text
UAV 状态
三维运动约束
执行前速度可行化
多跳几何
空对空信道
单跳容量
等时间与最优 TDMA
```

本次只实现纯数据和纯计算函数，不实现环境、奖励和强化学习。

------

## 一、创建的源码结构

```text
src/uav_multi_relay/
├── __init__.py
├── core.py
├── kinematics.py
└── communication.py
```

测试保持精简：

```text
tests/
├── test_smoke.py
└── test_physics.py
```

只新增一个测试文件 `test_physics.py`。

------

## 二、修改 `pyproject.toml`

在正式依赖中加入：

```toml
dependencies = [
    "numpy>=1.24",
]
```

保留现有 pytest 开发依赖。

不要加入：

```text
torch
gymnasium
scipy
pandas
matplotlib
```

------

## 三、实现 `core.py`

定义不可变数据类：

```python
UAVState
```

字段：

```python
name: str
position_m: np.ndarray
velocity_mps: np.ndarray
```

要求：

- 位置和速度都必须是形状 `(3,)` 的有限浮点数组；
- 非法形状或 NaN 应抛出 `ValueError`；
- 保存时复制数组，避免外部修改状态；
- 提供 `moved(position_m, velocity_mps)`，返回新状态，不修改原状态。

再定义：

```python
MotionLimits
```

字段：

```python
max_horizontal_speed_mps
max_climb_speed_mps
max_descent_speed_mps
max_horizontal_accel_mps2
max_vertical_accel_mps2
```

所有限制必须为正数。

------

## 四、实现 `kinematics.py`

### 1. 请求速度可行化

实现：

```python
make_velocity_feasible(
    requested_velocity_mps,
    current_velocity_mps,
    limits,
    delta_t_s,
) -> np.ndarray
```

按以下顺序处理：

1. 水平加速度；
2. 垂直加速度；
3. 水平合速度；
4. 上升速度；
5. 下降速度。

水平速度约束：

$$
\sqrt{v_x^2+v_y^2}
\leq V_{xy,\max}
$$

垂直速度约束：

$$
-V_{\mathrm{down},\max}
\leq v_z
\leq V_{\mathrm{up},\max}
$$

水平加速度约束：

## $$ \left| \mathbf v_{xy}^{\mathrm{new}}

\mathbf v_{xy}^{\mathrm{current}}
\right|
\leq
A_{xy,\max}\Delta t
$$

垂直加速度约束：

## $$ \left| v_z^{\mathrm{new}}

v_z^{\mathrm{current}}
\right|
\leq
A_{z,\max}\Delta t
$$

函数返回的是**实际可执行速度**。

不得先更新位置再裁剪位置。

### 2. 状态推进

实现：

```python
advance_state(
    state,
    applied_velocity_mps,
    delta_t_s,
) -> UAVState
```

位置更新：

# $$ \mathbf q[n+1]

\mathbf q[n]
+
\mathbf v^{\mathrm{applied}}[n]\Delta t
$$

要求：

- 不修改输入状态；
- `delta_t_s` 必须大于 0；
- 新状态保存实际执行速度。

------

## 五、实现 `communication.py`

本文件集中实现第一版通信纯函数。

### 1. 多跳节点序列

实现：

```python
ordered_nodes(
    high: UAVState,
    relays: tuple[UAVState, ...],
    low: UAVState,
) -> tuple[UAVState, ...]
```

返回：

```text
H, R1, R2, ..., RK, L
```

至少要求一个中继，且节点名称不得重复。

### 2. 链路几何

定义：

```python
LinkGeometry
```

字段：

```python
distance_3d_m
horizontal_distance_m
elevation_angle_rad
```

实现：

```python
compute_link_geometry(tx_position_m, rx_position_m) -> LinkGeometry
```

仰角：

# $$ \theta

\arctan
\left(
\frac{|\Delta z|}
{\sqrt{\Delta x^2+\Delta y^2}}
\right)
$$

水平距离为 0 时必须稳定返回，不得产生 NaN。

再实现：

```python
compute_chain_geometries(nodes) -> tuple[LinkGeometry, ...]
```

若有 $K$ 个中继，应生成：

$$
K+1
$$

条有序链路。

### 3. 天线增益

实现竖直短偶极子简化模型：

```python
dipole_gain(
    elevation_angle_rad,
    max_gain_linear,
    min_gain_linear,
) -> float
```

公式：

# $$ G(\theta)

\max
\left{
G_{\min},
G_{\max}\cos^2\theta
\right}
$$

参数均使用线性值，不在核心计算中使用 dB。

### 4. 信道增益

实现：

```python
channel_power_gain(
    distance_m,
    reference_gain_linear,
    reference_distance_m,
    path_loss_exponent,
    tx_gain_linear,
    rx_gain_linear,
    minimum_distance_m,
) -> float
```

公式：

# $$ h

\beta_0
\left(
\frac{d_{\mathrm{eff}}}{d_0}
\right)^{-\alpha}
G_tG_r
$$

其中：

$$
d_{\mathrm{eff}}=\max(d,d_{\min})
$$

必须明确：

```text
reference_gain_linear 不包含方向性天线增益
```

### 5. SNR 与容量

实现：

```python
snr_linear(
    transmit_power_w,
    channel_gain_linear,
    noise_psd_w_per_hz,
    bandwidth_hz,
    noise_figure_linear,
) -> float
```

# $$ \mathrm{SNR}

\frac{Ph}{N_0BF}
$$

实现：

```python
shannon_capacity_bps(
    bandwidth_hz,
    snr_value_linear,
) -> float
```

$$
C=B\log_2(1+\mathrm{SNR})
$$

### 6. TDMA

实现：

```python
equal_tdma_rate(capacities_bps)
```

返回：

```python
(rate_bps, fractions)
```

# $$ R_{\mathrm{eq}}

\frac{1}{M}\min_i C_i
$$

其中：

$$
M=K+1
$$

实现：

```python
optimal_tdma_rate(capacities_bps)
```

返回：

```python
(rate_bps, fractions)
```

# $$ R^*

\left(
\sum_i\frac1{C_i}
\right)^{-1}
$$

# $$ \tau_i^*

\frac{1/C_i}
{\sum_j1/C_j}
$$

若任意容量小于或等于 0：

- 端到端速率返回 `0.0`；
- 时间比例返回长度正确且有限的数组；
- 不得出现 NaN 或无穷值。

------

## 六、测试要求

所有新测试放入：

```text
tests/test_physics.py
```

至少测试以下内容：

### 运动

1. 水平请求速度过大时，最终水平范数不超过上限；
2. 上升和下降速度分别受到限制；
3. 一步速度变化不超过加速度限制；
4. `advance_state()` 使用实际速度更新位置；
5. 输入状态没有被原地修改。

### 通信

1. 四个中继生成六个节点和五条链路；
2. 垂直链路的水平距离为 0，仰角为 $\pi/2$；
3. 偶极子增益不会低于 `min_gain_linear`；
4. 信道增益中的 $\beta_0$ 与天线增益只相乘一次；
5. SNR 和容量为有限非负数；
6. 容量为：

```python
[10.0, 20.0, 30.0, 40.0, 50.0]
```

时，等时间 TDMA 速率为：

```python
2.0
```

1. 最优 TDMA 比例之和为 1；
2. 某一跳容量为 0 时，端到端速率为 0。

不要创建更多测试文件。

------

## 七、更新公开接口

在 `src/uav_multi_relay/__init__.py` 中继续保留：

```python
__version__ = "0.1.0"
```

只导出本阶段最常用的类型：

```python
UAVState
MotionLimits
```

不要把全部函数都堆进顶层包。

------

## 八、更新 README

只增加一个“当前已实现”章节，写明：

```text
UAV 位置和速度状态
分方向速度与加速度限制
执行前速度可行化
多跳几何与空对空信道纯函数
单跳容量
等时间与解析最优 TDMA
```

同时明确尚未实现：

```text
完整环境
安全距离过滤
奖励函数
多智能体强化学习
```

------

## 九、AGENTS.md 与 aaa.md

`AGENTS.md` 是当前 Codex 执行指令文件，必须保留。

执行期间：

- 严格按照本文件工作；
- 不删除 `AGENTS.md`；
- 不把它当成项目功能文档；
- 下一次任务会覆盖其中内容。

完成后覆盖写入 `aaa.md`：

```markdown
# 本次执行结果

- 阶段：1
- 任务：运动与通信物理核心
- 完成状态：
- 修改和新增文件：
- 测试命令：
- 测试结果：
- Commit ID：
- 当前分支：
- GitHub 推送结果：
- 计划偏差：
- 遗留问题：
```

------

## 十、验证与提交

执行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
pytest
```

全部通过后提交：

```bash
git add .
git commit -m "stage-1: add motion and communication core"
git push
```

然后将真实 Commit ID 和推送结果覆盖写入 `aaa.md`，再提交：

```bash
git add aaa.md
git commit -m "docs: record stage-1 result"
git push
```

最终执行：

```bash
git status
```

工作区必须干净。

------

## 十一、禁止事项

本阶段不要实现：

- Gymnasium 环境；
- H/L 航点轨迹；
- 多 UAV 碰撞过滤；
- 奖励函数；
- replay buffer；
- SAC 或 MASAC；
- 训练脚本；
- 绘图代码；
- 大量测试文件；
- 旧项目源码复制。

完成后只回复：

```text
测试结果
两个 Commit ID
推送结果
遗留问题
```

------

当前进度：

```text
项目初始化：已完成
阶段 1：等待 Codex 执行
总体进度：0 / 5
```