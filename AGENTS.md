对，上一份太长，Codex 容易出现三种问题：

- 抓不住当前唯一目标；
- 做了很多没要求的扩展；
- 在 Git、测试和 `aaa.md` 之间遗漏步骤。

以后每次计划控制为：

```text
一个目标
少量文件
明确禁止项
两三个测试命令
一个验收结果
```

当前只做仓库初始化。下面这份直接复制给 Codex。

------

# Codex 任务：初始化新项目仓库

仓库：

```text
https://github.com/wdwd200/uav-multi-relay
```

## 目标

初始化一个最小 Python 项目。
本次不实现运动、通信、环境或强化学习代码。

## 创建以下结构

```text
uav-multi-relay/
├── .gitignore
├── README.md
├── pyproject.toml
├── aaa.md
├── scripts/
│   └── check_install.py
├── src/
│   └── uav_multi_relay/
│       └── __init__.py
└── tests/
    └── test_smoke.py
```

## 文件要求

### `src/uav_multi_relay/__init__.py`

```python
"""Multi-relay UAV communication research project."""

__version__ = "0.1.0"
```

### `scripts/check_install.py`

导入 `uav_multi_relay` 并打印版本。

必须包含：

```python
if __name__ == "__main__":
    main()
```

不要修改 `sys.path`。

### `tests/test_smoke.py`

只测试：

```python
import uav_multi_relay


def test_package_import() -> None:
    assert uav_multi_relay.__version__ == "0.1.0"
```

### `pyproject.toml`

要求：

- Python `>=3.10`
- 使用 `setuptools`
- 使用 `src` 布局
- 项目名为 `uav-multi-relay`
- 版本为 `0.1.0`
- 开发依赖只加入 `pytest`

本次不要加入 PyTorch、Gymnasium、SciPy 等依赖。

### `README.md`

简要写明：

- 项目研究多中继单任务 UAV 通信；
- 逻辑拓扑为 `H → R1 → ... → RK → L`；
- 当前仅完成项目初始化；
- 安装命令；
- 测试命令。

### `.gitignore`

至少忽略：

```text
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
build/
dist/
outputs/
checkpoints/
```

## 验证

执行：

```bash
python -m pip install -e ".[dev]"
python scripts/check_install.py
pytest
```

所有命令必须成功。

## 禁止事项

本次不要：

- 创建大量空目录；
- 实现任何 UAV 模型；
- 实现通信公式；
- 实现环境；
- 实现强化学习；
- 从旧项目复制代码；
- 创建多个测试文件。

## GitHub

先提交项目初始化：

```bash
git add .
git commit -m "init: create project skeleton"
```

然后覆盖写入 `aaa.md`，内容只包括：

```markdown
# 本次执行结果

- 任务：初始化新项目
- 完成状态：
- 修改和新增文件：
- 测试命令：
- 测试结果：
- 初始化 Commit ID：
- 当前分支：
- GitHub 推送结果：
- 遗留问题：
```

填写真实结果后，再提交：

```bash
git add aaa.md
git commit -m "docs: record initialization result"
git push -u origin main
```

最终确认：

```bash
git status
```

工作区必须干净。

## 完成后回复

只回复：

- 测试结果；
- 两个 Commit ID；
- 是否推送成功；
- 是否存在遗留问题。

------

当前进度：

```text
新项目初始化：等待 Codex 执行
总体开发：0 / 5
```