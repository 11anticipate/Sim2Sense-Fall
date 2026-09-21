# AGENTS.md

## 项目目标

这是一个面向室内跌倒检测的无线信道感知研究仓库。项目链路为：

`Isaac Sim 运动/场景真值 -> Sionna RT 无线传播 -> CSI/CIR 样本 -> 时空模型 -> 报警事件`

## 每阶段必须执行

1. 开始前阅读 `task_plan.md`，确认当前阶段和未解决问题。
2. 完成一个阶段后立即更新 `task_plan.md`、`docs/progress.md` 和相关研究笔记。
3. 代码改动必须保留类型标注、输入校验和可复现配置；避免把路径、随机种子、设备或阈值写死在函数体中。
4. 先提供 CPU 可运行的 dry-run/smoke test，再依赖 Isaac Sim、Sionna、CUDA 或真实采集硬件。
5. 任何实验结果必须同时记录数据版本、场景划分、随机种子、模型版本和指标定义。
6. 仿真人体网格、原始 CSI/CIR、模型权重和渲染缓存默认不入 Git；只提交 schema、配置、脚本、文档和小型可公开样例。

## 代码规范

- Python 使用 `src/` 布局、Python 3.10+、类型标注和 `ruff`。
- 数据边界优先使用不可变 dataclass 或显式 schema；遇到缺失字段时快速失败。
- 文件路径使用 `pathlib.Path`；随机过程必须显式传入 seed。
- 不在库代码中打印调试信息；使用 `logging`，由 CLI 或实验入口配置日志级别。
- 任何仿真适配器都必须有 `dry_run=True` 的轻量路径，并对 Isaac Sim/Sionna 未安装给出明确错误。

## 验证

提交前至少运行：

```bash
python -m compileall src tests scripts
python -m pytest -q
uv tool run ruff check .        # 本机系统 Python 无 pip，用 uv 隔离运行
```

涉及室内场景时再跑（需要 GPU 与 Isaac Sim）：

```bash
python3 scripts/scenes/build.py --dry-run                 # CPU，无需 Isaac Sim
~/isaacsim/python.sh scripts/scenes/build.py --headless   # 导出 USD
~/isaacsim/python.sh scripts/scenes/verify.py             # 场景与物理 smoke test
```

注意：独立运行的 Isaac Sim 默认不会把 PhysX 挂到 USD stage 上，物理步数在涨但物体不动。
需要物理时必须先调 `sim2sense_fall.scenes.usd.activate_physics()`；只断言「漂移为 0」
会把「物理没跑」误判成通过，因此 `scripts/scenes/verify.py` 用「抬高后落回」作正向对照。

如果 GPU、Isaac Sim、Sionna 或网络不可用，必须在 `docs/progress.md` 中记录实际错误，不得把未运行写成已验证。

## Git 与远程

- 目标远程：`git@github.com:11anticipate/Sim2Sense-Fall.git`
- 使用 Conventional Commits。
- 当前工作树的 `.git` 目录由外部环境提供且可能只读；不能为了提交而删除或重建用户已有 Git 元数据。
