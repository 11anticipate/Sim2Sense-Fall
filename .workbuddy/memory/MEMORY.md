# 项目长期约定（Sim2Sense-Fall）

## 环境

- Isaac Sim 6.0.1-rc.7：`/home/gsh/isaacsim`。GUI 用 `./isaac-sim.sh`，脚本用 `./python.sh`。
- 命名空间是 `isaacsim.*`；`omni.isaac.core`、`isaacsim.core.api` 在本版本不存在。改用
  `isaacsim.core.simulation_manager` 与 `isaacsim.core.experimental.*`。
- `pxr` / `omni.physx` / `omni.usd` 只有 Kit 运行时（`SimulationApp`）启动后才可导入。
- 系统 Python 3.12 无 `pip`；`ruff` 用 `uv tool run ruff`。仓库无 `.venv`。
- GPU：RTX 4060 Laptop 8 GiB，`DISPLAY=:0`（Wayland），GUI 可直接运行。

## 代码约定

- `src/` 布局、类型标注、`from __future__ import annotations`、行宽 100。
- 数据边界用不可变 dataclass（`frozen=True, slots=True`），缺失/非法字段立即抛 `ValueError`。
- 路径用 `pathlib.Path`；随机过程显式传 seed；库代码不 `print`，用 `logging`。
- 重运行时（Isaac Sim / Sionna）必须惰性导入，CPU 上 `import` 对应模块不能失败，
  未安装时抛带可执行命令的明确错误。
- 每个阶段都要有 CPU 可跑的 `--dry-run` 路径，再依赖 GPU/Isaac Sim。

## 场景侧约定

- 几何用参数化图元拼装，不引入在线素材库；保证离线可复现、可 diff。
- 材质同时定义渲染（`UsdPreviewSurface`）、力学（摩擦/恢复/密度）与电磁
  （ITU-R P.2040 幂律系数 + Sionna 材质名）三套属性。
- 「规划（CPU，纯 Python）」与「落地（Isaac Sim / USD）」分离：`ScenePlan` 是消费边界。
- 动态家具必须是**组合刚体**：刚体根 Xform 承载 `RigidBodyAPI` + 单一质量，零件在局部
  坐标系下作碰撞体，不能让每个零件各自成为刚体。
- 场景是 Z-up、单位 1 m、`/World` 为 default prim；几何 prim 带
  `sim2sense:roomId/category/semantic/physicsMode/massKg/movable/tags` 自定义属性。

## 验证约定

- 判定「物理有效」必须做正向对照（把刚体抬高后落回），只断言「漂移为 0」会把
  「物理没跑」误判成通过。
- 未实际运行的检查不得写成通过；失败与阻塞记入 `docs/progress.md`。
- 生成物（USD/清单/报告）写 `artifacts/`，已被 `.gitignore` 排除。
