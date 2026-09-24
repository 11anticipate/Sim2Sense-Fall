# Sim2Sense-Fall

基于 Isaac Sim 与 Sionna RT 的室内无线信道跌倒检测研究骨架。

项目目标是生成受物理约束的人体跌倒与日常活动轨迹，将动态人体/场景状态映射为无线传播观测（CIR、CSI 及派生时频特征），再训练能够在未见房间、个体、链路和硬件条件下保持可靠性的检测模型。最终输出包括跌倒事件、置信度、报警延迟和可追溯的样本元数据。

## 研究边界

- Isaac Sim：人体运动、碰撞、场景和动态网格真值。
- Sionna RT：天线、频段、材质、多径传播和时域信道响应。
- 模型输入：CSI/CIR 或其派生表示；不把视频作为部署输入。
- 评测重点：漏报率、误报率、F1、报警延迟、跨域性能、仿真到真实差距。
- 隐私表述：项目以“不采集视频、最小化保存人体网格、限制原始无线数据访问”为隐私目标；不能把无线感知直接宣称为绝对零隐私风险。

## 当前状态（2026-09-24）

固定公寓、SMPL 蒙皮、多轴 PhysX 人体和真实 AMASS 导入已运行；键盘可控制前进、后退、
转向和停止，并显示实际物理姿态。当前使用有限根外力辅助，**整体动作质量尚未通过验收**：
仍有滑步、皮肤穿地/穿墙，以及用户新反馈的右臂摆动异常。

后续只新增蹲下、起立和摔倒及其切换；路线由用户键盘控制，不做主动避障。
起立按蹲姿回站姿规划，R 仍是显式复位。任务与验收顺序见 [当前计划](task_plan.md)。

固定公寓 + 单类实际物理后推跌倒 → Sionna 复数 CIR 已完成 smoke；规模数据、
动态家具同步、检测训练、域随机化与真实数据验证仍未完成。
现行指南、证据记录与历史文档统一从 [文档索引](docs/README.md) 查阅。

## 目录

```text
.
├── AGENTS.md                 # 协作与完成阶段后的文档更新要求
├── README.md
├── task_plan.md              # 阶段计划和阻塞项
├── notes.md                  # 文献证据和方案判断
├── pyproject.toml            # Python 工具与依赖边界
├── configs/
│   ├── baseline.yaml              # 可复现实验默认配置
│   ├── humans/                    # 人体、动作、键盘与根辅助配置
│   └── scenes/
│       └── indoor_apartment.yaml  # 室内场景声明（房间/墙/门窗/家具/材质）
├── data/                     # 原始/生成/处理数据边界说明
├── artifacts/                # 本地实验产物边界说明（导出的 USD 场景写在这里）
├── docs/
│   ├── README.md             # 全部工程文档索引与证据使用规则
│   ├── architecture.md       # 组件边界、数据流和评测设计
│   ├── data-contract.md      # CSI/CIR 样本契约
│   ├── indoor-scene.md       # 室内场景构建、导出与 GUI 查看
│   ├── human-simulation.md   # 当前人体链路与边界
│   ├── keyboard-control.md   # 键盘操作与实际证据
│   ├── history/              # 旧计划和旧指南，保留历史语境
│   └── progress.md           # 阶段日志、验证结果和阻塞项
├── scripts/humans/           # 预览、物理试验、键盘与人体检验
├── scripts/sionna/           # 公寓/人体网格到无线传播入口
├── scripts/scenes/           # 场景操作入口
│   ├── build.py              # 构建（CPU dry-run / headless / GUI）
│   ├── view.py               # 查看（俯视去顶 / 斜视去顶 / 外观）
│   └── verify.py             # 清单、USD 与物理验收
├── src/sim2sense_fall/
│   ├── schema.py             # 样本与事件数据结构
│   ├── validation.py         # 数据契约检查
│   ├── windowing.py          # 流式窗口和报警事件聚合
│   ├── simulators.py         # Isaac/Sionna 适配器协议与 dry-run
│   ├── humans/               # SMPL/AMASS、PhysX、控制与真值
│   ├── sionna/               # 几何导入、公寓转换与信道
│   └── scenes/               # 可复用场景实现
│       ├── __init__.py       # CPU 公共接口
│       ├── spec.py           # 场景 schema、YAML 加载与校验
│       ├── materials.py      # 视觉 / 力学 / 电磁材质
│       ├── furniture.py      # 参数化家具配方
│       ├── planner.py        # CPU 几何规划与清单
│       ├── numbers.py        # 数值边界校验
│       ├── geometry.py       # 世界坐标、越界与穿墙检查
│       ├── usd.py            # USD 导出与物理激活
│       ├── view.py           # 临时查看相机与去顶显示
│       └── verification.py   # USD 与清单逐项核对
├── tests/
│   ├── test_core.py
│   ├── humans/               # 人体数据、控制与回归
│   └── scenes/               # 场景单测与回归
│       ├── test_scenes.py
│       ├── test_scene_regressions.py
│       ├── test_scene_usd.py
│       └── test_scene_view.py
└── .gitignore
```

## 快速开始

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m sim2sense_fall.windowing --help
```

本机系统 Python 没有 `pip`，`ruff` 用 uv 隔离运行，不会污染其它环境：

```bash
uv tool run ruff check .
```

Isaac Sim 和 Sionna 是可选运行时，不在基础安装中强制拉取。实际入口分别使用 Isaac Python
和独立 Sionna 环境；CPU dry-run 不应依赖 GPU。环境与命令见对应指南。

## 键盘人体

公寓 USD 已导出后，在仓库根目录执行：

```bash
python3 scripts/humans/keyboard.py --dry-run --out artifacts/humans/keyboard_dry
~/isaacsim/python.sh scripts/humans/keyboard.py
```

W/S 前后移动，A/D 转向，空格停止，R 复位，Esc 退出。蹲下、起立和摔倒键位仍待实现。
当前操作说明、记录文件及已知问题见 [键盘控制](docs/keyboard-control.md)。

## 室内场景

场景声明在 `configs/scenes/indoor_apartment.yaml`（实际导出 18.48 m × 15.40 m；原始参数平面为 8.4 m × 7.0 m，两室一厅一厨一卫加走廊）。
规划部分是纯 Python，不需要 Isaac Sim；USD 落地需要 Isaac Sim 的运行时。

```bash
# 校验场景并写出可复现清单（不需要 Isaac Sim）
python3 scripts/scenes/build.py --dry-run

# 导出 USD（headless）
~/isaacsim/python.sh scripts/scenes/build.py --headless

# 场景与物理 smoke test（退出码 0 表示通过）
~/isaacsim/python.sh scripts/scenes/verify.py --settle-seconds 3

# 用 Isaac Sim GUI 查看已导出的场景
~/isaacsim/python.sh scripts/scenes/view.py --activate-physics --settle-seconds 2
```

导出产物写在 `artifacts/scenes/`（`.gitignore` 已排除）：`indoor_apartment.usda`、
`indoor_apartment.scene.json`（图元/材质/物理属性清单）和 `indoor_apartment.build_report.txt`。
完整说明见 [`docs/indoor-scene.md`](docs/indoor-scene.md)。

## 远程仓库

目标地址为 `git@github.com:11anticipate/Sim2Sense-Fall.git`。

历史 Git 状态见 [阶段日志](docs/progress.md)。分支、HEAD、远程差异须现场查询，
不要按历史指南切换分支或覆盖当前改动：

```bash
git status --short --branch
git remote -v
git log -1 --oneline
```

提交遵循 Conventional Commits；本次文档整理不包含提交或推送。
代码和人体资产的许可分别核对，SMPL/AMASS 原文件、人体网格和原始无线数据默认不入 Git。

## 文献依据

项目路线吸收了本地 Zotero 全文中的几个可核查结论：SiFall 的在线异常检测视角、DGSense 的域泛化与虚拟数据生成、CSI-Bench 的真实连续数据评测要求、CSI 深度模型的硬件/位置差异，以及数字孪生雷达研究对仿真保真度和表示方式的提醒。逐篇证据记录见 [`notes.md`](notes.md)。
