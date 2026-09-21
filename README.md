# Sim2Sense-Fall

基于 Isaac Sim 与 Sionna RT 的室内无线信道跌倒检测研究骨架。

项目目标是生成受物理约束的人体跌倒与日常活动轨迹，将动态人体/场景状态映射为无线传播观测（CIR、CSI 及派生时频特征），再训练能够在未见房间、个体、链路和硬件条件下保持可靠性的检测模型。最终输出包括跌倒事件、置信度、报警延迟和可追溯的样本元数据。

## 研究边界

- Isaac Sim：人体运动、碰撞、场景和动态网格真值。
- Sionna RT：天线、频段、材质、多径传播和时域信道响应。
- 模型输入：CSI/CIR 或其派生表示；不把视频作为部署输入。
- 评测重点：漏报率、误报率、F1、报警延迟、跨域性能、仿真到真实差距。
- 隐私表述：项目以“不采集视频、最小化保存人体网格、限制原始无线数据访问”为隐私目标；不能把无线感知直接宣称为绝对零隐私风险。

## 当前状态

已完成：

- 本地 Zotero 文献核对和项目证据笔记（见 [`notes.md`](notes.md)）。
- `src/` 包、样本 schema、窗口化基线和仿真适配器接口。
- 仓库目录、配置、测试、文档和阶段日志。
- Isaac Sim 室内场景：两室一厅一厨一卫加走廊，6 个房间、46 段墙、11 处门窗开口、
  34 件家具/灯具，含墙体/地板/家具的碰撞、摩擦、密度与电磁属性，可导出为 USD。
  说明、运行方式和 GUI 查看指令见 [`docs/indoor-scene.md`](docs/indoor-scene.md)。

待完成：

- GitHub 上游同步。当前环境对 `github.com` 的 DNS 解析失败，且工作树 `.git` 目录只读，详见 [`docs/progress.md`](docs/progress.md)。
- Sionna RT 接入、人体轨迹与 CSI/CIR 生成。
- 场景领域随机化、真实 CSI/CIR 数据导入、跨域实验和仿真到真实验证。

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
│   └── scenes/
│       └── indoor_apartment.yaml  # 室内场景声明（房间/墙/门窗/家具/材质）
├── data/                     # 原始/生成/处理数据边界说明
├── artifacts/                # 本地实验产物边界说明（导出的 USD 场景写在这里）
├── docs/
│   ├── architecture.md       # 组件边界、数据流和评测设计
│   ├── data-contract.md      # CSI/CIR 样本契约
│   ├── indoor-scene.md       # 室内场景构建、导出与 GUI 查看
│   └── progress.md           # 阶段日志、验证结果和阻塞项
├── scripts/
│   ├── build_indoor_scene.py     # 场景构建入口（dry-run / headless / GUI）
│   ├── view_indoor_scene.py      # 在 Isaac Sim GUI 中打开已导出场景
│   └── verify_indoor_scene.py    # 场景与物理 smoke test
├── src/sim2sense_fall/
│   ├── schema.py              # 样本与事件数据结构
│   ├── validation.py          # 数据契约检查
│   ├── windowing.py           # 流式窗口和报警事件聚合
│   ├── simulators.py          # Isaac/Sionna 适配器协议与 dry-run
│   ├── scene_materials.py     # 视觉 / 力学 / 电磁三类材质属性
│   ├── scene_spec.py          # 场景 schema、YAML 加载与校验
│   ├── scene_furniture.py     # 参数化家具配方
│   ├── scene_planner.py       # CPU 几何规划与可复现清单
│   └── isaac_scene.py         # USD 落地（几何/碰撞/刚体/材质/灯光）
├── tests/
│   ├── test_core.py
│   └── test_scenes.py
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

Isaac Sim 和 Sionna 是可选运行时，不在基础安装中强制拉取。后续适配器会通过环境检查和版本记录接入，避免在没有 GPU 的机器上导入失败。

## 室内场景

场景声明在 `configs/scenes/indoor_apartment.yaml`（8.4 m × 7.0 m，两室一厅一厨一卫加走廊）。
规划部分是纯 Python，不需要 Isaac Sim；USD 落地需要 Isaac Sim 的运行时。

```bash
# 校验场景并写出可复现清单（不需要 Isaac Sim）
python3 scripts/build_indoor_scene.py --dry-run

# 导出 USD（headless）
~/isaacsim/python.sh scripts/build_indoor_scene.py --headless

# 物理 smoke test（12 项检查，退出码 0 表示通过）
~/isaacsim/python.sh scripts/verify_indoor_scene.py --settle-seconds 3

# 用 Isaac Sim GUI 查看已导出的场景
~/isaacsim/python.sh scripts/view_indoor_scene.py --activate-physics --settle-seconds 2
```

导出产物写在 `artifacts/scenes/`（`.gitignore` 已排除）：`indoor_apartment.usda`、
`indoor_apartment.scene.json`（图元/材质/物理属性清单）和 `indoor_apartment.build_report.txt`。
完整说明见 [`docs/indoor-scene.md`](docs/indoor-scene.md)。

## 远程仓库

目标地址为 `git@github.com:11anticipate/Sim2Sense-Fall.git`。当前会话尝试读取该地址时收到 `Temporary failure in name resolution`，因此没有声称已经完成 clone、push 或上游代码核对。网络恢复且 `.git` 可写后，按以下顺序同步：

```bash
git fetch origin
git branch --show-current
git log --oneline --decorate -5
```

同步后先检查上游目录和许可证，再决定是否合并本地骨架；不要覆盖用户未审阅的上游文件。

## 文献依据

项目路线吸收了本地 Zotero 全文中的几个可核查结论：SiFall 的在线异常检测视角、DGSense 的域泛化与虚拟数据生成、CSI-Bench 的真实连续数据评测要求、CSI 深度模型的硬件/位置差异，以及数字孪生雷达研究对仿真保真度和表示方式的提醒。逐篇证据记录见 [`notes.md`](notes.md)。
