# 阶段进度

## 2026-09-21

### 阶段 1：环境与资料核对 — 已完成

- 项目目录原为空，仅包含外部提供的只读 `.git` 目录。
- 本地 Zotero 数据库可读，发现 187 个条目；相关全文位于 `/home/gsh/Zotero/storage/` 和 WorkBuddy 导出目录。
- 已检查本地无线跌倒检测文献和 PDF 文本层。

### 阶段 2：文献证据 — 已完成

- 完成 SiFall、DGSense、CSI-Bench、Chu et al.、TED-Net/DGNN 和数字孪生雷达路线的证据记录。
- 明确项目评测必须覆盖域偏移、连续流式数据、仿真到真实差距和报警代价。
- 证据边界记录在根目录 `notes.md`，没有把预印本或摘要外推成已验证事实。

### 阶段 3：仓库骨架与代码 — 已完成

- 建立 `src/sim2sense_fall` 包、schema、验证器、窗口化报警基线和 Isaac/Sionna 适配器协议。
- 加入 `pyproject.toml`、`configs/baseline.yaml`、`.gitignore` 与核心测试。

### 阶段 4：文档与管理约束 — 已完成

- 建立 `AGENTS.md`，要求每阶段更新计划、进度和证据文档。
- 完成 README、架构说明、数据契约和远程同步说明。
- 首次测试发现系统环境没有自动把 `src/` 加入导入路径，已在 `pyproject.toml` 固定 `pytest` 路径；同时修正了 `slots` dataclass 测试构造方式。

### 阶段 4 验证记录

- `python -m compileall -q src tests`：通过。
- `PYTHONPATH=src python -m sim2sense_fall.windowing --duration-s 10`：通过，生成 33 个窗口。
- `python -m pytest -q`：通过，4 passed。
- Python 3.10 兼容性检查：将 `StrEnum` 替换为 `str, Enum`，兼容 smoke test 通过。
- 清理了测试缓存；生成数据、模型和仿真缓存均由 `.gitignore` 排除。
- 首次 `pytest`：因导入路径和测试构造问题失败，已修复，待复跑。
- `ruff check .`：当前环境未安装 `ruff`，未将其写成通过。

### 阶段 5：GitHub 上游同步 — 阻塞

- 目标：`git@github.com:11anticipate/Sim2Sense-Fall.git`
- 实际错误：`ssh: Could not resolve hostname github.com: Temporary failure in name resolution`
- 另外当前 `.git` 目录为只读空目录，`git init` 返回 `Read-only file system`，无法创建 Git 元数据、提交或更新 remote。
- 当前状态：本地工作树文件已准备好，但不能声称已 clone、push 或完成上游核对。

### 阶段 5：Isaac Sim 室内场景 — 已完成

目标：搭建包含卧室、客厅、卫生间等典型室内空间的三维场景，配置墙体、地板与家具属性，
导出 USD，并给出 GUI 查看方式。

- 环境核对：Isaac Sim `6.0.1-rc.7` 位于 `/home/gsh/isaacsim`（`isaac-sim.sh` / `python.sh`），
  GPU 为 RTX 4060 Laptop（8 GiB），CUDA 13.2，显示环境 `DISPLAY=:0`。
  注意 Isaac Sim 6 使用 `isaacsim.*` 命名空间（`omni.isaac.core` 已不存在），且 `pxr`
  只在 Kit 运行时启动后才可导入。
- 新增场景管线：`scene_materials.py`（材质库）、`scene_spec.py`（schema + YAML）、
  `scene_furniture.py`（参数化家具配方）、`scene_planner.py`（CPU 几何规划）、
  `isaac_scene.py`（USD 落地）。
- 新增配置 `configs/scenes/indoor_apartment.yaml`：8.4 m × 7.0 m，两室一厅一厨一卫加走廊，
  6 个房间、46 段墙、11 处门窗开口、34 件家具/灯具。
- 新增脚本：`scripts/build_indoor_scene.py`（`--dry-run` / `--headless` / `--gui`）、
  `scripts/view_indoor_scene.py`（GUI 查看）、`scripts/verify_indoor_scene.py`（物理 smoke test）。
- 新增文档 [`indoor-scene.md`](indoor-scene.md)：场景说明、材质与家具属性表、运行方式、
  GUI 查看指令和验证记录。

#### 阶段 5 关键实现决定

- **三类材质属性一次定义**：视觉（`UsdPreviewSurface`）、力学（PhysX 摩擦/恢复/密度）、
  电磁（ITU-R P.2040-3 幂律系数 + Sionna 材质名）。电磁参数按频率求值而不是硬编码，
  换频段不会静默复用旧数值。
- **规划与落地分离**：几何规划是纯 Python，可在无 Isaac Sim 的机器上单测与评审；
  `isaac_scene.py` 只消费规划结果。`pxr` 采用惰性导入，CPU 上导入该模块不会失败。
- **组合刚体**：动态家具不能「按零件各成一个刚体」，否则首次接触就散架。规划阶段把它
  拆成刚体根 Xform（承载 `PhysicsRigidBodyAPI` + 单一质量）与局部坐标子碰撞体。
- **不做在线素材依赖**：家具全部用盒体/圆柱体参数化拼装，保证离线可复现、可 diff。

#### 阶段 5 验证记录

- `python -m compileall -q src tests scripts`：通过。
- `python -m pytest -q`：通过，27 passed（新增 `tests/test_scenes.py` 23 项，覆盖材质校验、
  场景校验、墙体开口分段、共享墙不重复、组合刚体、清单可序列化）。
- `uv tool run ruff check .`：通过，All checks passed（隔离运行，未改动 Isaac Sim 环境）。
  过程中用 `ruff check --fix` + `ruff format` 清理了新文件的导入顺序、废弃导入与行宽，
  并确认格式化前后场景清单逐字段一致（几何未被改变）。
- `python3 scripts/build_indoor_scene.py --dry-run`：通过。
- `~/isaacsim/python.sh scripts/build_indoor_scene.py --headless`：通过。
  产出 `artifacts/scenes/indoor_apartment.usda`（335 KB，344 prim，231 几何，231 碰撞体，
  1 刚体/13.56 kg，14 视觉材质，12 接触材质，8 灯光）与
  `indoor_apartment.scene.json`（180 KB 清单）。
- `~/isaacsim/python.sh scripts/verify_indoor_scene.py --settle-seconds 3`：14 项检查全部 PASS，
  含正向对照「刚体抬高 0.25 m 后落回原位，残差 0.0000 m」，并比对原文件 SHA-256
  确认验证过程不改动导出产物。
- `~/isaacsim/python.sh scripts/view_indoor_scene.py --activate-physics --settle-seconds 2`：
  GUI 实际打开并驻留，物理步进正常。

#### 阶段 5 踩到的坑（已修复，记录以免重复）

1. **`Sdf.Layer.Export` 导到自己的路径会静默留空文件**：用 `Usd.Stage.CreateNew(path)`
   建 stage 时必须 `Save()`，只有继承来的匿名 stage 才用 `Export()`。已在 `build_stage`
   中按来源分支处理，并加了「导出后文件不得为空层」的断言。
2. **USD prim 名不能以数字开头**：家具零件原命名 `00_frame` 会触发
   `Path must be an absolute path: <>` 这种完全看不出原因的报错。已改名为 `p00_frame`，
   并在 `build_stage` 前用 `Sdf.Path.IsValidIdentifier` 预校验整份规划。
3. **独立运行的 Isaac Sim 不会把 PhysX 挂到 USD stage**：时间线能播、物理步数在涨，
   但没有任何物体会动，`omni.physx.tensors` 报
   `Failed to get a valid attached USD stage id`。修复方式是
   `SimulationManager.enable_all_default_callbacks()` + `setup_simulation()`（封装为
   `activate_physics()`）。这也是为什么验证脚本必须做「抬高再落下」的正向对照——
   只测漂移会把这个故障判成通过。
4. **Kit 会接管 `sys.stdout`**：headless 下 `print` 的输出会被吞掉。构建报告改为同时写
   `sys.__stdout__` 和 `artifacts/scenes/*.build_report.txt`。
5. **`SimulationApp.close()` 会直接终止解释器**：`finally` 里必须先写报告再 `close()`。
6. **YAML 1.1 浮点指数必须带符号**：`2.4e9` 会被解析成字符串，必须写 `2.4e+9`。
   已在解析器里加了针对性错误提示。
7. **Kit 会把未知命令行参数转发给自己**：脚本在启动 `SimulationApp` 前需要清空
   `sys.argv`。
8. **Isaac Sim 打开场景会往内存图层塞 prim**：`omni.usd.get_context().open_stage()`
   之后，stage 上多出 `/Render` 和 4 个 `/OmniverseKit_*` 视口相机（共 7 个 prim），
   而 `Usd.Stage.Open` 走图层缓存也会读到这份被改过的图层——于是同一份 `.usda`
   会报出 344（文件真实值）和 351（进程内被改写后的值）两个数。验证脚本已改为在
   `.usda` 的临时副本上运行，并比对原文件 SHA-256，保证产物不被验证过程污染。

## 下一步

1. 用场景配置里的 `seed` 驱动房间布局、材质与家具的随机化，形成训练域族（对应 DGSense 路线）。
2. 在场景中加入人体（刚体或骨架）并导出与场景同时间基准的运动真值。
3. 接入 Sionna RT，把 `/World` 几何与 `sim2sense:em_*` 材质映射为传播场景，生成首条
   `ChannelSample` 并完成 CPU schema 校验。
4. 复核代理电磁材质与已安装 Sionna 版本 `itu_*` 数值的一致性。
5. 网络恢复后读取上游分支、许可证和目录，做一次非破坏性合并评估。
