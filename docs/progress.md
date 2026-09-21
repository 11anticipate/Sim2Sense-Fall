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

### 阶段 6：GitHub 上游同步 — 已完成（未推送）

- 目标：`git@github.com:11anticipate/Sim2Sense-Fall.git`
- 早前错误：`ssh: Could not resolve hostname github.com`，且 `.git` 只读。这两条**已解除**：
  本轮 `git ls-remote --heads origin` 与 `git fetch origin` 均成功，`.git` 可写。
- 当前事实：本地 `main` = `origin/main` = `c77a37d`
  （*chore: scaffold Sim2Sense fall sensing project*），0 ahead / 0 behind，顶层目录树一致。
  也就是说上游目前只有脚手架提交，没有需要合并的内容。
- 许可证核查：上游**没有 LICENSE / COPYING 文件**，默认即「保留所有权利」。在拿到明确许可前，
  不应假设代码可对外分发。
- 本轮动作：把场景工作提交到 `feature/indoor-scene` 分支（3 个 Conventional Commits），
  `main` 保持与上游一致、未被改动。
- 未做：**没有 push**，也没有开 PR。推送属于对外发布动作，等用户确认。
- 附注：`.workbuddy/`（Agent 工作记忆）目前是未跟踪状态，是否纳入版本管理由用户决定。

### 阶段 6 验证记录

- `git ls-remote --heads origin`：`c77a37dc…  refs/heads/main`，退出码 0。
- `git fetch origin`：成功，无新对象。
- `git rev-list --left-right --count origin/main...HEAD`：`0  0`。
- `git ls-tree --name-only origin/main` 与本地已跟踪的顶层条目逐项一致。
- `git status --short`：仅剩未跟踪的 `.workbuddy/`，无未提交的源码改动。

## 下一步

1. 确认是否把 `feature/indoor-scene` 推到远端并开 PR；同时决定 `.workbuddy/` 是否纳入版本管理。
2. 用场景配置里的 `seed` 驱动房间布局、材质与家具的随机化，形成训练域族（对应 DGSense 路线）。
3. 在场景中加入人体（刚体或骨架）并导出与场景同时间基准的运动真值。
4. 接入 Sionna RT，把 `/World` 几何与 `sim2sense:em_*` 材质映射为传播场景，生成首条
   `ChannelSample` 并完成 CPU schema 校验。
5. 复核代理电磁材质与已安装 Sionna 版本 `itu_*` 数值的一致性。
6. 上游没有 LICENSE，先与仓库所有者确认许可范围，再考虑任何对外分发。

## 2026-09-21 — 室内 USD 验收：证据采集

- 已读取计划、配置、场景规划/导出/查看/验证代码和现有 27 项测试。
- `python` 不存在（`command not found`），改用 `python3`：27 tests passed，compileall 通过；CPU dry-run 输出另存 `artifacts/acceptance/cpu/`，未覆盖既有产物。
- `uv tool run ruff check .` 被 snap 环境阻止：`required permitted capability cap_dac_override not found`。使用已安装的 `/home/gsh/.local/bin/ruff check .`：通过。
- GPU 读取报 `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`；提权的只读 `nvidia-smi` 未执行，自动审批服务报 `503 Service Unavailable`，属于审批服务故障而非安全拒绝。
- Isaac 原验证脚本实际运行成功，14 项 PASS（`artifacts/acceptance/isaac_verify.log`），PhysX 使用 CPU 回退；日志同时报 `NVML_ERROR_DRIVER_NOT_LOADED`、`No device could be created`、`Failed to open display`。这能证明本轮物理 smoke test 通过，不能证明 GPU 渲染或 GUI 正常。
- 直接导入 pxr 首次报缺包，其后报 `libusd_tf.so` 缺失；显式配置本机 bundled USD Python 和动态库路径后，OpenUSD 0.25.11 成功读取实际 USD：344 prim / 231 geometry / 0 camera。
- 实际几何坐标确认六块吊顶覆盖全部房间；发现地基地面重合、地毯穿墙/越界、冰箱穿墙。验收结论尚待汇总。

### 验收阶段 2 完成：问题复现与内部查看

- 已用实际 USD 包围盒验证越界/穿墙；当前全部家具仅有轴对齐或 90° 倍数旋转，因此所报告盒体与墙的交集为真实盒体体积交集，不是任意旋转 AABB 的误报。
- 故意传入不存在的 manifest，Isaac 验证日志明确 `verification: FAILED`，但命令退出码仍为 **0**。`SimulationApp.close()` 默认 fast shutdown 且默认 `exit_code=0`，吞掉了后续返回值。这一验收门禁漏洞尚未修复。
- 配置反例均被错误接受：家具位置 NaN（manifest 含 NaN）、`same-name` 与 `same_name` 清洗后路径冲突（244 规划图元只有 238 个唯一路径）、0.05 m 床尺寸生成负床垫/枕头尺寸。
- 新增 `scene_view.py` 与查看器 `--view top|roofless|exterior`；默认去顶俯视，临时 session layer 隐藏屋顶与灯具，保留原文件和碰撞体。
- 系统 Python：28 passed / 1 skipped（缺少 pxr）；Isaac bundled Python 追加本机 pytest 所在目录后，2 项查看模块测试 passed，覆盖图层隔离、碰撞体保留、模式切换和相机包含地板四角。首次 bundled Python 测试报 `No module named pytest`，已用本机既有测试包完成，无下载。
- compileall 和本地 ruff 通过。新增查看器的 GPU/GUI 尚未验证。

### 验收阶段 3 完成：报告与最终验证

- 完成 `docs/indoor-scene-review.md`：6 项可复现待修缺陷、研究用途边界、证据位置与整改顺序。
- 重新 headless 导出到 `artifacts/acceptance/rebuilt/` 成功，USD SHA-256 与原件完全相同：`2e5fb8ec7523e972b5ef3ffb7f545a1078ffbf43480fc208e5f801c783cf86cb`。
- 三种新查看模式在 Isaac 无界面下完成视口集成：相机切换正确；top/roofless 隐藏 6 块吊顶，exterior 恢复；231 碰撞体保留；源文件哈希不变。见 `inspection_view_checks.json` / `inspection_view.log`。没有 GPU 画面输出，GUI/RTX 视觉仍待验证。
- 从实际 USD 生成并检查了对比图与布局标注图；三维预览采用 CPU 深度缓冲，图上明确标注不是 RTX 截图。
- `task_plan.md` 当前状态已从“场景任务完成”改为“基础构建完成、整体验收需整改”。生成资产未加入 Git，未提交/推送；用户已有 `.workbuddy/` 保持未跟踪。

## 室内验收整改 — 实现与 CPU 回归

- R1：构建/验证/查看入口把最终错误码传给 `SimulationApp.close(exit_code=...)`；启动失败明确报错。验证器先检查清单，并拒绝 0/NaN 时长及无效落差对照。
- R2：地基上表面位于最厚地板底面，厚度改由 `foundation_thickness_m` 配置；较薄地板下增加混凝土找平层，所有行走表面仍为 z=0。
- R3/R4：三块地毯明确适配房间的尺寸，冰箱旋转 180° 使门/把手朝室内。增加 CPU 世界几何校验，处理动态家具父变换、旋转盒体与圆柱体，拒绝越界和穿墙；允许接触以及地毯与家具的合理叠放。
- R5/R6：完整计划检查规范化路径唯一性；配置和派生零件检查有限数值与正尺寸；JSON 禁止 NaN/Infinity。
- 验证器移除 344/231 等固定计数，以清单逐图元比较世界变换、尺寸、标签、碰撞、刚体质量以及渲染/物理/电磁材质。
- 实现后的第一轮 CPU 回归：70 passed / 1 skipped；ruff 通过。新增子进程测试模拟 Kit 立即退出，确认构建/验证的启动异常与运行期异常均返回 1。

### USD 与运行时复验

- 真实 OpenUSD 回归 9 项通过：完整导出匹配、地板/地基分离、6 种同计数错误资产被拒绝，以及此前查看模式测试。
- 完整系统测试：70 passed / 8 skipped（8 项需要 pxr 的测试已在 bundled USD 环境实际执行）；compileall 通过。
- `uv tool run ruff check .` 本轮报 DBus：`Process 2 is a kernel thread, refusing.`；已安装 `/home/gsh/.local/bin/ruff check .` 通过。
- 首次 Isaac headless 构建因无 GPU 触发图形错误弹窗，60 s 超时（退出 124）；其后验证因 USD 尚不存在而非零退出。改为 `DISPLAY= WAYLAND_DISPLAY=` 后 CPU 回退构建成功。
- GPU 提权只读检查仍被自动审批服务 503 阻止，未执行；没有绕过审批或修改系统驱动。
- 修复后的 USD 在 Isaac CPU 回退下通过 10 项检查，含逐图元/材质匹配、椅子稳定、0.25 m 抬升回落、源文件不变；退出码 0。

### 整改完成与默认资产更新

- 实际 Isaac 运行期负例（旧 USD / 新 manifest）返回 1，准确识别地基、三块地毯、冰箱的变换差异，确认不再出现失败返回成功。
- 已把通过验证的 USD/清单更新到默认 `artifacts/scenes/`；旧版本备份在 `artifacts/remediation/before/`。新 USD SHA-256 为 `38f062c3ca9d54e5ab8f2b2cc7428ce06cb85c7933d74530a3c3452958403b86`。
- 更新后实际 USD 几何审计：家具越界、家具/墙体体积交集均为空；地基顶面 -0.12 m，与地板底面接触，行走面仍为 0 m。
- 已生成并逐张检查修复后的 CPU 立体预览和平面图；未把 CPU 预览写成 RTX 截图。
- `docs/indoor-scene-remediation.md` 汇总六项闭环、测试结果、资产/源码哈希和后续边界；`task_plan.md`、研究笔记与场景文档已同步。未提交或推送。

## 场景目录整理 — 迁移完成

- 可复用实现迁入 `src/sim2sense_fall/scenes/`，三个入口迁入 `scripts/scenes/`，对应测试迁入 `tests/scenes/`。
- 同步模块导入、脚本仓库定位、测试子进程与当前文档；旧入口不保留兼容副本。
- 首轮 CPU 回归 70 passed / 8 skipped；下一步复验实际 USD 与物理行为。

### 目录整理 — 回归完成

- CPU 测试 70 passed / 8 skipped；bundled OpenUSD 测试 9 passed（包含被跳过的 8 项）。compileall、已安装 Ruff、diff 空白检查通过。
- `uv tool run ruff check .` 退出 46，实际错误仍为 DBus `Process 2 is a kernel thread, refusing.`，使用 `/home/gsh/.local/bin/ruff check .` 完成检查。
- 新入口 CPU 构建/清单验证通过；从 `/tmp` 使用绝对路径构建及查看帮助成功，验证迁移后的仓库定位。包发现包含 `sim2sense_fall.scenes`。
- Isaac 构建和验证均退出 0，物理/资产 10 项 PASS；日志有 `NVML_ERROR_DRIVER_NOT_LOADED`、`No usable CUDA device`，实际使用 CPU 回退，GPU/GUI 视觉未复验。
- USD SHA-256 仍为 `38f062c3ca9d54e5ab8f2b2cc7428ce06cb85c7933d74530a3c3452958403b86`，与迁移前逐字节一致。清单仅 generator 改为 `sim2sense_fall.scenes.planner`，默认清单已同步。
- YAML 仅材质模块路径注释变化，当前 SHA-256 为 `34109c0df85b82ad86d7fc367cc3beaca0b730361444a5e93814407df0e59d83`；历史验收报告的旧配置/清单哈希保留。
- 本轮为固定场景回归，无数据集划分、无模型；场景 `apartment_cn_two_bedroom`，seed=20260921，2.4 GHz。物理指标沿用整改记录：静置/回落各 3 s，抬升 0.25 m，最大轴向误差容限 0.02 m。
- 日志和含源码/资产哈希的摘要在 `artifacts/reorganization/`。场景配置与产物目录保持清晰边界，文档和协作指令已同步。
