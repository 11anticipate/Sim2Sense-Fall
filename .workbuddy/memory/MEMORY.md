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


## 人体阶段（阶段 7 之后）

- `src/sim2sense_fall/humans/` 是人体侧；`usd_human.py` 是该包中唯一接触 `pxr` 的模块，
  CPU 上导入它不能失败。
- **关节旋转约定**：SMPL 的蒙皮链把子关节偏移放在父关节的已姿态化坐标系里、旋转轴与静止姿态
  对齐。因此 USD 转动关节只要局部朝向为单位阵、轴标记与身体轴一致、`localPos0` 等于静止骨向量，
  就能逐字复现该链。`rig.forward_kinematics` 是这条链的**独立 CPU 实现**，物理结果必须与它比对
  （验收脚本要求 24 连杆误差 < 5 mm，当前为 0.00 mm）。
- **单位边界**：USD 转动关节的 limits 与驱动 target 是**度**，Isaac 的 `Articulation` 报**弧度**。
  蓝图把 limits 存成度，正好在 authoring 处不需要换算。任何新增的关节字段都要先确认单位。
- **Isaac 的物理张量视图只在时间轴播放时有效**；未 `play()` 读关节状态会抛
  `Instance's physics tensor entity is not valid`。
- `Articulation.get_world_poses()` **只返回根连杆**；逐连杆位姿要用 `RigidPrim(path)` 读。
  `RigidPrim` 的接触视图必须在构造时用 `contact_filter_paths` + `max_contact_count` 建立。
- 阻尼属性在 `PhysxSchema.PhysxRigidBodyAPI`（`physxRigidBody:linearDamping`），
  不在 `UsdPhysics.RigidBodyAPI`。
- **配置里区分 `physics_dt_s`（步长）与「速率」**：两者混用会让时间轴差 120 倍。
  试验循环里已有「时长自检」断言，别绕过它。
- **辅助必须显式记录**：稳定期骨盆托持 `settle_used_root_support`、试验期骨盆钉定
  `root_pinned_during_trial`、根参考是否被跟踪 `root_reference_tracked`。
  不允许出现「看不出来的支撑」。
- **扰动不等于标签**：跌倒/躺下的判定只用轨迹（阈值固定在配置的 `events` 段，先于试验）。
  躯干倾角用骨盆→颈部的实测方向，不是根朝向——只看根朝向会把弯腰判成直立。
- **门禁排除是正常结果**：穿透或发散的试验标 `invalid` 并被排除，`simulate.py` 对它们 `[SKIP]`
  而不是 `[FAIL]`；不要为了「全绿」放宽 `penetration_limit_m`。
- 资产（SMPL 模型、AMASS 序列）是注册制、不可再分发：放在 `data/humans/`（已 gitignore）或
  `SIM2SENSE_HUMAN_ASSETS`。缺失时流水线抛带注册地址的错误，**不自动下载**，也不拿替代物
  冒充真实人体。产物的 `body_representation` 只有 `capsule_proxy_surface` 与 `smpl_skin_mesh` 两种合法值。
- `pose_surface_points` 输出的是胶囊体表面代理点云（按物理姿态采样），不是蒙皮网格。

## 资产下载（SMPL / AMASS，注册制）

- `amass.is.tue.mpg.de` 与 `smpl.is.tue.mpg.de` 是**两套独立注册**，凭据不能互用。
- 端点 `download.is.tue.mpg.de/download.php?domain=amass&sfile=<相对路径>`：未带会话返回站点
  HTML；带 `PHPSESSID` 返回 `206` + `Accept-Ranges: bytes` → **服务器支持续传**。浏览器下大包
  反复失败是因为下载器不续传，不是站点不可达。
- 手工分段下载三条硬规则：① `curl -C` 与 `-r` 互斥（会报 error 33）；② **必须 `--retry 0`**，
  重传交给外层循环按 `stat` 实际字节数重算 Range，否则 curl 重传会重复写入导致数据错位；
  ③ 每个 part 只允许一个写入者（旧实例未退出生效会错位）。
- 优先用 `src/sim2sense_fall/humans/fetch.py::Session`（登录 + Range 续传 + 分段），
  只在它不可用时才手工 curl。
- 落盘后必做校验：`stat` 对总长 → `bzip2 -t` → 抽样 `load_amass_clip` → `git check-ignore`。
- `load_amass_library` 用 `rglob("*.npz")` 收全部 npz，非 `*_poses.npz` 会抛 ValueError，
  `--root` 要指向只含 poses 的目录（如 `amass_raw/CMU/38`），不要指到 `amass_raw`。

## 物理交互（2026-09-22 审计，详见 `docs/physics-interaction-audit.md`）

- **两种「根高度」约定不能混**：`HumanRigPlan.ground_offset_m` 是「体坐标系抬到最低胶囊
  落在 z=0」的偏移；`spawn_root_position` 被 `set_root_pose` / `forward_kinematics`
  当作**根连杆（pelvis）的世界坐标**。两者相差根连杆静止 z。~~当前 `rig.py:843-848`
  把前者当后者，导致人体每次出生悬空 0.2336 m~~ → **2026-09-22 实测更正：两份约定之差是
  −0.044589 m，方向是「下沉陷入地面 44.6 mm」而不是悬空 0.2336 m**（0.2336 是把
  `|pelvis.rest_position[2]|` 误读成偏差）。修正后 `pelvis_height_fraction=0.55` 的物理含义
  也随之改变。**P0-1 已修复**：`ground_offset` 改由 `forward_kinematics(plan, {}, (0,0,0))`
  求，出生点最低胶囊 z = −3.7e-7 m（原 −44.6 mm），`plan.py` 报 ground_offset 1.012637。
  任何新写「站立高度/出生点」的代码都要先确认是哪一个约定。
- **`omni.physics.tensors` 的接触视图在本机不可用**（Isaac Sim 6.0.1-rc.7）：
  `RigidPrim(..., contact_filter_paths=..., max_contact_count=...)` 必然抛
  `AssertionError: Physics contact view is not valid`，插件报
  `Pattern '/World/Human' did not match any rigid contact for filters`。
  与路径、类别、计数、`play()` 时机都无关，无过滤的位姿读取正常。
- **可用的接触通道**：`omni.physx.get_physx_simulation_interface().get_contact_report()`
  轮询，前提是 `PhysxSchema.PhysxContactReportAPI` 加在接触对两侧（人体胶囊 + 环境碰撞体）。
  返回 `(headers, data)`，含 `position / normal / impulse / separation / type`。
- **接触归因只能走几何，不能走句柄**（2026-09-23 实测，`/tmp/s2s-audit/probe_*.py`）：
  ① 报告里的 `actor*/collider*` 是**不透明数值句柄**，`proto_index0/1` 是无效哨兵
  `4294967295`，**不存在 句柄→prim 路径 的 API**（`actor = 2×collider − 256` 只是巧合）；
  ② `PhysxContactReportAPI` **按 ACTOR 生效，不按 collider**：摘掉全部 19 个人体胶囊的
  该 API，报告仍是 6 对 / 8 句柄（一对未少）；只标单个胶囊，仍是同样的 8 句柄；
  摘掉**环境**侧则 6 → 0，重挂 → 6。所以「人体侧有哪些接触」无法从报告里筛出。
  ③ 报告**落后一个物理步**，连续两次读取会跨越一次 `app.update()`。
  ⇒ 归因实现为**几何包含测试**（`_segment_in_volumes`：接触点落在实时世界胶囊体积内），
  记 `contact_attribution: "geometry"`；导出 `contact_segment`（连杆名）+
  `contact_is_support`（法向朝上且贴支撑面，`support_surface_height_m` 取环境包围盒最高面，
  不假设 z=0）。报告两「侧」的身份**故意不拆**。`attributed_contact_samples()` 保证
  **一次读取**同时给出点位与连杆名，避免跨步错配。
- **`verify.py` 的检查必须与运行时用同一个量**：它的「最低胶囊落在 z=0」在体坐标系里算，
  从不用 `spawn_root_position`，因此 PASS 会掩盖出生点 bug。
- **稳定期的「站立」不是地面支撑**：`simulate.py` 对 `root_mode=free` 每步
  `set_root_pose`。`settle_used_root_support: true` 是诚实记录，但首帧从不在力学平衡。
  （P0-3 未做：应改为有界支撑力 + 平滑释放，并记 `settle_support_kind` /
  `settle_released_at_s`。）
- `set_root_pose` 本身精确（写完立刻读偏差 0.0 mm），但每个物理步都会引入自由落体位移，
  所以「放置后隔几步再读」的对照读数会偏低。
- 24 连杆只有 19 个胶囊碰撞体：`left_foot/right_foot/head/left_hand/right_hand` 是叶关节，
  `_capsule_for` 返回 None。支撑接触落在踝部胶囊上，头/手可穿障。

## 坐标系与姿态判定约定（2026-09-23 修复，详见 `docs/mesh-orientation-defect.md`）

- **SMPL 原始文件的帧 ≠ 管线帧**，实测结论：
  - 文件：`X = 横向(左右)`、`Y = 上`、`Z = 前`（`basicmodel_neutral_lbs_10_207_0_v1.1.0`）；
  - 管线（`RestSkeleton` 文档）：`X = 前`、`Y = 左`、`Z = 上`；
  - 两者差一个**绕垂直轴的 90° 偏航**。任何读模型文件的新代码都要显式说明用哪个帧。
- **`up_axis_conversion` 只管 up，不能当整体换基**。它的 docstring 前提是「两帧共享 +X 前向」，
  对 SMPL 文件**不成立**。整帧导入必须用 `body_frame_conversion(up=, forward=, left=)`
  （三元组，强制 `det = +1` 拒绝镜像）；`up_axis_conversion` 只留给确实只需要 up 的路径。
- 导入帧的实测证据记在 `SmplModel.source_frame`（如 `up=y forward=z left=x`），并写进采集
  清单的 `body_model` 块 —— 单凭清单可以复现导入。
- **extent 不能判姿态**。SMPL 是 T-pose，臂展（1.825 m）本来就大于身高（1.796 m），
  所以「最高的轴是 Z」不是「站着」的判据；任何 extent 排列都分不出「站着张开手」和
  「躺着手张开」。判姿态只问解剖问题：① 头在不在脚上方；② 全程是否从立到平。
- **绕垂直轴旋转不变的检查全都看不见偏航**：stature 范围、FK 对独立 CPU 链、身高拟合收敛、
  胶囊包含、「mesh 会动」全部通过一个偏航 90° 的人体。导出侧必须至少有一条检查
  **比较 mesh 与 links**（不是 links 对 links —— 那在坏导出上也过），并点名解剖方向：
  `the body stands up in the first frame`（俯仰）+ `the mesh and the links agree on the
  body's facing`（偏航）。新增检查**必须配正向对照**（把 mesh 绕 Z 偏航 90° 后必须 FAIL），
  不会失败的检查不算证据。
- **`fit_mesh_to_rest_joints` 的配对必须取自共用关节名**，不能裸 argmin 按行号。rig 缩放到
  配置身高而 pkl 保持原身高，正确配对也差几十毫米（踝 53 mm、脚 91 mm）；≥1.75 m 时
  argmin 会**非双射**，静默返回 scale 1.0794 而不是 1.0459。确认配对只能用**关节间距离**：
  绝对坐标受绕原点缩放影响（纯 1.08 缩放会把正确配对判错），骨向量方向分不开共线脊柱关节。
  逐行比较要屏蔽被测行与配对列（自距恒为 0，否则主导均值）。
- **清单字段别按名字猜**：`hashes.mesh_source_sha256` 曾灌的是**动作**的 provenance
  （脚本动作恒 null，读起来像「皮肤不可标识」）。已拆成 `motion_source_sha256` +
  `body_model_sha256`，并新增 `body_model` 块。`points.link_names` / `joint_names` 也补上了 ——
  没有它们，（N, 24, 3）里定位不到某个连杆。

## 摔倒 Mesh 采集（`artifacts/humans/fall_mesh`，2026-09-23）

- 采集器 `scripts/humans/collect_fall_mesh.py`，`--fall-only` 取 `fall_reference` 标签的片段。
  产物每样本 `<id>.mesh.npz` + `<id>.points.npz`，**共享一条 `time_s`**。
- `fidelity` 只有两个合法值：`kinematic_replay`（纯 FK 回放，无重力/接触）与
  `physics_trial`（`simulate.py`）。**不得混入同一清单**。
- 独立复核用 `scripts/humans/verify_fall_collection.py`（第二实现，不 import 采集器）。
  「自 agree」不是证据，两边都过才是。
- 静态参考（`stand_neutral`）**不动是正确结果**；检查要写成「骨架动时 mesh 必须跟着动」，
  不能写成「mesh 必须动」。
