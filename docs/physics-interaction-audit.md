# 人体—世界物理交互审计（2026-09-22）

> 目的：在实现可靠摔倒检测之前，先把「人体模型与世界模型之间是否存在真实且稳定的物理交互」
> 这件事查清。本文只给根因、修复方案和验证标准，不重构无关模块。
>
> 方法：读代码 + 在本机实际运行 Isaac Sim 6.0.1-rc.7（headless，CUDA，RTX 4060 Laptop 8 GiB）。
> 每条结论都对应可复现的命令或产物；没有实际跑过的项都标注为未验证。

---

## 0. 结论摘要

**物理交互是存在的，但「稳定」和「可观测」这两条都没有满足。**

先说与前提不符的部分，避免按错误前提改代码：

- 物理引擎**已集成**。`activate_physics()` 把 PhysX 挂到 stage 上，
  `~/isaacsim/python.sh scripts/humans/verify.py` 本轮实跑 **31 项 PASS / 0 FAIL，退出码 0**
  （日志 `/tmp/s2s-audit/verify.log`）。正向对照是真的：把骨盆抬高 0.25 m 后
  **骨盆下落 1.3079 m**（从 1.4585 m 到 0.1506 m），水平位移 0.5559 m，最低体点 −0.0000 m。
  刚体、碰撞体、质量、重力、单位都不是缺失项。
- 所以「人体与世界完全不发生物理交互」不成立：`scripts/humans/simulate.py --headless` 的
  四向推动试验确实把骨盆推动了（`push_forward` 根位移 0.242 m，峰值躯干角 15.49°）。

真正成立的问题有四个，按严重度排：

| # | 问题 | 一句话证据 |
| --- | --- | --- |
| **RC-0** | **出生点把「体坐标系到地面的偏移」当成了「骨盆高度」→ 人体每次出生都陷入地面 44.6 mm，站立基准高 44.6 mm，跌倒阈值跟着偏高 24.5 mm** | 体坐标系偏移 1.246231 m 与「根连杆在原点时实测最低胶囊」1.290820 m 相差 **−0.044589 m**；差值恰为根连杆静止 z 与体坐标系原点的落点差 |
| **RC-1** | **接触事件/接触力在所有入口都读不到**，而且现用 API 在本机建不起来 | `trials_index.json` 7 条试验全部 `contact_forces_recorded: false`；`omni.physics.tensors` 报 `Pattern '/World/Human' did not match any rigid contact for filters` |
| **RC-2** | **观测/回放/稳定期三条通路用运动学直接写骨骼，绕过物理** | `view_amass.py:224-230` 每帧 `set_joint_positions`+`set_root_pose`+`reset_velocities`；`simulate.py:710-715` 稳定期每步传送骨盆 |
| **RC-3** | 碰撞过滤只覆盖 floor/ground，墙面与家具的接触被物理求解但从记录里消失 | `usd_human.py:577-605` + `simulate.py:641`；场景实有 46 段墙 + 127 件家具碰撞体被排除 |

另有 5 项次要缺陷（脚/头/手无碰撞体、场景无台阶、出生点不避墙、自由根不能站立污染易混淆集、
抬升对照读数不精确），列在 §3。

**对前置条件的结论**：当前的人体从未处于「被地面支撑」的状态——它出生时双脚陷入地面 44.6 mm、
稳定期被逐步传送、松手后必然倾倒。因此现在拿到的试验轨迹不能当成「有真实支撑的跌倒」，
RC-0 与 RC-2 必须先修。

> **口径更正（2026-09-23）**：本文初稿曾写「人体每次悬空 0.2336 m」。那是把
> `|pelvis.rest_position[2]|` 当成了悬空高度，属于误读。**实测**（`/tmp/s2s-audit/probe_gap.py`
> 与 CPU 正运动学双向对照）显示：两份约定之差是 **−0.044589 m**，方向是**下沉**而非悬空。
> 结论方向不变（约定确实错了、确实必须修），但数值以 44.6 mm 为准。详见 §3 RC-0。

---

## 1. 技术栈与运行入口（现状）

**依赖与工程配置**（`pyproject.toml`）

- `requires-python = ">=3.10"`，本机 3.12.3；运行时依赖只有 `numpy>=1.24`、`PyYAML>=6.0`；
  dev 依赖 `pytest>=8.0`、`ruff>=0.6`；`src/` 布局 + setuptools；`line-length = 100`，
  ruff 选 `E,F,I,UP,B`；pytest `testpaths=tests`、`pythonpath=src`。
- 入口脚本 `sim2sense-window = sim2sense_fall.windowing:main`。
- **没有** torch / sionna / mujoco：Sionna 阶段（阶段 8）尚未开始，
  `src/sim2sense_fall/simulators.py` 目前只是协议定义 + dry-run 适配器。

**Isaac Sim 运行时**

- Isaac Sim 6.0.1-rc.7，装在 `/home/gsh/isaacsim`；GUI 用 `./isaac-sim.sh`，脚本用 `./python.sh`。
- 命名空间是 `isaacsim.*`；`pxr` / `omni.physx` / `omni.usd` 只有 Kit 启动后才可导入，
  因此 `src/sim2sense_fall/humans/usd_human.py` 把 USD 导入全部推迟到函数内部。
- GPU RTX 4060 Laptop 8 GiB，driver 595.91.07，`DISPLAY=:0`；headless 可跑（本轮全部实测均 headless）。

**运行入口**（`scripts/humans/`）

| 入口 | 性质 | 是否走物理 |
| --- | --- | --- |
| `plan.py` | CPU 规划 + 骨架/动作契约 + 正运动学校验 | 否（纯 CPU） |
| `build.py --dry-run / --headless / --gui` | 导出带动画学人体的 USD；`--gui` 写显示 DOF | 仅 settle（默认 0 s） |
| `simulate.py --dry-run` | 参考动作的正运动学回放 + 标注 | 否，provenance 记为 `kinematic_reference_only` |
| `simulate.py --headless --all` | 真实物理试验 + 同步真值导出 | **是**（PD + 扰动） |
| `verify.py` | CPU / USD / 物理三层验收 | **是**（正向对照） |
| `view_amass.py` | AMASS 预览 | **否，纯运动学回放**（见 RC-2） |

---

## 2. 人体模型与世界模型各自的表示方式

### 2.1 人体：骨骼 + 蒙皮网格 + 胶囊碰撞代理，三层并存

| 层 | 表示 | 位置 |
| --- | --- | --- |
| 模型资产 | SMPL v1.1.0 neutral，6890 顶点 / 13776 三角面 / 24 关节 / 300 shape directions | `data/humans/smpl/`（注册制、已 gitignore）；由 `humans/assets.py` 加载 |
| 骨骼 | 24 连杆拓扑 | `humans/skeleton.py`、`humans/rig.py` |
| 物理 | 24 连杆 + 14 转动 DOF + 9 固定关节 + **19 个胶囊碰撞体**，1 个 articulation root | `humans/usd_human.py` 作者化到 `/World/Human` |
| 外观 | 视觉专用 `UsdGeom.Mesh` `/World/Human/Skin`，按**物理连杆位姿**逐帧蒙皮 | `humans/mesh_sequence.py` |

要点：

- **骨骼是手工静止骨架，只有关节中心来自 SMPL**：`joint_positions` 由 `J_regressor @ v_template` 求出，
  但 `mass_weight` 是手设相对权重（躯干≈53%、腿≈33%、臂≈14%），不是拟合的人体测量数据
  （`configs/humans/human_smpl_neutral.yaml:56-85`）。
- **物理体是胶囊代理，不是蒙皮网格**：`Human/Skin` 没有 `CollisionAPI`，19 个胶囊带
  `CollisionAPI` 且被设为 `visibility = invisible`（`usd_human.py:172-178`）。
- **关节约定**：SMPL 蒙皮链是 `G_child = G_parent @ [R_child | J_child − J_parent]`，
  因此 USD 转动关节用单位局部朝向 + 身体轴 token + `localPos0` = 静止骨向量即可逐字复现
  （`usd_human.py:211-219`）。`rig.forward_kinematics` 是这条链的独立 CPU 实现。
- **根模式**：`rig.root_mode = free`，骨盆是自由刚体（`configs/humans/human_smpl_neutral.yaml:45`）。
  总质量 72.0 kg，自碰撞关闭（胶囊在静止姿态故意重叠，打开会炸开）。
- **单轴简化**：每个关节只有一个 `y`（矢状面）转动自由度（同文件 97-127 行）。
  多轴路径有 CPU 单测但从未在 Isaac 中跑过。

### 2.2 世界：参数化图元 + 每房间一块地板 + 一个全局物理场景

- 声明 `configs/scenes/indoor_apartment.yaml` → CPU 规划 `scenes/planner.py` → USD
  `artifacts/scenes/indoor_apartment.usda`（18.48 m × 15.40 m，6 房间）。
- 实测该 USD：232 图元 / **231 个 `PhysicsCollisionAPI`** / 12 个物理材质 / 14 个视觉材质；
  **只有 1 个刚体**（家具中的可动件），场景本体全部是
  `sim2sense:physicsMode = "static"`。
- `sim2sense:category` 实测分布：`furniture` 127、`wall` 46、`opening` 27、
  `lighting_fixture` 18、`floor` 6、`ceiling` 6、`ground` 1、`furniture_body` 1。
- 重力是**作者化**的，不是默认值：

  ```
  def PhysicsScene "PhysicsScene"
  {
      vector3f physics:gravityDirection = (0, 0, -1)
      float physics:gravityMagnitude = 9.81
  }
  ```

- 地板：每个房间一块独立板（客厅 `scale (9.68, 6.6, 0.12)`、`translate (13.64, 3.3, −0.06)`），
  顶面统一在 **z = 0**，材质 `wood_floor`，物理材质 `contact_0p650_0p550_0p050`
  （静摩擦 0.65 / 动摩擦 0.55 / 回弹 0.05）。
  **6 块地板 + 1 块基础层同高 → 场景里没有任何台阶或高差。**

### 2.3 坐标系与单位（单一事实来源）

| 项 | 值 | 依据 |
| --- | --- | --- |
| 长度 | 米（`metersPerUnit = 1.0`） | `verify.py` 检查「exported stage uses metres and Z-up」PASS |
| 朝向 | Z-up | 同上 |
| default prim | `/World` | `usd_human.py:54` |
| 体坐标 | +X 前、+Y 左、+Z 上（右手系） | `configs/humans/human_smpl_neutral.yaml:5-9` |
| 关节限位/驱动 target | **度**（USD 侧） | `usd_human.py:212-227` |
| 关节状态 | **弧度**（Isaac 侧） | `usd_human.py:820-834` |
| 物理步长 | 1/120 s（`physics_dt_s`），与信道 50 Hz 分离 | `configs/humans/human_smpl_neutral.yaml:152-158`、`:218-232` |

单位边界已有针对性检查（限位往返误差 0.000007 度），这条是可靠的。

---

## 3. 根因清单（含代码位置与证据）

### RC-0 / P0：出生点混淆了两种坐标约定，人体每次陷入地面 44.6 mm

> **2026-09-23 更正。** 本节初稿把缺陷写成「人体每次悬空 0.2336 m」，并给出
> `spawn_root_z = 1.246231` vs `正确值 = 1.012637937`。**两组数字都是错的**：
> 前者把体坐标系偏移当成了要写入的根高度，后者又用它减了一次骨盆静止高度。
> 在同一份真实 SMPL 计划上重测后，实际量是下面这组。根因结论（两种约定混用）不变，
> 但**方向相反**：人体不是悬空，而是**踝部插进地板 44.6 mm**。

**位置**

- `src/sim2sense_fall/humans/rig.py:815-825`（`ground_offset` 定义）
- `src/sim2sense_fall/humans/rig.py:843-848`（`spawn_root_position` 直接复用 `ground_offset`）
- `src/sim2sense_fall/humans/rig.py:1010-1012`（`forward_kinematics` 把 `root_position`
  直接当作根连杆 = pelvis 的世界坐标）
- 消费点：`scripts/humans/verify.py:228-231`、`scripts/humans/simulate.py:703,763-766`、
  `scripts/humans/build.py:196`

**机理**

`ground_offset_m` 是「把整个**体坐标系**上移多少才能让最低胶囊落到 z=0」；
而 `spawn_root_position[2]` 被当作「**骨盆连杆**的世界 z」交给 `set_root_pose`。
骨盆连杆的静止原点不在体坐标系原点上，两个量相差骨盆的静止高度。

**重测后的实数（真实 SMPL 中性模型，`configs/humans/human_smpl_neutral.yaml`）**

```
约定 A（体坐标系求和）  ground_offset_m                 = 1.246231
约定 B（FK，根连杆置于原点后取最低点）                   = 1.290820
                                差值 B - A              = -0.044589 m
```

两种约定**只在根连杆的静止 z 恰为 0 时才相等**；SMPL 的根连杆 `pelvis`
静止在 z = **−0.233593**，所以两者差一个非零量。写入 `spawn_root_position` 的
是约定 A 的值，而运行时把它解释成根连杆的世界 z，于是最低胶囊落在
**z = −0.0446 m**——脚陷进地板 **44.6 mm**，不是悬空。

**证据（本机实跑，2026-09-23 重测）**

1. `HumanRigPlan.standing_root_height_m`（新增属性）与
   `forward_kinematics(plan, {}, root_position=plan.spawn_root_position)`
   共同给出最低胶囊 **−3.7e-7 m**（修复后），修复前为 **−0.044589 m**。
2. 修复后 CPU 计划阶段端到端跑通，`ground_offset_m` 报 **1.012637**，
   即「最低胶囊落在 z=0」这一条现在由 FK 量而不是体坐标系求和量来保证。

**连带影响（同源，必须一起修）**

- `scripts/humans/simulate.py:341` 与 `:906` 把
  `standing_pelvis_height_m = plan.spawn_root_position[2]`。于是
  `events.pelvis_height_fraction = 0.55` 实际算出 **0.685427 m**，
  而正确的阈值是 **0.709951 m**。
  **差 0.0245 m（24.5 mm）**，方向上仍是把「低位」判据放宽，
  压低跌倒检出率，但幅度是 24.5 mm 而不是初稿写的 128.5 mm。
- `src/sim2sense_fall/humans/events.py:177-179` 的兜底取「首帧骨盆高」，
  而首帧是**陷进地板**的状态，兜底值同样偏低。
- `verify.py:142-152` 的「CPU model puts the lowest capsule on the floor」之所以 PASS，
  是因为它在**体坐标系**里算 `最低点 + ground_offset`，从未用过 `spawn_root_position`——
  检查量与运行时使用的量不是同一个，所以这个 PASS 掩盖了 RC-0。

**修复（已完成）**

- `rig.py` 新增 `_rest_poses`（从根做 BFS，不依赖声明顺序，因为 `left_collar` 的父
  `spine3` 声明在其后）与 `_measure_clearance`，`ground_offset` 改由 FK 量求得。
- `validate_plan_geometry` 的出生点检查改为用
  `forward_kinematics(plan, {}, root_position=plan.spawn_root_position)`，
  容差 `_SPAWN_CLEARANCE_TOLERANCE_M = 1e-5`（出生 z 被舍入到 1e-6，不能用精确相等）。
- `simulate.py` 的 `standing_pelvis_height_m` 改为**实测稳定期骨盆高**
  （`settled_pelvis_height_m`），不再用解析出生高度。

### RC-1 / P0：接触事件与接触力在全部入口不可读，且现用 API 在本机必然失败

**位置**

- `src/sim2sense_fall/humans/usd_human.py:686` — `enable_contact_views: bool = False`（默认关）
- 没有任何入口打开它：
  - `scripts/humans/simulate.py:452` → `HumanRuntime(stage, plan)`（省略参数 → False）
  - `scripts/humans/verify.py:595` → 同上
  - `scripts/humans/build.py:249`、`scripts/humans/view_amass.py:181` → 显式传 `False`
- 失败结果落在 `usd_human.py:779-785`（`contact_error`）与 `usd_human.py:985-994`
  （`contact_force_magnitudes` 直接返回 `None`）

**证据**

`artifacts/humans/trials/trials_index.json`：

```json
"runtime_capabilities": {
  "contact_forces": false,
  "contact_error": "AssertionError: Physics contact view is not valid"
}
```

7 条试验全部 `"contact_forces_recorded": false`；每条 trial JSON 的
`label.metrics.used_contact_forces: false`、`label.reasons` 对应的
`contact_evidence: "none"`。也就是说，**「摔倒过程中的接触点」这一项从来没有被记录过**。

**更深一层（本轮实测定位）**

把 `enable_contact_views=True` 打开后，`omni.physics.tensors` 的接触视图在本机**仍然建不起来**：

```
[Error] [omni.physx.tensors.plugin] Pattern '/World/Human' did not match any rigid contact for filters
[Error] [omni.physx.tensors.plugin] Provided patterns for sensor and filters did not match any rigid contact entries
[Error] rigid_prim.py:1924 _on_physics_ready -> AttributeError: 'NoneType' object has no attribute 'check'
```

已逐一排除的可能原因（每种组合各跑一次）：

| 试探项 | 取值 | 结果 |
| --- | --- | --- |
| 传感器路径 | `/World/Human`（articulation root）、`/World/Human/left_hip`（普通连杆） | 都失败 |
| 过滤路径 | 6 块地板、基础层、墙、天花板 | 都失败 |
| `max_contact_count` | 8 / 16 / 64 / 256 | 都失败 |
| 构造时机 | `play()` 之前、`play()` + 8 步之后、落地 2 s 之后再建 | 都失败 |
| 刚体视图本身 | 无过滤的 `RigidPrim('/World/Human')` 读世界位姿 | **正常**（(9.32, 0.52, 1.246231)） |

→ 失败点在 `omni.physics.tensors` 的接触视图，不在路径筛选，
也不在 `sim2sense:category`。因此「打开开关」是必要但不充分的。

**可用的替代通道（本轮已实测成功）**

`omni.physx.get_physx_simulation_interface().get_contact_report()` 可用，前提是
**把 `PhysxSchema.PhysxContactReportAPI` 同时加到接触对的两侧**：

| 标注范围 | 订阅回调 | 轮询 `get_contact_report()` |
| --- | --- | --- |
| 只标注人体 19 个胶囊 | 注册了 | **0 对** |
| 只标注人体 19 个胶囊 | 未注册 | **0 对** |
| 人体 19 + 地板 6 + 基础层 1（25 个） | 未注册 | **5 对** |
| 人体 19 + 地板 6 + 基础层 1（25 个） | 注册了 | **5 对** |

拿到的单条记录（实测）：

```
type      = ContactEventType.CONTACT_PERSIST
position  = (8.5901, 2.137, 0.0)     # 接触点，z 恰好是地板顶面
normal    = (0.0, 0.0, 1.0)          # 地板法向
separation / impulse / face_index0/1 / material0/1 均可读
```

→ **接触点、法向、冲量与分离量都是可读的**，只是不能走 `omni.physics.tensors`。
`actor0/actor1` 与 `collider0/collider1` 是数值句柄而不是 prim 路径。

> **2026-09-23 实测更正（决定性）**：原先这里写「需要建立一张 句柄 → prim 路径 的映射表」，
> 实测证明**这条路走不通**，且不能用它做归因。证据（`/tmp/s2s-audit/probe_*.py`）：
>
> 1. **没有 句柄 → 路径 的 API**。报告结构体里 `proto_index0/1` 是无效哨兵 `4294967295`，
>    没有任何路径字段；`actor = 2 × collider − 256` 只是数值巧合，不是可用的身份。
> 2. **`PhysxContactReportAPI` 是按 ACTOR 生效的，不是按 collider**。对全部 19 个
>    人体胶囊摘掉该 API，报告仍是 6 对、8 个句柄，**一对都没少**；只给任意**单个**胶囊
>    加上该 API，得到的仍是同样的 8 个句柄。而摘掉**环境**侧的 API 则报告立刻清空
>    （6 → 0，重挂后 → 6）。因此「人体侧有哪些接触」根本无法从报告里筛出来。
> 3. **报告落后一个物理步**：连续两次读取会跨越一次 `app.update()`，「恢复后」的读数与
>    基线不一致；句柄本身也在两次读数之间漂移。
>
> **结论**：接触归因改为**几何判定** —— 拿报告里的接触点去和**实时世界坐标胶囊体体积**
> 做包含测试（`_segment_in_volumes`）。归因结果记在 `contact_attribution: "geometry"`，
> 每个接触点对应一个 `contact_segment`（连杆名）与 `contact_is_support`（是否压在支撑面上，
> 由法向 + 相对支撑面高度判断，`support_surface_height_m` 取环境包围盒最高面，不假设 z=0）。
> 报告两「侧」的身份**故意不拆**：拆不出来，硬拆就是埋雷。CPU 侧单元测试见
> `tests/humans/test_humans.py::test_segment_in_volumes_names_the_containing_limb`。
> 由于报告落后一步，`attributed_contact_samples()` 保证**一次读取**同时给出点位与连杆名。

#### 2026-09-23 追加实测：`support_surface_height_m` 的三层坑（已修）

`contact_is_support` 依赖「支撑面高度」这个标量。查这条链路时连踩三层，逐层都已实测钉死：

1. **`UsdGeom.BBoxCache.ComputeWorldBound` 返回的是错的参考系**（最初实现用的就是它）。
   房间地板 `translate z = −0.06`／`scale z = 0.12`，它的世界顶面应是 `0.00`；`ComputeWorldBound`
   返回 `z ∈ [−0.0, −0.0]`，而**它上方的天花板返回同一个值**（应为 2.85）。
   即：既不带 `xformOp:scale`，也不带 prim 自身的 translate。
   巧的是这个错值恰好压线通过 `_SUPPORT_HEIGHT_TOLERANCE_M`，所以**从未暴露**。
2. **改成手算时 `BBox3d` / `Range3d` 都没有 `TransformBy`**（两者 `hasattr → False`）。
   我按 USD 文档写的 `box.TransformBy(xform)` 直接抛 `AttributeError`；
   而当时的 `try/except` 会把异常吞成 `contact_samples = None` → **整段 121 帧接触历史悄悄归零，
   且全部检查仍报 PASS**。这比原 bug 更危险，已一并修掉（见下）。
3. **`ComputeLocalBound` 本身也是错的**：它返回一个**已缩放、但仍被 prim translate 偏移**的盒。
   地板实测 `local bound = (0, 0, −0.12) .. (8.8, 6.6, 0)`，而几何真值应是
   `(−0.06, −0.06, −0.06) .. (+0.06, +0.06, +0.06)`。于是再叠一次 translate 就变成 `−0.06`。
   另外它对 `render` / `proxy` / `guide` 三个 purpose **返回反向无穷哨兵盒**，不是可用回退。

**独立交叉验证**（不依赖任何 bbox API）：实际接触点全部落在 `z = 0.00000` 且 `normal_z = +1.0000`，
与场景 authord 的 `−0.06 + 0.06 = 0.00` 逐位吻合 → 地板顶面真值就是 **0.0**，不是 `−0.06`。
修好后 `support_surface_height_m = −1.34e−9`（浮点零），`contact_is_support` 从
**True=0 / False=632**（一个躺在地板上的人体却被判「无支撑接触」）变成
**True=456 / False=176**：456 条全部 `z = 0.00000`、`normal_z = +1.0000`（躺在地面上），
176 条是躯干/肩/肘撞到立面的冲击（高度最高 1.45 m）。这才是有物理意义的拆分。

**修法**：不再经 `BBoxCache`，直接从 prim 自身的 `size`/`radius`/`height` 属性构造居中几何盒
（`_prim_local_box`），再变换 8 个角点取最大 z（`_world_box_max_z`）。本场景全部碰撞体都是
`Cube`（197）或 `Cylinder`（34），两者都带 `size` 属性，因此这条路是完备的；未知类型返回 `None`
而不是假装单位立方体。回归测试（CPU 可跑、不依赖 `pxr`）见
`tests/humans/test_humans.py::test_world_box_max_z_maps_every_corner_not_just_the_local_maximum`
与 `::test_prim_local_box_is_centred_for_cube_and_cylinder`。

**同时修掉的「静默降级」**（第 2 层留下的地雷）：原本三处 `except ContactSourceUnavailable`
把「通道坏了」和「没碰到东西」混为一谈。现在三者分开：
- `simulate.py` 逐帧：通道中途失效 → **抛错并拒绝出结果**，不再把已采集的历史抹成空；
- `simulate.py` 稳定期门禁：通道可读但归因不到任何连杆 → 抛错（归因是纯包含运算，读得出点就
  不可能归因不到，所以这是访问器坏了，不是人体没碰地）；稳定期确实零接触 → 单独抛错；
- `support_surface_height_m()` 返回 `None` → 抛错（没有参考面就无法分类接触）。
新增检查 `contact points can be attributed to limbs` 与
`support surface height is readable and below the body` 在 `simulate.py` 与 `verify.py` 两侧都跑。

### RC-2 / P0：三条通道用运动学直接写骨骼，绕过物理

| 位置 | 行为 | 后果 |
| --- | --- | --- |
| `scripts/humans/view_amass.py:224-230` | 每帧 `set_joint_positions` + `set_joint_targets` + `set_root_pose` + `reset_velocities` | 骨盆被传送、速度被清零，**接触永远不可能产生反作用**。这正是「场景里有地板、有碰撞体，但看起来完全没有物理」的那条通路 |
| `scripts/humans/build.py:258-259` | 直接写显示 DOF | GUI 里人体是摆拍的 |
| `scripts/humans/simulate.py:767-768` | `if config.control.mode == "kinematic": set_joint_positions(values)` | 配置支持的 `kinematic` 模式整条绕过动力学（当前配置是 `pd`，所以未生效，但通路存在） |
| `scripts/humans/simulate.py:710-715` | 稳定期**每一步** `set_root_pose(pinned_root, pinned_quaternion)`；`pinned = plan.root_mode == "free"`（`:708`）对当前配置**恒为真** | 0.6 s 内 72 步把骨盆钉在站立位姿上。**「站着」不是因为地面在支撑，而是因为每步都在传送**。这一事实记在 provenance 的 `settle_used_root_support: true`，但试验的首帧从未处于力学平衡 |
| `scripts/humans/simulate.py:760-766` | `--pin-root` 在试验期做同样的事 | 记为 `root_pinned_during_trial: true`，但仍是运动学辅助 |

底层实现：`src/sim2sense_fall/humans/usd_human.py:940-946`（`set_root_pose`）、
`:948-962`（`reset_velocities`）、`:824-828`（`set_joint_positions`）。

已核实的旁证：`docs/human-simulation.md:58-63` 自己声明
`view_amass.py` 是 kinematic replay、「不适合用来判断墙体或地面的接触反作用」；
`task_plan.md:178` 也把 physics replay mode 列为待办。

### RC-3 / P1：碰撞过滤只覆盖 floor / ground，墙面与家具的接触被丢弃

- `src/sim2sense_fall/humans/usd_human.py:577-605` — `environment_contact_paths()`
  只看 `sim2sense:category ∈ ("floor", "ground")`（`DEFAULT_CONTACT_CATEGORIES`）。
- `scripts/humans/simulate.py:641` — 批次索引里把 `"contact_categories": ["floor", "ground"]` 硬编码。

后果：摔到墙上、摔到沙发上的接触被 PhysX 正常求解，但**不进入任何记录**。
需求里的「障碍物」接触因此无法与「地面」接触区分。
场景实测有 46 段墙 + 127 件家具 + 27 处开口 + 6 块天花板带 `CollisionAPI`，全部被排除在外。

### RC-4 / P1：场景没有台阶，也没有任何高差

- `configs/scenes/indoor_apartment.yaml` 的 6 个房间地板同高（顶面一律 z = 0）。
- 实测 `sim2sense:category` 取值只有
  `floor / ground / wall / opening / furniture / ceiling / lighting_fixture`，没有 `step`。
- 需求里的「台阶」当前**没有任何几何表示**，因此也无从验证踏步、踩空、绊倒这类交互。

### RC-5 / P1：脚、头、手没有碰撞体；支撑接触发生在踝部胶囊上

- `src/sim2sense_fall/humans/rig.py` 的 `_capsule_for` 对叶关节返回 `None`，
  24 连杆只生成 19 个碰撞体。
- 实测无胶囊的连杆：`left_foot`、`right_foot`、`head`、`left_hand`、`right_hand`。
- 后果一：人的支撑接触是**踝部胶囊**而不是脚掌 → 站立高度偏高。
  实测踝关节静止世界 z = 1.246231 − 1.1379 ≈ 0.108 m（真实成人约 0.075 m）。
- 后果二：**头和手可以穿过障碍物**（跌倒时头部撞墙不会被求解）。

### RC-6 / P2：出生点只避家具、不避墙

- `scripts/humans/common.py:222-270` — `scene_spawn_point()`：
  inset 只用 `room.wall_thickness + clearance_m`（`:239`），
  障碍列表只有家具（`:242-253`），目标函数只最大化「离家具最远」。
- 实测出生点 `(9.32, 0.52)`；客厅 y ∈ [0, 6.6]，**距 −Y 墙只有 0.52 m**。
  向 −Y 的侧倒会在很短的位移内撞墙，跌倒轨迹在四个方向上严重不对称。

### RC-7 / P2：自由根 + 单轴矢状面关节 ⇒ 站立与行走不可保持，污染易混淆动作集

- 配置：`configs/humans/human_smpl_neutral.yaml:97-127`，每个关节只有 `y` 轴。
- 实测：`trials_index.json` 中 `walk_in_place × none` 的标签是 **`fall`**，
  跟踪误差 **150.10°**；`control_loss` 同样 150.01°。
- 机理：单轴关节 + 中等刚度 PD 的自由根人体是倒立摆，必然倾倒；
  `docs/human-simulation.md:206-213` 已如实记录为「未达标」。
- 后果：**一个本该是 `no_fall` 的易混淆动作被标成了 `fall`**，
  验证集无法区分「跌倒」与「行走中倒下」。

已核实物理侧旁证（`/tmp/s2s-audit/probe_rest.py`，drives 保持静止位姿、跑 3 s）：

```
t=0           最低胶囊 +0.196150 m（left_ankle）   ← 悬空（RC-0）
t=3s          最低胶囊 −0.000237 m（right_wrist）  ← 已经躺平
pelvis z      0.165505 m（出生 1.246231 m）
14 个 DOF     髋/膝/踝全在 ±0.0°，肘 −8.5°，其余 < 4°
```

即：**关节几乎没动，整个人像一块板一样倒下去了**。这直接证明「站立」不是物理平衡。

### RC-8 / P2：抬升对照的读数偏低约 37 mm

- `scripts/humans/verify.py:452-456`：`place(args.lift_height)` 之后连做 4 次 `app.update()`
  才读 `lifted_root_z`。
- 实测（`/tmp/s2s-audit/probe_root.py`）：`set_root_pose` 本身是精确的
  （写完立刻读，偏差 **0.0 mm**），但每多一步物理就会掉；偏置 0.9 m 时一步就掉 49.7 mm。
- 于是本轮 `lifted_root_z` 读到 1.4585 m，而指令值是 1.2462 + 0.25 = 1.4962 m（差 37 mm），
  报告的「下落 1.3079 m」并不等于「抬升量 + 自由落体」。
  这不影响该对照的通过与否（0.25 m 门限远大于 37 mm），但读数语义应当说清。

---

## 4. 修复方案（按优先级）

### P0-1 修出生点约定（RC-0）——最小改动，收益最大

**改什么**：让 `ground_offset` 与 `spawn_root_position` 只用**一种**约定。

`src/sim2sense_fall/humans/rig.py:815-825`，把 `ground_offset` 改成在
**根连杆坐标系**里求（即让「偏移」直接就是根连杆的世界 z）：

```python
rest_poses = forward_kinematics(plan, {}, root_position=(0.0, 0.0, 0.0))
ground_offset = -min(
    float(link.capsule.lowest_world_z(rest_poses[link.name].translation))
    for link in colliders
)
```

`standing_height` 用同一批 `rest_poses` 重算，`spawn_root_position`（`:843-848`）保持不变。
改完之后 `spawn_root_position[2]` 就真正等于站立骨盆高度，
`simulate.py:341/:906` 传给标注器的 `standing_pelvis_height_m` 自动变正确，
`events.py` 的 55% 阈值无需改动即自动校准。

**同时要改的检查（否则 PASS 仍然掩盖问题）**：

- `scripts/humans/verify.py:142-152`：把 `link.capsule.lowest_world_z(link.rest_position)
  + plan.ground_offset_m` 换成「用 `forward_kinematics(plan, {}, root_position=plan.spawn_root_position)`
  求最低胶囊点」。检查量必须与运行时使用的量一致。
- `src/sim2sense_fall/humans/rig.py:927-975`（`validate_plan_geometry`，其中 `:957` 用同一写法
  `lowest_world_z(link.rest_position) + plan.ground_offset_m`）改成同一约定。

**涉及的数据结构与初始化流程**：`HumanRigPlan.ground_offset_m` 的语义
（体坐标系偏移 → 根连杆世界 z）、`HumanRigPlan.spawn_root_position`、
`RigPlan → forward_kinematics` 的调用、`build.py/simulate.py/verify.py` 三处
`set_root_pose(plan.spawn_root_position)` 的调用点（内容不变，语义变正确）。

**验收**：见 V1、V2a。

### P0-2 建立可用的接触通道（RC-1 + RC-3）

**第一步（1 行，必做）**：`scripts/humans/simulate.py:452`、`scripts/humans/verify.py:595`
改为 `HumanRuntime(stage, plan, enable_contact_views=True)`；
`build.py:249`、`view_amass.py:181` 保持 `False`（只做外观，不需要接触）。

**第二步（真正的修复）**：在 `HumanRuntime` 里加一条不依赖 `omni.physics.tensors` 的
接触来源，并**显式记录用的是哪一条**。

- 新增 `_enable_contact_report(stage, filter_paths)`：
  对 `self._link_paths` 下的 19 个胶囊**和** `filter_paths` 里的每个碰撞体
  `PhysxSchema.PhysxContactReportAPI.Apply(prim)` + `CreateThresholdAttr().Set(0.0)`。
  （两侧都要标，这是本轮实测出来的必要条件。）
- 新增 `contact_report()`：每物理步轮询
  `omni.physx.get_physx_simulation_interface().get_contact_report()`，
  返回 `headers, data`，按 `contact_data_offset / num_contact_data` 展开成
  `(pair, position, normal, impulse, separation)`。
- 建立 `collider_handle → 连杆名` 的映射（构造期按同一顺序遍历 stage 生成），
  把每条接触归到具体连杆；`actor*` 归到刚体。

  > **已废弃**：见 RC-1 的 2026-09-23 实测更正 —— 句柄映射不存在，且 API 按 actor
  > 生效，映射建出来也筛不出人体侧。实际落地为几何包含测试 + `contact_segment`。
- `capabilities` 增加 `contact_source ∈ {"tensors_view", "physx_contact_report", "none"}`
  与 `contact_pair_state`；`capabilities["contact_error"]` 保留。
- `contact_force_magnitudes()` 在该来源下改为「本步对所有接触的 `|impulse|` 之和」，
  并把每步的 `max` 也留一份（fall impact 是瞬时尖峰，平均会抹掉）。
- 导出侧：`export.py:357/427/555` 的 `contact_force_n` 语义不变，
  新增 `contact_points_xyz (N,3)` 与 `contact_owner (N,)`，
  让「接触点」这一项真正可读（RC-1 的核心诉求）。

**第三步（RC-3，一行 + 一次核对）**：
`usd_human.py` 的 `DEFAULT_CONTACT_CATEGORIES` 扩到
`("floor", "ground", "wall", "opening", "furniture", "furniture_body", "step")`，
并把 `simulate.py:641` 的硬编码换成从同一个常量取；
导出里保留 `contact_category`（每步每条接触属于哪一类），
这样「地面撞击」与「撞墙/撞家具」可以分开统计。

**为什么要单独记来源**：`events.py:495-502` 已经把
「几何推断的代理事件」与「求解器报出的实测接触」分开（`contact_evidence`）。
新增通道后必须继续把它标成 `measured_contact`，
而几何路线（§6 方案 D）只能标 `geometric_proxy`——不允许混称。

**涉及的数据结构与初始化流程**：
`HumanRuntime.capabilities`、`HumanRuntime._rigids`、`TrialRecording.contact_force_n`、
`GroundTruth` 的 NPZ 字段与 `trials_index.json` 的 `runtime_capabilities`、
`scripts/humans/migrate_trials_index.py`（旧批次索引要能识别新字段缺失）。

**验收**：见 V2d、V2e、V3a、V3d。

### P0-3 取消稳定期的逐步传送（RC-2 的 `simulate.py:710-715`）

**最小改动**：把「每步 `set_root_pose`」换成**有界支撑力**，并记录释放时刻。

- 首选：稳定期改用 `runtime.apply_force("pelvis", (0, 0, +w))`，
  `w` 取略小于重力的固定值（例如 0.98 × 72 kg × 9.81），
  让脚部在这段时间内**真正承重**；释放时把该力平滑降到 0（例如 0.1 s 线性）。
- 保底：仍用传送，但把释放后的瞬态排除在记录窗口之外，
  即 `execute_trial` 在 `settle_seconds` 之后**再跑一小段（如 0.3 s）不记录**，
  直到 `|骨盆竖直速度| < stabilization_threshold_s` 才开帧（配置项已存在）。

**必改的记录**：`settle_used_root_support` 保留，但增加
`settle_support_kind ∈ {"none", "kinematic_teleport", "bounded_force"}` 与
`settle_released_at_s`。这样「稳定期用了什么支撑」不可能再被误读。

**影响面**：`simulate.py:693-716`（稳定循环）、`test_*trial*` 相关断言、
`trials_index.json` 的 provenance 字段。

**验收**：见 V3b、V3c。

### P0-4 给查看器加真正的物理回放（RC-2 的 `view_amass.py`）

**最小改动**：`view_amass.py` 增加 `--physics-replay` 分支：

- 只用 `runtime.set_joint_targets(values)`，**不再**每帧
  `set_joint_positions` / `set_root_pose` / `reset_velocities`；
- 保留既有 PD 增益（或按 `--control-scale` 缩放）；
- 根由重力和接触决定；若必须托住骨盆以便观察，用
  `--root-support` 显式开启并在屏幕/日志大字标注「ROOT SUPPORTED」；
- 蒙皮继续用**实测**连杆位姿驱动（`skin_mesh_sequence_frame(mesh, runtime.link_poses())`），
  这样皮肤和碰撞代理不会分家。

默认仍保留现有运动学预览（它适合核对重定向），但把 banner 改成
`AMASS preview (KINEMATIC REPLAY -- not physics)`，避免再被当成物理证据。

**验收**：见 V2d（在该模式下），以及录屏中人体应能与家具发生可见的阻挡、绕行、卡住。

### P1-1 给脚/头/手补碰撞体（RC-5）

- `rig.py` 的 `_capsule_for` 对叶关节改为：若叶关节有父骨方向，则沿**父骨方向的末段**
  生成一个小胶囊（foot 用 0.09 m 半径 + 0.05 m 柱长；hand 用 0.045 m；
  head 用 0.11 m 球近似 → 柱长 0），而不是返回 `None`。
- 后果必须同步：碰撞体数 19 → 24，`verify.py:178-182` 的
  「capsule count matches the planned colliders」会自动跟着 plan 走（它比较的是 plan），
  `build.py:219-223` 同理；但 `docs/` 里所有「19 胶囊」的表述要一起更新。
- 站立高度会因此下降约 0.03 m（脚掌承重替代踝部胶囊），RC-0 修好后再校准一次。

### P1-2 加台阶（RC-4）

- `configs/scenes/indoor_apartment.yaml` 增加一个可选的 `steps:` 段
  （位置、宽、深、级数、每级高），`scenes/planner.py` 为每级生成一个静态立方体，
  带 `sim2sense:category = "step"` 与 `sim2sense:stepIndex`。
- 物理侧不需要新代码（沿用静态碰撞体 + 现有摩擦材质）；
  只需把 `step` 加进 `DEFAULT_CONTACT_CATEGORIES`（P0-2 第三步已含）。
- 电磁侧沿用 `materials.py` 的材质三属性写法加一种 `step_edge` 材质即可。

### P2-1 出生点加入墙体项（RC-6）

`common.py:242-270`：把「离墙距离」并入目标函数，
例如要求候选点同时满足 `离最近墙 ≥ 1.2 m`（不足则回退并记录），
或在同分点中优先选「离墙最远」的点。`--spawn-x/--spawn-y` 已存在，可直接用于人工指定。

### P2-2 停止让 `walk_in_place` 污染易混淆集（RC-7）

在自由站立达标之前：
`simulate.py` 的 `gates` 已经分别报告 `tracking_gate_applies` / `tracking_within_tolerance`，
只需把「跟踪未达标且标签为 `fall` 的 `no_fall` 参考动作」标成
`label_credible: false` 并在索引里写原因（当前 `label_credible` 只看
「fall 必须有 impact」，不覆盖「行走被判成跌倒」）。
真正的解法（多轴关节 + 平衡控制，或 Isaac Lab）见 §6 方案 B。

### P2-3 抬升对照读数（RC-8）

`verify.py:452-456` 改成「`place()` → 立刻读 `lifted_root_z` → 再 `hold(1.5)`」，
并断言 `lifted_root_z` 与指令值之差 < 1 mm；把这段经历写进注释，
理由与文件里既有的「天花板对照」注释同源。

---

## 5. 验证步骤（可量化）

### V1 — CPU 层（不需要 Isaac，可直接进 pytest）

```bash
python3 -m pytest -q
python3 scripts/humans/plan.py
```

必须新增并通过的断言（放进 `tests/humans/`）：

1. `forward_kinematics(plan, {}, root_position=plan.spawn_root_position)` 的最低胶囊点
   ∈ [−1e−6, +1e−6] m。**当前值是 +0.233594 m，应当 FAIL。**
2. `plan.spawn_root_position[2]` 与
   `plan.ground_offset_m + plan.link(plan.root_link).rest_position[2]` 相等（1e−9）。
3. `plan.spawn_root_position[2]` 与 `plan.ground_offset_m` **不相等**
   （显式钉住两种约定的差异，防止回退）。
4. 用 `spawn_root_position[2]` 推出的 55% 阈值与真实站立骨盆高的 55% 一致（1e−9）。

### V2 — Isaac 层（正向对照，扩展 `scripts/humans/verify.py`）

```bash
~/isaacsim/python.sh scripts/humans/verify.py --out /tmp/s2s-audit/verify2
```

| 编号 | 检查 | 量化门限 | 当前值 |
| --- | --- | --- | --- |
| V2a | 静止停留：把人体放到 `spawn_root_position`、DOF 归零、drives 开启，跑 1.0 s | 全程最低胶囊 ∈ [−0.005, +0.020] m，且骨盆下降 < 0.05 m | 起步 **+0.196 m** → FAIL（P0-3 未做） |
| V2b | 不穿透：全程无胶囊低于 −0.05 m | 与现有一致 | PASS（−0.0000 m） |
| V2c | 抬升回落正对照：抬高 0.25 m | 下落 ≥ 0.23 m，且 `lifted_root_z` 与指令值差 < 1 mm | 下落 1.3079 m（读数差 37 mm） |
| V2d | 接触可读：贴地状态下 | ≥ 1 对接触，接触点 z ∈ [−0.02, +0.02] m，法向与 +Z 夹角 < 5° | 实测可拿到：点 (8.5901, 2.137, 0.0)、法向 (0,0,1) |
| V2e | 接触负对照：把人体整体抬高 1.0 m 悬空 | 接触对 **= 0**（防止把常驻噪声当成接触） | 待补 |
| V2f | 接触归因 | 每条接触都能落到一个 `contact_segment`（连杆名）；`contact_attribution == "geometry"`；`support_surface_height_m` 可读且等于实测支撑面 | **PASS（GPU 实跑）**：`verify.py` 报 `attribution=geometry, limbs=['left_ankle','right_ankle']`；`support_surface_height_m = −1.34e−9`（地板真值 0.0，与实测接触点 `z = 0.00000` 吻合）；单次试验 632 行接触跨 121 帧，`is_support` True=456 / False=176 |
| V2g | 稳定期不传送 | `settle_support_kind != "kinematic_teleport"`，或 `settle_released_at_s` 到首帧 ≥ 0.3 s 且释放后骨盆竖直速度 < 阈值 | **仍未做（P0-3）**：当前恒为传送 |
| V2h | 保留现有 31 项 | 全部 PASS | 31 PASS / 0 FAIL |

### V3 — 试验层（`scripts/humans/simulate.py --headless`）

```bash
~/isaacsim/python.sh scripts/humans/simulate.py --headless --all --out /tmp/s2s-audit/trials2
```

| 编号 | 检查 | 量化门限 |
| --- | --- | --- |
| V3a | 接触力真的被写进产物 | `trials_index.json` 每条 `contact_forces_recorded: true`；`label.metrics.used_contact_forces: true`；`contact_evidence: "measured_contact"` |
| V3b | 首帧必须在被支撑状态 | 首帧 `pelvis_height` 与站立骨盆高（1.012637937 m）之差 < 2%；首帧最低体点 ∈ [−0.02, +0.02] m |
| V3c | 辅助必须显式 | `settle_used_root_support: false`，或 `settle_support_kind` + `settle_released_at_s` 均在 provenance 中 |
| V3d | 撞击与接触对得上 | `first_impact_s` 与「接触冲量首次超过静止噪声 5 倍」的时刻之差 ≤ 2 个物理步（2/120 s） |
| V3e | 标签可信度 | `walk_in_place × none` 不得为 `fall`；若仍为 `fall`，必须 `label_credible: false` 并写明原因 |
| V3f | 时间轴自检 | `|times[-1] − (frames−1)·step_s| < 1e−9`（已有断言，保留） |

### V4 — 回归

```bash
python -m compileall src tests scripts
python3 -m pytest -q
uv tool run ruff check .
~/isaacsim/python.sh scripts/scenes/verify.py --settle-seconds 3
```

`rig.py` 与 `verify.py` 改动后，`artifacts/scenes/indoor_apartment.usda` 不应发生变化
（`verify.py` 已有「the verified base scene was not modified」哈希检查，保留）。

---

## 6. 替代路径与取舍

### 方案 A（推荐）：留在 Isaac/PhysX，修上面四件 P0

- **成本**：改 5 个文件（`rig.py`、`usd_human.py`、`verify.py`、`simulate.py`、`common.py`）
  + 1 个新测试文件；无新依赖；现有 31 项验收、CPU 测试和场景管线全部保留。
- **效果**：拿到真实接触力/接触点/法向；人体凭空高度从 0.234 m 降到 0；
  跌倒阈值自动校准；稳定期有真实承重。
- **边界**：**不解决**自由站立与行走（RC-7）——那需要真正的平衡控制。
- **风险**：~~`omni.physx` 接触报告的 `actor/collider` 返回数值句柄，
  句柄→路径映射需要实现并做 V2f 验证；若映射不可靠，则接触只能归到「人体整体」，
  达不到「接触点归属到具体连杆」。~~
  **已实测落定**：句柄→路径映射不存在，且该 API 按 actor 生效（见 RC-1 更正）。
  接触点归属**改走几何包含**，已能给出具体的 `contact_segment`（连杆名），
  风险从「映射是否可靠」变为「接触点是否落在实时胶囊体积内」——
  后者由 `_segment_in_volumes` 的 CPU 单元测试守住。

### 方案 B：Isaac Lab + 模仿学习 / 多轴关节平衡控制

- **成本**：高。新依赖栈（IsaacLab）、重新定义 articulation 与 actuator 配置、
  多轴关节重规划、动作重定向与训练预算；现有 `verify.py` 的多数对照要重写。
- **效果**：这是**唯一**能同时解决「自由站立/行走」（RC-7）与「自然跌倒」的路径，
  也是让 `walk_in_place`、`sit_stand` 这些易混淆动作真正可用的前提。
- **取舍**：应放在方案 A 之后。先有可读的接触与正确的初始条件，
  再去花训练预算，否则训练出来的控制器无法用接触真值评估。

### 方案 C：人体侧换用 MuJoCo / mujoco-warp，Isaac 只做渲染与 USD

- **成本**：中高。多一套物理栈要同步；USD/蒙皮导出与 Sionna 侧可原样复用。
- **效果**：接触力、接触点、接触对是一等公民，且求解器对铰接人体更稳；
  MuJoCo 的接触模型更可解释。
- **取舍**：失去「单一运行时」的简单性，且 `simulate.py` 的真值导出、
  事件标注、批次索引都要改成消费第二套状态；场景侧（墙/家具/地板）需要重新导入。
  只有在方案 A 的接触通道最终证伪时才值得。

### 方案 D（保底）：保持运动学人体，接触完全由几何推断

- **成本**：最低。`simulate.py` 已经在算 `body_points` 与 `min_body_point_z`，
  只需要把它们与场景碰撞体的 AABB 求交，输出「接触时刻 + 接触部位 + 最近面」。
- **效果**：**接触时刻和接触部位可用**（事件标注本来就用轨迹，不依赖接触力），
  但没有接触力，也无法反映真实反作用。
- **取舍**：必须如实标成 `contact_evidence: "geometric_proxy"`，
  产物里绝不能写成 `measured_contact`。可以作为方案 A 落地前的过渡，
  也可以作为方案 A 中映射表失败时的降级路径。

---

## 附：本轮实测证据与复现

所有产物在 `/tmp/s2s-audit/`（临时目录，未入库）：

| 文件 | 内容 |
| --- | --- |
| `verify.log` | `scripts/humans/verify.py` 实跑：31 PASS / 0 FAIL，退出码 0 |
| `probe_gap.py` / `gap.log` | 根姿态约定证据：`live lowest = CPU FK(live state) = +0.196150 m`；`CPU FK(rest) = +0.233594 m` |
| `probe_contact.py` / `contact.log` | `omni.physics.tensors` 接触视图两种构造顺序都失败，含插件原始报错 |
| `probe_view.py` / `view.log` | 排除传感器路径/过滤路径/类别：全部 `AssertionError: Physics contact view is not valid` |
| `probe_late.py` / `late.log` | 排除延迟构造与 `max_contact_count`：16/64/256 全失败；PhysX 场景为 GPU dynamics + GPU broadphase + TGS |
| `probe_api.py` / `api.log` | 枚举本机可用接触 API：`get_physx_simulation_interface().get_contact_report / subscribe_*_contact_report_events`；`PhysxSchema.PhysxContactReportAPI` 存在 |
| `probe_report2.py` / `probe_matrix.py` | **可用通道**：人体 19 + 地板 7 双标注 → `get_contact_report()` 返回 5 对，含接触点/法向/冲量 |
| `probe_rest.py` / `rest.log` | 站立 3 s：起步悬空 +0.196 m，末态躺平 pelvis z = 0.166 m，DOF 几乎未动 |
| `probe_untag.py` / `probe_single.py` | **接触归因的决定性证据**：摘掉全部 19 个人体胶囊的 contact-report API → 报告仍为 6 对 / 8 句柄（一对未少）；只标单个胶囊 → 同样 8 句柄；摘掉环境侧 → 6 → 0 → 重挂 → 6。证明 API 按 actor 生效、且无句柄→路径映射 |

复现关键命令：

```bash
# 验收（本机实测 PASS）
~/isaacsim/python.sh scripts/humans/verify.py --out /tmp/s2s-audit/verify

# 接触归因能力探测（证明句柄映射不可行）
~/isaacsim/python.sh /tmp/s2s-audit/probe_untag.py
~/isaacsim/python.sh /tmp/s2s-audit/probe_single.py

# 根姿态约定（不需要 Isaac，秒级）
python3 - <<'PY'
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts/humans")
from common import (load_inputs, DEFAULT_CONFIG, DEFAULT_ASSETS, DEFAULT_MOTIONS,
                    resolve_spawn_point, DEFAULT_SCENE_CONFIG)
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig, forward_kinematics
c, r, _ = load_inputs(config_path=DEFAULT_CONFIG, assets_path=DEFAULT_ASSETS,
                      motions_path=DEFAULT_MOTIONS)
b = select_body(r, model_id=c.skeleton.model_asset, allow_procedural=True)
p = plan_human_rig(c, rest=fit_rest_skeleton(c, b.model.mesh().rest_skeleton()),
                   spawn_xy=resolve_spawn_point(DEFAULT_SCENE_CONFIG, None, None))
poses = forward_kinematics(p, {}, root_position=p.spawn_root_position)
low = min(poses[l.name].transform_point(l.capsule.center)[2]
          - abs(float((poses[l.name].rotation @ l.capsule.direction)[2]))
            * l.capsule.cylinder_length_m / 2 - l.capsule.radius_m
          for l in p.links if l.capsule is not None)
print("spawn_root_position =", p.spawn_root_position)
print("ground_offset_m     =", p.ground_offset_m)
print("lowest capsule at spawn = %+.6f m  (should be 0.000000)" % low)
PY
```

**未验证项（不得写成已验证）**：墙体/家具接触响应试验、台阶交互、
物理回放模式下的接触反作用、多轴关节链在 Isaac 侧的运行、
稳定期改为有界支撑力后的首帧平衡（P0-3）、
以及几何归因在**真正跌倒轨迹**上的召回率（当前只在站立步行试用例上跑通）。
