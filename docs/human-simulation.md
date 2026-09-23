# SMPL 人体、动作与跌倒仿真（阶段 7）

> 2026-09-22 修复复验：CPU/USD/Isaac 分层验收已通过。`stage7-review.md` 保留早期问题证据；当前 Isaac 证据为 [`artifacts/humans/human_verify_repair6.json`](../artifacts/humans/human_verify_repair6.json)。AMASS 本地导入器和摔倒候选筛选已完成并通过合成 `.npz` smoke test，但本机仍没有真实 AMASS 序列；自由站立/行走控制仍未完成。

> **二次验收更新（2026-09-22）**：上一段资产结论已被本机事实纠正。`data/humans/smpl/` 中已有 SMPL v1.1.0 neutral/male/female，neutral 已通过实际加载和静止蒙皮复验；AMASS 原始动作序列仍未取得。逐帧网格导出契约见 [mesh-export.md](mesh-export.md)。

本文件描述阶段 7 的**实际实现边界**：哪些内容已在本机真实运行并通过验收，哪些只有
CPU 验证，哪些因资产未取得而尚未验证。所有结论都对应可复现的命令与产物。

## 1. 结论速览

| 子阶段 | 状态 | 依据 |
| --- | --- | --- |
| 7.1 资产核查与 CPU 数据契约 | **部分完成**：SMPL v1.1.0 neutral 已取得并通过 CPU 加载/静止蒙皮；AMASS 本地导入器已完成，但真实序列未取得 | `scripts/humans/plan.py`；`scripts/humans/import_amass.py`；`artifacts/humans/assets_audit.json` |
| 7.2 单人体导入与基础物理 | **已完成分层验收**：24 连杆胶囊体关节链和实际 SMPL `Human/Skin` USD 网格已写入；CPU/USD/Isaac 结构、映射、重力回落和地面检查通过 | `scripts/humans/build.py`；`artifacts/humans/human_verify_repair6.json` |
| 7.3 动作重定向与受控物理运动 | **部分完成**：运动学回放与关节 PD 跟踪已验证；**自由站立/行走未达标** | PD 最大误差 1.540°（容限 15°，逐 DOF、碰撞隔离）；站立验收见 §6 |
| 7.4 跌倒与易混淆动作验证集 | **部分完成**：受控外力与控制能力下降两类扰动已实现；滑倒/绊倒**按计划推迟** | `configs/humans/human_smpl_neutral.yaml` 的 `perturbations` |
| 7.5 同步真值与后续接口 | **部分完成**：逐帧固定拓扑网格已接入 CPU、USD 和导出契约；Isaac 物理姿态驱动网格的稳定性待复验 | `docs/mesh-export.md`；`GroundTruth` NPZ/JSON |

**未验证的上边界（必须如实传播）**

- **SMPL 已导入 CPU 路径**。neutral v1.1.0 已从本机 pickle 加载，实际读取 6890 顶点、13776 面、24 关节和 300 个 shape directions，并验证静止蒙皮复现。Isaac 逐帧稳定性仍需重跑，不能把 CPU 通过扩大成 Isaac 全部通过。
- **没有使用任何 AMASS 序列**。当前跌倒/躺下等标签来自脚本化参考动作与脚本化扰动，
  不是真实采集动作。
- 人体、动作与标定均在**固定室内场景**中完成，未做体型/动作批量生成（属阶段 9）。

### AMASS 本地导入与摔倒候选筛选

入口为 `scripts/humans/import_amass.py`，只读取用户已下载并解包的本地 `.npz`，不会联网下载。
它校验 `poses (N,156)`、`trans (N,3)`、帧率、`gender` 和 `betas`，记录源文件 SHA-256，
将 SMPL-H 的 52 关节身体块重定向到 SMPL 的 24 关节，并执行 Y-up→Z-up 坐标转换。
`--fall-only` 依据躯干角度、根部下降、向下速度和单轴关节残差筛选候选；这是运动学候选筛选，
不是临床跌倒标签，候选仍需经过 Isaac 物理复核。

```bash
python3 scripts/humans/import_amass.py --root /path/to/AMASS \
  --out artifacts/humans/amass_import --fall-only
python3 scripts/humans/simulate.py --dry-run \
  --amass-root /path/to/AMASS --amass-fall-only
```

本轮用两个合成 `.npz`（一个候选摔倒、一个站立）实跑入口，结果为 `2 sequences scanned, 1 fall candidates`；
CPU 单测为 `200 passed / 8 skipped`。合成结果仅验证格式和管线，不代表已取得真实 AMASS。

官方 AMASS 页面是注册制下载，需要用户自己的 MPI 账号和非商业科研许可。仓库已提供认证下载入口，先列出账号可见的子集，再选择少量数据：

```bash
python3 scripts/humans/fetch_assets.py --site amass --list
python3 scripts/humans/fetch_assets.py --site amass --select CMU Transition --unpack
~/isaacsim/python.sh scripts/humans/view_amass.py \
  --amass-root /path/to/authorized/AMASS --fall-only
```

本轮合成播放复验使用 `amass__fall` 共 241 帧，生成 `artifacts/humans/amass_preview.usda`；
该文件用于检查播放链路，不能替代真实 AMASS 序列。

### 不下载 AMASS 也能在 Isaac Sim 里看动作（2026-09-23 新增）

`view_amass.py` 的 `--amass-root` 现在是**可选**的。省略它就改读内置的
`configs/humans/motions.yaml`，播放的是同一个 `MotionClip` 类型，回放循环一字未改 ——
两种来源只差一个 loader。这是本机（没有 AMASS 授权数据）**今天就能看到摔倒动作**的入口：

```bash
# 单个内置摔倒参考
~/isaacsim/python.sh scripts/humans/view_amass.py --motion fall_forward_reference

# 库里全部 fall_reference，依次播放
~/isaacsim/python.sh scripts/humans/view_amass.py --fall-only

# 任何其它内置动作
~/isaacsim/python.sh scripts/humans/view_amass.py --motion squat
```

`--fall-only` 的判据随来源改变：给了 `--amass-root` 用 AMASS 筛选器，否则用库自身的
`fall_reference` 标签（与 `collect_fall_mesh.py --fall-only` 同一判据）。脚本动作的
provenance 不是 AMASS，而 `screen_amass_clip` 按设计会对非 AMASS 输入抛错，所以不能
对它跑筛选器。输出文件名（`motion_preview.usda` / `amass_preview.usda`）与报告横幅
同样随来源分开 —— 不能让报告写着「AMASS preview」却播着脚本动作。

**kinematic replay 与 P0-3 无关。** 查看器不跑控制环，也不依赖人体能否靠 PD 自己站住；
它每帧直接写 DOF 与根位姿。P0-3 影响的是「物理自己能不能站住」，也就是 `simulate.py`
的自由根试验，不是「能不能看」。

`view_amass.py` 是 **kinematic replay**：每帧直接写入 DOF 和根部位姿，并清零速度；
SMPL `/World/Human/Skin` 是 visual-only，19 个胶囊只是隐藏的碰撞代理。因此查看器适合
确认动作重定向、根位姿、蒙皮与代理的同步，不适合用来判断墙体或地面的接触反作用。
当前物理碰撞验收入口是 `scripts/humans/verify.py`：它已验证地面回落和不穿透；墙体代理
虽已写入场景，但尚未有专门的墙体接触响应试验。后续需增加独立的 physics replay mode，
用 PD target 驱动并保留真实 PhysX 接触，再用实际 link pose 更新蒙皮。

### 人体 GUI 查看修复

此前 `build.py --gui` 没有接入场景查看器的去顶相机，而且没有把已加载的 SMPL 顶点传给 USD，因此会看到屋顶遮挡，Stage 中也只有胶囊碰撞代理。当前入口已修复：`/World/Human/Skin` 写入 6890 个 SMPL 顶点和 13776 个三角面；显示姿态配置让肩臂自然下垂，并将同一 DOF 姿态写入 GUI 的 PhysX articulation。19 个胶囊继续承担碰撞，但在视图中不可见，避免与皮肤重叠。GUI 默认使用 `human` 临时 session layer：按人体边界取景并隐藏 ceiling/lighting fixture；`roofless` 仍可用于整屋斜视。屋顶和胶囊隐藏只影响当前查看会话，碰撞体和磁盘上的基础场景仍保留。

## 2. 目录与入口

```
src/sim2sense_fall/humans/     skeleton / assets / rotations / motion / config
                               rig / events / export / usd_human
scripts/humans/                common.py plan.py build.py simulate.py verify.py
configs/humans/                assets.yaml human_smpl_neutral.yaml motions.yaml
tests/humans/                  CPU 测试（当前全量 200 passed / 8 skipped）
artifacts/humans/              rig 清单、资产审计、验证报告、真值样本
```

| 命令 | 作用 |
| --- | --- |
| `python3 scripts/humans/plan.py` | CPU：资产审计 + 骨架/动作契约 + 刚体规划 + 正运动学校验 |
| `python3 scripts/humans/build.py --dry-run` | CPU：只出规划清单 |
| `~/isaacsim/python.sh scripts/humans/build.py --headless` | 导出带动画学人体的 USD（不覆盖原场景） |
| `~/isaacsim/python.sh scripts/humans/build.py --gui` | 导出并打开 GUI；默认用 `human` 视角聚焦人体、隐藏屋顶并应用自然下垂姿态 |
| `~/isaacsim/python.sh scripts/humans/build.py --gui --view roofless` | 导出并打开 GUI；隐藏屋顶、按整套房间使用室内斜视相机 |
| `~/isaacsim/python.sh scripts/humans/build.py --gui --view top` | 导出并打开 GUI；隐藏屋顶并使用顶视相机 |
| `python3 scripts/humans/simulate.py --dry-run` | CPU：对参考动作做正运动学回放并标注 |
| `~/isaacsim/python.sh scripts/humans/view_amass.py --fall-only` | 在 Isaac Sim 中预览**内置**的摔倒参考动作（不需要 AMASS） |
| `~/isaacsim/python.sh scripts/humans/view_amass.py --motion <id>` | 在 Isaac Sim 中预览任一内置动作 |
| `~/isaacsim/python.sh scripts/humans/view_amass.py --amass-root /path/to/AMASS --fall-only` | 在 Isaac Sim 中预览已授权的原始 AMASS 候选摔倒动作 |
| `~/isaacsim/python.sh scripts/humans/simulate.py --headless --all` | 真实物理试验 + 真值导出 |
| `~/isaacsim/python.sh scripts/humans/verify.py` | 分层验收（CPU / USD / 物理） |

## 3. 骨架与关节映射（7.1 / 7.2）

**已核实的事实**：SMPL 为 24 关节，关节 0 是 `pelvis`，父表为拓扑排序；
SMPL 与 SMPL-H 共享关节 0–21，因此 AMASS（SMPL-H）的身体块可以直接映射；
AMASS 每帧 156 个姿态参数 = 52 关节 × 3，其中前 3 个是根朝向，其后 21 组是身体关节
1–21，其余 30 组是**手指**。结论：SMPL 的 `left_hand` / `right_hand`（关节 22/23）
在 AMASS 里**没有对应项**，本流水线将其保持在静止姿态，并把这一事实写进 clip 元数据
（`dropped_finger_joints` / provenance notes），而不是假装映射完整。

**程序化替代**：`joint_positions` 是手工设定的 1.70 m 成年人站立骨架，**不是** SMPL 数据。
`plan.py` 会输出 `skeleton_source: "procedural nominal rest skeleton (not SMPL model data)"`。
当模型文件存在时，关节中心改由 `J_regressor @ v_template` 求出，
`skeleton_from_kintree_table` 还会把模型自带的 `kintree_table` 与仓库中的常量比对，
不一致就报错而不是静默重定向。

**关节链构造**：每个关节按配置生成 1 个转动关节（`rotations` 有 N 个轴时生成 N 个
串联转动关节，中间插入零长度 proxy 连杆）。默认配置全部单轴，proxy 数量为 0；
多轴路径有 CPU 单测覆盖（`test_multi_axis_joints_become_a_proxy_chunk`），
但**未在 Isaac Sim 中运行过**。

**旋转约定**：SMPL 的蒙皮链是 `G_child = G_parent @ [R_child | J_child − J_parent]`，
即子关节偏移由父关节的已姿态化坐标系承载、旋转轴始终与静止姿态对齐。因此
USD 转动关节只要局部朝向为单位阵、轴标记与身体轴一致、`localPos0` 设为静止骨向量，
就能**逐字复现**该链，不需要额外的坐标系换算——也就不存在能藏住符号错误的换算步骤。

**质量与惯量**：`mass_weight` 是手工设定的相对权重（躯干≈53%、腿≈33%、臂≈14%），
归一化到配置总质量，**不是**拟合的人体测量数据；规划清单里保留了原始权重以便复核。
规划器还会算出实心圆柱近似的惯量用于体检，但 USD 只写质量，惯量交给 PhysX 从碰撞体推导。

## 4. 分层验收（7.2 / 7.3）

`scripts/humans/verify.py` 分三层报告，三层失败的成因不同，不能互相替代。

**CPU 层**：极限值有限且 `low < high`；静止姿态最低胶囊点恰好落在 z=0；
头部高于骨盆；关节运动方向逐项断言。

**USD 层**：米制、Z-up、唯一 articulation root、转动关节数 = 规划自由度（14）、
胶囊数 = 规划碰撞体数（19）。

**物理层（正对照）**：

| 检查 | 结果 |
| --- | --- |
| 配置的物理步长生效 | 请求 1/120 s，引擎报告 1/120 s |
| DOF 名称集合与规划一致 | 14 个全部一致 |
| 关节极限往返 | 最大偏差 7×10⁻⁶ 度（USD 存度、Isaac 报弧度） |
| 自碰撞按配置关闭 | 引擎 `False`，规划 `False` |
| **物理姿态 = CPU 正运动学** | 6 组姿态、24 连杆，最大误差 **0.00 mm** |
| PD 跟踪 | 最大关节误差 1.540 度（预先登记容限 15 度，逐 DOF、碰撞隔离） |
| 抬高后落回（正对照） | 抬高 0.25 m 后骨盆下落 0.279 m |
| 不穿透地板 / 不飘散 | 最低体点 0.0000 m；水平位移 0.011 m |

前两项曾经失败并已定位：

- 一开始把 `(0, 0)` 当作出生点，而公寓是**逐房间**铺地板，该点落在房间之外，
  人体直接穿过世界下落 40 m。现在出生点由场景配置推导：
  在最大房间里按网格采样，取离所有家具最远、且避开墙厚的点
  （`common.scene_spawn_point`，默认结果 `(4.52, 0.52)`）。
- 空中回放对照最初抬到 +2.0 m，即**高于 2.7 m 的天花板**，实测误差 317 mm 来自与天花板
  碰撞；改为 +0.9 m（地板与天花板之间）后误差降到 0。检查代码里记录了这段经历，
  以免后人重犯。
- 对照的参照量也调整过：起初比较的是**指令角**，而驱动器会把关节拉回零位，
  差值被误算成映射误差；现在比较的是**引擎自报的实测关节角**，
  这样测的才是关节坐标系映射，而不是驱动器的跟随能力。

## 5. 动作、扰动与事件标注（7.3 / 7.4）

`configs/humans/motions.yaml` 里是 9 段**解析式**参考动作：站立、弯腰、下蹲、坐下、
原地行走（正弦生成器），以及前倒/后倒/侧倒/伸手失衡四种跌倒参考。
它们的 `provenance.kind` 一律为 `scripted`，不会被当作采集数据。

**参考动作是旋转参考，不是跌倒轨迹。** 跌倒参考的根高度按 `0.5·L·(cos φ − 1)`
下降（L 为骨盆到地面距离、φ 为倾倒角）；完整的刚体绕足翻倒律会让身体横躺后其他胶囊
扫到地板以下，折半后骨盆足够高。就算这样它们仍然是**运动学目标**：
真实的跌倒由物理试验产生，`simulate.py --dry-run` 会打印穿透深度供核对而不是假设为零。

**扰动只有两类真正落地**：

- `force`：对指定连杆施加恒定外力（四向推）；
- `control_failure`：从 `start_s` 起把关节驱动增益按 `control_scale` 缩放。

`support_loss`（滑倒/失去支撑）在配置模式中定义但**刻意不发布**：它需要运行时改写
地板碰撞体或摩擦，本仓库未验证过；阶段计划本身也把滑倒/绊倒推迟到自由站立与行走通过
验收点之后，而该验收点未达到。运行时会把它标记为「未施加」，不会写成已生效的扰动。

**标注规则**（阈值全部写在 `configs/humans/human_smpl_neutral.yaml` 的 `events` 段，
在跑任何试验之前就固定下来，对所有试验一致适用）：

- `imbalance_onset`：躯干倾角首次超过 `trunk_onset_fraction × trunk_angle_deg`，或骨盆
  首次低于站立高度的 `pelvis_height_fraction`；
- `first_impact`：**onset 之后**首次有体点进入离地 `impact_height_m` 且下降速度超过
  `impact_speed_m_s`。从 onset 起算很关键——行走时脚每次落地都会触发这个条件；
- `stabilisation`：撞击之后体点速度连续 `settle_window_s` 低于 `settle_speed_m_s`。

躯干倾角取自**骨盆到颈部的实测方向**，不是根朝向：弯腰时骨盆可以不转，
只看根朝向会把「弯着腰」判成直立。这一点有专门测试（`test_trunk_axis_sees_a_bend_the_root_cannot`）。

标签取值与含义：

| 标签 | 含义 |
| --- | --- |
| `fall` | 姿态在 `max_transition_s` 内进入低位、保持 `min_low_frames` 帧、且发生撞击、且最后仍低位 |
| `controlled_lowering` | 进入低位但下降缓慢（`< controlled_descent_speed_m_s`）且无撞击：下蹲、坐下、主动躺下 |
| `recovered` | 失衡后回到直立 |
| `no_fall` | 从未失衡，或只是部分倾倒未达低位 |
| `invalid` | 发散或穿透超过 `penetration_limit_m`，排除而不标注 |

另有独立的 `final_posture` 字段（`upright` / `lying`），与标签正交：
「主动躺下」是 `controlled_lowering` + `lying`，「下蹲」是 `controlled_lowering` + `upright`。

**这些阈值是项目自定义的启发式规则，不是经过临床或生理验证的跌倒判据**，
不得被描述为已验证的跌倒检测器。

## 6. 明确未达标项

- **自由站立与行走控制未达标**：单轴矢状面关节 + 中等刚度 PD 的自由根人体在重力与接触下
  会倒下（本次验证中抬高后骨盆下落 0.279 m 并未回到原位）。这不是 bug，而是阶段计划
  预设的结果：人体导入成功**不以**自然站立/行走为条件。要达标需要多轴关节与更强的平衡
  控制（或模仿学习 / Isaac Lab），属于后续工作。
- **7.3 的「受辅助跟踪」与「无辅助自由运动」必须分开报告**：本次只做了锚定/空中条件下的
  跟踪与回放，没有声称无辅助自由运动达标。
- **多轴关节链未在 Isaac Sim 中运行**。
- **`support_loss` 扰动未实现**。
- **无模型时的胶囊体仍只是代理**；新导出会标记为 `capsule_proxy_mesh`，真实模型路径标记为
  `smpl_skin_mesh`，两者都按**物理姿态**生成世界坐标顶点。

## 7. 真值导出契约（7.5）

每次试验导出两个文件：

`<name>.npz`（数组，`allow_pickle=False`）

- 两条时间轴：`time_physics_s`（物理步）与 `time_channel_s`（无线采样率），
  各自的帧数、插值方法与两个速率一起记录在 json 里；
- `root_position` / `root_quaternion_wxyz`、`joint_positions_rad` /
  `joint_velocities_rad_s`、`link_positions`（逐时刻**物理**姿态）、
  `body_points` + `body_point_owners`；
- `reference_joint_positions_rad`（控制器被要求跟踪的参考，便于算跟踪误差）；
- `mesh_vertices_xyz`、`mesh_faces`、`channel_mesh_vertices_xyz`（固定拓扑、世界坐标、米制、Z-up）；
- `contact_force_n`（若运行时能取到，见下）与 `phase_label_index`。

`<name>.trial.json`（可读记录）

- `provenance`：场景与场景哈希、人体 id、刚体规划哈希、配置哈希、动作 id/类型/哈希、
  控制模式、根模式与是否使用了世界锚定、体表表示方式、扰动细节（含**是否真的生效**）、
  seed、物理步长、无线采样率、插值方法、站立高度、总质量、生成脚本、许可说明；
- `label`：标签、有效性、`final_posture`、onset / impact / stabilisation / 过渡时间、
  峰值与末态躯干角、峰值下降速度、最低骨盆高度、最低体点、判据与全部证据数值。

**数据划分**：`export.group_by` 为 `[subject, sequence]`，每条试验输出一个
`split_key`；按该键分组即可保证同一人物或同一原始序列不会跨训练/测试边界。
本阶段的脚本化动作用 `subject=scripted`，接入 AMASS 后自动带上真实人物与序列标识。

**接触力**：Isaac 的接触视图必须在构造 `RigidPrim` 时传 `contact_filter_paths` 与
`max_contact_count` 才有效，且只对**地板/找平层**（`sim2sense:category` 为
`floor` / `ground`）开启——跌倒撞击正是相对支撑面定义的。取不到时
`contact_force_n` 为空数组、能力与原因写入 `trials_index.json`，不会伪造零值。
事件判定本身只用轨迹，不依赖接触力。

## 8. 尚未验证 / 待办

1. **Isaac 运行时修复已通过**：repair3 曾因控制试验中的人体碰撞和传送后残余速度污染结果，出现 PD 57.064° 与非有限 PhysX 变换；repair6 已逐 DOF 隔离碰撞、重置速度并通过，最大误差 1.540°。GPU/CUDA 与接触力视图仍是环境/可选能力边界。

2. **SMPL 资产已取得并接入 CPU/USD**：`data/humans/smpl/` 中 v1.1.0 neutral/male/female 可加载；neutral 已验证 6890 顶点、13776 面、24 关节、300 个 shape directions，CPU 静止蒙皮漂移为 `2.22e-16 m`。`build.py --headless` 会写入 `Human/Skin`，而碰撞仍使用胶囊体。待 Isaac 数值稳定后，再核对每帧物理 link pose 到 USD/NPZ 网格的时间对齐。

3. **旧批次索引已迁移**：`scripts/humans/migrate_trials_index.py --write` 已将两个 `trials_index.json` 的 `physics_dt_s` 修为 `1/120 s = 0.008333333333333333`，并增加 `physics_hz: 120.0`；脚本会校验被引用 trial JSON 的 provenance。

   官方下载页（已核对）只提供三项：`1.0.0 for Python 2.7 (female/male, 10 shape PCs)`、
   `1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)`、`UV map in OBJ format`。
   由此有三条必须记住的结论：

   - **v1.0.0 没有 neutral 模型**，只有 female/male；neutral 是 v1.1.0 才加入的。
   - **v1.1.0 带 300 个 shape PC**（不是 10），因此它是本项目的首选：
     阶段 9 的体型随机化轴需要它。
   - **两者都是 Python 2.7 的 pickle**。加载器因此会先试 ASCII、失败后用 `latin1` 重试，
     并把实际生效的编码写进日志；`shapedirs` / `posedirs` 常见为 `chumpy` 包装，
     加载器用替代类解包底层 numpy 数组，解包不出来时给出「先用参考加载器转换一次」的
     可执行建议，而不是让流水线静默依赖已停止维护的 chumpy。
   - 下载页**不公布文件名**。登记表把文件名写成候选列表（`filename_candidates`），
     所以命名差异会表现为"以另一个名字找到"，而不是"资产缺失"——后者与"没下载"
     完全无法区分。
4. 取得 AMASS 后先做**候选动作筛选**：本仓库不预设 AMASS 有足够跌倒序列或可靠标签，
   筛选结果必须先记录再加进验证集。
5. 多轴关节链的 Isaac 侧验证；`support_loss` 的实现与验证。
6. 自由站立/行走控制（多轴关节 + 平衡控制，或模仿学习）。
