# 项目长期约定（Sim2Sense-Fall）

## 环境与工具链

- Isaac Sim 6.0.1-rc.7：`/home/gsh/isaacsim`。GUI 用 `./isaac-sim.sh`，脚本用 `./python.sh`。
  命名空间 `isaacsim.*`（`omni.isaac.core` 不存在；用 `isaacsim.core.simulation_manager` 与
  `isaacsim.core.experimental.*`）。`pxr`/`omni.*` 只有 `SimulationApp` 启动后才能 import。
- 系统 Python 3.12 无 pip；`ruff` 用 `uv tool run ruff`。GPU：RTX 4060 Laptop 8 GiB，`DISPLAY=:0`。

## 代码约定

- `src/` 布局、类型标注、`from __future__ import annotations`、行宽 100；数据边界用
  frozen dataclass，非法值立即 `ValueError`；`pathlib.Path`；seed 显式传；库代码用 logging。
- 重运行时（Isaac/Sionna）惰性导入，CPU 上 import 不能失败；每阶段要有 CPU `--dry-run`。
- 生成物写 `artifacts/`（已 gitignore）；未实际运行的检查不得写成通过，失败记 `docs/progress.md`。
- 判「物理有效」必须正向对照（抬高后落回），只断言漂移 0 会把「物理没跑」误判为通过。

## 场景约定

- 参数化图元拼装，不用在线素材库；材质同时定义渲染/力学/电磁（ITU-R P.2040 + Sionna 名）。
- `ScenePlan`（CPU 纯 Python）是消费边界；动态家具是组合刚体（根 Xform 承载 RigidBodyAPI + 单一质量）。
- Z-up、米、`/World` default prim；prim 带 `sim2sense:roomId/category/...` 属性。

## 人体侧核心约定

- `src/sim2sense_fall/humans/`；`usd_human.py` 是唯一 import `pxr` 的模块。
- **关节旋转**：USD 转动关节局部朝向单位阵 + `localPos0`=静止骨向量 即复现 SMPL 链；
  `rig.forward_kinematics` 是独立 CPU 实现，物理必须与它比对（<5 mm，当前 0.00 mm）。
- **单位**：USD 关节 limits/target 是**度**，Isaac `Articulation` 报**弧度**；`physics_dt_s`（步长）
  与「速率」是两回事，混用差 120 倍。
- **Isaac 物理 API 坑**：张量视图只在时间轴 play 后有效；`get_world_poses()` 只返回根连杆，
  逐连杆用 `RigidPrim`；阻尼在 `PhysxSchema.PhysxRigidBodyAPI`（`physxRigidBody:linearDamping`）。
- **接触通道**：`RigidPrim(contact_filter_paths=...)` 在本机必然 AssertionError（tensor 视图坏），
  已从 simulate/verify 关掉（请求它=每次 4368 行 [Error]）。用
  `omni.physx.get_physx_simulation_interface().get_contact_report()` 轮询，两侧挂
  `PhysxContactReportAPI`；归因走**几何包含测试**（报告的 actor 句柄不透明、无 API 映射回
  prim 路径、报告落后一个物理步）。详见 `docs/physics-interaction-audit.md`。
- **两种根高度不能混**：`ground_offset_m`（体坐标系抬到最低胶囊 z=0）vs `spawn_root_position`
  （pelvis 世界坐标）。P0-1 已修：ground_offset 由 `forward_kinematics` 求，出生点误差 −3.7e-7 m。
- **辅助必须显式记录**（`settle_used_root_support` / `root_pinned_during_trial` /
  `root_reference_tracked`）；扰动≠标签，标签只用轨迹（阈值在配置 events 段，先于试验）。
- 24 连杆只有 19 胶囊（foot/head/hand 叶关节无碰撞体）；门禁排除的试验标 invalid/[SKIP] 是正常结果。
- 资产注册制不可再分发：`data/humans/`（gitignore）或 `SIM2SENSE_HUMAN_ASSETS`；缺失时抛
  带注册地址的错误，不自动下载。`body_representation` 只有 `capsule_proxy_surface` /
  `smpl_skin_mesh`；`fidelity` 只有 `kinematic_replay` / `physics_trial`，不得混入同一清单。

## 坐标系约定（2026-09-23 偏航缺陷修复后）

- SMPL 文件帧 `X=左右/Y=上/Z=前` ≠ 管线帧 `X=前/Y=左/Z=上`，差 90° 偏航；整帧导入用
  `body_frame_conversion(up=, forward=, left=)`（det=+1 拒绝镜像），`up_axis_conversion`
  只留给只需 up 的路径。证据记 `SmplModel.source_frame`。
- extent 判不了姿态（T-pose 臂展>身高）；判姿态问解剖问题（头在脚上方吗）。新增检查必须配
  正向对照（mesh 偏航 90° 后必须 FAIL）。`fit_mesh_to_rest_joints` 配对取共用关节名，
  确认用关节间距离，屏蔽自距。详见 `docs/mesh-orientation-defect.md`。

## AMASS 资产与下载

- `amass.is.tue.mpg.de` 与 `smpl.is.tue.mpg.de` 是两套独立注册。端点支持 Range 续传（206+
  Accept-Ranges）。优先 `src/sim2sense_fall/humans/fetch.py::Session`；手工 curl 三规则：
  `-C` 与 `-r` 互斥；`--retry 0`（重传按 stat 重算 Range）；一个 part 一个写入者。
- `load_amass_library` 用 rglob 收全部 npz，`--root` 指向只含 `*_poses.npz` 的目录。
- 本机已有 2198 条（CMU 2088 + Transitions 110），去重切片 2146。

## 当前状态与卡点（2026-09-24 更新）

- 已跑通：固定公寓 + SMPL 有限 PD 静态站立/后推物理跌倒 → 完整公寓 Sionna 复数 CIR（阶段 8 smoke）。
- 手臂 T-pose 已修（肩轴 y→x、肘 y→z；站立横向半跨度 91.3→22.3 cm）。
- **原「clip→DOF 多轴分解」卡点已解除**：现役 rig 是**多轴链** 57 DOF / 62 连杆 / 22 碰撞体，
  每个记录关节展开成 3 个转动关节（主摆轴**最后**）。DOF 名后缀 `__dof1/__dof2` **不是**旋转轴，
  轴必须从 `plan.joints[i].axis` 读；按 `name.split("__")[0] == joint` 归组链。
  旧「14 DOF 单轴 / 0 条可表达」结论作废。
- **P0-A 右臂摆动已复现、定位并实施修复（09-24）**：源 AMASS 窗口本身右臂摆幅小，
  管线忠实传递（PhysX 实际与目标差 < 0.1°）。修复＝换源序列（`CMU/08/08_04`、
  `CMU/08/08_11`）。**但第 4 项判定为「未通过」**：实跑分段后前进肩 R/L 0.516
  （门槛 0.75）、肘 0.292（门槛 0.55）；后退 0.546 / 0.256。跳变与扭转已通过
  （1.20×→1.00×、扭转 ≤0.478×）。新线索：CPU 层肩比 2.332/1.426 而实跑仅 0.516/0.546
  → 差距在实跑时被 PD 跟踪/根辅助/接触吃掉，转 P0-B。见 `docs/arm-swing-audit.md`。
- **聚合口径必须按「被控动作」分段**：整段 `np.ptp` 会把不同动作的极值相除
  （曾得出肩比 0.828 / 肘比 0.731 的假通过）。工具
  `artifacts/humans/arm_capture_motions/walk_strip.py` → `<run>/motion_analysis.json`，
  回归 `tests/humans/test_motion_swing_analysis.py`。转向不推进步态相位，
  摆动判定按 `SWING_COMMANDS={(1,0),(-1,0)}` 限定前进/后退。
- **摆幅分数的分母不能用窗口自身极值**（递减窗口的收尾分数按构造恒为 0）：
  参考幅度与参考**谷值**必须成对传入（`reference_span_deg` + `reference_trough_deg`）。
- **步态周期两种时钟不要混**：源片段 **1.200 s**（145 帧）＝一个周期；
  stride 1.286 m；**3.214 s 是 1.286 ÷ 0.4 m/s 的墙钟播放时间**。
  `report["cycle_period_s"]` 是**一次 demo 演示**的长度，不是步态周期。
- 新发现独立缺陷：`keyboard.yaml` 的 `speed_m_s: 0.4` vs 源片段自身 1.11 m/s → 播放 0.36×，
  动作慢约 2.8×，直接抬高脚底滑速。不是右臂不对称的原因。
- 其他剩余：视口姿态显示（逐帧写 stage 皮肤会作废 PhysX tensor 视图，已回滚）、自然行走/
  平衡恢复、多方向有效跌倒与网格穿地修复、动态家具同步、人体 EM 校准、50 Hz 数据集。
- 人体出生点离墙 0.52 m（P2-1 未做）；`view_amass.py` 是 kinematic replay，physics replay
  mode（PD target 驱动 + 真实接触 + 实际 link pose 驱动蒙皮）待实现。
- CPU 307 passed / 9 skipped；bundled USD 9 passed。当前 git HEAD `a79bef6`，
  分支 `feature/fall-mesh-capture`。

## 方向与判据约定（2026-09-24 用户明确）

- **主方向**：SMPL+AMASS 控制人体在 **Isaac Sim** 里完成各种动作（行走/蹲下/起立/摔倒）。
  这才是实际使用的内容；其余都是为它服务的。
- **判据必须是模拟器实测，不能是几何代理。** 用户明确：**没必要为了可以 CPU 测试丢失准确性**。
  同类问题一律优先读 PhysX 的接触报告、记录的实际关节/根状态、实际渲染，
  把 CPU 侧的 dry-run / FK / 几何代理降级为**预筛**，不作为验收。
  几何代理可以留作快速回归，但凡与实测冲突**以实测为准**，并在文档里标明代理的失效边界。
- 反面教材（本轮）：`docs/support-mask-audit.md` 用「胶囊最低点相对该脚周期最低点」判「脚在地面上」，
  先把**参考**当身体量，得出「44% 掩码帧离地、12% 明确悬空」；换成 PhysX 实际接触后
  （`docs/mask-vs-contact-audit.md`）误报其实只有 1-5 mm（阈值级），
  而真正的病是**漏报**：右脚真实地板接触 **46-56%** 不被掩码认作支撑。
  代理没发现主症、还夸大了次症。
- 现有 CPU 侧仍保留的价值：`--dry-run`、schema/配置校验、单元测试、`audit_stance_ik.py`
  这类消融扫描。它们**不构成**对物理行为的结论。

## 支撑掩码缺陷（2026-09-24 实测，待修）

- `load_gait` 用**水平踝位移**推支撑相 + `|stance_speed − gait.speed_m_s| ≤ 0.15` 门。
  实测（PhysX 接触为准，四条会话）：**右脚真实地板接触 46-56% 不被认作支撑**（左脚 10-31%）；
  误报 16-28% 但脚只差 1-5 mm。→ 锚点从不下在右脚真实支撑上。
- 参考侧同一方向的独立证据：前进片段右脚退行速度比左脚快 **25%**（0.966 vs 0.721 m/s），
  而 `gait.speed_m_s` 合并中位 0.777 **由左脚定出**。
- 已实施开关 `keyboard.yaml::anchor_from_contact`（**出厂 false**）：
  true 时用 `contact_control.measured_support_feet()`（读 PhysX 接触）替换模型的
  `supporting_feet`；空集不回落。A/B 见 `artifacts/humans/anchor_from_contact/{off,on}/`。
- 另：`root_assist.yaml` 新增 `max_vertical_lift_fraction_of_weight`（**出厂 1.0**），
  用来夹住执行器总向上力；0.7 实测是回退（地面 122→212 N 但滑动 22%→38%、关节误差 5.70→8.71°）。
- 工具：`audit_friction_budget.py`（摩擦/载荷预算）、`audit_support_mask.py`（掩码几何）、
  `audit_mask_vs_contact.py`（掩码对实测接触）。前两个是预筛，第三个是判据。
