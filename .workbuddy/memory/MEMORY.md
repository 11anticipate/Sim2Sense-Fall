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

## 当前状态与卡点（2026-09-23）

- 已跑通：固定公寓 + SMPL 有限 PD 静态站立/后推物理跌倒 → 完整公寓 Sionna 复数 CIR（阶段 8 smoke）。
- 手臂 T-pose 已修（肩轴 y→x、肘 y→z；站立横向半跨度 91.3→22.3 cm）。
- **核心卡点：clip→DOF 多轴分解**。planner 已支持多轴链，映射器（AMASS 逐关节旋转→当前
  14 DOF 单轴关节）尚未分解；抽样 120 条得 28 条跌倒候选，0 条可被当前单轴 rig 表达，
  卡点全在肘/肩。
- 其他剩余：视口姿态显示（逐帧写 stage 皮肤会作废 PhysX tensor 视图，已回滚）、自然行走/
  平衡恢复、多方向有效跌倒与网格穿地修复、动态家具同步、人体 EM 校准、50 Hz 数据集。
- 人体出生点离墙 0.52 m（P2-1 未做）；`view_amass.py` 是 kinematic replay，physics replay
  mode（PD target 驱动 + 真实接触 + 实际 link pose 驱动蒙皮）待实现。
- CPU 246 passed / 8 skipped；bundled USD 9 passed。
