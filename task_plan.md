# Sim2Sense-Fall 项目计划

## 目标

建立一个可复现、可扩展的室内无线信道跌倒检测研究仓库：以 Isaac Sim 生成受物理约束的人体运动与场景状态，以 Sionna RT 生成 CIR/CSI 等无线观测，再训练可解释、可评测的跌倒检测模型，并保留从仿真到真实数据的验证边界。

## 阶段

- [x] 阶段 1：检查项目目录、协作规范、Git 状态和本地 Zotero 文献
- [x] 阶段 2：核对核心文献，形成项目相关的证据记录和研究路线
- [x] 阶段 3：建立仓库目录、Python 工程配置、数据契约和最小代码骨架
- [x] 阶段 4：编写 README、架构说明、路线图和阶段进度文档
- [x] 阶段 5：搭建 Isaac Sim 室内场景（卧室/客厅/卫生间/厨房/次卧/走廊），配置墙体、地板、家具属性并导出 USD
- [x] 阶段 6：同步 `git@github.com:11anticipate/Sim2Sense-Fall.git` 并核对上游内容
- [ ] 阶段 7：接入 SMPL 人体，完成骨架、PhysX 动作与跌倒仿真、同步真值导出的最小闭环（SMPL/Isaac 分层验收已通过；AMASS 导入器与候选筛选已完成，真实序列与自由站立/行走仍未完成）
      （**SMPL v1.1.0 neutral 已取得并通过 CPU 资产/静止蒙皮复验；AMASS 原始序列未取得，已用合成 `.npz` 完成导入 smoke test**）
- [ ] 阶段 8：接入 Sionna RT，完成首个端到端 smoke test（场景与动态人体 → CIR/CSI 样本）
- [ ] 阶段 9：用显式 `seed` 驱动场景、体型与动作随机化，形成训练域族

## 关键问题

1. 仿真人体轨迹、动态网格与无线信道样本如何用统一时间戳对齐？
2. 如何区分跌倒检测能力与场景、硬件、个体的域偏移？
3. 哪些指标能够覆盖漏报、误报、报警延迟和仿真到真实的性能变化？
4. 生成数据和真实采集数据如何在隐私、许可和可复现性上分开管理？

## 已做决定

- 本地 Zotero 和随附 PDF 是当前文献事实源；网络元数据在网络恢复前不作为已核实事实。
- 代码采用 `src/` 布局、类型标注、`ruff`、`pytest` 和配置文件驱动。
- 原始数据、生成数据、模型权重和渲染缓存默认不入 Git；仓库只保留小型示例、元数据和生成说明。
- 先实现稳定的数据契约与 dry-run，再接入重量级 Isaac Sim / Sionna 运行时。
- 场景几何用参数化图元拼装，不依赖在线素材库；同一材质同时定义渲染、力学和电磁三套属性。
- 场景规划（CPU）与 USD 落地（Isaac Sim）分离：规划可脱离 Isaac Sim 单测和评审，USD 只消费规划结果。
- 2026-09-22：用户确定采用 **SMPL** 作为人体模型，先完成人体与动作物理闭环，再接入完整 Sionna 链路。AMASS 作为动作参考来源，需核实版本并显式适配到 SMPL；不默认切换到 SMPL+H/SMPL-X，也不将 Isaac 动画人物作为主人体模型。
- 人体采用三层结构：SMPL 骨架与蒙皮网格、PhysX 分段刚体与关节、动作跟踪与平衡控制。运动学回放、被动 ragdoll 和受控物理运动分别标记，不能混称为物理真实动作。

## 错误与阻塞

- 2026-09-21：**已解除** — 早前 `github.com` DNS 解析失败；现在 `git ls-remote origin` 正常，
  `origin/main` 停在 `c77a37d`，与本地 `main` 完全一致（0 ahead / 0 behind）。
- 2026-09-21：**已解除** — 早前工作树 `.git` 是只读空目录；现在 `.git` 可写，存在 `main`
  分支与提交 `c77a37d`，且 `origin` 已指向目标仓库。场景工作已提交到
  `feature/indoor-scene` 分支，`main` 保持不动。详见 [`docs/progress.md`](docs/progress.md)。
- 2026-09-21：独立运行的 Isaac Sim 不会把 PhysX 挂到 USD stage 上，物理步数在涨但没有任何物体运动。已封装 `activate_physics()`（`enable_all_default_callbacks` + `setup_simulation`）修复，并加入「抬高后落回」的正向对照测试，避免把「物理没跑」误判成「场景稳定」。
- 2026-09-21：上游仓库**没有 LICENSE 文件**，默认即「保留所有权利」。在明确许可前不应假设
  代码可以对外分发或复用。
- 2026-09-21：**未做**推送。`feature/indoor-scene` 只存在于本地，是否推送与开 PR 待用户确认。

## 当前状态

### 摔倒 Mesh 采集（2026-09-23，分支 `feature/fall-mesh-capture`）

- [x] 修掉 SMPL 导入被偏航 90° 的真缺陷（文件帧 `X=横/Y=上/Z=前` vs 管线帧
      `X=前/Y=左/Z=上`；旧的 `up_axis_conversion` 只能保证 up 不变，表达不了这个偏航）。
      详见 [`docs/mesh-orientation-defect.md`](docs/mesh-orientation-defect.md)。
- [x] 修掉 `fit_mesh_to_rest_joints` 按行号裸 `argmin` 配对的隐患（≥1.75 m 时非双射，
      静默给出错误体尺）。改为按共用关节名配对 + 关节间距离确认 + 相似性断言。
- [x] 导出闸门补上两条「绕垂直轴旋转看不掉」的检查（俯仰 + 朝向），并用正向对照
      （把 mesh 绕 Z 偏航 90° 必须 FAIL）证明它们真的会失败。
- [x] `--fall-only` 采集 4 个摔倒片段的人体 3D Mesh + `(x, y, z)` 序列到
      `artifacts/humans/fall_mesh/`；`fidelity: kinematic_replay`。
- [x] 独立第二实现复核（`verify_fall_collection.py`，不 import 采集器）+ 目视确认图。
- [ ] 用 `simulate.py` 的 `physics_trial` 采集**带动力学**的摔倒轨迹（被 P0-3 卡住）。
- [ ] AMASS 真实序列导入（`--write-slice` 已实现，未对真实树跑过）。

### 人体 GUI 查看修复（2026-09-22，已完成）

- [x] 确认截图原因：人体构建入口未调用场景去顶相机；SMPL 网格未传入 USD 导出。
- [x] 接入临时去顶视角、真实蒙皮和静态预览，保留完整场景与碰撞。
- [x] 修复 T-pose：配置驱动的左右肩下垂和轻微肘弯同时用于 SMPL 蒙皮和 GUI PhysX DOF。
- [x] 隐藏 19 个内部胶囊的渲染可见性并保留 CollisionAPI，避免碰撞代理与皮肤重叠。
- [x] 新增 `human` 聚焦视角并设为人体 GUI 默认视角；物理房间尺寸保持不变，画面按人体边界取景。
- [x] 完成 CPU/Isaac USD 结构检查并同步查看命令；当前隔离环境 GPU/GUI 画面仍需在宿主机显示会话运行。
- [x] 修复 GUI 首次启动时 PhysX tensor 尚未初始化的问题：`play()` 后等待 4 次 Kit update 再写入显示 DOF。

**室内场景 R1–R6 整改完成，已通过 CPU、实际 USD 和物理复验**：`configs/scenes/indoor_apartment.yaml` 描述的 6 房间住宅可导出为
`artifacts/scenes/indoor_apartment.usda`（232 图元 / 231 碰撞体 / 14 材质），
当前验收器 10 项检查通过（CPU 回退），失败负例实际返回非零；详见 `docs/indoor-scene-remediation.md`。构建与 GUI 查看指令见 [`docs/indoor-scene.md`](docs/indoor-scene.md)。

**2026-09-22 复审修复状态：阶段 7 的 CPU/USD/Isaac 分层验收已通过。** 资产审计已纠正：项目内存在并已加载 SMPL v1.1.0 neutral/male/female 三个模型，neutral 的 6890 顶点、13776 面、300 shape directions 和静止蒙皮复验通过；AMASS 原始序列仍未取得。PD 验收已隔离人体碰撞并重置传送后的速度，最新 headless Isaac 运行通过，14 个 DOF 最大跟踪误差 1.540°（容限 15°），驱动关闭负对照、重力回落、地面穿透和场景哈希检查均通过。历史问题与证据仍见 [`docs/stage7-review.md`](docs/stage7-review.md)。固定室内场景中的 CPU/USD 数据链路
「一个 SMPL 拓扑人体 → 动作/跌倒 → 同步骨架、几何与接触真值」已跑通；Isaac 最新分层验收已通过：
24 连杆 / 14 自由度 / 19 胶囊体，物理姿态与 CPU 正运动学逐连杆一致（0.00 mm，24 连杆），
`scripts/humans/verify.py` 的 CPU/USD/Isaac 前置检查通过。真值试验两批：自由根动力学 7 项（fall=2、no_fall=5）、
带记录的骨盆钉定辅助 6 项（no_fall=4，2 项被有效性门禁排除）。

**当前资产边界：SMPL 已取得并可加载，AMASS 仍未取得。** 真实 SMPL 路径现在导出
`smpl_skin_mesh`；无模型时才回退为明确标记的 `capsule_proxy_mesh`/`capsule_proxy_surface`。
动作仍为 scripted，不能声称已经完成 AMASS 动作接入。逐帧网格契约见
[`docs/mesh-export.md`](docs/mesh-export.md)。

## 室内场景子任务

- [x] 创建卧室、客厅、卫生间、厨房、次卧及走廊配置，设置静态碰撞、材质与摩擦属性。
- [x] 创建实际 Python 文件、CPU dry-run 和 USD 导出入口。
- [x] 运行验证并记录结果，提供本机 GUI 启动指令。

## 下一步

1. 用已通过的 Isaac 分层验收继续检查逐帧 mesh 的有限性、拓扑和时间对齐；同时取得 AMASS 后做候选动作筛选并记录文件哈希、子集、人物、序列、帧率、坐标和许可。
2. 用 CPU 已验收的逐帧 `mesh_vertices_xyz`/`mesh_faces` 真值接入 Sionna RT（阶段 8）：把环境与逐时刻人体几何
   映射成传播场景，生成首条 `ChannelSample`；同时复核场景代理电磁材质，并独立定义人体电磁属性。
3. 自由站立/行走控制（多轴关节 + 平衡控制，或模仿学习），以及多轴关节链与 `support_loss`
   的 Isaac 侧验证。
4. 在固定场景闭环通过后，用显式 `seed` 扩充布局、材质、体型、动作及扰动变化，并建立无人物/动作来源泄漏的数据划分。

## 阶段 7 — SMPL 人体与 PhysX 动作仿真

### 7.1 资产核查与 CPU 数据契约

- [x] 核实 SMPL 模型文件的版本、来源、许可与可用性；模型参数及原始人体资产不入 Git。
      已核实：本机 `data/humans/smpl/` 存在 v1.1.0 neutral/male/female，neutral 已实际加载；SMPL 与 AMASS 均为 MPI 注册制非商业科研许可，流水线不自动下载；
      文件名、pickle 布局等次要事实来自公开二级资料，注册后需复核（`verified: false`）。
- [ ] 筛选少量 AMASS 短动作，记录来源子集、原始序列、人物、模型表示、帧率和许可；不预设存在足够的跌倒序列或可靠动作标签。**导入器、来源哈希、Y-up→Z-up、SMPL-H→SMPL 重定向和候选筛选已完成；当前资产根中未发现真实 AMASS 序列，待用户提供已授权数据**。
- [x] 明确 AMASS 表示到 SMPL 的兼容关系及转换/重定向方案，核对关节顺序、局部旋转和静止姿态，不直接混用不同人体模型的姿态参数。
      已实现并从公开来源核对：24 关节拓扑与父表、SMPL/SMPL-H 共享 0–21、AMASS 156 参数布局、
      SMPL 手部关节 22/23 在 AMASS 中无对应项（保持静止并记录）、Y-up→Z-up 变换。
- [x] 定义不可变人体/动作数据结构与配置：模型版本、体型、骨架映射、根位姿、关节旋转、单位、坐标系、时间戳、动作来源及 seed。
- [x] 提供无需 Isaac/模型下载即可运行的小型合成骨架 dry-run，检查骨架拓扑、有限数值、维度、旋转合法性、时间连续性和可复现性。
      `scripts/humans/plan.py`：全部 PASS；52 项 CPU 测试覆盖。

交付与验收：资产清单、格式契约、版本适配说明和 CPU smoke test；缺失或不兼容输入应明确失败。当前 neutral SMPL 资产已通过 CPU 加载和静止蒙皮复验，AMASS 仍未验证。

### 7.2 单人体导入与基础物理

- [x] 将骨骼与已取得的 SMPL neutral 网格写入 USD，检查尺度、朝向与静止姿态。
      当前 USD 构建会写入 `Human/Skin`（6890 顶点、13776 三角面）；碰撞仍由 19 个胶囊体承担。
- [x] 建立 SMPL 骨架到 PhysX 关节结构的显式映射；用胶囊体构建身体分段碰撞体，配置质量、惯量、关节轴/限制和自碰撞过滤。
      24 连杆 / 14 自由度 / 19 胶囊体；USD 中为 14 转动关节 + 9 固定关节 + 1 articulation root。
- [x] 质量分配、碰撞形状、接触参数和求解步长均由配置驱动，记录参数依据与近似边界。
- [x] 执行关节活动范围、单关节驱动、重力下落、支撑接触与室内碰撞检查，监测穿透、关节约束误差和数值发散。
      CPU/USD 前置检查通过；2026-09-22 repair6 Isaac 完整验收通过（证据：`artifacts/humans/human_verify_repair6.json`）。物理姿态与 CPU 正运动学在 24 连杆上误差 0.00 mm，PD 最大误差 1.540°。

交付与验收：一个可查看骨架、蒙皮及碰撞体的人体资产；自由下落作为物理正向对照，并验证实际物理姿态驱动蒙皮。此阶段允许被动倒下，不将自然站立/行走作为导入成功条件。

### 7.3 动作重定向与受控物理运动

- [x] 先验证动作重定向及运动学回放，再通过有限驱动力的关节控制跟踪参考动作。
      回放逐位精确；Isaac PD 跟踪最大误差 1.540°（预先登记容限 15°，逐 DOF 试验并隔离人体碰撞）。
- [ ] 按站姿与重心调整、弯腰/下蹲、坐下/起立、行走/转身的顺序推进，检查脚部滑移、接触、轨迹误差与平衡状态。
      **部分完成**：弯腰、坐下、原地行走在带辅助条件下可执行；下蹲因驱动器不足被有效性门禁排除。
- [x] 根部固定、辅助力或调试支撑必须显式记录；区分受辅助的跟踪测试与无辅助的自由运动。
      稳定期骨盆托持与试验期骨盆钉定分别记录为 `settle_used_root_support` / `root_pinned_during_trial`；
      自由根试验另记 `root_reference_tracked`。
- [x] 将自由站立与行走设为独立控制验收点；若基础控制无法达标，再评估模仿学习控制器与 Isaac Lab，不把大规模训练设为人体导入的前置条件。
      **结论：未达标**。单轴矢状面关节 + 中等刚度 PD 的自由根人体在重力下会倒下（抬高 0.25 m 后骨盆下落 0.279 m
      且不回原位）。按计划这不构成人体导入失败，但必须如实报告为未达标。

交付与验收：少量可追溯的日常动作及控制配置；分别报告回放与物理跟踪结果。参考动作不能直接替代真实仿真输出；误差指标、容差和失败判定须在验收运行前定义并保存。

### 7.4 跌倒与易混淆动作验证集

- [x] 先构建前倒、后倒、侧倒，以及坐下、躺下、弯腰、下蹲等日常动作小样例；保留失衡后恢复的情况。
      `configs/humans/motions.yaml`：9 段解析式参考动作，tags 区分 confusable 与 fall_reference。
- [x] 通过受控外力和控制能力下降触发失衡。**支撑丢失未实现**：需要运行时改写地板碰撞或摩擦，未验证；
      按计划推迟到自由站立/行走通过之后（该验收点未达标）。运行时会把它标记为未施加而非静默忽略。
- [x] 保存初始姿态、扰动位置/方向/时刻、控制强度及接触过程；依据实际轨迹标注事件，不把施加扰动直接当作跌倒标签。
      证据：四向推动试验全部标为 `no_fall`，无扰动的控制力失效试验标为 `fall`。
- [x] 明确失衡开始、首次撞击、最终稳定等时间点的定义，检查跌倒与主动躺下等动作的标签边界。
      阈值固定在配置中，先于任何试验；另有 `final_posture` 与标签正交。

交付与验收：带轨迹、接触记录及事件依据的小型验证集；排除发散、异常穿透和无效触发样本。被动 ragdoll 仅作为基线，不据此宣称已模拟自然人体跌倒。

### 7.5 同步真值导出与后续接口

- [x] 导出统一时间轴上的根位姿、关节姿态/速度、接触事件、人体几何及动作阶段；几何跟随实际物理姿态。
      新增固定拓扑 `mesh_vertices_xyz`/`mesh_faces` 与 `channel_mesh_vertices_xyz`；SMPL 路径为真实蒙皮网格，无资产时为明确标记的胶囊代理网格。
- [x] 记录场景/模型/动作/控制器版本、体型参数、seed、仿真参数及验收指标；按人物和原始动作序列分组的 `split_key` 已输出。
- [x] 物理步长与无线采样间隔分别配置，明确插值/重采样方法，为 Sionna 提供按时刻获取人体几何的接口。
      120 Hz 物理 / 50 Hz 信道（非整数比），四元数走球面插值、标量走 smoothstep；旧批次索引尚未迁移。
- [x] 完成 CPU、USD 和 Isaac 分层验证；`repair6` headless 运行通过 CPU/USD/关节映射、PD、驱动关闭负对照、重力回落和地面穿透检查。宿主机设备节点已恢复，用户提供的 `nvidia-smi` 已成功识别 RTX 4060；Isaac GPU smoke test 仍需在真实宿主机命令中复验，接触力视图仍是可选能力。

交付与验收：现有固定室内场景中“一个 SMPL 人体 → 动作/跌倒 → 同步骨架、网格与接触真值”的闭环；先通过结构映射、物理响应、姿态一致性和时间对齐验收，再扩展体型、动作、房间与批量生成。

### 代码组织与当前边界

- 实现位于 `src/sim2sense_fall/humans/`，入口、配置和测试分别位于 `scripts/humans/`、`configs/humans/`、`tests/humans/`；场景模块继续负责环境，人体模块负责身体、动作和控制。
- 暂不开展手指/面部精细控制、软组织仿真、大规模训练或批量数据生成。
- AMASS 官网/动作序列仍待取得和许可核对；本机 SMPL neutral 已加载。宿主机 NVIDIA 设备节点已恢复；当前隔离执行会话仍无法访问宿主机设备，因此人体 GPU/GUI 效果需在宿主机 Isaac 命令中复验。

### 2026-09-22 本轮房屋与 AMASS 预览修复

- [x] 房屋平面按配置 `layout_scale_xy=2.20` 水平扩大到当前尺寸的 2 倍；CPU dry-run 与 Isaac headless 导出的 manifest/USD 均应为 `18.48 m × 15.40 m`，墙高不变。
- [x] 新增 `scripts/humans/view_amass.py`：读取已授权的原始 AMASS `.npz`，筛选候选动作，同步更新 SMPL 网格、PhysX DOF、根位姿，并使用人体聚焦相机。
- [x] 修复 Transitions 预览的根坐标与同步：AMASS 根平移/旋转先按首帧归一化，SMPL 网格和 PhysX 使用同一帧根姿态；`amass__sit_stand_poses` 首帧不再漂浮或横向倾倒。
- [x] 指定 `--motion` 时直接加载目标 AMASS 文件，避免启动前扫描整套动作库；明确查看器是运动学回放，碰撞代理仍由 PhysX 保留，碰撞响应验收使用独立物理检查入口。
- [x] 用合成 AMASS fixture 在 Isaac headless CPU PhysX 上播放 `amass__fall` 241 帧并通过；产物为 `artifacts/humans/amass_preview.usda`。该 fixture 只验证播放链路，不是真实 AMASS 数据。
- [ ] 真实 AMASS 仍待用户用自己的 MPI 账号下载/解包；官方数据需要注册和非商业科研许可，不能匿名绕过认证。
- [x] 用户下载的 Transitions 已校验并解压：110 个原始 `.npz`，全部通过 AMASS 字段/重定向导入；项目筛选得到 0 个候选跌倒动作，作为坐下/起立/走路/爬行等易混淆动作源。
- [ ] CMU 下载未完成；当前 Downloads 中没有可用 CMU 压缩包，需要从官网重新开始或使用浏览器的续传入口。
- [x] 复核用户录屏 `/home/gsh/Videos/Screencasts/Screencast from 2026-09-22 23-30-10.mp4`（约 7.05 s）：`amass__sit_stand_poses` 表现为站立→下坐/后仰、腿部抬起→恢复站立，符合坐下/起立参考动作的预览语义。
- [x] 明确录屏边界：`view_amass.py` 是逐帧传送位姿的运动学回放，录屏不能证明墙体或地面碰撞响应，也不能把该动作标为跌倒；当前 PhysX 正向碰撞证据仍来自 `scripts/humans/verify.py`。
- [ ] 后续实现 physics replay mode：用有限刚度/阻尼的 PD target 驱动关节，保留真实 PhysX 接触和实际 link pose 驱动蒙皮；该模式与当前 visual/kinematic preview 分开验收。

## 2026-09-21 室内 USD 批判性验收

- [x] 核对现有 USD、配置、查看入口和测试覆盖，复现内部不可见问题。
- [x] 检查布局、物理与参数边界，运行 CPU 检查和可用的 Isaac 验证。
- [x] 交付验收报告与可查看的内部证据，更新进度和研究笔记。

首次验收结论为需整改；现 R1–R6 已修复，查看入口提供临时去顶俯视/斜视；GPU/GUI 视觉待有显示环境复验。

上述整改已完成，原始缺陷证据保留在 `docs/indoor-scene-review.md`；当前修复与复验见 `docs/indoor-scene-remediation.md`。

## 验收整改 R1–R6

- [x] 修复运行入口错误退出码，增加进程级负例；让验证依据 manifest 而非固定计数。
- [x] 修复地基层叠与家具布局，增加 CPU 几何越界/穿墙检查。
- [x] 严格校验有限数值、派生尺寸和规范化路径，加入回归测试。
- [x] 重新构建 USD、执行 CPU/Isaac 验证、刷新预览及整改记录。

范围：先完成当前固定室内场景可靠性基础；人体、Sionna、门铰链和机器人导航仍为后续阶段。

## 场景代码目录整理

- [x] 将场景实现集中到 `src/sim2sense_fall/scenes/`，入口集中到 `scripts/scenes/`，测试集中到 `tests/scenes/`。
- [x] 同步导入、仓库路径定位及当前使用文档，保留历史验收记录的证据语义。
- [x] 执行 CPU、OpenUSD 和 Isaac 回归，并确认目录迁移没有改变场景几何。

目录整理已完成：CPU 70 passed / 8 skipped，bundled USD 9 passed，Isaac CPU 回退 10 项 PASS；新旧 USD 字节一致。当前入口统一见 `scripts/scenes/`，详细证据见 `docs/progress.md`。
