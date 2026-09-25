# 阶段进度

本文件保留各轮原始记录；正文中的“当前”“下一步”“仍未做”只对应其日期和配置。
当前状态与任务统一见 [计划](../task_plan.md)，全部指南与历史审计见 [文档索引](README.md)。

## 2026-09-25 姿态过渡接触一致化（蹲/坐/弯共用机制）

- **根因分离（两个独立缺陷）**：①过渡/保持的根高与关节构型独立线性混合——参考
  几何本身脚不贴地；②混合把脚在体坐标系里平移——支撑脚被拖行。原动作矩阵实测：
  蹲下误差 25.1°/滑移 0.908、起立 0.72、蹲姿保持无支撑 1708 ms。
- **竖直闭包（actions.apply）**：有 rig 时根高改为混合构型的接触闭包
  （`-min(capsule_bottom(active_contacts))`，FK 逐帧导出）——参考几何构造上保证
  最低支撑触点贴地；接触集 = 前一保持 ∪ 目标姿态声明集（坐下时骨盆接管、起立时
  脚接管），request 重定向/起身完成路径同步更新。闭包是位姿的纯函数，中途重定向
  天然连续。
- **水平补偿**：根 xy 按双脚体坐标偏移均值反向平移，脚原地踩住（残差 = 双脚
  相对漂移）。
- **GPU 实测（24.2 s 矩阵）**：蹲下误差 25.1°→**8.4°**、滑移 0.908→**0.469**；
  起立滑移 0.72→**0.377**、误差 8.7→**5.2°**；行走/转向门无回归（0.026–0.048）。
- **未过门的两个残余，如实定性**：
  1. 过渡滑移 0.37–0.47：刚体脚在踝角变化下绕掌缘枢轴的接触点迁移（无脚趾
     自由度的 hull 固有），非脚平移拖动；根治需脚部保形接触或步态式重踏步。
  2. 蹲姿保持"无支撑"1708 ms **不是物理失承**：同窗接触冲量 205/205 帧、
     平均 342 N（脚在承重），浮空 13–18 mm 的是**蒙皮**——深蹲踝俯仰下 hull
     角接触与 LBS 皮肤固有间隙。质量门 support 口径用皮肤间隙 ≤5 mm 判定，
     对非平脚姿势（蹲/坐）需要预登记「主接触支撑」口径（与坐地姿态同族决策），
     未改门前如实记为失败。
- 验证：414 passed / 10 skipped（新增接触闭包测试：保持高度复现投影姿态高、
  过渡全程最低触点贴地、重定向连续性、sit 接触集含骨盆）、compileall、Ruff、
  CPU dry-run 通过。

## 2026-09-25 前馈采纳为基线、失力摔倒重构与配置清理

- **采纳**：`reference_feedforward_scale: 1.0` 与 `root_bob: true`（2 cm 深度钳，
  见下）合入 `keyboard.yaml`、`locomotion_acceptance.yaml`、`acceptance_matrix.yaml`
  作为新基线；删除被取代的 `keyboard_feedforward.yaml`、`keyboard_amass_direct.yaml`
  与两份 `*_ff` 验收变体；`keyboard_amass_hybrid.yaml` 保留为 AMASS 摆动形状
  （未过门待驱动决策）的唯一入口。
- **失力摔倒重构**：旧 F = 350 N 后推脉冲 + 全阻尼 → 实测是「绕脚踝的刚体倾斜」，
  膝/髋/踝总折叠仅 18°、关节速度中位 1°/s（用户反馈"像保持关节角度不变"）。
  两项机制修改：①`set_control_scale(scale, damping_scale)`——摔倒时驱动阻尼按
  `fall_damping_scale: 0.15` 缩放（全新阻尼是已测的求解器 NaN 域 2864°/s，全阻尼
  终端折叠速度 ~19°/s 僵直；0.15 为有界黏性域，恢复路径 `set_control_scale(1.0)`
  自动写回原始阻尼）；②`fall_force_n` 350→60 N——推力脉冲会让地面反力与腿轴对齐
  卸掉膝矩，60 N 只作方向偏置，重力主导屈膝坍缩。实测（24.2 s 矩阵）：
  膝/髋/踝折叠 18°→**39°**、坍缩耗时 0.74→**1.5 s**（瘫坐节奏）、末态根高
  0.19→**0.27 m**（瘫坐堆高于平躺，物理自洽）、impact/fallen 事件照常成立、
  errors=[] 无求解器 NaN。
- **行走门**：动作矩阵行走/转向全活动通过（0.022–0.063 m/s，优于或等同基线）。
- **未决回归（诚实记录）**：52.7 s 长时协议 backward 关节误差 24.0°（门 15°、
  基线 5.8°）——单一 W→S 反向、单关节（right_collar）5 帧振荡（~7 Hz ±20°）。
  消融已排除：前馈（关掉仍 23.9）、bob 深度（钳 2 cm 仍 24.0）、两者同关（24.0）；
  该窗关节目标与基线**逐位相同**（max diff 0.00°）、基线同窗实测安静（3.45°），
  而实测根水平速度振荡 6×——确定性代码合并产物，非混沌。三个配置变体结果
  完全一致（24.0）证明其与 bob/ff 无关；源头在本轮与坐/起身并行会话合并后的
  未提交改动中，需提交后 git 二分定位。稳态行走误差与基线等同。
- 清理并删除的本轮中间 dry-run 产物：`artifacts/humans/verify_*`。
- 验证：413 passed / 10 skipped（新增阻尼缩放测试 2 条、ActionConfig 边界 1 条）、
  compileall、Ruff、三配置 CPU dry-run 通过。

## 2026-09-25 弯腰/坐地/起身加入键盘控制（B/N/G 键）

- 素材来源为本地库调研结论：弯腰 = CMU/115_01 @2.30 s（trunk 83.8°，限位内最深帧）；
  坐地 = Transitions/mazen_c3d `sit_stand` @2.675 s（限位内坐地帧，五接触点固有
  平面散布 ~48 mm）；起身 = CMU/140_01 全片段（6.91 s 躺→站，轴残差 0、限位超差 0）。
- 实现：`ActionState` 泛化为多姿势注册表 + 快照式过渡（姿势切换连续，`V` 从任一姿势
  回站）；`load_posture` 支持声明接触集的有界 IK 落地投影（弯腰双脚残差 0，
  坐地骨盆锚定、肢体残差 ≤44 mm，全部入 provenance）；起身 = 摔倒后辅助回放
  （实测位置/朝向锚定，0.5 s 混入，恢复位置驱动与辅助，fall 记录独立保留）。
  键位映射/配置校验/导出器标签/键盘文档同步；`braking_for_crouch` 更名
  `braking_for_action`。并行会话同时在改步态 swing（teleop/contact_gait/keyboard），
  本轮改动与其区域不重叠。
- 验证：单测 8 条新增（含 140 片段加载与坐地投影的实测口径），
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 全套 405 passed / 10 skipped；compileall、
  Ruff、CPU dry-run（`/tmp/kb_dry_new_actions`，含新姿势与片段装配）通过。
- Isaac 实跑（34.5 s 演示协议 W→停→C/V→B/V→N/V→F→G→站，120 Hz，
  `artifacts/humans/new_actions_20260925/`）：errors=[]，0 复位；摔倒 20.32 s 请求 →
  21.09 s 撞击躺平 → 24.4 s 按 G 物理起身回放 7.4 s → 33.6 s 站立。
  **摔倒后自主物理起身链路成立，R 传送不再是唯一出口。**
- 分动作门（`quality_audit.json`，重算自实际蒙皮/接触）：弯腰保持**全门通过**
  （皮肤 p95 −1.68 mm、滑速 p95 0.053 m/s、无支撑 0 s、关节 6.31°）；
  弯腰过渡仅滑速 0.175 略超 0.15；坐地保持关节 2.92°、骨盆贴地承重，
  但足底门不适用（坐地是骨盆+手的新支撑形态，验收门口径待预登记）；
  坐下过渡关节 26.95°、起身段关节/滑速失败——与蹲过渡同源（关节空间混合路径
  与地面不一致），为下一个接触一致化整改项。蹲本回合未动（保持 24.55°）。
- 未宣称：坐地/起身的正式验收（门口径未预登记前不勾选）；正常动作矩阵整体
  质量仍 false。

## 2026-09-25 步态自然性混合方案与骨盆前馈控制器（对照专项）

- 背景：接触重定向行走验收通过但观感僵硬（平顶摆脚、刚性平脚、根部零起伏）。
  用户要求「AMASS 数据 + 逐步脚位置不滑 + 改控制器」。
- **AMASS 脚轨迹实测**：源剪辑支撑相踝滑移 p50 0.16–0.25 m/s、触地水平冲击
  0.9–1.2 m/s、踝抬 14.5 cm、支撑占比仅 ~42%（含双足离地窗口）；travel 掩码把右脚
  支撑滚动误判为摆动（右脚掩码缺陷的又一实例）→ 摆动窗检测改用世界系脚速度阈值
  （0.3×源速 + 最小 run 过滤），跨周期接缝 run 解缠（补一个步距 travel）。
- **AmassSwingShape（contact_gait.py）**：从重定向周期提取各脚摆动弧（纵向位移/
  侧向弓/矢状俯仰），按指令/源步幅比缩放，clearance 下限 + 4.5 cm 抬升上限 +
  0.45 rad 俯仰 tanh 软钳 + 5 帧环形低通（摆动检测用原始运动学，平滑只作用于
  轨迹）；端点重基到零高度/零俯仰、raised-cosine 时间弯曲保证触地/离地零世界
  速度。`ContactFootPlanner` 按方向+脚别查表替代解析五次曲线；支撑锚点/刹车/
  反向/可达性逻辑不动。CPU 测试 11 条（锚点闭合、端点零速、重定向、提取、规划器
  契约）。
- **骨盆前馈控制器（root_control/keyboard）**：AMASS 无发力数据（纯运动学），由
  参考运动学 Newton-Euler 反推——`configuration_com`（rig.py，连杆质量加权）+
  bake/`fit_contact_idle` 存逐帧 COM-根偏移表（接缝闭合）；`root_wrench` 增
  `feedforward_accel_m_s2`：水平全额 m·a、垂直按同重力分数（匀速行走 a_com≈0，
  弹簧不再按设计供给推进力）；`reference_feedforward_scale` 配置（默认 0 = 完全
  旧行为），assist_blend 门控、模式切换清历史、±5 m/s² 钳制。
- **GPU 消融（10.3 s demo，两份配置）**：
  - `keyboard_feedforward.yaml`（解析摆动 + root bob + 前馈 1.0）：**全门通过**，
    前/后滑速 p95 0.019、关节 2.9–3.8°、无支撑 17 ms、站立 0.013/6.1°——与已验收
    基线逐项等同，前馈+bob **零回归**；complete_recording_window=false 为 30 Hz
    交互记录已知末帧覆盖口径。
  - `keyboard_amass_hybrid.yaml`（AMASS 形状 + bob + 前馈）：滑速 0.80/0.62、
    关节 12.6–17.1° **未过门**。诊断链闭合：弧线激励超驱动带宽（目标速度 p99.9
    仍触 8 rad/s 限幅、踝 p95 滞后 ~10°）→ 根滞后 → 辅助弹簧水平剪切 600–770 N
    超摩擦预算（Fz≈350–480 N 时 ~500 N 上限）→ 支撑脚滑移；触地滞后冲击为次因。
    历轮迭代：俯仰钳 34°→21°、抬升帽+平滑+前馈 1.22→0.80——需求侧可压但收敛到
    解析路径，自然性增益同减。
- **结论**：骨盆前馈为可交付改进（解析路径上零回归）；AMASS 摆动形状机制打通但
  受限于现有关节驱动预算——过门需更强驱动（改动已验收资产）或真平衡控制器
  （独立课题），列为待办而非本轮完成项。GUI 对照入口：两份 YAML 直接 GUI 运行。
- 并行会话说明：本轮与坐/起身姿态会话同树并行（actions/keyboard.py 交叠），合并
  后全套 **411 passed / 10 skipped**、compileall、Ruff 通过；期间观察到的姿态测试
  失败为彼会话 WIP，合并后消失。

## 2026-09-25 新增纯 AMASS 回放对比配置（GUI 自然性对照）

- 用户反馈接触重定向方案行走观感僵硬，要求对照直连 AMASS 回放。新增
  `configs/humans/keyboard_amass_direct.yaml`：与 keyboard.yaml 相同源剪辑/速度/摩擦，
  但去掉 `contact_cycle`、`contact_planner`、`locomotion_root_assist` 与
  `max_stance_slip_m_s`，`stance.enabled: false`——运行时无任何腿部 IK 修正，
  关节目标即重定向后的 AMASS 周期（含源数据自身的根部起伏与支撑滑移）。
  代码零改动：`contact_planner`/`stance` 均按配置键存在与否启用。
- 预期该配置**不通过滑步门**（历史直连基线前/后滑速 p95 0.810/1.153 m/s，门 0.15），
  仅用于视觉自然性对比与后续「从 mocap 提取脚轨迹再净化」候选的参照。
  CPU dry-run 通过：`artifacts/humans/keyboard_amass_direct_dry/`，
  modes=[backward, forward, stand]，1236 帧，contact_planner=None。

## 2026-09-25 滑步专项：限定行走验收完成

- 根因与实现：修正禁飞误压摆动脚；世界足部完整位姿规划协调根平移/摆动，停车和反向另算落点；
  以连续IK和腿长可达性提前起步避免转向过伸。3 cm摆脚配合普通模式0.6重力补偿减少触地弹起；
  蹲/起保留原0.7，0.35 s平滑切换，F仍关闭辅助。控制仅写关节目标和有界外力，普通运动无传送。
- 同24.2 s综合协议`scoped_matrix`前/后/左转/右转滑速p95=0.028/0.017/0.029/0.025 m/s，
  行走与站立六项门全过。原同协议前/后0.810/1.153 m/s。52.7 s `accepted_long`全部门通过，
  各活动p95≤0.027 m/s、最长无支撑41.7 ms、最大关节5.824°；皮肤贴地/穿地5 mm门通过。
- 120 Hz独立皮肤材料点位移也确认改善（长时前/后p95=0.00630/0.00638 m/s）。
  旧“世界脚速再减根指令速度”0.4 m/s论证错误，已撤回并修审计脚本。
- 实际根向上辅助仍约体重68–70%，不是自主平衡。52.7 s只有普通模式；辅助作用域修改前后，
  综合协议C之前13 s的实际根/关节/根力/目标逐值完全一致。完整证据及失败候选见
  [专项报告](slip-resolution-2026-09-25.md)，产物`artifacts/slip_resolution_20260925/`。
- 默认keyboard与两份验收配置参数一致，控制源码/配置快照与SHA256已保存；compileall、Ruff、
  diff检查通过，完整CPU **386 passed/10 skipped**，OpenUSD **14 passed**。
  uv tool run ruff check . 本轮因 PyPI 代理连接拒绝（os error 111）失败；本地已安装 `/home/gsh/.local/bin/ruff check .` 全部通过，详见 `uv_ruff.log` 与 `ruff.log`。
- 蹲下24.889°、蹲姿足部无支撑1.708 s、起立滑速0.592 m/s及穿地仍失败，故综合会话整体质量false。
  普通30 Hz交互的末帧覆盖不足会被准入拒收；原生120 Hz验收窗完整。未生成训练数据/CIR。
  右臂、墙/家具、自主平衡、实时性能仍待办；120 Hz完整采集RTF约0.27，阶段7/8仍未整阶段通过。

## 2026-09-25 按计划实施：足底凸包、实测质量门与数据准入

- 已实现逐动作/左右行走转向的实际皮肤、接触、滑速、关节和无支撑门；完整记录窗也成为准入条件。
  增加原生120 Hz实际网格采集、120→50 Hz导出，拒绝30→50 Hz伪高频。
- 默认足底盒改为SMPL足部表面凸包，质量/质心/惯量固定作对照；短demo支撑脚皮肤离地p95
  30.88→0.253 mm。8组控制对照仍都未通过滑速门，未采用失败的辅助/PD/锚点调参。
- 统一24.2 s协议实跑前后走、W+A/D、蹲/起、F、R；诊断导出13段均匀50 Hz，
  全部标无效、禁入训练。下蹲关节误差21.8°、蹲姿无支撑最长1.71 s，动作门仍失败。
- 跌倒标签需冲击+低根高/倾角，falling/fallen合并事件；reset_id在降采样前分段。
  默认拒收运行/质量失败会话，RT入口拒收diagnostic。没有生成可训练数据或CIR。
- 原生物理频率记录发现初始/复位穿地约14.8 mm，已分开出生高度和根施力偏置；最终复测见
  [整改实测](control-completion-2026-09-25.md)。证据在`artifacts/control_completion_20260925/`。
- OpenUSD环境最初缺PhysxSchema注册，补齐本地plugin/library路径后，凸包/材质/场景测试14 passed。
  同时修正旧测试中的Usd.Stage.Get、材质属性名和binding查询API。
  独立场景CPU dry-run、Isaac重新导出、抬高0.25 m再落回正向物理smoke均通过。
  完整CPU套件已达372 passed/10 skipped；新增记录窗用例与最终检查见整改实测补记。
  Ruff使用本地已安装版本（先前uv联网代理拒绝）。质量离线脚本首版误用rig.total_mass_kg，
  已修正为配置实际字段skeleton.mass_kg并用真实会话成功重算。
- 最终默认语义矩阵`reset_corrected/`：初始/复位左右足底0.551/0.165 mm，正常动作足部穿地门通过；
  前/后滑速0.810/1.153 m/s仍失败。跌倒全身最低−7.13 mm。额外手凸包候选令蹲姿质量全门通过，
  但正常动作又出现−7.02 mm穿地，**候选保持关闭**；源配置与证据在`hand_hull.yaml`及`hand_hull/`。
  完整最终测试 **374 passed/10 skipped**；compileall、Ruff、diff检查通过，OpenUSD14 passed。
  实际默认导出/RT双重拒收和13段50 Hz诊断时间轴已验证；计划A、B1、D完成，B2、C、E保留未完成。

## 2026-09-25 独立复核：浮空确认、计划口径与数据准入

完整分析见 [浮空与完成度复核](progress-review-2026-09-25.md)，现行待办已同步 task_plan.md。

- 当前默认配置 CPU dry-run 与 Isaac 10.300 s demo 完成，errors=[]，最大关节误差11.842°；
  全会话滑速p95=0.385 m/s，动作验收失败。前进纯步态实际skin最低Z中位16.94 mm、
  p95=21.27 mm、89.6%采样帧全部网格>10 mm；根向上力中位为体重72.7%。
  10 mm为描述阈值，不是新的验收门。原穿地门仍true，确认其不检查悬空。
- 分动作滑速p95前进1.041/后退0.825 m/s，不能用混入大量站立的总体p95替代行走验收。
  单次固定seed=20260922、固定公寓、无train/test；模型/数据/场景/源代码哈希保存在
  `artifacts/review_20260925_float/current_metrics.json`，原始实际状态、接触、mesh和截图均保存。
- 纠正计划：摩擦材质与μ扫描已实现；self_collisions=false不代表自碰撞通过；EM参数已有
  Gabriel/IFAC来源但未实测校准；历史综合会话及公寓物理CIR产物现已缺失，区分历史运行与现存证据。
- 发现数据准入缺陷：导出器可仅凭falling模式标有效fall，不强制冲击/倾角证据，也未阻止
  验收失败会话；单帧reset标记可能在30Hz采样时漏掉。批量生成保持暂停。
- 本轮没有修改控制器、默认配置或场景；只审查、运行诊断和同步文档。推荐先补验收，分离
  碰撞几何与承重问题，再协调PD/根辅助/支撑控制，统一动作矩阵通过后恢复小规模无线包。
- 验证：compileall通过；默认pytest 87 passed/10 skipped，禁用外部插件补跑357 passed/10 skipped；
  后者才覆盖完整当前套件。uv联网取ruff报代理连接拒绝（os error111），本地ruff替代通过。
  OpenUSD跳过项、完整场景smoke、Sionna本轮未重跑。计划修改前全文已归档至
  `docs/history/task-plan-before-2026-09-25-independent-review.md`。

### 2026-09-25（续）支撑腿承重：机制建成，暴露参考滑步为必要前置

- **支撑控制器**：`StanceFootController` 新增积分支撑（`support_gain 0.1 /
  support_max_m 0.08` 入出货配置）：身体低于目标高度时，锚定支撑脚的锚点 z 按比例
  下压（支撑脚被压向地面、腿伸展承重），不沉时 0.97/step 衰减，摆动复位。
  回归 `test_contact_control.py` +3（增长/衰减/禁用/摆动复位/参数校验）。
- **A/B**（`support_ab/`，长走协议，μ=1.0）：地面法向载荷 **18.5%（off）→ 23.3%（on）
  → 46.6%（on + 重力补偿 0.7→0.4）体重**——承重机制本身建成。
- **意外的诚实发现**：载荷压上去后滑速 p95 反升 0.208→0.926、锚点释放 645→1890——
  脚被压实后，**参考自身支撑相滑步**（源素材 0.134 m/s）通过摩擦被传导。与 μ 扫描
  （摩擦↑→滑速↓）合起来定出因果结构：滑速 ≈ 参考拖拽 × 法向载荷 × 摩擦——三者
  单独动任何一个都会被另外两个吃掉。**修复顺序必须是：参考足种植 → 载荷转移**。
- 配置决策：`support_gain 0.1` 保留（机制无害，关节 5.01°），`gravity_compensation 0.4`
  暂不采用（等足种植落地后再做载荷转移）。
- 门禁：pytest **357 passed / 10 skipped**、Ruff、compileall 全过。

#### 2026-09-25（续3）行走悬空（脚浮空）根因与修复

- **用户报告**：脚浮在空中（截图显示明显间隙）。量化：行走期皮肤最低点距地
  **中位 19 mm、p95 31.5 mm**（站立 3.7 mm 正常）；全局门（会话最低 3.0 mm）被
  单次冲击帧掩盖。
- **根因两层**：①根辅助托住 0.82 体重 → 脚失重，悬在关节跟踪误差高度；
  ②参考支撑相本身悬空 +8.8/+2.6 mm（ground_amass_clip 只把第 0 帧锚到地面）。
- **修复**：①烘烤把支撑相脚 z 统一种植到 0——期间修掉一个真缺陷：z 修正增量
  只在摆动分支写回、双脚支撑帧全部丢失（写回移到帧级，旧计数器显示
  stance_plant_frames=0 的假象即此）；②`root_z_offset_m: -0.015` 整体下沉身体。
  实测：行走悬空 p95 **31.5 → 21.7 mm**（−31%），滑速 +16%（0.203→0.236，
  浮空-拖滑权衡）、关节 8.65°（门内）。
- **碰撞体脚底带**：脚盒改为只贴脚底 4 cm 带（`FOOT_SOLE_BAND_M`）——全脚盒随踝
  俯仰以虚拟棱角着地（盒角 5.9 mm 着地、脚底皮肤 38 mm 的脱开实测）。
- 剩余：中位 ~17 mm 的悬空需要支撑下压/锚点/掩码统一重构（press 依赖的 planted
  掩码漏报一半右脚真实触地）。门禁：pytest **357 passed / 10 skipped**、Ruff 全过。

## 2026-09-25（续2）足种植实验、结论修正、镜头自由化

- **足种植实验（假说被否证）**：为验证「先种植参考、再压载荷」的顺序，把支撑相
  世界系种植 + 摆动最小急动度回摆烤进参考（`stance_plant_speed_m_s`，must=speed_m_s）。
  CPU 层：种植后支撑相滑动 18 mm/s、触地 35-55 mm/s（原 400-800）、闭合 1.54° 保持。
  **但物理层反而变差**（slip p95 0.208→0.801，关节 4.87→7.35°）——因为**原始参考的
  支撑相本来就已种植**（18 vs 19 mm/s 无差异），加载滑步的根因不在参考几何，而在
  **高法向载荷下 PD 跟踪与地面反力的相互作用**。种植与支撑下压均已回退为禁用
  （代码+测试保留：`stance_plant_speed_m_s` / `support_gain`）。
- **修正后的行动项**：滑步最后差距（0.203 vs 0.15）需要 PD 增益/根辅助刚度/
  支撑压紧的**整体控制重调**——已列为下一动作控制任务。
- **键盘 GUI 镜头**：`camera_follow: false` 入出货配置，视口相机交还用户自由调节；
  `true` 恢复跟随。
- 出货确认（`plant_ab/shipped/`）：errors []、关节 **4.87°**、滑速 p95 **0.203**、
  皮肤 3.0 mm、释放 636——与 μ 扫描最优点一致。
- 门禁：pytest **357 passed / 10 skipped**、Ruff、compileall、dry-run 全过。

## 2026-09-25（下半）动作控制专项：A/D 门控、摩擦材质与 μ 扫描、行走中摔倒

- **用户决策与执行**：①原地 A/D 转向禁用（控制器仅在 W/S 按住时接受转向指令，
  marching/turn-参考路径移除，`test_turn_gating.py` 3 条）；②过早生成的数据
  （batch01/fall_session_01/sionna 输出）按指示删除，数据生成推迟到动作控制达标后。
- **人体摩擦材质**（P0-B 悬案前置）：`author_human(friction=...)` 为 22 个碰撞体
  author 单一 `UsdPhysics.MaterialAPI`（该构建无 UsdPhysics.Material prim 类、
  无 combine 属性——USD 默认 average），`human_friction` 配置化，不配置=旧行为。
- **μ 扫描判定**（长走协议 ×3，有效 μ=合成(μ_human,0.55)）：
  0.30/0.525/1.00 → 滑速 p95 **0.334/0.292/0.208** m/s、超阈帧 12.4/10.4/**6.8%**、
  关节 3.92/4.00/**5.48°**。**结论：摩擦单调有效（−38%）但单独关不了 0.15 门**
  ——与摩擦审计预测一致，余项为根辅助垂直载荷与参考支撑滑步（P0-B 根因）。
  μ_human=1.45/1.30（鞋底橡胶）入出货默认。
- **行走中触发摔倒（P1-A 核过）**：`fall_walk_trigger/` W 保持中按 F → 1.3 s 后
  冲击倒地，errors []。前/站立/行走触发均实测可用；侧向摔倒待控制达标后随批量重做。
- 门禁：compileall、pytest **354 passed / 10 skipped**（friction 材质 3 条需 pxr 环境跳过）、
  Ruff、dry-run 全过。

## 2026-09-25 摔倒识别数据管线闭环（物理摔倒→mesh→真实公寓 CIR）

- **缺口盘点**：既有 fall_mesh 采集是 CPU FK 回放的脚本参考动作（kinematic_replay、
  无物理）且早于当前全部管线改动；Sionna 侧此前只跑过 floor_wall 诊断场景。
  识别摔倒缺：物理摔倒 mesh 导出、非摔倒对照、真实公寓 CIR、真摔素材扩充。
- **新增 `export_session_mesh.py`**：keyboard 会话（recording.npz 30 Hz 世界系 mesh +
  control.npz 模式时间线 + report.json fall.events）→ 采集契约分样本导出。
  标签来自**轨迹与会话事件**（fall 须有会话冲击证据或摔倒模式段），拒绝互相矛盾的会话；
  渲染抖动时线性重采样到均匀网格。回归 8 条。**期间修掉躯干角公式缺陷**
  （误用 R00 当 R22，站立被标 90° 谎报躺平——yaw-only 回归钉住）。
- **综合物理会话 `fall_session_01/`**（26.9 s，errors []）：行走→蹲(0.572 m)→起立→
  摔倒(F，17.49 s 冲击)→躺平→复位→行走。**P1-A「同会话演练蹲/起/摔并核回标签」核过**。
- **导出 12 样本**（fall×2/walk×2/crouch×2/stand_up/stand×4），拓扑 13776 面恒定、
  有限、时间均匀严格增；躯干角修正后证据自洽（蹲 25.8°、躺平 93-97°、站立 3-6°）。
- **Sionna RT 真实公寓 CIR**（sionna-rt 2.1.0 @ `~/.local/opt/sionna/bin/python`）：
  `falling_00` 物理摔倒与 `forward_00` 行走对照各 12+ 帧，全部 PASS
  （人体改变复数 CIR / 运动在信道可见 / 确定性 / 有限值）。
  **首条「真实公寓 × 物理摔倒 → CIR」链路闭环**，产物 `artifacts/sionna/fall_session_01/`。
- **AMASS 真摔重筛**（30 Hz 全库，`amass_rescreen_20260925.json`）：2100 条 →
  **183 候选 → 70 个 multiaxis rig 可表达**（旧 rig 0/28）。
- 计划重定位：P0-B 摩擦/μ 扫描与 P0-C 自主平衡为数据质量整改，不阻塞数据管线
  （数据生成只需根辅助被诚实声明）。门禁：compileall、pytest **354 passed / 9 skipped**、
  Ruff、dry-run 全过。

## 2026-09-25 行走拖滑归因+抬脚修复、原地抬脚转向、计划矛盾修正

- **用户缺陷**：①行走时脚一直滑动；②原地按 A/D 不抬脚、双脚贴地像滑冰转向。
- **滑速归因**（新常驻工具 `audit_slip_attribution.py`，`slip_attribution.json`）：
  接触帧目标侧脚速 p50 0.40/p95 0.97 m/s ≫ 根速度波动 0.02、关节跟踪 0.03
  ——主犯是**参考在脚贴地时命令它前移**（摆动抬脚中位仅 15-32 mm；运行时抬脚 IK
  每步从新鲜参考重出发、0.087 rad 预算无法累积成 ~20° 的真抬脚）。物理接触在“刹车”：
  实际滑速低于命令滑速。
- **修复**（`bake_swing_clearance`，`swing_lift_m: 0.05` 入出厂配置）：把摆动抬脚
  （平滑台阶剖面：两端零斜率软着陆+平顶防拖行）与支撑穿地清除（最深 −31 mm→0）
  **离线烤进参考**；端点差严格保持（接缝两侧增量取平均）。剖面与“落地零速冻结”
  前后向各被一轮实跑否证（砸地 1500 N+collar 37°；冻结使前进关节翻倍 8.21°），证据留痕。
- **lift5 A/B**（`slip_lift_ab/`）：后退关节 11.39→**7.86°**、滑速 p50 0.081→0.068、
  p95 1.178→0.991；前进关节 4.59→**3.91°**、p95 0.343→0.449（触地瞬态集中，两种配置
  均在 0.15 门外——滑速门槛整体仍归 P0-B）。**期间自查出并修正我的配置生成缺陷**：
  bwd demo 误复制 W 键致角色撞墙，lift1–3 后退数据作废。
- **原地抬脚转向**：新工具 `screen_turn_clips.py`（2198 条全库筛查；第一版 yaw 公式
  atan2(x,z) 错误已修——162 候选作废重筛得 126）+ `verify_turn_clips.py`（重定向+FK
  抬步验证）→ 选定 **CMU/83/83_56**（177°、6 步 3+3 对称）。接线：`turn` 步态配置
  （`support_mask_from: foot_height`）、控制器 `turning` 模式（根停+转速>8°/s 激活、
  按参考节奏步进、heading 随指令连续旋转、转向期禁用支撑锚）、`turn_playback_scale`
  减速参数。A/B：off 基线 A 段 **0% 双脚离地**（纯滑转）；on **52% 离地**（真踏步）、
  yaw 跟满 269°、皮肤过门；1.0× 源步频（2.4 步/s）关节误差 22.83° 超门，
  **0.6× 减速后 13.05° 回到 15° 门内**（`turn_playback_scale: 0.6` 入出厂配置）。
  视觉复核留待用户 GUI 实测。
- **计划审查修正（09-24 遗留矛盾）**：用 dof_names 索引的复跑**再现** P0-A W 段
  0.516/0.292——此前“暂撤回”的只是归因推断，数字与“第 4 项未通过”判定维持。
- 新增回归：`test_swing_lift.py`（5 条）、`test_turn_gait.py`（4 条）。
  CPU 门禁：compileall、pytest、Ruff、干跑全过（终值见 09-25 记录尾部）。

## 2026-09-24 一瘸一拐物理侧判定（长时 Isaac A/B ×4，通过）+ 步幅对称审计工具

- **判定**：把 09-24 的「行走一瘸一拐」从参考侧结论推进到物理侧结论。
  协议：demo 内置 R 复位循环（[2 s 站立, 10 s W, 1 s 制动, R] × 3 = 45 s），
  {old, new 窗口} × {前进, 后退} 四次 **GPU** 实跑（`artifacts/humans/limp_ab/`，
  `errors: []`，复位计数与设计一致）；后退用 heading −90 + S 绕开 −Y 方向
  1.33 m 的墙。判据预登记在 `configs/humans/stride_symmetry_gate.yaml`
  （stagger ≤ 50 mm 为主，步幅比 0.90–1.11 为辅）。
- **结果**：物理侧前后错位（逐整周期、体面前后轴）前进 231.6 → **4.6 mm**、
  后退 113.8 → **−1.1 mm**，四项判定两 fail 两 pass，fail 的全是 old 窗口——
  **窗口修复在 PhysX 里成立，跛行消失**；目标/实际错位差 ≤ 2.4 mm，跟踪忠实，
  与 P0-A「PhysX 忠实执行目标」一致。同相位俯视/侧视对照图
  `montage_{top,side}_{forward,backward}.png`（侧视被家具遮挡，判读以数值为准）。
- **两个假读数教训（都已修并有回归测试）**：(1) 第一版把**左右轴**当前后轴
  （root 旋转第二列 vs 第一列），量出「new 193 mm 未修复」的假象——那是站距+相位快照；
  (2) 第一版用 `Δphase<0` 判回绕，而后退播放相位**递减**，每步都像回绕，
  顺带发现 A/B demo 的 S 段只有 0.2–0.33 周期，其 716 mm「错位」是整周期规则要拒绝的
  相位快照。工具 `scripts/humans/audit_stride_symmetry.py`（常驻、CPU、只读，
  `render_recording.py` 同轮补了 `--times`/`--follow` 并修掉 float32 相机崩溃）。
- **步幅比值是假阴性指标**：old 前进比值 0.989「近乎对称」但用户看到跛；
  new 前进 1.084 过门槛且不可见。可见跛行的签名是恒定错位——门槛以此为主判据。
- **同协议副产物**：前进滑速 p95 0.610 → **0.261 m/s**（−57%）、锚点释放
  1575 → 636、关节误差 6.086° → 4.589°；后退滑速不变（0.789 → 0.783）。
  滑速全部仍 > 0.15 门槛，归 P0-B 未解。
- **残留**：new 前进步幅比 1.084（L 比 R 长 8%）未深挖（非可见跛行症状）；
  转向中的步幅未测（转向不推进步态相位）；判定为单公寓单速度 0.4 m/s、
  每方向 2–6 个整周期，无划分。用户 GUI 复测待做。
- CPU 门禁：compileall 通过、pytest **338 passed / 9 skipped**（331 → 338，+7 为
  `test_stride_symmetry.py`）、Ruff 全过。Isaac 实跑日志见各 run 目录。

## 2026-09-24 键盘控制现状核对 + 膝关节外展缺陷修复

- **键位现状纠正（双向）。** `C=蹲下 / V=起立 / F=摔倒` 早已实现并在 Isaac 跑过：
  配置哈希 `d65a68b1`（与当时 `keyboard.yaml` 逐位相同）的 47.5 s 人工会话里七种模式全部出现
  （forward 3788 / stand 709 / crouching 215 → crouch 269 → standing_up 215 / falling 133 → fallen 367），
  蹲姿高 0.572 m，跌倒请求 14.97 s → `right_hand` 首次触地 16.07 s（4.52 Ns）→ 同帧判躺平。
  但 `docs/keyboard-control.md`、`docs/human-simulation.md`、`docs/architecture.md` 三处仍写"尚未配置键位"，
  计划里 P1-A 五个框全未勾 —— 已按代码与实跑证据改口径并回填。
  顺带撤回我自己上一轮"首次撞击标签滞后"的判断：数据里手先撑地、躯干 107 Ns 在后，标签是对的。
- **修掉一处会把结论反读的报告缺陷。** `report.json::fall` 的 `requested_time_s/impact_time_s/final_state`
  读的是 `ActionState` 活状态，会话中途按一次 R 就把已发生的跌倒报成 `None/locomotion`，
  真相只剩 `fall.events[]` —— 这正是它看起来"从没测过"的原因。改为从 `events` 末条派生，
  新增 `outcome`（fallen / impacted_not_fallen / no_body_floor_impact）与 `state_at_exit`、`falls_requested`。
- **门禁现状更正。** 本轮重跑：pytest **326 passed / 9 skipped**（`docs/progress.md` 上一条记的 307 已落后），
  compileall、Ruff 通过；**Ruff 此前挂着一条 E501**（未跟踪的 `scripts/humans/assess_keyboard.py:113`，138>100），
  按 AGENTS.md 提交前门禁当时实际不通过，已修。RTF 我又量一次 0.39（与 0.45/0.40 同量级）。
- **用户实测新缺陷：行走时膝关节外展过大、像劈叉向两侧张开。已定位并修复。**
  三级排除：重定向链忠实（三轴链乘回与源旋转矩阵比，19 个关节最大测地误差 **0.000°**）；
  纯步态目标膝侧向间距 0.177–0.261 m 偏宽但不劈叉；**支撑脚 IK** 把它变成
  **0.049–0.375 m**，膝侧向轴单帧注入 **35.02°**、髋 27.63°，93.9% 的步有腿 DOF 被改 >15°，
  0.7 rad 预算被打满，锚点残差 170 mm。与 PhysX/PD/根辅助无关（CPU 重放里没有那三层）。
- **我自己上一个假设被实测否证并撤回**：曾猜"步频与根速度耦合导致锚点漂走"。把根速取
  0.40 / 0.777 / 1.071 m/s，单支撑窗漂移 118 / 140 / 139 mm —— 与速度无关，是参考的几何属性：
  被标"支撑"的脚在源里自己中位滑 0.134（前进）/ 0.108（后退）m/s。
- **改动**：`max_correction_rad` 0.7→**0.087**；新增关节空间加权
  `δ = W⁻¹Jᵀ(JW⁻¹Jᵀ+λI)⁻¹e`，`abduction_weight=1000`（**扫出来的**：侧向注入 4.98°@1 /
  4.41°@100 / **0.53°@1e3** / 0.07°@1e4，惩罚集由 FK 探针实测而非轴字母）；残差 >30 mm **释放锚点**
  并累计进 `report.json::stance_control`；`max_stance_slip_m_s=0.15` 只对参考自身踩得住的帧建锚。
  复测：侧向注入 **35.02°→4.98°**、>10° 步 32–36%→**0%**、最大关节误差 **12.786°→5.719°**、
  皮肤最低点 +2.98 mm 通过、根辅助峰值 873.3→841.8 N。单因子回退均复现不出症状，须两者同开。
- **仍未解决（不粉饰）**：约束不再伤害姿态，但也几乎不成立 —— 97.5% 的步贴着 5° 预算、
  残差 96–111 mm、10.3 s 会话释放 197 次，滑速 p95 **0.651 > 0.15**（比修复前 0.598 略差，
  符合预测：不再靠变形换"看起来没滑"）。下一步三条路：按"落地可保持性"预筛换源 /
  锚点改为随根平移 / 直接进 P0-C。视觉确认仍需用户本人 GUI 走一段。
- **新增工具与测试**：`scripts/humans/audit_stance_ik.py`（实跑链路 CPU 重放 + `--variant` 单因子消融，
  常驻）；`tests/humans/test_contact_control.py` +5、`test_teleop.py` +1，每条带正向对照；
  产物 `artifacts/humans/stance_ik_audit{,_gate_off}/`、`stance_ik_prior_sweep/`、`knee_fix_isaac/`。
- **顺带修掉一处轴角色标签缺陷**：`gait_seam_audit.py::AXIS_ROLE` 声称 x=屈曲/y=外展，
  与 rig 配置头部正相反；实测腿链 y 才是屈曲（30° 使踝前后 18 cm）、x 是外展（左右 19 cm），
  而臂链同一个字母又是别的角色、且随静息骨架（程序化 vs SMPL）改变 —— 字母→解剖角色的全局表
  必然对至少一条链是错的。改为按 FK 实测输出角色（只改口径不改行为，摆臂比值数字按幅度选轴，未受影响）。

## 2026-09-24 P0-A 右臂摆动异常复现与定位

- 新增 `scripts/humans/arm_capture.py`：在不改出厂配置的前提下，于**同一次 Isaac 会话内**
  按方位循环驱动相机（`--azimuths`），每个循环复位到出生点后重放同一 demo，
  从而让同一仿真时刻被四个方位各拍一次。相机方位与拍摄计划全部来自 CLI 并写入报告。
- 实机运行（4 方位 × 7 时刻 = 28 张，1280×960）：`artifacts/humans/arm_capture_azimuths/`。
  联合图 `arm_azimuth_montage.png`（行为方位、列为同一仿真时刻）。
  受本会话模型无法读取 PNG 的限制，画面判读需由用户或具备视觉能力的模型完成；
  本条目结论全部来自可复算数值。
- 复现保真：最大关节误差 **13.787°**（`right_ankle__dof1`）、根辅助峰值 **759.39 N**，
  与出厂 `gui_tilt` 的 13.787° / 759.80 N 逐位吻合，证明复现的是同一条链路。
  皮肤最低点 −10.09 mm、脚底滑速 p95 0.998 m/s —— 与出厂基线同样**未过门**，无回归也无改善。
- 新增 `scripts/humans/gait_seam_audit.py`：CPU 上量化循环接缝与控制目标左右对称性。
  前进/后退均为 **0 个接缝离群 DOF**，最大「接缝步/周期内最大步」0.683 / 1.000，
  主摆轴最小限位余量 58.99°，肩左右零位移相关 −0.761 / −0.878（反相正确）。
- **结论**：右臂摆幅偏小的直接原因是**源 AMASS 在当前裁剪窗口内的自然不对称**，
  不是重定向、多轴分解、限位、接缝或控制器缺陷。主摆轴 ptp 左肩 20.929°/右肩 10.541°
  （比 0.504）、左肘 21.665°/右肘 5.941°（比 0.274），而 PhysX 实际与目标偏差 **< 0.1°**，
  即 rig 精确执行了参考。源 npz 与重定向后的比值逐项相同，且右腿同样偏小（腿链独立），
  排除右侧 rig/增益缺陷；整段滑窗统计显示本窗口的 0.373 落在**第 11 百分位**，
  即这是源序列中最安静的右臂区间之一，而非典型行走。
- **修复尚未实施**。审计文档列出被否证的假设与可选修复方向：
  [右臂摆动审计](arm-swing-audit.md)。
- 顺带记录（不在 P0-A 范围）：`keyboard.yaml` 的 `speed_m_s: 0.4` 与源片段自身速度
  1.11 m/s（后退 1.29 m/s）不匹配，回放速率仅 **0.36×/0.31×**，即动作被放慢约 2.8 倍。
  这既独立影响观感，也直接放大脚底滑速，应作为 P0-B 的输入。
- CPU 门禁：compileall 通过、pytest **281 passed / 9 skipped**、Ruff 全过。
  本轮实际运行了 Isaac（非 dry-run），日志见运行目录下 `run.log`。

## 2026-09-24 P0-A 第 4 项：修复方向筛选（方案 1 被否证）

- 先落门槛后跑扫描：新增 `configs/humans/arm_symmetry_gate.yaml`（预登记）与
  `scripts/humans/arm_fix_window_search.py`（在给定片段内按步长枚举窗口，
  对每个窗口做重定向并测肩/肘/膝主摆轴左右比、反相相关、速度、循环数、接缝）。
- **方案 1（在本片段内重选窗口）不成立**。对现役片段
  `data/humans/amass_raw/Transitions_mocap/mazen_c3d/walkbackwards_stand_poses.npz`
  （7.867 s）以 1.042 s 窗口、**0.05 s 步长**扫描 137 个候选：

  ```
  ERROR arm_fix_window_search: no window passed the pre-registered gate (137 candidates)
  ```

  **0 / 137 通过**。根因是该片段的判据互相冲突：反相相关 ≤ −0.70 的窗口只出现在
  4.4–5.9 s（受试者减速进入站立过渡，速度 0.15–0.45 m/s、仅 1 个周期），
  而 ≥1 周期且速度达标的窗口在 0.9–1.5 s（两肩同相，相关 +0.48…+0.68）。
  最关键的是**肘**：1.0–6.0 s 全段右肘主摆轴比仅 **0.206–0.546**，均低于门槛 0.55。
  反例：4.70 s 窗口肩比 1.049（看似完美对称），但肘比仅 **0.242**，
  比出厂窗口的 0.274 更差——该片段右肘全程欠驱动。
- 门槛留痕（不按结果放宽）：首版 `min_leg_ratio 0.50 / min_speed_m_s 0.75 /
  min_cycles_per_window 2` 在本片段内不可同时满足，属门槛与素材不匹配，
  按依据修正为 0.35 / 0.45 / 1 并在配置文件中逐条写明原因；
  编码实际缺陷的 `min_shoulder_ratio` 与 `max_antiphase_correlation` **未放宽**。
  修正后重跑仍为 0 / 137。
- 证据：`artifacts/humans/arm_fix_search/forward_window_search.json`；
  审计文档新增「修复尝试」章节：`docs/arm-swing-audit.md`。
- 新增 `scripts/humans/arm_source_screen.py`：因 AMASS `.npz` 不含活动标签、
  CMU 文件名是数字编号（`32_01_poses`），改为**按测量**而非按文件名识别行走
  （先用根位移与膝摆动周期预筛，再测臂对称性），用于筛选替代源序列。
- 新增 `artifacts/humans/arm_capture_before_after/make_before_after.py`：
  同方位、同仿真时刻的前后配对联合图与数值对照，用于第 4 项要求的「同相机前后对照」。

## 2026-09-24 P0-A 第 4 项：修复实施与同相机前后对照验证

- 依据上一轮筛选（方案 2 成立、方案 1 被否证），把 `configs/humans/keyboard.yaml` 的
  `forward` / `backward` 换成 `CMU/08/08_04` @0.0 s 与 `CMU/08/08_11` @0.4 s（均 1.2 s）；
  `idle` 保持原片段（近静止姿态，无摆臂可言）。配置里写明了换窗原因与旧窗口的实测比值。
- **CPU 层复算**（`gait_seam_audit.py`）：前进肩比 0.408 → **2.332**、肘比 0.294 → **1.942**；
  后退肩比 0.478 → **1.426**、肘比 0.378 → **1.970**。接缝离群仍为 0，
  且主摆轴最小限位余量**变大**（前进 12.66° → 15.13°），说明改善不是靠逼近限位换来的。
- **Isaac 实跑复跑**（修复后四方位 × 7 时刻 = 28 张，`artifacts/humans/arm_capture_fixed_azimuths/`）：
  最大关节误差 **13.787° → 6.378°**（最差 DOF 由 `right_ankle__dof1` 变为 `right_collar`）；
  皮肤最低点 **−10.09 mm → −1.11 mm**，由**不过门转为通过** 5 mm 门；
  根辅助峰值 759.39 N → 734.34 N；脚底滑速 p95 0.998 → 0.523 m/s（仍不过 0.15 门）。
  质量门由「跟踪 ✅ / 穿地 ❌ / 滑移 ❌」变为「跟踪 ✅ / 穿地 ✅ / 滑移 ❌」。
- PhysX 主摆轴实际摆幅（整段 10.3 s 取 `np.ptp`）：右肩 10.587° → **21.594°**、
  右肘 5.883° → **10.429°**。目标与实际差值全部 < 0.5°，证明 rig 仍忠实执行参考轨迹，
  改善来自源窗口本身。
  ⚠️ **此处先前记的比值 0.828 / 0.731 已作废，见下一节**。
- **前后对照可比性已机器校验**：`before_after_comparison.json::same_run_checks` 逐项比对
  两次运行的 scene/rig/root_assist 哈希、seed、周期、拍摄计划、采样率、物理步长、
  方位计划与相机参数，要求**全部相同**；同时要求 `keyboard_sha256` **必须不同**，
  否则脚本直接报错退出。28 格全部对齐、无缺格。
- **同轮发现并修复一个会反读物理结论的报告缺陷**：`arm_capture.py` 的
  `root_displacement_m` 把世界 `x` 标注为 "forward"，而出厂 `heading_deg 90` 使角色面朝 +Y，
  于是沿 +Y 笔直前进 4.8 m 被报告为「横向漂移 **1.206 m**、前进仅 0.0024 m」。
  新增 `src/sim2sense_fall/humans/travel.py::root_displacement_in_body_frame`，
  用逐步根朝向投影到人体 forward/left 轴，并剔除循环复位瞬移步
  （实测单步 0.003 m vs 复位跳变 1.197 m，阈值 0.05 m 区分充分）。
  修复后读数：前进 **4.80 m**、横向漂移 **0.0009 m**。
- 新增配对回归测试 `tests/humans/test_root_displacement.py`（13 条），含**正向对照**：
  把修复前的世界轴规则原样重跑，要求它在同一 fixture 上给出错误答案——
  否则「修复后通过」可能只是因为 fixture 太弱。
- CPU 门禁：compileall 通过、pytest **294 passed / 9 skipped**（281 → 294，+13 为本轮新增）、
  Ruff 全过。实机运行日志见各运行目录 `run.log`。

## 2026-09-24 P0-A 第 4 项：分段复跑判定（**未通过**，并更正上一节的错误比值）

- **更正**：上一节依据 `report.json::joint_span_deg`（整段 10.3 s 取 `np.ptp`）报告的
  肩比 0.828 / 肘比 0.731 **不成立**。左右臂的极值落在**不同被控动作**上
  （右肩极值在 t=0.00 静息与 t=2.08 前进，左肩极值在 t=7.09 静息与 t=3.88 前进），
  相除等于比较无关时刻。新脚本 `artifacts/humans/arm_capture_motions/walk_strip.py`
  按被控动作分段重算。
- **分段结果**（三次实跑，`<run>/motion_analysis.json`）：

  | 区段 | 前进（前） | 前进（后） | 门槛 | 判定 |
  | --- | --- | --- | --- | --- |
  | W 前进 肩 R/L | 0.195 | **0.516** | ≥ 0.75 | ❌ |
  | W 前进 肘 R/L | 0.185 | **0.292** | ≥ 0.55 | ❌ |
  | W 前进 跳变比 | 1.20× | **1.00×** | ≤ 1.05 | ✅ |
  | W 前进 扭转比 | 0.802× | 0.478× | ≤ 1.05 | ✅ |
  | S 后退 肩/肘 R/L | — | 0.546 / 0.256 | 0.75 / 0.55 | ❌ |
  | A 转向 肩跨度 | 0.04° | 0.05° | 不适用 | — |

- **结论**：右臂卡住**仍存在**（修复使肩比 +2.6×、肘比 +1.6×，但均未过门槛）；
  **跳变已消除**（1.20× 越界 → 1.00×）；**扭转无异常**（最大 0.478×）；
  转向不推进步态相位，按「不适用」处理而非判失败。
- **CPU 层与实跑层的差距是新的线索**：CPU 层肩比 2.332 / 1.426，实跑仅 0.516 / 0.546。
  差距只能来自实跑时 PD 跟踪、根辅助或接触吃掉了右臂指令 —— 转 **P0-B** 处理。
- **期间修掉四个会让结论反读的分析缺陷**：整段 ptp 冒充摆幅；窗口不足一个完整摆动时
  低估幅度（源片段 1.200 s 一个周期，按 0.4 m/s 播放占 3.214 s 墙钟，demo 的 3.0 s
  W 段起于峰值收于 0.60 幅度处，实测 17.42° 而全摆幅 21.59°）；分数分母用窗口自身
  极值（递减窗口的收尾分数按构造恒为 0，须与参考谷值成对传入）；复位瞬移被算成跳变。
  口径更正：「步态周期 3.214 s」应写作「源周期 1.200 s / 墙钟播放 3.214 s」。
- 新增回归测试 `tests/humans/test_motion_swing_analysis.py`（13 条，每条配正向对照）。
- CPU 门禁：compileall 通过、pytest **307 passed / 9 skipped**（294 → 307，+13）、Ruff 全过。
- **`task_plan.md` 的 P0-A 第 4 项不打勾**。

## 2026-09-24 需求收敛与工程文档整理

- 用户确认新增动作只需蹲下、起立、摔倒；不做主动避障，路线由用户键盘控制。
  起立先按蹲姿到站姿规划；倒地后的地面起身未自动纳入，R 仍明确是传送复位。
- 新登记用户人工行走反馈：左臂摆动较正常，右臂异常。列为 P0-A。
  （后续：09-24 已独立复现、定位与实施修复；实跑分段复算仍未过门槛，见上文两节。）
  计划对照原始 AMASS、循环目标、PhysX 实际肩肘腕和显示，不预设根因或强行镜像。
- 计划保留碰撞体拟合、支撑脚/滑动、自主平衡、摔倒标签、手动路线接触及性能验收。
  拳击、坐椅、抓握、搬物和蹲行不属于本轮新增动作；阶段 7 仍未完成。
- 重写 `task_plan.md` 和人体当前指南；原内容保存在 `docs/history/`，历史审计保留原路径并加日期/替代证据导航。
- 增加 `docs/README.md`，覆盖根目录、数据目录和全部工程 Markdown；更新 README、架构、键盘、场景、网格、无线契约及 Sionna 入口。
  撤除现行说明中的 AMASS 缺失、GPU 不可用、Sionna 未接入等旧状态；旧 Sionna dB 结果在原表旁明确标为已撤回。
- 本轮核对代码/配置与既有报告，没有改源代码、控制参数或动作配置，没有新增 Isaac/GPU 试验；
  13.787°、9.86 mm、0.751 m/s 等均引用先前键盘实测，不算本轮新结果。
- 文档校验：25 份工程 Markdown 全部纳入索引；144 个本地链接无缺失，归档后的相对链接已修复。
  键盘/动作预览/Sionna 的 CLI 帮助与现行指南参数一致；compileall、Ruff、diff check 通过，
  pytest 为 281 passed / 9 skipped。没有新增 Isaac/场景物理运行，基础回归不代表右臂或动作质量已通过。
- 迁移补丁首次因同路径多操作被工具拒绝，未产生修改；拆分归档与新建后完成。

## 2026-09-24 键盘交互与实时蒙皮

- 新增 `scripts/humans/keyboard.py`，支持 W/S 前进后退、A/D 转向、空格制动、
  松键站立、R 复位、Esc 退出。根控制为显式有限外力，普通移动不传送。
- 物理回调覆盖每个时间步；显示网格预创建、仅更新点坐标，GUI 连续操作及复位正常。
  自动测试通过实际 Carb 键盘事件队列，1236 步与 1236 回调一致；运行中视口截图已查看。
- 最终 GUI 跟踪最大误差 13.787° 通过 15° 门槛，前进 1.098 m，制动末段速度 <0.028 m/s。
  仍有 9.86 mm 皮肤穿地及脚底切向速度 p95=0.751 m/s，整体动作质量不通过。
- 挡墙对照根最大 Y=2.355 m（墙面 2.49 m），2836 接触报告点，1599 个有 >0.01 Ns 冲量。
  最大根目标偏离 0.10 m，保留阻挡响应；局部皮肤越墙约 51.2 mm 仍需修正。
- CPU 281 passed / 9 skipped，compileall、Ruff、diff check 和场景 dry-run 通过。
- 说明、启动命令、模型/数据/seed/指标及截图见 [键盘控制](keyboard-control.md)。
  阶段 7 仍未完成，自主平衡和严格接触质量未因键盘接入而自动解决。

## 2026-09-24 AMASS 动作实测复核

- GPU 审批服务 HTTP 503 曾阻塞；用户修改权限后 RTX 4060 实测恢复，已实际运行 Isaac。
- 修复局部/世界坐标混用、接触路径解码与 pelvis 归属、末端碰撞体缺失、速度目标缺失、限位漏检。
- 物理基线复现通过，但自由 sit_stand 实测失败，截图确认前栽及头部穿地，不能作为成功动作。
- 侧向行走 3 s 有限根力辅助试验通过：最大关节误差 8.147°，根位置最大误差 24.52 mm，
  根姿态最大误差 3.264°，皮肤无穿地，853 个实际地板接触点，根位移约 (0.407,1.981,0.012) m。
  这是外部力辅助的物理跟踪，非无辅助平衡；根没有逐帧传送。
- 全长侧行 7.275 s 辅助通过：最大关节误差 8.147°、根位置 24.6 mm、根角 4.382°，
  最低皮肤 -3.15 mm。全长后退 7.867 s 在出生点 (12,1.455) 辅助通过：10.793°、
  27.4 mm、3.895°、最低皮肤 -3.70 mm。两者 `unassisted_action_reproduced=false`。
- 同相机参考/实际截图已查看；挡墙对照实际产生 919 个墙接触点（178 帧），
  峰值冲量 26.207 Ns，根被阻挡；约 25.2 mm 局部皮肤越过墙表面仍需解决。
- 自由侧行失败（37.304°、根偏离 2.110 m）；快速拳击失败（33.34°）；
  蹲行因左膝副轴超限 8.535° 提前拒绝。坐地、支撑切换及自主平衡未验收通过。
- 最终坐地复测：全长因左髋副轴超限 2.992° 被预检拒绝；可表达的前 3 s
  仍有 38.434° 最大关节误差、27.95 mm 皮肤穿地，不能作为完成的坐地起立动作。
- 最终人体 Isaac 基线通过：22 碰撞体、PD 误差 11.462°、无驱动负对照 85°、
  重力回落 1.0176 m、末段漂移 0.037 mm。CPU 277 passed / 9 skipped，
  compileall、Ruff、git diff --check 通过。`uv` 的旧代理连接拒绝在清除该命令代理后解除。
- 场景独立重建 `scene_rebuild/` 并通过 Isaac 复验，动态椅子抬高 0.25 m 后落回，
  原公寓场景未改写。
- 产物 `artifacts/amass_audit_20260924/`；完整方法、数据版本、seed、指标、截图与复现命令
  见 [本轮审计](amass-physics-audit-2026-09-24.md)。阶段 7 仍部分完成。

## 2026-09-23 — 手臂 T-pose 修复与两个可视化缺陷定位

- **根因**：`human_smpl_*.yaml` 把肩和肘都声明成绕 `+Y` 的单轴关节，而 SMPL rest 的手臂指向 `+Y`（侧平举），所以绕 `Y` 转是**绕手臂自身轴的扭转**。实测肩绕 `Y` 转 90° 只把手腕移动 **0.034 m**，绕 `X` 才移动 **0.536 m**。2026-09-22 记录的"配置驱动肩下垂修复 T-pose"因此是一个静默空操作：`default_pose_rad` 的 ±π/2 从来没有把手臂放下来过。
- **改法**：肩轴 `y→x`（内收/外展，限位左 `[-120,75]`、右 `[-75,120]`），肘轴 `y→z`（左 `[-145,10]`、右 `[-10,145]`）。下垂角实测为左 **−105°** / 右 **+105°**（腕到髋 0.067 m，含自然外展角；−90° 时是 0.130 m）。
- **中性姿态只声明一次**：新增 `rig.apply_neutral_pose(clip, plan, neutral_rad)`，把 `visualization.default_pose_rad` 作为参考动作的基线加进对应轴槽位，`build_reference` 在重采样前应用。理由：程序化骨架的 rest 手臂已下垂、SMPL 资产的 rest 是 T-pose，若把 ∓105° 烤进每个动作文件，同一段动作在两种骨架下含义相反，且 rest 一变就全体过期。角度写入会校验关节限位，越界直接报错。
- **实测结果**：站立参考残差 **0.000000°**（单轴 rig 完全能表达新参考）；记录网格的站立横向半跨度 **91.3 cm → 22.3 cm**。Isaac 两条试验仍全通过：站立 drop 1.26 mm / tilt 4.55° / drift 6.71 mm，`push_backward` 仍判 `fall`（撞击 0.93 s、末姿 0.31 m），且最低体表点由 −0.0321 m 改善到 **−0.0167 m**（手臂不再扫地板）。跟踪误差 0.827° / 13.743°，容限 15°。
- **顺带清掉的动作**：`walk_in_place` 原有的左右肩反相正弦通道（绕 `Y`、幅值 18°）被删除——它是绕臂轴扭转，视觉上什么都不做；`reach_then_topple` 重写为"手臂抬离躯干 + 屈肘"，并在 notes 里写明它**不是**前伸，因为该 rig 没有肩屈曲自由度。
- **发现一：GUI 从来没显示过真实姿态。** `build_human_stage` 只在建模时用静止模板写过一次 `/World/Human/Skin`，试验循环从不更新它；该 prim 是骨盆的刚性子节点，所以视口里人体会跟着骨盆倾倒但四肢不动，手部还会看起来"脱落"。逐帧蒙皮只写进记录数组，不写进 stage——**证据和画面来自两条不同的代码路径**。
- **发现二：直接逐帧写 stage 皮肤会打断物理。** 试过在 `execute_trial` 每步写 6890 个顶点（先世界坐标→人体偏移出画，再改骨盆局部坐标→正确），第二条试验在 `runtime.joint_positions_rad()` 处抛 `AssertionError: Instance's physics tensor entity is not valid`，`[FAIL] trials completed without error`。逐帧 stage 写会作废 PhysX tensor 视图。该实现已完整回滚（含 `HumanRuntime.set_display_skin`），当前树恢复绿色；视口姿态显示需要另找路径（每 `app.update()` 而非每物理步写、或试验结束后单独回放捕获）。
- 验证：`python3 -m pytest -q` = **249 passed / 9 skipped**（新增 `test_neutral_pose_offsets_deviations_without_choosing_a_rest_pose`），`ruff check .` 通过，`compileall` 干净，Isaac headless 两条试验 PASSED。姿态对比图见 `artifacts/review_20260923/gui_20260923/pose_before_after.png`。
- `simulate.py` 新增 `--hold-seconds`（配合 `--gui`，`-1` = 关窗前一直显示）：报告打完就 `app.close()`，人眼看不到刚跑完的跌倒。实测单条试验无 hold 在 `[34.0s]` 关闭、`--hold-seconds 8` 在 `[43.1s]` 关闭，退出码 0。`view_amass.py` 的 GUI 分支本来就有 `while app.is_running()`，不需要该开关。
- **仍未修**：试验视口只显示建模时写死一次的静止模板（四肢不动）。`view_amass.py --motion stand_neutral --capture-dir` 能看到真实垂臂姿态，因为它每帧写皮肤；两条路径的差别已记录在案。
- **上一条已修（同日追加）**：视口现在能看到真实姿态。做法是 **物理结束后回放**，不是物理循环中写入：
  - `usd_human.write_display_skin(stage, points, faces)` 在 `/World/DisplaySkin` 建一个**世界坐标、articulation 子树之外**的显示网格，并在会话层隐藏建模时写死的那个 `/World/Human/Skin`（非破坏式，与去顶同一策略）。
  - `simulate.py play_back_trial()` 在 `finally` 里把最后一条已完成试验的 `mesh_vertices_xyz` 按记录时钟（`PLAYBACK_FPS = 20`，`time.sleep` 追时钟）回放，然后才 `--hold-seconds`。
  - **为什么不逐帧写**：实测归因修正过一次。headless + 每帧写 + reset 全通过；GUI + 每帧写 + reset 会在下一条试验的 `joint_positions_rad()` 抛 `Instance's physics tensor entity is not valid`；而**完全不写、只做 6 次 reset** 又全部通过。所以触发条件是"GUI 下每帧改 stage"，不是 reset 本身，也不是写入本身。回放路径在物理停止后才碰 stage，从根上避开这个组合。
  - 实测：GUI 双试验 `human simulate: PASSED` + `replayed 101 recorded frames`，无张量失效；截图 `artifacts/review_20260923/gui_20260923/isaac_playback_final_pose.png` 显示人体**仰卧于地板上、双臂垂放**，Stage 面板里 `DisplaySkin` 存在，此前"手部脱落/四肢冻结"的现象消失。


## 2026-09-23 — AMASS 整机基修复、批量容错与筛选口径统一

- **缺陷**：`retarget_amass_clip` 用 `up_axis_conversion("y","z")` 搬运 AMASS 的关节旋转，该基只保证 up 不变、表达不了偏航，把人体的左右轴放到了管线的前向轴上。这正是 `docs/mesh-orientation-defect.md` 记录过、网格路径已用 `body_frame_conversion` 修掉的同一类错误，AMASS 路径当时漏改。
- **实测证据**：修复前髋/膝/踝/脊柱的屈伸能量落在管线 `x`（中位 15–19°），与 rig 声明的 `y` 完全错位；改用 `AMASS_BODY_FRAME = up=y, forward=z, left=x` 后同样的屈伸落到 `y`（膝 19.0°/19.3° 左右对称）。关节槽位映射另做独立确认：AMASS `poses` 第 10、11 槽在 40/40 条抽样序列中恒为零，正是 SMPL 两个叶子 foot 关节的特征，说明槽位 `k` 就是项目拓扑的第 `k` 个关节，不存在重排序。
- **改动**：`motion.py` 新增公开 `BodyFrame` 与 `AMASS_BODY_FRAME`（含推导依据），`retarget_amass_clip` 改收整机基并删除 `source_up_axis`/`target_up_axis` 两个错误默认旋钮；`import_amass.py` 增加资产交叉校验，模板实测帧与 AMASS 假定帧不一致时直接失败。
- **批量容错**：`load_amass_library` 原来遇到第一条不合格序列就抛错，实测 `amass__01_05_poses` 一处 122.8° 跳变即让 2198 条的筛查整体失败。现返回 `AmassLibraryLoad(clips, failures)`，逐文件跳过并记录原因，仅在全库无一条可读时报错。
- **筛选口径**：`screen_amass_clip` 原来自带 25° 阈值，而运行期 `joint_values_from_clip` 用 1e-6 rad 直接抛错，两条规则互相矛盾，且投影逻辑重复三处。现在筛选直接调用运行期那条规则（`joint_values_from_clip` / 新抽出的 `axis_residuals`），并把"是不是跌倒"和"rig 能不能表达"拆成 `fall_candidate` 与 `rig_expressible` 两个字段分别上报，拒绝原因点名卡住的关节。
- **真实数据结果**（`--limit 120`）：120 文件读取、1 条不可读被记录、119 条完成筛查，**28 条跌倒候选**（修复前该数字被合并进"off-axis 超限"而显示为 0），其中 **0 条可被当前单轴 rig 表达**；28 条的卡点全部在上肢（肘 23、肩 5）。
- **下一阶段的硬事实**：把肘改成 `['y','z']`、肩改成 `['x','y','z']`（planner 已支持多轴链，实测 DOF 从 14 增至 20）后候选仍是 0/28。原因是 `axis_residuals` 对同一根关节的每个 DOF 都独立读取同一个原始 axis-angle 向量，只排除自己那一轴，因此第二个 DOF 永远不会降低第一个 DOF 的残差；同理，多轴关节的驱动分量只在角度很小时才近似有效。也就是说**多轴链在 planner 里已经实现，但 clip→DOF 映射器没有多轴分解**，这是 AMASS 接入的真实剩余工作量。
- 验证：`python3 -m pytest -q` = **248 passed / 8 skipped**，`ruff check .` 全部通过，`compileall` 干净；`scripts/humans/plan.py` 与 `simulate.py --dry-run --all`（54/54 标签）均 PASSED。本轮只动 CPU 路径，未重跑 Isaac/Sionna。

### 追加：Isaac 实跑复核、GUI 看不到人体（有屋顶）与两处使用陷阱

- **物理零回归确认**：用 `~/isaacsim/python.sh scripts/humans/simulate.py --config configs/humans/human_smpl_stable.yaml --motions configs/humans/standing_validation.yaml --trial stand_neutral:none --trial stand_neutral:push_backward` 复跑，label、tracking 0.776/16.931°、`config_sha256`、`scene_sha256`、`motion_sha256` 与 `artifacts/review_20260923/final_physics` 归档**逐项一致**；同一条命令连跑两次结果逐位相同，说明该试验在当前环境内是确定性的。
- **陷阱一（我自己先踩的）**：`stand_neutral` 在默认 `motions.yaml` 里是 **1.0 s**，在 `standing_validation.yaml` 里才是 **5.0 s**。用默认值时推搡后只剩 0.6 s 窗口，人被诚实判成 `no_fall / partial topple`，看起来像物理回归，实际是参数不一致。`motion_sha256` 覆盖编译后的 clip，所以换动作库一定会变哈希——这一点应优先于"结果变了"的猜测去核对。
- **陷阱二（用户复现时暴露）**：`simulate.py --gui` 从不调用去顶视角，公寓是封闭的，视口只能看到屋顶，而试验照样报 PASSED。`view_amass.py`、`build.py`、`scripts/scenes/view.py` 三处各自重复"配置相机 + 把视口指过去"，`simulate.py` 是漏掉的那一处。
- **修复**：`scenes/view.py` 新增 `apply_inspection_view(stage, mode, aspect_ratio, require_viewport)`，把"作者相机"和"选中相机"绑成一个动作；`simulate.py` 新增 `--view {human,top,roofless,exterior}`（默认 `human`，仅 `--gui` 时生效）并调用该 helper；三处重复实现全部收敛到它。此前 `configure_inspection_view` 的返回值在个别调用点被丢弃，正是"截图成功但画面里没有人体"的成因。
- 验证（真 Isaac 解释器 + bundled OpenUSD 实测）：相机路径 `/InspectionCamera`、屋顶 `invisible`、屋顶**仍是碰撞体**、源图层字节未变、视锥覆盖人体 z=0/1/2 三点；headless 下 `get_active_viewport()` 仍返回对象，所以 `require_viewport=True` 只在完全没有 Kit 的解释器里抛错，GUI 入口才使用它。`python3 -m pytest -q` = **248 passed / 9 skipped**（新增 1 项需 pxr，在 CPU 解释器下跳过，已在 bundled USD 内手工验过同等断言），`ruff check .` 通过。
- 捕获相机改为逐帧跟随人体（原来只在第 0 帧取景，位移较大的动作会走出画面而捕获仍报成功）。
- **新发现的独立缺陷（尚未修）**：AMASS 预览没有竖直锚定。`normalize_root_motion` 只锚首帧平移与朝向，根高度直接取序列值；实测 `CMU/01/01_02` 源 `trans` 的 up 分量跨度 **4.01 m**、首帧 **-0.211 m**，回放到第 2896/4345 帧时骨盆已在 **+2.30 / +2.84 m**，第 1448 帧在 **-0.66 m**，所以画面里看不到人不是相机没跟上，而是人体在天花板上方或地板下方。该竖直漂移在旧 `up_axis_conversion` 与新整机基下完全相同（两者都把源 Y 映射到管线 Z），**不是本轮帧基修复引入的**。
- **对筛选结论的影响**：`screen_amass_clip` 的 `root_drop_m` 与 `peak_down_speed_m_s` 用的就是这个会漂的根高度，因此"120 条抽样得 28 条跌倒候选"只能作为**上界**，不能当作已验证的候选数；候选判定需要改成以地面/最低体表点为参考，而不是以序列根高度为参考。


## 2026-09-22 — Transitions 录屏动作表现复核

- 复核用户提供的 `/home/gsh/Videos/Screencasts/Screencast from 2026-09-22 23-30-10.mp4`（约 7.05 s）：画面从站立开始，中段下坐并后仰、腿部抬起，后段恢复站立。
- 该表现与 Transitions `amass__sit_stand_poses` 的坐下/起立参考动作一致；它是动作预览证据，不是跌倒标签，也不代表已经完成 PhysX 接触响应验收。
- `view_amass.py` 当前每帧直接写入 DOF 和根部位姿并清零速度，属于运动学回放。墙体和地面碰撞代理仍写入 stage，但接触反作用会被下一帧传送覆盖；要验收碰撞应使用独立物理入口或后续 physics replay mode。
- `DISPLAY= WAYLAND_DISPLAY= ~/isaacsim/python.sh scripts/humans/verify.py` 的最新实际结果：19 个胶囊碰撞代理恢复，重力回落 `1.3051 m`，最低人体点 `-0.0000 m`，地面碰撞已通过。墙体代理已写入场景，但当前验收未覆盖墙体接触响应，不能把它概括为墙/地面碰撞均已完成。

## 2026-09-22 — AMASS 查看器启动路径与碰撞边界核对

- 指定 `--motion amass__sit_stand_poses` 时，查看器现在按动作 ID 直接定位单个 `.npz`，不再先解析 Transitions 的全部 110 个序列；未指定动作时仍保持完整库扫描和确定性排序。
- 实测隔离会话从进程启动到 headless 完成约 18 s，其中 Isaac Sim/Kit 扩展初始化约 13.4 s；日志显示当前隔离会话没有可用 CUDA/NVML，RTX 初始化失败后回退 CPU，这部分不能用仓库代码进一步压缩。首次启动还会建立 shader/材质缓存。
- `view_amass.py` 的 19 个胶囊代理确实带有 `UsdPhysics.CollisionAPI`，SMPL `/Skin` 明确是 visual-only。该入口是运动学回放：每帧写入 DOF/root 位姿并清零速度，所以墙/地面的 PhysX 反作用会被下一帧传送覆盖，不能把它当作碰撞响应实验。真实碰撞正向对照仍由 `scripts/humans/verify.py` 的重力回落和地面穿透检查承担。
- 重新运行 `DISPLAY= WAYLAND_DISPLAY= ~/isaacsim/python.sh scripts/humans/verify.py`：实际 Isaac CPU PhysX 通过，19 个碰撞代理恢复，抬高后骨盆下降 1.3051 m，最终最低人体点 `-0.0000 m`；地面碰撞链路有效，墙体代理已写入但本次检查未覆盖墙体接触响应。日志仍记录当前隔离环境的 CUDA fallback。

## 2026-09-22 — Transitions sit-stand 预览根位姿修复

- 用户复现 `view_amass.py --motion amass__sit_stand_poses` 时，人物曾因直接使用 AMASS 序列的全局 `trans`/root rotation 而出现在房间外并横向倾倒；蒙皮又没有使用同一根旋转，导致视觉皮肤与 PhysX 胶囊代理错位、穿墙。
- 新增 `normalize_root_motion()`：以首帧为局部锚点，使用 `t[k] - t[0]` 和 `R[k] @ R[0].T`；查看器的 `forward_kinematics()`、SMPL skin 和 `HumanRuntime.set_root_pose()` 现在消费同一帧归一化根位姿。
- 根旋转连续性检查改用旋转测地距离，避免跨越 180° 时主值轴角向量被误判为 359° 跳变。
- CPU 回归：`201 passed / 8 skipped`，Ruff 和 compileall 通过。真实 Transitions `amass__sit_stand_poses`：980 帧、120 Hz、8.158 s；Isaac headless CPU PhysX 播放和 USD/JSON 导出通过，`root motion anchored` 检查通过。
- 当前隔离执行仍记录 `NVML_ERROR_DRIVER_NOT_LOADED` / `cuInit failed (100)`，所以这次 headless 结果是 CPU PhysX fallback；宿主机 GUI 需要用用户已恢复的 NVIDIA 会话重新运行命令观察画面。

## 2026-09-22 — 人体 GUI 姿态、碰撞代理与取景修复

- 用户截图中的三项视觉问题已定位并修复：SMPL neutral 的水平 rest pose 造成手臂穿墙；胶囊碰撞体被错误显示造成皮肤与内部结构分离；整屋包围盒取景使人体显得过小。
- `configs/humans/human_smpl_neutral.yaml` 新增配置驱动的 `visualization.default_pose_rad`：左右肩分别为 `+pi/2`、`-pi/2`，左右肘为 `-0.20` rad；左肩限位扩展到 110°。CPU 正运动学实测左右上臂向下约 0.27 m，SMPL 6890 顶点姿态全部有限。
- `scripts/humans/build.py` 用上述 DOF 姿态生成 posed SMPL 网格，并在 GUI 启动时对同一组 PhysX DOF 写入 position/target；`--view human` 按 `/World/Human` 边界取景并隐藏屋顶，现为默认视角。`roofless`、`top`、`exterior` 仍可选。
- `usd_human.py` 将 19 个胶囊设置为不可见，同时保留 `UsdPhysics.CollisionAPI`；headless USD 检查确认 `19/19` 隐藏、`19/19` 仍是碰撞体，SMPL 网格为 6890 顶点 / 13776 三角面。
- 验证：`python3 -m pytest -q` 为 `199 passed / 8 skipped`；compileall 与 Ruff 通过；`DISPLAY= WAYLAND_DISPLAY= timeout 180 ~/isaacsim/python.sh scripts/humans/build.py --headless --name human_display_repair` 通过。Isaac 日志仍记录隔离环境不可见宿主机 GPU（NVML/CUDA CPU fallback），不把它写成 GPU 渲染通过。
- 复核补充：新增显示姿态回归后完整测试为 `200 passed / 8 skipped`，本机 `/home/gsh/.local/bin/ruff check .` 通过；`uv tool run ruff check .` 仍被运行环境的 DBus transient scope 错误阻断（`Process 2 is a kernel thread, refusing`），不影响已完成的本机 Ruff 检查。
- 用户在宿主机 GUI 首次运行时报告 `AssertionError: Instance's physics tensor entity is not valid. Play the simulation/timeline to re-initialize it`。根因是 GUI 分支在 `runtime.play()` 后立即写 DOF，Kit 尚未完成 tensor articulation 初始化；已补上 4 次 `app.update()`，与 `verify.py`/`simulate.py` 的已验证顺序一致。CPU 回归与 headless 构建复验继续通过；当前隔离环境没有显示会话，GUI 需在宿主机重新运行确认画面。

## 2026-09-22 — 人体 GUI 与真实蒙皮显示修复

- 复现了用户截图：`scripts/humans/build.py --gui` 未接入场景查看器的 roofless 相机，因此完整屋顶遮挡室内；同时该入口虽然加载 SMPL，却没有将网格传入 `build_human_stage()`，生成的 USD 只含胶囊碰撞体。
- 已修复 `scripts/humans/build.py`：默认 `--view roofless`，支持 `top` / `roofless` / `exterior`，通过临时 session layer 隐藏屋顶并设置相机；`--view exterior` 可恢复完整外观。
- 已修复 `build.py`、`verify.py`、`simulate.py` 的 USD 导出参数：真实 SMPL 网格传入 `/World/Human/Skin`。在 headless 复验中实际输出 `6890 skin verts`、`13776` 三角面、24 links、19 colliders，最终 `human build: PASSED`；基础场景哈希检查通过。
- 复验：`python3 scripts/humans/build.py --dry-run` 通过；`python3 -m pytest -q` 为 `199 passed / 8 skipped`；本机 Ruff 和 compileall 通过。当前隔离会话的 Isaac 日志仍因没有宿主机 GPU/显示而 CPU fallback，但 USD 结构与蒙皮计数已验证。

## 2026-09-22 — AMASS 导入与 GPU/CUDA 复核补充

- 已新增 `src/sim2sense_fall/humans/amass.py` 和 `scripts/humans/import_amass.py`：本地 AMASS `.npz` 扫描、字段校验、源 SHA-256、SMPL-H 52→SMPL 24 重定向、Y-up→Z-up、来源元数据和摔倒候选筛选；`scripts/humans/simulate.py` 可通过 `--amass-root` / `--amass-fall-only` 接入 dry-run/仿真选择。
- 新增 `tests/humans/test_amass_import.py`，覆盖字段缺失快速失败、来源哈希、坐标转换、候选摔倒和确定性排序。用 `/tmp/synth-amass-import` 合成 fixture 实跑 `import_amass.py`：`2 sequences scanned, 1 fall candidates`；合成数据仅验证管线，不能当作真实 AMASS。
- 当前资产根扫描未发现真实 AMASS `.npz`，因此真实子集、人物、序列、许可和物理复核仍待用户提供已授权数据；项目不会自动下载注册制数据集。
- GPU/CUDA 复核结果：PCI 设备 `01:00.0` 为 RTX 4060 Max-Q，内核 `nvidia` 驱动 `595.91.07` 已绑定，`nvcc 12.0`、`libcuda.so.1`、`libcudart.so.12` 和 `libnvidia-ml.so.1` 存在，`systemd-udevd` 与 `nvidia-persistenced` 均运行；但 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 均不存在，`/sys/class/misc` 也没有 NVIDIA 节点，`nvidia-smi` 返回 `couldn't communicate with the NVIDIA driver`。`udevadm test` 显示规则会调用 `/sbin/ub-device-create`，但该工具在当前受限会话无权限创建节点；`sudo` 被 `no new privileges` 阻断。结论是宿主机 udev/设备节点或容器权限问题，非仓库代码缺少 CUDA toolkit。
- 因此 Isaac 日志中的 `NVML_ERROR_DRIVER_NOT_LOADED` / `No usable CUDA device present` 仍属环境限制；本轮没有手工 `mknod`、卸载/重载驱动或修改系统服务，GPU/RTX 渲染不能写成已通过。宿主机管理员需在真实系统会话修复 udev 设备节点后再复验 `nvidia-smi`、CUDA sample 和 Isaac GPU backend。

### 宿主机修复后的复验（用户终端，2026-09-22）

- 用户在真实宿主机执行 udev 规则重载、`udevadm trigger` 和 `nvidia-persistenced` 重启后，`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`、`/dev/nvidia-modeset` 和 `nvidia-uvm-tools` 已恢复。
- 用户提供的 `nvidia-smi` 已成功返回：RTX 4060 Laptop GPU，驱动 `595.91.07`，驱动报告 CUDA `13.2`，显存 `1720 MiB / 8188 MiB`，GPU 利用率 `40%`。这证明宿主机 NVML 和字符设备链路已经恢复；此前的 GPU 故障已解决。
- 本仓库的受限执行会话仍看不到宿主机 `/dev/nvidia*`，所以不能从该隔离会话代替用户运行 Isaac GPU smoke test。真实宿主机下一步运行 `~/isaacsim/python.sh scripts/humans/verify.py` 和 `~/isaacsim/python.sh scripts/humans/build.py --headless`，检查日志中不再出现 `NVML_ERROR_DRIVER_NOT_LOADED` / `No usable CUDA device`，并用 `nvidia-smi` 观察 Isaac 进程显存占用。
- `nvcc` 工具链仍是本机 CUDA 12.0，而 NVIDIA 驱动向后兼容并报告 CUDA 13.2；这是驱动支持版本与本地编译工具包版本不同，不是当前设备故障。

## 2026-09-22 — 房屋扩大与 AMASS 预览复验

- `configs/scenes/indoor_apartment.yaml` 新增 `layout_scale_xy: 1.10`。缩放同时作用于房间 origin/size、门窗宽度与偏移、家具位置和水平尺寸，墙高保持原值；CPU manifest 和 Isaac headless USD 实测 footprint 为 `9.24 m × 7.70 m`，面积为 `71.148 m²`。
- 新增 `scripts/humans/view_amass.py`。它只读取已授权的原始 AMASS `.npz`，在同一帧更新 SMPL skin、PhysX articulation DOF 和 root pose，并调用 `configure_inspection_view(mode="human")`；不把导入产物 NPZ 当作原始 AMASS 输入。
- 用 `/tmp/synth-amass-import/Subject1/fall.npz` 播放 `amass__fall`：Isaac headless 启动、PhysX 激活、241 帧播放和 USD 预览导出全部通过，退出码 0；产物为 `artifacts/humans/amass_preview.usda`。合成 fixture 不能作为真实 AMASS 实验结果。
- 首次预览失败是 `Prim` 直接调用 `GetPointsAttr()`，已改为 `UsdGeom.Mesh(prim)` 后复验通过。误把 `/tmp/amass-import-result`（retargeted NPZ）作为原始输入会明确失败，这是输入边界而非静默降级。
- 本次隔离 Isaac 日志仍报告 `NVML_ERROR_DRIVER_NOT_LOADED`、`cuInit failed (100)` 并回退 CPU PhysX；宿主机 `nvidia-smi` 已由用户复验正常，GPU Isaac smoke test 仍需在宿主机显示/设备会话运行。
- 官方 AMASS 列表访问已实际尝试：凭证 dry-run 通过并识别 4 个配置项，但带现有代理访问返回 `Connection refused`；去掉代理后返回 TLS `SSLV3_ALERT_HANDSHAKE_FAILURE`。因此本轮没有把在线下载写成完成，待网络/代理恢复后可直接重跑 `fetch_assets.py --site amass --list`。

## 2026-09-22 — Transitions 本地下载核验

- `/home/gsh/Downloads/Transitions.tar.bz2` 已通过 `bzip2 -tv`，归档内有 110 个 `.npz`，字段实测包含 `poses (N,156)`、`trans (N,3)`、`mocap_framerate`、`gender` 和 `betas`；解压后占用约 261 MB，存于 `data/humans/amass_raw/Transitions_mocap/`。
- `scripts/humans/import_amass.py --root data/humans/amass_raw --out artifacts/humans/amass_transitions_import` 已通过：110 sequences scanned，全部完成 SMPL-H→SMPL 重定向；候选筛选为 `0 fall candidates`。这批数据可用于 `sit_stand`、`walk`、`crawl`、`run` 等易混淆动作，不能写成已取得跌倒样本。
- 真实 Transitions 动作 `amass__sit_stand_poses` 已在 Isaac headless 中播放 980 帧并通过，产物和元数据写入 `artifacts/humans/amass_preview.usda/.json`；本次运行仍是当前隔离环境的 CPU PhysX fallback。
- CMU 下载状态：此前看到的 `.part` 临时文件和 0 字节目标文件目前均已从 Downloads 消失，当前没有可校验的 CMU 压缩包；需要官网重新下载或由浏览器对仍存在的任务执行续传。

## 2026-09-22 — 房屋再次扩大

- 用户要求把当前房屋再扩大到 2 倍。`layout_scale_xy` 已从 `1.10` 调整为 `2.20`，因此当前实际导出 footprint 从 `9.24 × 7.70 m` 变为 `18.48 × 15.40 m`；房间高度、墙厚和竖向家具尺寸保持不变。
- 房间 origin/size、门窗宽度/偏移、家具位置和水平尺寸继续由同一个缩放锚点统一变换，避免房间与家具各自缩放造成坐标错位；CPU 测试期望和 USD manifest 将同步刷新。

## 2026-09-22 — 阶段 7 二次验收与逐帧网格导出（当前）

### 修复后复验（repair6）

- 修复 `scripts/humans/verify.py` 的 PD 验收：每个 DOF 单独跟踪，传送后清零根部/关节速度，并在控制试验期间暂时关闭 19 个**人体自身**碰撞体，避免固定室内家具/墙体把控制误差污染成关节跟踪失败；重力和地面检查前恢复全部碰撞体。
- 新增 `HumanRuntime.reset_velocities()` 与 `set_body_collisions_enabled()`，并将逐 DOF 的 target/reached/error 写入 `human_verify.json`，保留可审计证据。
- 最新 headless Isaac：[`artifacts/humans/human_verify_repair6.json`](../artifacts/humans/human_verify_repair6.json)，**PASSED**；PD 最大误差 `1.540°`（容限 `15°`），逐 DOF target/reached/error 及 stiffness/damping 均写入 JSON，驱动关闭负对照、19 个碰撞体恢复、抬高后重力回落、地面穿透和基础场景哈希均通过。运行仍记录无 NVIDIA/CUDA 设备，因此这是 CPU PhysX 回退，不是 GPU 渲染验收。
- 新增 `scripts/humans/migrate_trials_index.py` 并迁移两个旧批次索引：`physics_dt_s` 从错误的 `120.0` 修为 `0.008333333333333333`，同时加入 `physics_hz: 120.0`；迁移脚本会读取被引用 trial JSON 的 provenance 做一致性校验。

- 资产结论已纠正：`data/humans/smpl/` 中存在并可加载 SMPL v1.1.0 neutral、male、female 三个 pickle；审计为 **3/5 个声明模型可用**。neutral 实测 6890 顶点、13776 三角面、24 关节、300 个 shape directions，静止姿态线性蒙皮最大漂移 `2.22e-16 m`。因此“SMPL 尚未下载”不成立。
- AMASS 仍未取得：在配置资产根及项目数据目录中未发现 AMASS 原始 `.npz` 动作序列；当前动作仍为 `scripted`。不能把现有动作称为 AMASS。
- 新增 [`docs/mesh-export.md`](mesh-export.md) 和 `humans/mesh_sequence.py`：固定拓扑、逐帧世界坐标 `(x,y,z)`、米制、Z-up、物理时间轴与信道时间轴均有校验。
- `scripts/humans/build.py` 与 `simulate.py` 现在共享同一实际 SMPL 资产和尺度拟合；CPU 规划的 `ground_offset_m` 已从此前程序骨架的 `0.9911 m` 对齐到 SMPL 规划的 `1.2462 m`。
- `GroundTruth`/NPZ 现在写入 `mesh_vertices_xyz`、`mesh_faces`、`channel_mesh_vertices_xyz`、顶点归属和拓扑哈希。无真实模型时使用明确标记的 `capsule_proxy_mesh`，不会伪称 SMPL。
- CPU 验收：`compileall` 通过；`196 passed / 8 skipped`；本机 Ruff 通过；`uv tool run ruff check .` 仍被环境 DBus 错误阻断（`Process 2 is a kernel thread, refusing`）。`scripts/humans/plan.py` 通过并验证真实 SMPL；`scripts/humans/simulate.py --dry-run` 通过，并实际写出每个动作的 `<motion>.mesh.npz`（`stand_neutral`: `(121, 6890, 3)` 顶点、`(13776, 3)` 面、全部有限），代理网格无零面积三角形。Isaac 数值稳定性曾在 repair3 失败，已由 repair6 复验修复。
- 下一步：在已通过的 Isaac 分层验收上继续检查每帧 link pose→mesh 的有限性、拓扑一致性和 USD/NPZ 时间对齐；取得 AMASS 后再做动作筛选与重定向；随后进入 Sionna RT 首条 CIR/CSI smoke test。
- Isaac repair3 的失败记录保留为历史证据：[stage7-final-isaac.log](../artifacts/humans/stage7-final-isaac.log)。其后的 repair6 已通过；失败原因是控制试验未隔离人体碰撞且复用传送后的速度状态，已在代码中修复并由逐 DOF 证据复验。

## 2026-09-22 — 阶段 7 第一次批判性复审（历史快照）

- 完整验收不通过，阶段 7 恢复未完成状态；程序骨架/CPU 契约部分可用。详见 [`stage7-review.md`](stage7-review.md)。该快照的资产扫描结论已被二次验收纠正。
- 当时误报两个配置资产根不存在、SMPL 审计 0/5；二次验收确认项目内已有 3 个 SMPL v1.1.0 pickle，并通过 neutral CPU 加载与静止蒙皮复验。AMASS 原始序列仍未发现。
- CPU：compileall、134 passed / 8 skipped、本机 Ruff 通过；plan/build/simulate CPU 路径通过。`uv tool run ruff check .` 退出 46，DBus `Process 2 is a kernel thread, refusing.`。
- 六项待整改：伪模型导致真实蒙皮假通过及禁止降级无效；PD 目标重复转弧度；体点/方向速度混用导致假撞击；索引把 120 Hz 写成 120 s；联合键不保证人物隔离；日常动作超跟踪容差仍标可用。
- Isaac 结构与前 3 个姿态比较通过后报 `ValueError: quaternion contains a non-finite value`，退出 1；同时有 `NVML_ERROR_DRIVER_NOT_LOADED`、接触视图 `AttributeError: 'NoneType' object has no attribute 'check'`。历史全部 PASS 未复现，根因未定位，不能只归为 GPU 缺失。
- 日志：`artifacts/humans/stage7-review-isaac.log`。固定场景 `apartment_cn_two_bedroom`、seed=20260922、程序胶囊代理；无新训练/数据划分，指标与哈希见验收报告；GPU/GUI 未验证。

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

## 2026-09-22 — 阶段 7：SMPL 人体、动作与跌倒仿真

实现与验收细节见 [`human-simulation.md`](human-simulation.md)。本节只记录实际跑过的东西与失败教训。

### 交付概览

- 新增 `src/sim2sense_fall/humans/`（skeleton / assets / rotations / motion / config / rig /
  events / export / usd_human）、`scripts/humans/`（common / plan / build / simulate / verify）、
  `configs/humans/`（assets / human_smpl_neutral / motions）、`tests/humans/test_humans.py`（52 项）。
- `python3 scripts/humans/plan.py`：全部 PASS。
- `python3 -m pytest -q`：122 passed / 8 skipped（8 项需 pxr，在 bundled USD 环境另跑）。
- `/home/gsh/.local/bin/ruff check .`：All checks passed。
- `python3 -m compileall -q src tests scripts`：通过。

### Isaac Sim 实际运行

- 早期 `~/isaacsim/python.sh scripts/humans/verify.py` 记录为**全部 PASS**，但该记录不能代表当前状态；后续使用实际 SMPL 规划重跑后未复现。
  物理姿态与 CPU 正运动学在 6 组姿态、24 连杆上误差 **0.00 mm**；
  关节极限往返最大偏差 7×10⁻⁶ 度；抬高 0.25 m 后骨盆下落 0.279 m；最低体点 0.0000 m；
  PD 跟踪最大误差 2.0 度（预先登记容限 15 度）；物理步长 1/120 s 生效。
  产物 `artifacts/humans/human_verify.json`。
- `~/isaacsim/python.sh scripts/humans/simulate.py --headless`（自由根，7 项）：
  `fall=2`（`control_loss`、`walk_in_place`）、`no_fall=5`（无扰动站立与四向推动）。
- `~/isaacsim/python.sh scripts/humans/simulate.py --headless --pin-root`（带记录的骨盆钉定，6 项）：
  `no_fall=4`（站立、弯腰、坐下、原地行走），2 项被有效性门禁排除（下蹲、后倒参考）。
- `~/isaacsim/python.sh scripts/humans/build.py --headless`：导出带动画学人体的 USD。

### 最新复验状态（2026-09-22）

- `python3 -m compileall -q src tests scripts`：通过；`python3 -m pytest -q`：**196 passed / 8 skipped**；本机 `/home/gsh/.local/bin/ruff check .`：通过；`uv tool run ruff check .` 仍被 DBus 环境错误阻断。
- `scripts/humans/plan.py`：真实 SMPL neutral 资产审计和 CPU 规划通过；`scripts/humans/simulate.py --dry-run`：通过，`stand_neutral.mesh.npz` 为 `(121, 6890, 3)`、`(13776, 3)`，数值有限。
- repair3 的失败结果（PD 57.064°、非有限残差、回落 0.1127 m）已被 repair6 复验替代；不要将该历史快照当作当前状态。当前 CPU/USD/Isaac 分层验收通过，AMASS 动作筛选和自由站立/行走控制仍未完成。

### 踩到的坑（均已修复，记录以免重复）

1. **`UsdPhysics.RigidBodyAPI` 没有阻尼属性**：`CreateLinearDampingAttr` 属于
   `PhysxSchema.PhysxRigidBodyAPI`（属性名为 `physxRigidBody:linearDamping`）。已单独封装，
   schema 缺失时记录「未写入阻尼」而不是静默跳过。
2. **`Articulation.get_world_poses()` 只返回根连杆的位姿**（双连杆也返回一行）。逐连杆世界位姿
   必须用 `RigidPrim(path).get_world_poses()` 读。
3. **连杆名与骨骼关节名不同**：根连杆的 prim 名取自 articulation 路径（`Human`），不是 `pelvis`。
   运行时的名字索引改为按规划路径构造，并在名字对不上时直接报错。
4. **接触视图必须在构造时创建**：`RigidPrim(..., contact_filter_paths=..., max_contact_count=...)`；
   否则 `get_net_contact_forces` 抛断言。当前实现仍取不到（记录为能力缺失与原因，不伪造零值）。
5. **物理张量视图只在时间轴播放时有效**：未 `play()` 就读关节位置会抛
   `Instance's physics tensor entity is not valid`。运行封装新增 `play()`/`pause()`。
6. **出生点必须在房间地板上**：公寓是逐房间铺地板，硬编码 `(0, 0)` 落在房间之外，
   人体直接穿过世界下落 40 m。改为从场景配置推导（最大房间内网格采样，取离家具最远且避开墙厚的点）。
7. **空中对照的高度不能穿过天花板**：抬到 +2.0 m 会撞 2.7 m 天花板，实测误差 317 mm。
   改到 +0.9 m 后误差为 0。检查代码里写下了这段经历。
8. **比对参照量必须是引擎自报的关节角**：比 `指令角` 时驱动器正把关节拉回零位，差值被误算成映射误差；
   改成用实测关节角做正运动学，测的才是映射本身。
9. **物理步长与物理速率的混淆**：`execute_trial` 把「速率」当「步长」用，导致 1 s 试验的时间轴导出成
   14400 s，稳定期被压缩成一步，于是所有试验都从倒地的身体开始。已修正并加入时长自检断言。
10. **骨盆高度门限的量纲**：`pelvis_height_fraction` 原先乘的是**身高**，0.55×1.70 m 相当于直立骨盆高度的
    94%，于是重力下正常的 6 cm 下沉被判成「姿态丧失」。改为乘**站立骨盆高度**。
11. **世界锚定（`root_mode: anchored`）不可用**：PhysX 对 `body0 = 空` 的固定关节返回非有限四元数。
    现在模拟入口显式拒绝该模式并给出替代路径，而不是产出 NaN 产物。
12. **参考动作必须与关节折叠自洽**：跌倒参考的根高度按「半个刚体绕足翻倒律」下降，
    否则身体横躺后其他胶囊体会扫到地板以下；下蹲参考的骨盆下降量与腿部折叠不匹配时，
    双脚会穿过地板并被有效性门禁排除（这是门禁在正常工作，不是门禁出错）。

### 明确未验证 / 未达标

- **该历史记录中的“真实 SMPL 蒙皮网格与 AMASS 序列均未取得”已过时。** 当前 SMPL neutral 已实际加载并可导出 `smpl_skin_mesh`；AMASS 序列仍未取得。无模型时才使用明确标记的 `capsule_proxy_mesh`。
- **自由站立与行走控制未达标**（阶段计划允许不以此作为导入成功条件）。
- 多轴关节链未在 Isaac Sim 中运行；`support_loss` 扰动未实现。

---

## 2026-09-23 物理交互 P0 修复（阶段 7 之后）

对照 `docs/physics-interaction-audit.md` 的根因清单，本轮完成 P0-1 与 P0-2，并在修复过程中
发现并修掉了第三条更隐蔽的缺陷。全部结论均有 GPU 实跑证据。

### P0-1 出生/站立高度两种约定混用 —— 已修复

`rig.py` 把 `ground_offset_m`（体坐标系量）当 `spawn_root_position`（根连杆世界坐标）用，
导致出生点系统性偏移。现已统一到同一约定，`verify.py` 双侧确认：

- `[PASS] CPU model puts the lowest capsule on the floor -- lowest capsule point at +0.000000 m`
- `[PASS] spawn height and ground offset use one convention -- spawn root z 1.012637 m, ground offset 1.012637 m`

> 更正：早先审计里写的「出生悬空 0.2336 m」是把 `|pelvis.rest_position[2]|` 误读成悬空量。
> 两份约定的真实差值是 **−0.044589 m**，方向是**下沉**而非悬空。已在审计文档中更正。

### P0-2 接触通道 —— 已修复并改为几何归因

`omni.physics.tensors` 的接触视图在本机不可用（插件报
`Pattern '/World/Human' did not match any rigid contact for filters`，随后
`rigid_prim.py:1924` 抛 `AttributeError: 'NoneType' object has no attribute 'check'`）。
可用的替代是 `omni.physx` 的 `get_contact_report()`，但它返回的是**一对不透明数值句柄**，
且实测 `PhysxContactReportAPI` 是**按 actor 而非按 collider** 生效的（摘掉全部 19 个人体胶囊，
报告仍是同样 6 对；摘掉环境侧才清空）。因此**逐连杆归因不可能从句柄做出来**。

改成**几何包含**判定：拿报告里的接触点去和实时世界坐标胶囊体做包含测试。GPU 实测：
`attribution=geometry, limbs=['left_ankle', 'right_ankle']`，单次试验 632 行接触覆盖 121 帧。

同时消除一处**静默降级**：原先三处 `except ContactSourceUnavailable` 会把「通道坏了」和
「没碰到东西」混为一谈，一次异常就能把整段接触历史抹成空而所有检查仍报 PASS。现在：
逐帧中途失效 → 抛错拒绝出结果；稳定期可读但归因不到连杆 → 抛错；稳定期确实零接触 → 单独报错。

> **2026-09-23 追加：把「请求坏视图」这件事本身也修掉了。** P0-2 第一步曾把
> `simulate.py` / `verify.py` 的 `enable_contact_views` 打开成 `True`（理由是「必要但不充分」）。
> 实测这个请求有代价、无收益：只要请求，`omni.physics.tensors` 就为**每条过滤路径**打一条
> `[Error]`，同一场景单次运行共 **4,368 行**；更关键的是致命的一步发生在事件回调里——
> `RigidPrim._on_physics_ready` 由 `SimulationManager` 以 weakref proxy **异步派生**，
> 其中 `PhysxRigidContactView.check()` 解引用空 `_backend` 抛出的 `AttributeError`
> （`'NoneType' object has no attribute 'check'`）**不经过脚本里任何 try/except**，直接进
> app 的 stderr；时间线停止后 physics-ready 会再触发一次，所以 `--gui --hold-seconds -1`
> 按中断时看到的那一屏正是它。两个入口现改为 `enable_contact_views=False`，`HumanRuntime`
> 新增 `contact_tensor_view_reason`，把「没请求」与「请求了但坏了」「没有可请求的对象」
> 分成三种事实记录。
>
> 真机对照（`/tmp/probe_contact_views.py`，同一 stage 各跑一次，两个方向都跑通）：
> 请求 → **4,368 行 `[Error]`**；不请求 → **0 行 `[Error]`**，`[Warning]` 只剩显卡/手柄等
> 环境噪音。端到端复跑用户原命令（去 GUI）
> `simulate.py --trial stand_neutral:none --trial stand_neutral:push_backward` 全 PASS：
> 接触 20 对、归因 `left_ankle/right_ankle`、每试 601 帧接触历史照旧
> （`push_backward` 覆盖 9 个身体段），`contact_source=physx_contact_report` 不变。

### 新发现并修复：支撑面高度被 bbox API 静默算错

`contact_is_support` 依赖「支撑面高度」标量，这条链路连踩三层：

1. `UsdGeom.BBoxCache.ComputeWorldBound` 返回**错的参考系**——不带 `xformOp:scale` 也不带
   prim 自身 translate。地板世界顶面应为 `0.00`，它返回 `−0.0`；其上方的天花板返回同一个值
   （应为 2.85）。错值恰好压线通过容差，所以从未暴露。
2. 改手算时 `BBox3d`/`Range3d` 在本构建里**都没有 `TransformBy`**（`hasattr → False`），
   按 USD 文档写的调用直接抛 `AttributeError`，又被上面的 `try/except` 吞掉。
3. `ComputeLocalBound` 本身也错：返回**已缩放但仍被 prim translate 偏移**的盒
   （地板实测 `z ∈ [−0.12, 0.0]`，几何真值 `[−0.06, +0.06]`），再叠一次 translate 就成了 `−0.06`。
   它对 `render`/`proxy`/`guide` 还返回反向无穷哨兵盒。

**独立交叉验证**（不依赖任何 bbox API）：实测接触点全部落在 `z = 0.00000`、`normal_z = +1.0000`，
与场景 authord 的 `−0.06 + 0.06 = 0.00` 吻合；`docs/progress.md` 早先独立记录的
「地基顶面 −0.12 m，行走面仍为 0 m」也一致。

**修法**：绕开 `BBoxCache`，从 prim 自身 `size`/`radius`/`height` 构造居中几何盒再变换 8 个角点。
本场景碰撞体全是 `Cube`(197) 或 `Cylinder`(34) 且都带 `size`，此法完备；未知类型返回 `None`。

**效果**：`support_surface_height_m` 从 `0.0`（错值压线）变为 `−1.34e−9`（浮点零，真值）；
`contact_is_support` 从 **True=0 / False=632** —— 一个躺在地板上的身体却判「无任何支撑接触」——
变为 **True=456 / False=176**：456 条全部 `z = 0.00000`、`normal_z = +1.0000`（躺在地面上），
176 条是躯干/肩/肘撞立面的冲击（高度最高 1.45 m）。回归测试 CPU 可跑、不依赖 `pxr`。

### 本轮验收

```bash
python -m compileall -q src tests scripts   # OK
uv tool run ruff check .                    # All checks passed!
python -m pytest -q                         # 213 passed, 8 skipped
DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/verify.py                  # 31 PASS / 0 FAIL
DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/simulate.py --trial stand_neutral:none
```

`verify.py` 关键行：

```
[PASS] contact channel is readable -- 2 pairs on the probe step, source physx_contact_report
[PASS] reported contact points resolve to body limbs -- attribution=geometry,
       limbs=['left_ankle', 'right_ankle']
[PASS] support surface height is readable -- -1.341104505225843e-09 m
[PASS] raised body falls back under gravity -- pelvis descended 0.9338 m from 1.2289 m
[PASS] no body geometry is pushed through the floor after settling -- lowest body point -0.0000 m
```

### 仍未做（诚实记录）

- **P0-3**：稳定期仍每步 `set_root_pose` 传送骨盆。「站着」不是因为地面支撑，而是因为每步都在传送；
  `settle_used_root_support: true` 是诚实记录，但首帧从不处于力学平衡。这是目前最大的剩余 P0。
- **P0-4**：`view_amass.py` 的 physics replay mode 未实现。
- 后果之一：`stand_neutral`（一个**站立**片段）仍被标成 `fall`（`peak_descent_speed 6.20 m/s`、
  `final_trunk_angle 95.9°`、最低点 `−0.0406 m` 穿地 41 mm）。根因就是 P0-3。

---

## 2026-09-23 摔倒 Mesh 采集（分支 `feature/fall-mesh-capture`）

用户指令：P0 改完后提交并开新分支，采集**摔倒动作**的人体 3D Mesh + 三维坐标序列
`(x, y, z)`，为导入 Sionna RT 做准备；**只采集摔倒**；要求「多方面确认」准确性。

### 修掉的缺陷：SMPL 导入被偏航 90°（真缺陷）

详见 [`mesh-orientation-defect.md`](mesh-orientation-defect.md)。摘要：

- **文件帧 ≠ 管线帧**。授权 pkl 是 `X=横向(左右) Y=上 Z=前`；管线（`RestSkeleton` 文档）
  是 `X=前 Y=左 Z=上`。两者差一个绕垂直轴的 90°。
- 旧代码用 `up_axis_conversion("y","z")`，它**只保证 up 不变**（docstring 的前提「两帧共享
  +X 前向」对 SMPL 文件不成立），于是把「文件 X(横向)」映射到「管线 X(前)」——人体侧着站，
  而 up 仍朝上，所以它自己的检查全过。
- 修为 `body_frame_conversion(up=, forward=, left=)`（完整解剖三元组，强制 `det=+1` 拒绝镜像）
  + `_derive_body_frame()` 从关节实测三元组并交叉验证 + `_check_standing_frame()`。
  证据记入 `SmplModel.source_frame`（现报 `up=y forward=z left=x`）。
  `_dominant_axis` 删除；`up_axis_conversion` 保留给只需要 up 的 AMASS retarget 路径。
- 实测：模板 extent 由 `[1.5341, 0.1546, 1.4963]` → `[0.2905, 1.7451, 1.7174]`；
  与程序化骨架的方向余弦 ≥ 0.9965。

### 顺带修掉的隐患：`fit_mesh_to_rest_joints` 的关节配对

原实现按行号裸 `argmin` 配对，而 rig 缩放到配置身高、pkl 保持原身高，**正确配对**也会差
几十毫米（踝 53 mm、脚 91 mm）。实测 ≥1.75 m 时 argmin 变成**非双射**，静默返回
scale `1.0794` 而非真实的 `1.0459`。现改为**按共用关节名取对应** + 用**关节间距离**确认
（绝对坐标受绕原点缩放影响会把正确配对判错；骨向量方向分不开共线脊柱关节）。
另断言拟合残差是纯相似：实测 **0 µm**。

### 导出闸门：补上「旋转看不掉」的检查

**原有检查全部绕垂直轴旋转不变**（stature 范围、FK 对独立 CPU 链、身高拟合收敛、胶囊包含、
「mesh 会动」），所以偏航 90° 全过 —— 这正是缺陷能活下来的原因。新增两条，
**比较 mesh 与 links**（links 对 links 在坏导出上也过）：

| 检查 | 钉住 | 实测 |
| --- | --- | --- |
| `the body stands up in the first frame` | 俯仰 | 头面顶点 z `+1.5955` > 踝面 z `+0.0907` |
| `the mesh and the links agree on the body's facing` | 偏航（朝向） | 两者横向轴都是 `y`（1.8252 / 0.3038 m） |

**正向对照已做**：把导出网格绕 Z 偏航 90°（extent 正好复现 `[1.8252, 0.3038, 1.7963]`）→
两条都 `FAIL`，报 `mesh lateral axis 'x'` vs `links lateral axis 'y'`。
不会失败的检查不算证据。另把 `stand_neutral` 会误判 FAIL 的「mesh 必须动」改成
「骨架动时 mesh 必须跟着动」（静态参考不动是正确结果）。

### 采集产物

```bash
PYTHONPATH=src python3 scripts/humans/collect_fall_mesh.py --fall-only --overwrite
PYTHONPATH=src python3 scripts/humans/verify_fall_collection.py      # 独立第二实现
PYTHONPATH=src python3 scripts/humans/preview_fall_meshes.py         # 目视确认
```

`artifacts/humans/fall_mesh/`（gitignore，不入库）：

| sample | 帧 | 顶点 | 面 |
| --- | --- | --- | --- |
| `fall_forward_reference` | 145 | 6890 | 13776 |
| `fall_backward_reference` | 145 | 6890 | 13776 |
| `fall_lateral_reference` | 145 | 6890 | 13776 |
| `reach_then_topple` | 121 | 6890 | 13776 |

每样本两个文件（`<id>.mesh.npz` 几何 + `<id>.points.npz` 坐标序列）**共享一条 `time_s`**；
顶点数/面数跨帧不变，单一拓扑 sha256 `19710f11eb74fe51…`。
`coordinate_system: world_z_up_xyz`、`units: m`、`axis_order: xyz`。
`fidelity: kinematic_replay` —— **纯 FK 回放，无重力/无接触**，与 `simulate.py` 的
`physics_trial` 严格区分，不得混入同一清单。

清单新增 `body_model` 块（授权文件 sha256 / 身高 / 顶点面数 / 实测 `source_frame`）与
`points.link_names` / `joint_names`，满足 `AGENTS.md` 的模型版本要求并让 `(N,24,3)` 可定位。
`hashes.mesh_source_sha256` 曾误灌**动作**的 provenance（脚本动作恒 null），
已拆为 `motion_source_sha256` + `body_model_sha256`。

### 本轮验收（CPU，全部实跑）

```bash
python -m compileall -q src tests scripts   # OK
uv tool run ruff check .                    # All checks passed!
python -m pytest -q                         # 227 passed, 8 skipped
PYTHONPATH=src python3 scripts/humans/collect_fall_mesh.py --fall-only --overwrite
                                            # 全部 [PASS]，含新增两条解剖检查
PYTHONPATH=src python3 scripts/humans/verify_fall_collection.py
                                            # ALL CHECKS PASS (4 samples)，独立实现
```

目视确认（`fall_preview.png`）：首帧为张臂站立、皮肤与连杆重合、三个片段各自向
前/侧/后倒至平躺。

### 仍未做（诚实记录）

- 本轮采集**只有 `kinematic_replay`**。真正带动力学的摔倒轨迹需 `simulate.py` 的
  `physics_trial`，而它仍被 **P0-3** 卡住（稳定期每步 `set_root_pose`，PD 撑不住 → 自由落体）。
- AMASS 侧：`--write-slice` 已实现但**未对真实树跑过**（2198 个 npz，stem 重名）。
- `fall_lateral_reference` 的事件标签是 `invalid`（穿地 −0.219 m）—— 纯 FK 回放不查地板，
  这是预期结果而非新 bug；**不能**为了「全绿」放宽 `penetration_limit_m`。

---

## 2026-09-23 查看动作与 P0-3 的边界（分支 `feature/fall-mesh-capture` 续）

用户问：是否必须先做完 P0-3 才能在 Isaac Sim 里看到人物动作？

**不是。** 两件事正交：

- **看得到** 靠 **kinematic replay**：查看器每帧直接写 DOF 与根位姿并清零速度，不跑控制环。
- **P0-3 管的是「物理自己能不能站住」**：`simulate.py` 稳定期每步 `set_root_pose` 传送骨盆
  撑着人体，释放后 PD 撑不住就塌。这只影响自由根试验（`physics_trial`）。

### 修掉的真实缺口：查看器只认 AMASS

`view_amass.py` 的 `--amass-root` 原本是 `required=True`，因此在没有授权 AMASS 数据的机器上
**根本无法看任何动作**——而内置动作库就在旁边，回放循环本来也不关心里来源。改为可选：

```bash
~/isaacsim/python.sh scripts/humans/view_amass.py --motion fall_forward_reference
~/isaacsim/python.sh scripts/humans/view_amass.py --fall-only
```

`--fall-only` 判据随来源改变（AMASS 筛选器 / 库自身的 `fall_reference` 标签）；输出文件名与
报告横幅同样分开，不允许报告写「AMASS preview」却在播脚本动作。文件未改名
（8+ 处文档引用它）。

### 顺带修掉：`verify.py` 的「不发散」判据原来在测缺陷

原判据是无驱动自由落体后骨盆水平位移 ≤ **固定 1.0 m**。修复前它 PASS 是**巧合**——人体被
偏航 90°，倒向 0.52 m 外那面墙被挡住（0.4263 m）。修复后人体按应有方向倒，位移 1.3855 m。

探针实测（逐帧骨盆轨迹）：

```
direction = -2.0 deg from +X        ← 正好是管线前向 +X
horizontal displacement = 1.3855 m  (dx +1.3846, dy -0.0495)
final pelvis height     = 0.3029 m  ← 躺平
```

**这是帧修复正确的独立佐证**，不是新 bug。判据改为断言它名字真正声称的两件事：会停下来
（末段 200 ms 爬行 **0.005 mm**）+ 位移不超过自身体型的可达范围
（`2.0 ×` 站立骨盆高度 = 2.0253 m，覆盖倒伏弧 + 撞地滑移）。界由实测站立高度推出，不是照观测值调。

### Isaac 侧复验（本分支首次跑 GPU）

`DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/verify.py` → **`human verify: PASSED`**

```
[PASS] physics pose matches the CPU model with left_knee at +55 deg -- largest link error 0.00 mm over 24 links
[PASS] joint limits round-trip through USD degrees and Isaac radians -- largest limit mismatch 0.000007 degrees
[PASS] the body does not diverge across the room -- pelvis moved 1.3855 m horizontally and crept 0.005 mm over the final 200 ms (limit 2.0253 m = 2 x the 1.0126 m standing pelvis height)
[PASS] no body geometry is pushed through the floor after settling -- lowest body point -0.0000 m
```

### 新定位的开放项（属于审计 P2-1）

`scene_spawn_point` 选的是「离家具最远」的点，结果落在客厅 `(9.320, 0.520)`，而客厅 bounds 为
`x 8.800..18.480 / y 0.000..6.600` → **离 −X 与 −Y 墙各只有 0.52 m**。向前(+X)倒有 9.16 m 空间，
倒向 ±Y 会在 0.5 m 处撞墙。**需要空间的动作与试验（摔倒采集尤其）都受它影响**，建议优先于 P0-3 处理。

---

## 2026-09-23 全量验证：Isaac 显示 / 物理交互 / Mesh 检查 / Sionna 导入

用户要求：动作在 Isaac Sim 里正确显示、物理世界交互正确、检查导出的 Mesh 图、
查本机 Sionna 环境并验证导入成功。

### 1. 全量测试门禁

```bash
python -m compileall -q src tests scripts   # OK
uv tool run ruff check .                    # All checks passed!
python3 -m pytest -q                        # 238 passed, 8 skipped
DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/verify.py    # human verify: PASSED
DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/simulate.py --headless \
    --trial stand_neutral:none --trial fall_forward_reference:none   # human simulate: PASSED
```

### 2. Isaac Sim 里动作正确显示（新增截图能力）

`view_amass.py` 新增 `--capture-dir` / `--capture-frames`：逐帧等间隔回放并截取视口，
确定式（不看墙钟），可直接出图。修了两个让它"成功却没拍到"的坑：

- `configure_inspection_view` 返回相机路径，**调用方必须把它赋给视口**；此前被丢弃，
  视口一直用默认相机，截到的是房间一角。
- `capture_viewport_to_file` 返回的是 capture delegate，**不是 awaitable**；
  要等的是 `delegate.wait_for_result()`。最后一帧还要再 pump 几帧才会落盘，
  否则留下 0 字节的 `.cap-*` 临时文件。

结果：6/6 帧落盘，逐帧像素差 0.995..2.083（修复前各帧完全相同）。
截图见 `artifacts/humans/isaac_captures/`（`fall_zoom_sheet.png`）。

### 3. 截图暴露并修复：人体出生点穿墙（审计 P2-1）

截图里**人体的手臂穿出了外墙**。量出来：出生点 (9.320, 0.520) 离 −X/−Y 墙各 0.52 m，
而 1.70 m 人体的 T-pose 臂展 1.83 m（半幅 0.91 m）→ 手臂出墙 0.39 m。

根因：`scene_spawn_point` 只保证**家具**间距（`SPAWN_CLEARANCE_M = 0.40`），
没把人体自身的外延算进去。修复：新增 `human_spawn_clearance_m(standing_height_m)`
= `SPAWN_CLEARANCE_M + 0.55 × 身高`（1.70 m → **1.335 m**），由
`resolve_spawn_point(..., standing_height_m=...)` 传入；6 个调用点全部接上。
新出生点 **(10.255, 1.455)**，离两墙各 1.455 m。采集数据随之重跑。

### 4. 物理世界交互

`simulate.py` 实跑：接触通道 18 对、接触点归因到肢体（`left_ankle`/`right_ankle`）、
支撑面高度可读、标签由轨迹推出、`human simulate: PASSED`。
张量接触力视图仍然不可用，按设计 `[SKIP]`（改用 `physx_contact_report`）。

**P0-3 仍未做，且这次量化确认了它**：`stand_neutral`（站立）被标成 `fall`
（0.19 s 姿态失稳、0.28 s 撞击、末态身高 0.31 m = 躺平）。原因不变：
稳定期每步 `set_root_pose` 撑着，释放后 PD 撑不住。
**交互本身（接触/重力/归因/不穿透）是对的；错的只是"自由站立"这一项。**

### 5. 导出 Mesh 检查（数值 + 图）

- **拓扑**：`V=6890 F=13776 E=20664`，**Euler = 2，边界边 0，非流形边 0**
  → 封闭、水密、亏格 0 的曲面，无破洞/翻面/退化拓扑。
- **面积**：全程 1.991 m²（reach_then_topple 1.981..1.991），无塌缩或爆炸；
  最小三角形 2.4e-07 m²（非退化）。
- **蒙皮贴着骨架**：每个顶点到最近连杆的距离 p95 = 0.172 m、max = 0.218 m，全程不变
  （若蒙皮脱离骨架这个值会暴涨）。
- 图：`artifacts/humans/fall_mesh/fall_preview.png`（序列）、
  `artifacts/humans/isaac_captures/fall_zoom_sheet.png`（Isaac 视口）。

### 6. Sionna RT 导入（新模块 + 已验证）

环境：`/home/gsh/.local/opt/sionna/bin/python`，**sionna-rt 2.1.0**、mitsuba 3.9.1、
torch 2.11.0+cu130、CUDA 可用（RTX 4060）。`verify_sionna.py` 全通过。
详见 [`sionna-import.md`](sionna-import.md)。

新增 `src/sim2sense_fall/sionna/`（`mesh_import.py`，运行时惰性导入）与
`scripts/sionna/import_fall_mesh.py`。API 事实（写错会静默失败）：

- 几何不能走 `Scene.add()`（只收 Tx/Rx/Material），要走 **`Scene.edit(add=...)`**；
- `SceneObject(mi_mesh=...)` 可直接吃内存里的 `mitsuba.Mesh`（`mitsuba.traverse` 填）；
- `RadioMaterial` 是**场景级**资源，`edit(remove=)` 不会摘掉 → 每帧新建会在第二帧撞名，
  必须注册一次复用（`scene_radio_material`）；
- `capture_viewport_to_file` 的返回值**不是 awaitable**，要等 `wait_for_result()`。

人体电磁材质：ITU 表里**没有人体组织**（只有 19 种建材，含 `vacuum`）。
默认取软组织常数 ε_r=51.0、σ=2.16 S/m、厚 0.02 m，
`HumanMaterial.as_dict()` 带 `provenance: "modelling assumption, not measured"`，
**正式出结果前需补文献**。`vacuum` 是错误选项——它让电磁意义上不存在人体。

实测（12 帧，3.5 GHz，正向对照=同一场景有无人体）：

```
[PASS] the body changes the channel (positive control) -- 最大 +6.03 dB
[PASS] the channel changes as the body falls -- 全程散布 10.15 dB
[PASS] the ChannelSample satisfies the data contract
```

逐帧可解释：站立时 **−4.10 dB**（身体挡住直射径），倒平后 **0.00 dB**（不再遮挡，
信道回到无人体基线），最后一帧 **+6.03 dB**（变成反射体）。
这正是跌倒检测所依赖的"信道随姿态变化"。
产物：`artifacts/sionna/fall_import/`（`.cir.npz` + `.import.json`）。

### 明确边界

- Sionna 场景用的是**内置 `floor_wall`**，不是本项目的公寓（公寓→Sionna 是另一件事）。
- 人体几何是 `kinematic_replay`，传播是真的 Sionna RT —— 信道是非物理轨迹的函数。
- 12 帧、max_depth=4 是**导入验证**，不是数据集生成。


## 2026-09-23 独立复审（进行中）

- 起点 6084df8，原总结只到 c6761de，工作树已有审计文档修改，已保留。
- 原 238 passed / 8 skipped 已复现。发现 Sionna 丢弃虚部、逐帧改变时延网格、按文件名标 fall、未保存完整 ChannelSample 元数据；原 dB 数值撤回。
- 已实现复数 sinc CIR / 固定绝对时延网格 / 等间隔源帧 / 显式 seed / 轨迹标签 / 完整 provenance。公寓 231 个几何部件转换通过 CPU，GPU 实际求得 60–89 条路径（5 帧试验）。
- IFAC/Gabriel 来源已核对并写入 docs/human-em-material.md，Muscle 3.5 GHz 参数更正为 εr=51.44423，σ=2.55752 S/m。
- 稳定期改为真实积分而非位置写入；无辅助站立仍在诊断，几种 PD/足部几何实验失败，失败产物保存在 artifacts/review_20260923，未标成成功。
- 环境：沙箱 GPU 不可见，提权后 RTX 4060 可用；默认网络代理 127.0.0.1:7897 不监听，直连公开来源成功；uv snap 报 DBus Process 2 is a kernel thread，改用已有 /home/gsh/.local/bin/ruff。

## 2026-09-23 本轮完成状态（取代上方“进行中”）

- 完整复核与复现命令见 [verification-2026-09-23.md](verification-2026-09-23.md)。
- 静态站立：5秒、无根支撑；下降1.195 mm、漂移7.593 mm、倾角4.668°、关节误差0.776°。新 stable 配置保留有限力矩，足部使用显式平足胶囊近似。
- 后推物理跌倒：0.833 s失稳、0.933 s撞击代理，最低网格点−32.1 mm，通过既有−50 mm门槛；非零穿透如实记录。站立/后推最终两项均usable。
- 公寓231个几何部件 + 实际后推人体 → 11帧复数CIR，在Sionna CUDA实跑通过；相对无人非相干增益约−11.74..−0.013 dB，旧数值撤回。
- 材质来源已核对；相机最终四帧已目视确认；AMASS真实2198条树已切片，3条CMU导入通过。
- 最终门禁：compileall/ruff/diff通过，pytest 246 passed / 8 skipped，bundled USD 9 passed，scene/human Isaac验收通过。
- 未完成：自然行走/恢复、前推及控制失效穿地、动态家具传播同步、人体EM测量校准、50Hz数据集和检测模型。当前是单场景两类状态的smoke，不是完整研究系统。
- 产物：artifacts/review_20260923/final_physics、final_physics_channel、captures_final、verification_summary.json。大文件保留在忽略目录，未提交/推送。

## 2026-09-24 app.update() 时序缺陷实测与修复（Fix B）

- **实测**：`scripts/humans/probe_loop_timing.py`（自由落体正对照）在本 build
  （Isaac Sim 6.0.1-rc.7，python kit）上测得一次 `app.update()` 推进**固定 1/60 s
  物理时间**（dt=1/120 s 时每 update 2 步；60 updates = 120 步 = 1.0 s，球落
  4.946 m ≈ ½g·1.0²）。与 `omni.kit.loop-isaac` 开关无关（两模式逐字节一致），向
  python kit 注入 GUI kit 的 `--/app/runLoops/main/manualModeEnabled=true` 与
  `rateLimitEnabled=false` 也无效（参数确实传到 kit 命令行，行为不变）。日志：
  `artifacts/humans/probe_loop_baseline.log`、`probe_loop_enabled.log`、
  `probe_loop_manual.log`。
- **确定性步进器**：`SimulationManager.step(steps=n, update_fabric=False)` 恰好推进
  n 步且时钟与自由落体吻合（60 步 = 0.5 s，球落 1.247 m vs 期望 1.226 m）。
  注意两个新的静默原生崩溃源：`update_fabric=True` 与 `timeline.set_play_speed(0.5)`
  —— 均 exit 0、无 traceback（`probe_loop_halfspeed.log`）。
- **修复**：`scripts/humans/common.py::step_physics(steps)` 封装 manager.step 并读回
  步数计数断言（步进器静默失败会显式 RuntimeError）。verify.py 的 `hold()` 与
  rest-sample 窗口、simulate.py 的 settle 窗口与逐帧试验循环全部换用；
  `app.update()` 只保留在注册/视口/回放位置。`GRAVITY_SETTLE_SECONDS` 3.0→6.0
  （旧标定在 2× 时序下完成，实际物理时长 6 s；换算保真）。
- **复验**：`artifacts/humans/verify_multiaxis_stepper.log` 全绿 37 PASS / 0 FAIL。
  重力正对照：pelvis 自 1.2381 m 下降 1.1031 m，末段 200 ms 蠕动 0.085 mm
  （门槛 5 mm）；PD 跟踪最大误差 11.512°<15°；沉降后最低点 −0.0000 m 无穿地。
- `src/sim2sense_fall/scenes/usd.py::step_simulation` 的 docstring 已改为记录实测
  事实（每 update 固定 1/60 s）；scenes 侧秒数窗口仍是旧标定，使用前需重测。
- CPU 门禁：compileall 通过、ruff 全过、pytest **269 passed / 9 skipped**。

## 2026-09-24 AMASS 根运动锚定缺陷与修复（Fix A）

- **实测**：失败试验 `trials_multiaxis_amass.log` 的 −5.0 m pelvis 是两个缺陷叠加：
  (1) `simulate.py` 把采集坐标系 `root_translation[0]` 直接当公寓世界位姿，clip
  10_05 的 pinned xy=(11.211, −0.683) 不在任何房间 → 身体坠入虚空；(2) settle 窗口
  在 2× 时序下实际 1.2 s，自由落体足够坠到 −5.0 m。
- **修复**：`simulate.py:main` 筛选循环先 `normalize_root_motion`（与
  `view_amass.py:181` 一致）再 screen，试验共享同一锚定坐标系。
- **CPU 筛选对照**（57 DOF 多轴 rig，`/tmp/amass_subset` 前 12 条，归一化后）：
  12/12 `rig_expressible=True`（残差 0.000 rad），**3 条 accepted 跌倒候选**：
  `135_02`（trunk 172.7°，drop 0.120 m）、`17_01`（177.6°，0.199 m）、
  `22_03`（63.1°，0.778 m）。10_05 归一化后 fall=True→False：raw 的候选是被采集
  全局朝向（整段倾斜）抬出来的假象；归一化后「第 0 帧直立」更符合「从站立跌倒」
  的试验语义。
- **Isaac 物理试验**（`--amass-root /tmp/amass_subset --amass-limit 12
  --amass-fall-only`，多轴 rig，headless）：结果见
  `artifacts/humans/trials_amass_anchored.log` 与 `artifacts/humans/trials_amass_anchored/`。

## 2026-09-24 足部摩擦能否解决滑步（P0-B 子项，只读分析）

问：能不能像真实世界一样给脚加摩擦力解决滑步。答：**不能单独解决**，已定量。
新增常驻工具 `scripts/humans/audit_friction_budget.py`（CPU、不需要 Isaac、只读已有会话），
证据 `artifacts/humans/friction_budget/report.json`，完整口径见
[足部摩擦审计](foot-friction-audit.md)。本轮没有改任何物理参数。

- **先修了两个盲区。** (1) 人体碰撞体**没有任何物理材质**——`human_trial.usda` 里只有
  `PhysicsCollisionAPI`/`PhysxCollisionAPI`/`PhysxContactReportAPI`，没有 `MaterialBindingAPI`，
  摩擦走 PhysX 默认 0.5 再与地板 0.55 平均 → 有效 μ≈0.525，且**不进任何配置、不进 provenance**。
  所以「把脚的摩擦调大」当前无处可调，这也正是配置里 `support_loss` 被标为未实现的原因。
  (2) **摩擦力从未被测量**：`contact_samples` 存的是接触报告的 `impulse`，在全部归档会话上
  实际检查 **3286 个非零样本全部与接触法向平行，最大偏差 2.22e-16**——该字段只有法向分量。
- **摩擦需求改为从动力学反推**：`F_c = m(a_COM+g) − F_assist`，`a_COM` 由记录关节角走
  `rig.forward_kinematics` 求质心后两次差分。**正向对照**：重建法向载荷 vs 接触日志实测，
  `cycles` 为 p50 **120 N vs 122 N**，`locomotion_v4` 为 **122 N vs 122 N**，方法自检通过。
- **`cycles`（2040 帧，10413 个踝-地板样本）**：地面承担体重 0.17，根部外力竖直 p50
  **581 N = 0.82 体重**（配置写 0.7）；`mu_required` p50 0.198、p95 0.674、p99 1.401；
  超过假定 0.525 的帧 11.5%，超过 1.0 的 2.0%；滑动帧 355/1933 = 18.4%，其中
  **切向需求 ≥ 上限的只有 86 帧 = 24.2%**，且它们的 slip p50 **0.358 m/s 小于**
  其余四分之三的 **0.528 m/s**——摩擦受限的那批不是最严重的。
  滑动帧的 `mu_required` 直方图在 **[0.5,0.6) 最高单峰（69/355）**，与「默认 0.5 平均地板 0.55」
  推出的 0.525 吻合，等于用数据自认了有效系数；该区间滑动率 48% 对整体 18.4%，差 2.6 倍。
- **稳健性与跨会话**：平滑窗 3/5/9/15/25 帧时「摩擦受限滑动占比」为
  20.4/24.2/31.4/39.1/44.2%，最坏也不到一半；`locomotion_v4` 为 18.5%。
  反向对照 `actions_final`（**根部辅助关闭**）地面承担 1.00 体重，`mu_required` p50 **0.004**、
  滑动帧 1.1%。注意它只做站立/蹲起，不是同类动作的干净开关对照。
  `locomotion_v6/v8` 显示 52.5%/62.7%，但那里地面实测法向 p50 仅 2 N，分母趋零，不作证据。
- **机制**：摩擦上限 μN 与 N 成正比，辅助举走 0.82 体重把上限压到真实世界的约 1/5；
  切向需求由 57 个 PD 目标加一个上限 1500 N 的根部弹簧决定，摩擦冻住脚只会把需求转成
  关节误差或继续被弹簧拖（弹簧 1500 N 对脚底 64-120 N，每次都赢）；
  且 P0-B 已归档的「参考自身支撑脚中位滑 0.134/0.108 m/s、锚点残差 111 mm」说明
  摩擦留不住一个目标在动的脚。
- **未验证假设**：脚部代理是 0.228 m 胶囊而非平面脚掌，绕长轴滚动时库仑摩擦不提供阻力矩；
  现有滑速口径量接触点切向速度，**纯滚动不会被捕获**。本轮未量测，不作结论。
- CPU 门禁：compileall、`ruff`（新脚本）全过、pytest **326 passed / 9 skipped**。

## 2026-09-24 路线 2 实跑：让地面承重（否证）+ 路线 3 起步（一个假设被否证）

承接同日足部摩擦分析。用户选定执行路线 2 → 3。

### 路线 2：把法向载荷还给地面 —— 已实施、已实跑，**预测被否证**

- **改动**：`root_wrench` 原只加 `gravity_compensation_fraction·m·g`，竖直位置弹簧随后仍能
  再往上推，故执行器实际举 0.82 体重而非配置的 0.70。新增
  `max_vertical_lift_fraction_of_weight` 夹住执行器**总向上力**，使配置值成为真天花板；
  校验不得 >1、**不得低于补偿比例**（否则夹的是补偿本身，报错直接教你去降补偿）。
  `configs/humans/root_assist.yaml` 出厂值 **1.0**。回归 3 条 + **变异检查**
  （删掉夹取那一行，2 条立刻失败）。
- **A/B**（同 seed 同 10.3 s demo 各一条确定性运行，`artifacts/humans/assist_cap_ab/{before,after}/`；
  `before` 复现改动前行为：关节 5.703°、滑速 p95 0.6573，对上历史 5.719° / 0.651）：

  | 指标 | cap 1.0（改动前） | cap 0.70 |
  | --- | ---: | ---: |
  | 地面法向 p50 | 122 N（0.17 体重） | **212 N（0.30 体重）** |
  | 执行器竖直 p50 | 586 N（0.83 体重） | **494 N（0.70 体重）** |
  | μ_req p50 / p95 | 0.206 / 0.832 | **0.163 / 0.642** |
  | 需求超 μ 帧 | 14.2% | **8.6%** |
  | 滑动帧占比 | 22.0% | **38.1%（更差）** |
  | 其中摩擦受限 | 36.1% | **13.6%** |
  | 滑速 p95 | 0.6573 | **0.6739（更差）** |
  | 超 0.15 样本 | 14.0% | **30.7%（更差）** |
  | 最大关节误差 | 5.703° | **8.713°（更差）** |

- **读数**：载荷侧完全按设计工作（地面正好是反事实预测的 212 N，执行器被压回 0.70，
  摩擦受限滑动 −62%），**但总滑动 +73%、滑速与跟踪同时变差**——腿扛不动多出的载荷，
  骨盆下沉，下沉变成脚的运动。**因此 0.70 按实测回退，出厂取 1.0**（同旋钮最松端）。
- **口径保留**：cap=1.0 在 max 处确实夹到 706.3 N（旧行为无夹取），故该臂比真旧行为略多载荷，
  这次对比对 `after` 的劣势略有低估；每臂一次确定性 demo，不是统计。
- CPU 反事实预测 cap 0.70 时摩擦受限滑动 −54%，实测 −31% 但总滑动 +83% → 净负。
  反事实冻结运动、只移动 μ 的分母，高估收益，与其「上界」标注一致。

### 路线 3：参考自身踩不住的来源 —— 起步，**否证一个我自己的假设**

- **否证**：把根运动由「常数中位速度」换成「源片段自己的根平移」**几乎不改变漂移**
  （travel-support 掩码内世界踝速 p50：左 0.115→0.114、右 0.098→0.148 m/s）。
  漂移在根输运**上游**，是重定向周期自身的性质。
- **线索（不得当结论）**：前进片段 `CMU/08/08_04` 左右支撑退行速度差 34%
  （0.721 / 0.966 m/s；`gait.speed_m_s` 为合并中位 0.777），后退片段两腿一致（1.0905 / 1.0798）。
  不对称是 09-24 为修右臂新换的前进源特有的。
- **未做**：源片段（重定向前）自身左右支撑退行速度对照；支撑掩码正确性检查
  （188/290 腿-帧被标 travel-support，偏高，可能把摆动相算成支撑）。
- CPU 门禁：compileall、ruff 全过、pytest **329 passed / 9 skipped**（326→329）。

## 2026-09-24 路线 3 实测：支撑掩码本身不合格，P0-B 前提数字需重述

用户要求「按推荐来，但要注意实测测试」，故本项全程实跑（CPU、只读、无 Isaac）。
新增常驻审计 `scripts/humans/audit_support_mask.py`，证据
`artifacts/humans/support_mask_audit/audit.json`，文档 [支撑掩码审计](support-mask-audit.md)。

- **掩码定义**：`load_gait` 只用**水平位移**（脚 x 与根前进反向）推支撑相，并用
  `|stance_speed − gait.speed_m_s| ≤ 0.15` 加门。**从不看脚的高度，也不看世界系静止。**
- **工具自检先行**：审计里的骨盆相对退行速度，其**合并中位数必须逐位等于 `gait.speed_m_s`**
  （实测两条片段均为 True），否则比较的不是同一个量。
- **A. 高度**：掩码帧的脚高于自身周期最低点 p50 **12.6 mm（前进）/ 18.0 mm（后退）**，
  p90 54.1/32.0，**max 63.6/40.5 mm**；>50 mm 占 12%/0%。
  **44.1%（前进）/ 39.9%（后退）的掩码帧脚不在最低点附近**；加出厂门后前进片段
  「明确悬空」由 **6.9% 升到 12.0%** —— 门按速度筛帧，把摆动中段留下、贴地的删掉。
  脚自身最低点实测 −2.27/−8.92（前进）、−0.53/**−12.98**（后退）mm，故「高于最低点」≈离地高度。
- **B. 可踩住**（脚在地面 **且** 世界系 <0.05 m/s）：掩码帧里只有 **35.0%（前进）/ 43.3%（后退）**
  满足，其余 65-70% 是构造上不可达的锚点；出厂门**丢掉 40.7%（前进）/ 10.0%（后退）**可踩帧
  （`travel` 原始掩码一帧不漏，是门加出来的错）。
  → **P0-B 归档的「参考支撑脚中位滑 0.134/0.108 m/s」建立在一个多数帧踩不住的集合上，
  既高估参考缺陷、又丢掉四成可踩帧；重述前不得作为换源/改锚点的依据。**
- **C. 参考能否踩住**（每帧取两脚世界水平速度较小者）：源根平移下 <0.05 m/s 的帧占
  **40.7%（前进）/ 34.5%（后退）**，常数根速下只有 **26.9% / 21.4%**。
  **修正上一条的过头结论**：上一条说「换成源根几乎不改变漂移」，那只在**掩码中位帧**上成立；
  在**最慢脚**这个更该用的量上源根明显更好（+/13.8 pp、+/13.1 pp）。
  但**两条臂都不够好：59-66% 的帧两只脚都在动**，参考在多数帧里根本没有落地脚。
- **D. 左右不对称口径更正**：在骨盆相对量、且两腿都被标的相位窗上复算，
  前进片段左 0.736 / 右 0.960 m/s，**差 23.4%**（上一条误写的 34% 系比较口径错误，已更正）；
  合并中位 0.777 由**左腿**定出（左腿占 188 掩码帧的 100）。后退片段两腿差 **1.0%**（对称）。
  后退共同相位窗仅 2/12 bin，其差值不可引用。
- **未做/不得声称**：源片段**重定向前**自身的左右对照未做，故**不得**归因不对称到源或重定向；
  「在地面上」判据（胶囊最低点相对该脚周期最低点，容差 20 mm）未与真实接触报告交叉验证。
- **下一步据此改变**：不是「换源」，而是**用锚点真正需要的几何量重建掩码**
  （在地面上 **且** 世界系静止 **且** 用源根），再看还剩多少可下锚的帧。
- 未新增 pytest 覆盖：现有 `tests/humans/` 无一加载真实 gait（都会引入 AMASS 数据依赖），
  本工具按仓库既有「常驻审计脚本」模式（同 `audit_stance_ik.py`、`gait_seam_audit.py`）交付。
- CPU 门禁：compileall、ruff 全过、pytest 329 passed / 9 skipped。

## 2026-09-24 判据换成 PhysX 实测接触（用户方向调整）

用户明确：主方向是「SMPL+AMASS 在 Isaac Sim 里把动作做出来」，**没必要为了可 CPU 测试
丢失准确性**。据此把判据从几何代理换成模拟器实测，并对上一条几何审计做修正。
新工具 `scripts/humans/audit_mask_vs_contact.py`，证据
`artifacts/humans/mask_vs_contact/all.json`，文档 [掩码对实测接触](mask-vs-contact-audit.md)。

- **方法**：对已有实跑的每帧每脚读三个独立陈述——掩码的 `claimed`（来自 `gait_phase`）、
  **PhysX 的 `contacted`**（法向冲量 > `slip_min_impulse_ns`）、以及由**记录的实际关节状态**
  做 FK 得到的脚高度（用于归因）。只统计纯步态帧（`gait_weight ≥ 0.95`）。
- **四条独立会话（实测）**：

  | 会话 | 计入帧 | 左 claimed/contacted | 右 claimed/contacted | 否掉掩码的支撑 | 漏掉的真接触 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | before | 396/1236 | 159/154 | **113/157** | 18.4% | 28.6% |
  | after | 396/1236 | 159/168 | **113/185** | 18.0% | 36.8% |
  | cycles | 1332/2040 | 572/625 | **385/735** | 16.5% | 41.2% |
  | locomotion_v4 | 396/1236 | 159/153 | **113/205** | 27.9% | 45.3% |

- **主症是漏报，不是误报**：**右脚真实地板接触 46.5/54.1/54.8/56.1% 不被认作支撑**
  （左脚 10.4/17.9/25.3/30.7%）。**误报的帧上脚只差 1-5 mm**（三条会话；locomotion_v4 为 19 mm），
  是接触阈值边缘。→ **上一条几何审计夸大了误报**（它拿参考当身体量），
  但对漏报方向的估计正确（几何版 40.7% vs 实测右脚 46.5%，同量级）。
- **与参考侧不对称互相印证**：几何审计量到右脚退行速度比左脚快 25%、
  `gait.speed_m_s` 由左脚定出；本文件在**物理侧**独立得到同一只脚被系统性漏掉。
  两套数据、两个方法，指向同一因果链。
- **已实施**：`contact_control.measured_support_feet()`（读 PhysX 接触，空集不回落）；
  `keyboard.yaml::anchor_from_contact`（**出厂 false**）；`keyboard.py` 把接触报告改为
  回调开头**读一次**（pre-step 回调，语义不变，此前读两次会重复计数）。
  回归 2 条 + **变异检查**（删地板判定 → 1 条失败）。
- **A/B 未判定**：`artifacts/humans/anchor_from_contact/{off,on}/` 两侧数值出来前，
  **不得**声称接触驱动改善了滑动或跟踪。
- **约定已写入项目长期记忆**：判据用模拟器实测，CPU/FK/几何代理降级为预筛，
  冲突以实测为准。
- CPU 门禁：compileall、ruff 全过、pytest **331 passed / 9 skipped**（329→331）。

### 补：`anchor_from_contact` A/B 实跑结果（同 seed 同 10.3 s demo，各一条确定性运行）

`artifacts/humans/anchor_from_contact/{off,on}/`。`off` 逐位复现文档基线（关节 5.703°、
force_peak 728.2、slip p95 0.6573、above_tol 14.0%、释放 197 次、残差 0.1113 m
—— 正是 P0-B 归档的 197 次 / 111 mm），对照臂干净。

| 指标 | off（出厂） | **on（接触驱动）** |
| --- | ---: | ---: |
| 报告滑速 p95 | 0.6573 m/s | **0.6097（−7.2%）** |
| 超 0.15 m/s 的样本 | 14.0% | **13.0%** |
| **锚点最大残差** | 0.1113 m | **0.0970 m（−12.8%）** |
| 锚点释放次数 | 197 | 192 |
| 最大关节误差 | **5.703°** | 7.561°（变差，仍 <15°） |
| 根外力峰值 | 728.2 N | 767.3 N |
| 皮肤最低 Z | +2.98 mm | +2.99 mm |

**判定：方向对、幅度不够、且有代价。** 它瞄准的两件事都改善（滑速 p95 −7.2%、
锚点残差 −12.8%，后者正是 P0-B 归档「残差 111 mm、锚点不可达」那一项），
**但没有过门槛**（滑速 p95 0.61 仍远高于 0.15），且**关节跟踪变差**（5.703°→7.561°）。
后者可解释：替换式闸门会在「掩码说摆动、PhysX 说接触」的帧上也下锚，
IK 与摆动目标对着干，误差转到关节上——这是**下一步的主要矛盾**。
一条确定性 demo/臂，不是统计。
**未测**：改为「模型 ∧ 接触」的**交集**（只删不加）是否更好。

## 2026-09-24 用户实测：行走一瘸一拐（左腿迈得比右腿大）—— 已定位到参考构造并修复

用户实测反馈：走起来左腿迈得比右腿大很多。全程按「判据用实测/实际管线」做。
新工具 `scripts/humans/fit_gait_windows.py`，证据
`artifacts/humans/gait_window_fit/fit.json`，文档 [步幅窗口拟合](gait-window-fit.md)。

- **量化**（走发货管线：`gait.sample` + `gait.tilt` + 前向运动学）：两脚平均前后偏置
  **forward 247.1 mm**（左脚平均在骨盆前 +108.6、右脚在后 −138.4）、**backward 95.8 mm**；
  步幅比 L/R 0.880 / 1.076。**判据用前后偏置，不是步幅比**——247 mm 的固定错位就是
  用户看到的「左腿一直伸在前面」。
- **根因**：`Gait.sample` 的周期性靠一条线性斜坡伪造
  `q = interp(phase) − phase·(joints[-1] − joints[0])`，**只有窗口是整数步幅时才无害**。
  配置窗口 1.2 s = 144 帧都不是整步幅：端点差 forward **16.95°**、backward **32.87°**；
  而 08_04 真实步幅 ~160 帧 = 1.333 s（近邻姿态匹配 2.80° 与全局周期性两个独立方法都指向它）。
  两腿相差半步幅 → 同一条每-DOF 斜坡在两条腿上呈现**符号相反的准静态偏移** → 两脚固定错位。
- **修复**：forward `start_s 1.9667 + duration_s 1.3083`、backward `0.4 + 1.4833`。
  工具先用**全局周期性**定期（序列性质，用整段比单起点稳），再用**端点闭合**选起点，
  并**拒绝半步幅**（判据「lag-2P 周期性优于 lag-P 即半步幅」，实测拦下了 100-102 / 84-86 帧候选）。
  所有数字都由**调用 `load_gait` 覆盖窗口**得到，走发货管线，不是重新实现。工具幂等。
- **参考侧结果**：环路闭合 16.95→**1.54°** / 32.87→**3.33°**；
  前后错置 247.1→**0.1 mm** / 95.8→**5.2 mm**；步幅比 0.880→**1.020** / 1.076→**0.997**；
  每周期根前进量 932→1154 mm。
- **口径警告**：根前进量变了 ⇒ **改动前后的结果不可直接比较**（含滑速与关节误差历史基线）。
- **物理侧 A/B 待判定**：`artifacts/humans/gait_window_ab/{old,new}/`，出数前不得声称物理上已消失。
- CPU 门禁：compileall、ruff 全过、pytest 331 passed / 9 skipped。

## 2026-09-25 用户实测：站立时腿是弯的 —— 与防滑不冲突，已修复

用户反馈：滑步解决后正常站立时膝盖明显弯着（截图）。问题：两者是否不可兼得。答案：不冲突，
站立弯腿是 idle 被钉在步行高度帽造成的，步行防滑机制不用动。

- **定位**（CPU 量化，工具 `scripts/humans/audit_idle_posture.py`）：`keyboard.py` 的
  `fit_contact_idle` 用 `min(g.height_m.min())`=0.8609 m 拟合站姿。该高度帽是步行必须的
  （支撑相极点脚在髋前后 0.5×0.6×0.62=0.186 m，腿要够到地面），但站立时脚在髋正下方，
  同一根高让腿多出 4.7 cm 富余，两段腿在近伸直区 cos(φ/2)=d/L 的敏感性把它换成 37-43° 屈膝
  （原始 mocap 站姿在自身高度 0.9077 m 只有 14°/8.5°）。`reach_fraction: 0.97` 不是主犯：
  按 0.97 步幅帽站高 0.8842 拟合仍有 24-32° 屈膝。
- **第一次修复尝试（fraction=1.0，完全直腿）被物理否证**：stand slip p95 0.010→0.797 m/s、
  左脚接触占比 0.485。滑移全部集中在 spawn/R 复位后的 settle 与站稳阶段：3° 近奇异直腿是
  轴向刚性支柱，接触脉冲间歇（0↔7.6 Ns），身体反复弹起离地、每次冲击脚底打滑 2-3 m/s。
  基线 37-42° 屈膝时接触恒定安静（imp 1.30 恒定）。结论：站立需要保留少量屈膝作轴向柔顺
  （人放松站立也有 ~10-15°）。
- **最终修复**：新增 `idle_stand_height_fraction: 0.988`（keyboard.yaml，校验在
  `load_keyboard_config`，范围 (0.9, 1.0]），idle 改按站高×0.988 拟合 → 目标屈膝约 8-20°。
- **第二个新缺陷（fraction=0.988 暴露）**：S 松开停车时 stand 模式 slip 0.022 过了，但
  无支撑 66.7 ms>50 ms 门、左髋 dof1 误差 16.0°>15° 门。定位：身高混合是线性斜坡，斜坡
  结束那一帧速率从 +0.097 m/s 硬切到 0，身体带着动量被伸直的腿顶起弹跳（vz 振荡、脉冲
  尖峰 15.9/20.0 Ns）。基线不弹是因为旧配置 idle 高度==步态高度，停车没有身高变化。
  **修复**：`TeleopController.advance` 的 weight 从限速率线性改为指数趋近
  （`w += (target-w)·min(1, dt/transition_s)`），混合速度平滑衰减到零，消除末端动量注入。
- **GPU 验收**（RTX 4060 实 GPU，demo 协议，`artifacts/humans/keyboard/`）：
  forward slip p95 0.019 / 误差 3.8°、backward 0.019 / 2.9°、stand 0.012 / 6.2°、
  双脚接触 0.96/0.97，全部活动门通过（overall accepted: True）。基线 stand 0.010/1.0
  ——滑步水平与弯腿基线等同，站姿侧视渲染（`renders7/`）膝盖基本直。
- **口径**：站立根高 0.8609→0.8966（实测沉降后 0.879）；跌倒阈值参考
  `fallen_height_fraction` 随 idle 高度变成真实站高。站立直腿视觉收益与滑步门同时满足，
  「不可兼得」不成立。
