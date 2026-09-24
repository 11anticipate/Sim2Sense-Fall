# 文献与方案笔记

[工程文档索引](docs/README.md) · [当前计划](task_plan.md)。本页的旧实现记录按日期保留，
“未取得 AMASS”“GPU 不可见”等历史表述不覆盖 2026-09-24 的实际导入和 GUI 证据。

## 2026-09-24 用户需求与后续验证依据

- 新增动作收敛为蹲下、起立、摔倒；起立先按蹲姿回站姿，地面起身另属更复杂任务。
  运动方向由用户键盘决定，不需要主动避障；墙/家具阻挡、支撑和碰撞反应仍必须正确。
- 用户在人工正常行走时发现右臂摆动异常，左臂较正常。这是待复现的视觉反馈，尚无根因证据。
  排查链为原始 AMASS → 裁剪/周期修正目标 → 实际 PhysX 关节 → 蒙皮；
  左右不完全对称可能来自原片段，不能把对称性本身设成正确性的充分条件。
- 三个新动作不能只增加键位：需要支撑切换、状态合法性、受阻处理、摔倒时辅助策略和标签区分。
  特别是蹲姿低骨盆高度不等于跌倒；摔倒请求不等于实际跌倒，R 复位不等于起立。
- 既有物理证据统一见 [键盘实测](docs/keyboard-control.md) 和 [AMASS 审计](docs/amass-physics-audit-2026-09-24.md)。
  本次只整理计划/文档，未新做动作实验。文献事实章节与原始失败记录保持原意。

## 2026-09-22 — Transitions 录屏动作表现与物理边界

- 用户录屏 `/home/gsh/Videos/Screencasts/Screencast from 2026-09-22 23-30-10.mp4` 约 7.05 s；观察到站立、下坐/后仰并抬腿、恢复站立三个阶段。这与 `amass__sit_stand_poses` 的坐下/起立参考动作相符，不应标为摔倒样本。
- `view_amass.py` 仍是 kinematic replay：每帧写入 DOF/root 位姿并清零速度。即使 19 个胶囊带 `PhysicsCollisionAPI`，墙/地面的接触反作用也会被下一帧传送覆盖；录屏不能作为碰撞响应证据。
- 当前碰撞证据应分开表述：`verify.py` 已验证人体胶囊的地面回落和不穿透（重力回落 1.3051 m、最低人体点 -0.0000 m）；墙体碰撞代理已写入场景，但尚未用专门试验验收墙体接触。
- 后续 physics replay mode 需要：PD target 驱动而非每帧 teleport；保留 PhysX 接触；用实际 link pose 更新 SMPL 蒙皮；记录接触事件和时间轴；与仅用于观察动作的 visual preview 分开报告。

## 2026-09-22 — 查看器启动与碰撞边界

- 指定动作 ID 后直接加载同名 `.npz`，避免为显示一个动作解析整套 110 条 Transitions；未指定动作时仍扫描全库。
- Isaac Sim 启动慢的主要时间来自 Kit/RTX/PhysX 扩展初始化和材质/着色器缓存。隔离运行实测约 13.4 s 到 `Simulation App Startup Complete`，当前会话因 CUDA/NVML 不可见而走 CPU fallback；这不是 AMASS 文件读取耗时。
- 人体碰撞代理仍在：19 个胶囊有 `PhysicsCollisionAPI`，但皮肤网格不参与碰撞。`view_amass.py` 是 kinematic replay，每帧设置位姿并清零速度，因而不适合观察墙/地面的物理反作用；需用 `scripts/humans/verify.py` 或后续受控物理播放验证碰撞。
- 最新 `verify.py` headless 实测通过：19 个人体碰撞代理、重力回落 1.3051 m、最低人体点 `-0.0000 m`。因此“没有碰撞”只适用于查看器的传送式回放表现，不适用于 PhysX 验收入口。

## 2026-09-22 — Transitions sit-stand 根位姿与蒙皮同步修复

- `sit_stand_poses.npz` 的首帧 AMASS `trans` 是序列自身的 Y-up 局部坐标，首帧 root rotation 也是 SMPL-H 的全局校准姿态，不能直接叠加到房间出生点。查看器现在按首帧归一化：`t[k] - t[0]`、`R[k] @ R[0].T`。
- 同一帧归一化根位姿同时驱动 CPU 正运动学、SMPL 蒙皮和 PhysX articulation root；首帧因此保持在房间 spawn 点并直立，动作语义是坐下/站起过渡。
- 根旋转的连续性验证采用旋转测地距离，以正确处理跨 180° 的主值轴角表示。
- 实际验证：`python3 -m pytest -q` 为 `201 passed / 8 skipped`，compileall/Ruff 通过；Transitions 980 帧 headless 播放通过，产物 `artifacts/humans/amass_preview.usda/.json`。隔离会话仍是 CPU PhysX fallback，宿主机 GUI 画面需用户用 NVIDIA 已恢复的会话复验。

## 2026-09-22 — 人体自然姿态与聚焦查看修复

- 截图中的手臂穿墙不是墙体碰撞失效，而是 SMPL neutral 的水平 rest pose 未经过显示姿态变换。新增配置驱动显示姿态：`left_shoulder=+pi/2`、`right_shoulder=-pi/2`，左右肘 `-0.20` rad；同一姿态用于 CPU LBS 网格和 GUI PhysX DOF。
- 胶囊体是物理碰撞代理，不是第二层皮肤。USD 现在给 19 个 capsule 设置 `visibility=invisible`，但继续保留 `UsdPhysics.CollisionAPI`，并写入 `sim2sense:collisionProxy=true` / `sim2sense:renderable=false`。
- 新增 `human` inspection view：按 `/World/Human` bounds 加边距取景、隐藏屋顶和灯具；人体 GUI 默认不再按整套公寓缩放。房间几何和碰撞尺寸没有修改。
- 本轮证据：CPU 199 passed / 8 skipped、compileall、Ruff；Isaac headless `human_display_repair` 检查 24 links、14 DOF、19 capsules、19 invisible capsules、6890 skin verts、13776 faces，并通过基础场景哈希检查。隔离运行的 NVML/CUDA 报错仍只表示 GPU 不可见，不代表宿主机驱动状态。
- GUI 首次运行暴露 Isaac tensor 初始化竞态：`runtime.play()` 后立即设置 DOF 会报 `Instance's physics tensor entity is not valid`。GUI 构建入口现与已有 verify/simulate 入口一致，在 `play()` 后执行 4 次 `app.update()` 再设置 position/target；CPU 回归 200 passed / 8 skipped，headless build 继续通过。

## 2026-09-22 — 人体 GUI 遮挡与 SMPL 网格导出修复

- 用户截图复现了两个独立问题：人体入口没有调用已存在的 `configure_inspection_view()`，所以屋顶遮挡室内；`build.py` 也没有把 `body.model.mesh()` 传给 USD author，导致只显示碰撞胶囊。
- 修复后，`build.py` 默认 `--view roofless`，视图修改写入 session layer，不改变屋顶碰撞或基础 USD；`build_human_stage()` 接收经过 rig 尺度拟合的 SMPL vertices/faces，`/World/Human/Skin` 实际写入 6890 vertices / 13776 faces。
- Isaac headless 结构复验通过：24 links、14 joints、9 fixed joints、19 colliders、6890 skin verts；GPU 图形仍受当前隔离执行环境限制，宿主机 GUI 命令需由用户终端运行。

## 2026-09-22 — AMASS 与 GPU/CUDA 修复边界

- AMASS 现在有可审计的本地导入管线：`poses (N,156)` / `trans (N,3)` 校验，SMPL-H 身体块重定向到 SMPL 24 关节，Y-up→Z-up，源文件 SHA-256 和许可元数据随 manifest 保存。候选摔倒筛选使用躯干角度、根部下降、向下速度和单轴残差；它是候选生成规则，不是临床标签。
- 真实 AMASS 仍未取得。本轮合成 fixture 只证明入口和筛选逻辑可运行：2 条序列扫描、1 条候选摔倒；不能将合成结果描述为数据集实验。
- GPU 证据链显示驱动模块和 CUDA 用户态库存在，但字符设备节点缺失：`nvidia-smi` 失败，`/dev/nvidia*` 不存在，udev 规则依赖 `/sbin/ub-device-create`。当前会话无 `CAP_MKNOD`、`sudo` 受 `no new privileges` 阻断，因此仓库无法自行完成宿主机设备节点修复。
- 后续用户已在真实宿主机重启 udev/persistence 服务并恢复全部 `/dev/nvidia*` 节点；其 `nvidia-smi` 成功识别 RTX 4060、驱动 595.91.07、CUDA 13.2、8 GiB 显存。GPU 驱动故障已解除；仍需在宿主机运行 Isaac headless 命令确认 Isaac backend 实际使用 GPU。当前隔离执行会话不能代替该验证。

## 2026-09-22 — 房屋扩大与 AMASS 预览

- 房屋采用配置驱动的水平缩放 `layout_scale_xy=1.10`，从原始 `8.4 × 7.0 m` 变为 `9.24 × 7.70 m`；房间高度不缩放。场景 CPU 计划、manifest 和 Isaac headless USD 的 footprint 一致。
- `view_amass.py` 的输入是原始 AMASS `.npz`（必须含 `poses` 和 `trans`）；`import_amass.py` 生成的 retargeted NPZ 是输出格式，不能再次作为 AMASS 原始输入。该边界已通过一次明确的失败和一次成功播放验证。
- 合成 `Subject1/fall.npz` 在 Isaac headless CPU PhysX 播放 241 帧并导出 `artifacts/humans/amass_preview.usda`，证明“原始 AMASS → SMPL-H→SMPL 重定向 → mesh/DOF 同步播放”链路可运行；真实数据仍待授权下载。
- 官方在线获取入口已集成到 `scripts/humans/fetch_assets.py`，使用自己的 MPI 账号凭证后可列出/选择 AMASS 子集；本轮网络直连仍受代理握手失败限制，未把远程下载写成完成。
- 实测 `fetch_assets.py --site amass --list`：默认代理为 `Connection refused`，去掉代理为 TLS `SSLV3_ALERT_HANDSHAKE_FAILURE`；账号凭证本地存在且 dry-run 通过。网络恢复后无需改代码即可列出并选择子集。

## 2026-09-22 — Transitions 已下载，CMU 中断

- Downloads 中的 `Transitions.tar.bz2` 是完整 bzip2 归档，含 110 个 AMASS `.npz`；当前导入器全部读取成功，字段和坐标转换契约满足要求。
- Transitions 的 110 条动作经现有保守筛选得到 0 条候选跌倒，但含 `sit_stand`、`walk`、`crawl`、`run` 等日常/易混淆动作。真实 `sit_stand` 已完成 Isaac headless 预览，980 帧通过。
- CMU 的中断临时文件当前已不在 Downloads，不能从现有文件续传；应回到官网同一 `CMU → SMPL+H G` 下载项重新开始，或在浏览器下载管理器中对仍保留的任务点击 Resume。磁盘剩余约 45 GB，重新下载前应确认空间。

## 2026-09-22 — 房屋扩大到当前尺寸的两倍

- 将 `layout_scale_xy` 从 `1.10` 改为 `2.20`。原始基础平面 `8.4 × 7.0 m` 经统一缩放后为 `18.48 × 15.40 m`，即当前 `9.24 × 7.70 m` 的 2 倍；高度不变。
- 缩放仍在场景 schema 边界一次性应用到房间、开口和家具，人体 spawn 点通过同一场景配置重新计算。

## 2026-09-22 阶段 7 复审补充

### 修复后结论（repair5）

- `scripts/humans/verify.py` 已修复 PD 验收污染：逐 DOF 单独跟踪、传送后清零根部/关节速度，控制试验暂时关闭 19 个自身碰撞体，重力与地面检查前恢复碰撞。
- headless Isaac repair6 通过：PD 最大误差 1.540°（容限 15°），逐 DOF 目标/实测/误差和当前 stiffness/damping 已记录，驱动关闭负对照、重力回落、地面穿透和场景哈希检查均通过；运行使用 CPU PhysX 回退，GPU/CUDA 不可用。
- 两个旧批次索引已由 `scripts/humans/migrate_trials_index.py` 迁移，`physics_dt_s=0.008333333333333333`，并新增 `physics_hz=120.0`。

- 历史复审时的结论是 SMPL CPU/USD 网格接入和胶囊碰撞原型部分通过、Isaac 完整验收不通过；该结论已由 repair6 更新。当前 AMASS 未取得，自由站立/行走仍未完成；证据见 `docs/progress.md` 与 `docs/stage7-review.md`。
- 文件存在、成功加载、用于实际几何、物理姿态驱动蒙皮是四个不同验收条件；当前 CPU/USD 路径已接入真实 neutral 网格，Isaac 逐帧稳定性仍未通过。
- PD 中程测试重复转弧度，75 度目标变为 1.309 度，历史 2 度误差不可外推为动作跟踪合格。
- 轨迹启发式撞击不是测得接触；近地体点和速度必须对应同一体点/方向。水平移动负例已被错误标为 fall。
- 联合人物/序列键只隔离组合，不能保证跨人物泛化。受控动作需单独检查跟踪，不能以数值未发散作为动作完成证据。
- repair3 曾出现非有限四元数并退出 1；repair6 已重现并通过修复后的验收。官网资料本轮未重新核验。

## 文献事实源

- 本地 Zotero 数据库：`/home/gsh/Zotero/zotero.sqlite`
- 本地文献整理：`/home/gsh/WorkBuddy/2026-09-21-19-38-52/无线信号人体摔倒检测文献综述.md`
- 本地 PDF 集合：`/home/gsh/WorkBuddy/2026-09-21-19-38-52/pdfs/`
- 核心条目导出：`wireless_falldetection_zotero.ris`、`wireless_falldetection_zotero.bib`

## 核心证据记录

### E01 — SiFall

- Source: Ji, Xie, Li, *SiFall: Practical Online Fall Detection with RF Sensing*, SenSys 2022 / arXiv:2301.03773，全文 PDF 已在本地。
- Source type: full paper
- Supports: 跌倒是不可控、不可复现的事件；可以学习正常活动分布，把跌倒视为异常；在线增量学习和动态分段适合部署。
- Method / dataset / metric: 商用 Wi-Fi CSI；16 名受试者；连续在线处理；前端去噪、提取动态成分，后端 FallNet。
- Limitation: 真实跌倒稀缺，异常检测依赖正常活动分布；在线适配可能发生误适配和灾难性遗忘。
- Project relevance: 仿真数据不应只做“跌倒分类样本”，还要生成正常活动、异常活动和在线流式窗口。
- Claim strength: supported

### E02 — DGSense

- Source: Zhou et al., *DGSense: A Domain Generalization Framework for Wireless Sensing*, arXiv:2502.08155，全文 PDF 已在本地。
- Source type: full paper / preprint
- Supports: 环境、位置、个体会造成域偏移；目标域无数据的域泛化比依赖目标域样本的域适应更贴近部署；虚拟数据生成和 episodic training 可用于学习域无关特征。
- Method / dataset / metric: 虚拟数据生成器；ResNet + attention 空间特征；1D CNN 时间特征；episodic training；包含 acoustic fall detection 评估。
- Limitation: 预印本；通用框架的具体收益依赖每个无线任务的域构造质量。
- Project relevance: Isaac Sim / Sionna 应显式随机化房间布局、材质、人体参数、链路和噪声，形成训练域族，并保留完全未见域测试。
- Claim strength: supported

### E03 — CSI-Bench

- Source: Zhu et al., *CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing*, NeurIPS 2025 Datasets and Benchmarks / arXiv:2505.21866v2，全文 PDF 已在本地。
- Source type: full paper / benchmark paper
- Supports: 受控实验室数据、单一硬件和分段录制会限制真实泛化；连续日常数据和标准化划分是必要的评测条件。
- Method / dataset / metric: 26 个室内环境、35 名用户、超过 461 小时有效 CSI；跌倒、呼吸、定位、运动源等任务；提供标准化 split 和 baseline。
- Limitation: 新基准的社区横向验证仍在积累；真实跌倒属于长尾事件。
- Project relevance: 项目评测应采用按房间/个体/硬件留出，而不是随机切片；仿真数据应提供可导出的标准格式，方便与 CSI-Bench 类真实基准对齐。
- Claim strength: supported

### E04 — Deep Learning-Based Fall Detection Using WiFi CSI

- Source: Chu et al., *Deep Learning-Based Fall Detection Using WiFi Channel State Information*, IEEE Access 2023，全文 PDF 已在本地。
- Source type: full paper
- Supports: 端到端深度模型可以减少手工特征；主径/非主径位置和硬件差异会显著影响信号。
- Method / dataset / metric: 700+ CSI 样本、22 名志愿者、4 类室内环境、多种跌倒与日常活动；以降采样和 reshape 为主要预处理。
- Limitation: 混合环境随机划分可能高估跨环境泛化；同频干扰和硬件差异未被充分解决。
- Project relevance: README 中不能把高准确率直接写成真实部署保证；必须报告留一环境测试、误报率和报警延迟。
- Claim strength: supported

### E05 — TED-Net + DGNN

- Source: Cho et al., *WiFi based Human Fall and Activity Recognition using Transformer based Encoder Decoder and Graph Neural Networks*, arXiv:2504.16655，全文 PDF 已在本地。
- Source type: full paper / preprint
- Supports: CSI → 骨架姿态 → 图动作识别的中间表示有助于可解释性；姿态估计误差会向下游传播。
- Method / dataset / metric: 三天线 CSI；CNN + Transformer 的 TED-Net 估计骨架；定向 GNN 进行动作识别；公开多模态数据 + 20 人跌倒数据集。
- Limitation: 预印本；多目标、强遮挡和姿态误差传播仍需验证。
- Project relevance: Isaac Sim 可以提供 3D 关节/网格真值，同时保留 CSI-only 模型作为隐私友好的最终输入，姿态只作为训练辅助或可解释性输出。
- Claim strength: supported

### E06 — Radar-Based Fall Detection for Assisted Living: Digital-Twin Representation Case Study

- Source: Ratto et al., *Radar-Based Fall Detection for Assisted Living: A Digital-Twin Representation Case Study*, PerCom Workshops 2026 / arXiv:2601.11938；本地 PDF 文件名为 `radar_digitaltwin.pdf`。
- Source type: full paper / preprint
- Supports: 数字孪生可以缓解高冲击真实跌倒数据的伦理和采集难题；表示方式（时序/时频而非静态图）会显著影响跌倒检测。
- Method / dataset / metric: 单房间数字孪生；平衡跌倒/非跌倒数据；统一小型 CNN 比较时频图、时间图和静态 RDM。
- Limitation: 电磁散射、人体模型和场景保真度决定仿真结论能否外推；单房间且缺少真实交叉验证。
- Project relevance: 本项目需要单独记录仿真参数和真实验证，不把合成数据性能等同于现实性能；优先输出时序 CIR/CSI，而不是只生成静态图像。
- Claim strength: supported

## 综合判断

1. 研究缺口不是“再训练一个 CNN”，而是把可控的人体/场景物理变化与无线信道变化对齐，并用未见域和真实数据检验迁移。
2. Isaac Sim 负责运动和场景真值；Sionna RT 负责多径、材质、天线和频段下的传播观测；深度模型只消费 CSI/CIR 及其派生特征。
3. 数据契约必须同时保留 `scene_id`、`subject_id`、`hardware_profile`、`activity`、`timestamp`、`channel_representation` 和 `simulator_version`，否则无法做域划分和复现。
4. “零隐私风险”应改成工程目标或风险边界：无线感知不采集视频，但 CSI 仍可能包含活动、位置和身份信息；项目要记录最小化采集、访问控制、脱敏和不上传原始人体网格的策略。

## E07 — 室内 USD 验收证据（2026-09-21）

- 事实源：当前 `artifacts/scenes/indoor_apartment.usda`，用 OpenUSD 0.25.11 读取并提取世界变换、边界、语义与材质，记录于 `artifacts/acceptance/usd_geometry.json`；没有借用此前 GUI 成功记录作为本轮视觉证据。
- 当前源文件无 Camera；六块吊顶为不透明实心几何。默认外部视角不能用来验收内部。
- 地基范围 z=[-0.30,0]，六块地板均 z=[-0.12,0]，顶面重合；复合材料/接触参数与后续 RT 几何需要复核。
- 实际家具边界：走廊地毯 x=[6.5,8.5], y=[2.1,5.1]，走廊范围 x=[0,8.4], y=[3,4.2]；卧室地毯 y 到 3.05；冰箱与厨房北墙发生 0.01–0.03 m 穿透。
- 14 项物理检查本轮在 CPU 回退下通过；这不检验 GUI/RTX、空间通行性、摩擦系数数值响应或无线传播正确性。

### E07 补充：验收门禁与边界

- 故障注入：`--manifest artifacts/acceptance/intentionally_missing.scene.json`，日志 FAIL、进程退出 0，见 `artifacts/acceptance/isaac_failure_exit_probe.log`。应将失败码传给本机版本支持的 `SimulationApp.close(exit_code=...)`，并对启动失败、断言失败和正常成功三条 CLI 路径做进程级验证。
- 对照输入：NaN 坐标、标识符规范化碰撞、过小正尺寸均能穿过当前 schema/planner。输入合法不等于派生几何合法；检查应在配置边界与完整计划输出两处执行。
- 单椅自由落体只证明重力和支撑碰撞生效，不证明地面摩擦、接触材质切换或通行净空正确。

### E07 最终交付

- 验收报告：`docs/indoor-scene-review.md`；结论为需整改，R1–R6 未在本次审查中改动。
- 新查看入口已通过 USD 单元测试和 Isaac 无界面视口集成；临时去顶显示与完整仿真几何分离。重新导出与原始资产哈希一致。
- 接下来应先修错误退出码和真实几何，再开展 GPU 视觉、机器人可达性以及分材质摩擦/无线传播验证。

## E08 — 验收整改设计

- 地板保持 z=0 统一行走面；基础承托面为 `-max(floor_thickness)`，较薄地板补齐找平层。默认场景仍为一个基础板，不改变地板/房间数量。
- 几何验收针对真实配方零件而非标称家具尺寸，采用旋转盒体分离轴及圆柱-盒体距离检查；房间边界、墙相交和楼板支撑独立检查。机器人通行和一般家具之间碰撞仍是后续任务。
- 验证负例需要在进程边界测试；仅 mock `close()` 为普通返回函数会掩盖原来的退出码问题。

### E08 复验与交付

- R1–R6 已闭环：CPU 70 tests passed；依赖 USD 的测试在 bundled 环境另跑 9 passed；真实 Isaac 正例退出 0、旧资产/新清单负例退出 1。
- 默认 USD 已更新，实际几何无家具越界/穿墙；地基顶面为 -0.12 m。详细版本、容差与证据见 `docs/indoor-scene-remediation.md`。
- 量化摩擦标定、机器人可达性、真实 GPU 光照和 Sionna 传播仍未验证，不能据本轮资产检查推断其效果。

## E09 — 场景模块边界整理

- 场景 schema、材质、配方、规划、几何校验与 USD 适配集中在 `sim2sense_fall.scenes`；命令行生命周期留在 `scripts/scenes/`。
- 包入口只公开 CPU 数据与规划接口，导入包无需安装 Isaac/pxr。下游通过 ScenePlan/USD 消费场景。
- 此阶段仅移动目录与更新导入；以重新导出的 USD 字节哈希检验行为一致性。清单 generator 随模块名更新。

### E09 验证结论

- 目录迁移后 USD 字节哈希不变；清单除 generator 外结构完全相同，可将后续场景变更与本次目录重构区分。
- 70 项 CPU、9 项 bundled USD 测试和 10 项 Isaac CPU 回退检查通过；GPU 渲染仍待具备设备的环境复验。
- 当前维护入口见 README 与 `docs/indoor-scene.md`；旧验收报告保留历史证据，迁移回归摘要见 `artifacts/reorganization/validation_results.json`。

## 阶段 7 关键事实与证据边界（2026-09-22）

### 已核实的事实（有公开一手来源）

- **SMPL 骨架**：24 关节，关节 0 为 `pelvis`，父表拓扑排序；关节名与父表由
  `smplx` 参考实现、Meshcapade 骨架文档与多个公开 SMPL 加载器一致复现。
  本仓库把该表写成常量 `SMPL_KINEMATIC_PARENTS`，并在模型文件存在时用其 `kintree_table`
  反向校验，不一致就报错而不是静默重定向。
- **SMPL-H / AMASS 兼容关系**：AMASS 原生表示是 SMPL-H（22 身体关节 + 30 手部关节 = 52 关节），
  每帧 156 姿态参数；论文配置用 16 个 betas 与 8 个 DMPL。SMPL 与 SMPL-H **共享关节 0–21**，
  所以 AMASS 的身体块可直接映射到 SMPL；但 SMPL 的关节 22/23（`left_hand` / `right_hand`）
  在 AMASS 里**没有对应项**（SMPL-H 接下来的关节是手指），只能保持静止。
- **许可**：SMPL 与 AMASS 均为 Max Planck Society 的**非商业科研**许可，且都需注册后下载。
  本仓库不自动下载任何模型或数据集，缺失时抛出带注册地址与搜索路径的错误。

### 官方下载页一手核对结果（2026-09-22）

页面只提供三项，无文件名：

| 条目 | 内容 |
| --- | --- |
| version 1.0.0 | **for Python 2.7**（female/male，**无 neutral**），10 shape PCs |
| version 1.1.0 | **for Python 2.7**（female/male/**neutral**），**300 shape PCs** |
| UV map | OBJ 格式 |

由此**推翻**本文件早先的三条记录：v1.0.0 并没有 neutral 模型；neutral 与 300 个 shape PC
都属 v1.1.0；"v1.1.0 含 chumpy" 不再是待核实的传闻，而是与"for Python 2.7"这一标注一致的
高概率事实（Py2 pickle 正是需要 `latin1` 解码与 chumpy 解包的原因）。
加载器现在同时处理这两件事，并会自动检测实际编码与包装方式。

### 来自公开二级资料、尚未一手复核（一律标记 `verified: false`）

- SMPL 发布文件的**具体文件名**：下载页不公布，登记表以候选列表形式声明
  （`filename_candidates`），命名差异表现为"以另一名字找到"而不是"资产缺失"。
- `shapedirs` 的实际宽度：文件名里的 `10` 指前 10 个分量，与页面标注的 300 不一致，
  必须以实际文件为准；登记表把 `betas: 300` 作为**声明值**而非静默假设。
- SMPL-H v1.2 的 `SMPL_*.pkl` 文件名与内部布局。
- AMASS 各子集（CMU、KIT、Transitions、SFU、MPI_HDM05、BMLmovi、ACCAD、HumanEva 等）的
  具体序列、帧率与坐标约定均**未查看**。

### 必须传播的边界

- **不预设 AMASS 含足够的跌倒序列或可靠跌倒标签**。取得访问权后必须先做候选动作筛选并记录结果，
  再决定它们能否进入验证集。
- 本阶段 CPU/USD 路径已使用真实 SMPL neutral 蒙皮网格（`smpl_skin_mesh`）；无模型时才使用明确标记的胶囊体代理（`capsule_proxy_surface`）。Isaac 逐帧物理姿态驱动网格仍未稳定通过；本阶段没有任何一方数据来自真实采集。
- 跌倒/躺下的标签来自**项目自定义的启发式规则**（阈值固定在配置里、先于试验），
  不是经过临床或生理验证的跌倒判据；引用时必须如此描述。
- 自由站立与行走控制**未达标**：单轴矢状面关节 + 中等刚度 PD 的自由根人体在重力下会倒下。
  这不是人体导入的失败条件（阶段计划明确允许），但不得反向表述为「控制已可用」。
