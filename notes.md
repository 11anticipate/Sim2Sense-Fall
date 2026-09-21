# 文献与方案笔记

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
