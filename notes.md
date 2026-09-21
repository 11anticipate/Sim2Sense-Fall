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
