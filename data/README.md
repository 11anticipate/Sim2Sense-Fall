# 数据目录

[工程文档索引](../docs/README.md) · [无线数据契约](../docs/data-contract.md)。

- `raw/`：真实采集的原始 CSI/CIR；默认不入 Git。
- `generated/`：Isaac Sim + Sionna RT 生成数据；默认不入 Git。
- `processed/`：清洗、对齐、切窗后的派生数据；默认不入 Git。
- `metadata/`：可公开的 schema、数据版本、划分和许可说明，可纳入 Git。
- `humans/smpl/`：本地授权 SMPL 模型；原始模型参数不入 Git。
- `humans/amass_raw/`：本地授权 AMASS 原始序列；按子集/人物/序列及源哈希追溯，不入 Git。

仿真网格、实时截图、控制/接触记录与验收报告默认写入仓库 `artifacts/`，不与源码或数据契约混存。
动作候选、通过物理验收的动作、可用于无线训练的样本是不同状态，需逐级登记。

每个数据版本都应保留生成配置、随机种子、仿真器版本、硬件配置和校验摘要。未经脱敏和许可核对，不要将受试者数据、人体网格或原始信道文件提交到远程仓库。
