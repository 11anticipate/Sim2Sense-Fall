# 工程文档索引

整理日期：2026-09-24。运行方式查指南，待办查计划，试验结论查对应日期和配置的原始报告。
本次仅整理文档，没有重跑 Isaac，也没有修复右臂或加入新动作。

## 当前入口

| 文档 | 职责 |
| --- | --- |
| [项目 README](../README.md) | 项目状态、目录与快速启动 |
| [task_plan.md](../task_plan.md) | 唯一当前待办：右臂、接触/平衡、蹲下/起立/摔倒、手动路线交互、性能验收 |
| [AGENTS.md](../AGENTS.md) | 项目协作、验证与资产管理规则 |
| [系统架构](architecture.md) | 模块边界、控制链路与研究风险 |
| [人体仿真](human-simulation.md) | SMPL/AMASS/PhysX 入口与已验证边界 |
| [键盘控制](keyboard-control.md) | 现有按键、配置、输出和实时截图 |
| [室内场景](indoor-scene.md) | 公寓构建、导出、查看与场景检查 |
| [网格导出契约](mesh-export.md) | 实际/参考网格、时间轴及不同入口格式 |
| [无线数据契约](data-contract.md) | ChannelSample 与数据划分边界 |
| [Sionna 导入](sionna-import.md) | 当前公寓 CIR 入口与早期导入实验 |
| [人体电磁材料](human-em-material.md) | 参数来源、频点与建模假设 |
| [数据目录](../data/README.md) | 原始、生成、处理、元数据与人体资产管理 |
| [阶段进度](progress.md) | 按轮次保留的执行记录；旧“当前”仅指当轮 |
| [文献与方案笔记](../notes.md) | 文献事实、方案判断与日期明确的研究笔记 |

## 实测与历史证据

| 文档 | 使用边界 |
| --- | --- |
| [AMASS 物理审计，09-24](amass-physics-audit-2026-09-24.md) | 坐标、接触、辅助跟踪、失败动作和挡墙；早于键盘实时显示 |
| [独立复核，09-23](verification-2026-09-23.md) | 特定配置站立/后推跌倒与完整公寓复数 CIR smoke |
| [AMASS 多轴缺陷](amass-replay-defect.md) | 多轴链、DOF 重排、时钟等修复过程，不代表全部动作通过 |
| [SMPL 朝向缺陷](mesh-orientation-defect.md) | 模型静止坐标/配对缺陷；AMASS 世界基后续修正见 09-24 审计 |
| [物理交互审计，09-22](physics-interaction-audit.md) | 原始问题与局部追加修复；当前待办以计划为准 |
| [阶段 7 首次复审](stage7-review.md) | 历史失败、旧资产扫描和数据泄漏风险，不作为现状摘要 |
| [室内场景首次验收](indoor-scene-review.md) | R1–R6 原始问题 |
| [室内场景整改](indoor-scene-remediation.md) | R1–R6 修复实测；旧 GPU 阻塞为历史环境 |
| [旧项目计划](history/task-plan-before-2026-09-24-review.md) | 整理前完整快照，原阶段细项和历史证据保留 |
| [旧人体指南](history/human-simulation-before-2026-09-24-review.md) | 整理前完整快照，不按旧“未取得/未运行”判断当前能力 |

## 证据使用规则

1. 以实际代码、配置、报告和实测画面为准；文档给出证据入口，不能替代运行验证。
2. 运行完成、参考跟踪通过、物理交互通过、无辅助平衡通过分别报告。
3. `view_amass.py` 是运动学预览；`simulate.py` 的记录后渲染和 `keyboard.py` 的实时视口截图注明来源。
4. 旧 GPU 阻塞、AMASS 缺失、单轴 rig 和接触归因说明只适用于对应日期，不能覆盖后续结果。
5. 用户右臂反馈尚未独立复现。现有截图与全身最大误差不能替代专门的左右臂诊断。
6. 新动作范围仅蹲下、起立、摔倒；主动避障不做。历史提及拳击、坐椅或抓握不构成当前承诺。

实验产物位于 Git 忽略的 `artifacts/`，本机可查看；克隆仓库后历史图片/报告可能不存在。
代理内部目录 `.workbuddy/`、`.mimosa/` 不属于工程使用文档，本轮保持原状。
