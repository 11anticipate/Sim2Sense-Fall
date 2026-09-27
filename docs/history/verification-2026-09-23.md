# 2026-09-23 独立复核、整改与当前进度

> 本页为指定日期/配置的实测报告，保留当轮事实与失败结果。
> 后续 AMASS/键盘状态及新任务见 [人体指南](human-simulation.md)、[计划](../task_plan.md) 与 [文档索引](README.md)。

## 结论

原总结方向基本正确，但其中信道数值、跌倒标签、自由站立和“无穿透”的表述不能直接采信。本轮已完成：

1. 修正复数信道、固定时延采样、标签和元数据；撤回旧 −4.10/+6.03/10.15 dB 数字。
2. 新增明确的静态站立配置，5 秒自由根站立与后推物理跌倒通过。
3. 实现全部 231 个公寓几何部件到 Sionna，实际物理跌倒已生成合法复数 CIR。
4. 核对 Gabriel/IFAC 人体材质来源，保留均匀肌肉等效薄层的建模边界。
5. 修复聚焦相机并逐帧查看截图；复核网格拓扑；核实真实 AMASS 库与切片入口。

起点 HEAD=6084df8（原总结截至 c6761de）。进入任务时 docs/physics-interaction-audit.md 已有修改，保留。当前修复尚未提交或推送。

## 原分析逐项复核

| 原说法 | 复核结论 |
|---|---|
| 238 passed / 8 skipped | 起点复跑一致；修复后 246 passed / 8 skipped，额外 bundled USD 9 passed |
| 截图不同证明姿态可见 | 只能证明像素更新。复查曾看到外墙和移动影子；相机已移到室内方向，实际看图确认人物完整 |
| 0.40 + 0.55×身高是几何保证 | 是包络经验近似；当前人体起始位置有效，不保证任意体型或整个动作扫掠体不碰墙 |
| Euler=2 等于无破洞/翻面 | 边界边与非流形边为零、共享边方向一致已独立复算；尚未做三角形自交检测 |
| 站立被标 fall 是“误标” | 旧人体确实倒下，轨迹标签可以是正确的；失败的是站立控制，不能强改标签 |
| 6084df8 已释放辅助 | 末步 remaining=1，权重从未降至零；仍逐步写根位置与旋转。已移除自由根稳定期写入 |
| 全部物理交互正确、不穿透 | 仅部分接触、重力、归因成立。新试验前推/控制失效仍因网格穿地超限被剔除 |
| Sionna +6.03 dB / 散布10.15 dB | 不成立：旧代码只取 Paths.a 的实部，不能据此解释遮挡或反射 |
| 四个跌倒参考片段都是 fall | 不成立：清单中两个 controlled_lowering、一个 invalid、一个 no_fall；旧脚本按文件名误赋 FALL |
| 8 项 Sionna 环境自检 | 历史文档打印编号实际为1–7；本轮用真实公寓与物理人体端到端检验更直接 |
| 工作树干净、P0-3 仍完全未做 | 已过时：进入任务时多了6084df8和未提交审计修改 |

## 静态站立与物理跌倒

新增 configs/humans/human_smpl_stable.yaml、configs/humans/standing_validation.yaml。原 neutral 配置继续保留为弱 PD 基线。

- 显式启用 horizontal_foot_capsules：把踝—趾斜胶囊的支撑轴放平，保留原最低点、半径、骨架和蒙皮。它是足底碰撞近似，不是人体测量。
- 髋、膝、踝与脊柱的 stiffness ×20、damping ×5，保留原有限 max_force；这是刚性站姿基线，不是自然人体平衡或行走控制器。
- 重置时清除根/关节速度并恢复驱动倍率；自由根稳定期与试验期均不写根位姿。显式 --pin-root 仍记录为辅助模式。
- 独立站立门槛：骨盆下降≤0.05 m、躯干倾角≤15°、水平漂移≤0.05 m。未通过时试验不可用并返回失败。
- 外力试验在扰动前检查跟踪误差；扰动后仍记录全部误差，但按实际轨迹、撞击与穿透验收，不能用“必须保持站姿”排除跌倒。
- 原验证器无条件打印 usable=True 已修复，日志与 gates.usable 一致。

最终验收：artifacts/review_20260923/final_physics/trials_index.json。

| 项目 | 实测 |
|---|---|
| 无扰动5秒站立 | no_fall；无根部支撑 |
| 最大骨盆沉降 | 0.001195 m |
| 最大水平漂移 | 0.007593 m |
| 最大躯干倾角 | 4.6683° |
| 站立关节最大误差 | 0.7757° |
| 后推跌倒前跟踪误差 | 0.8177° |
| 后推跌倒 | 0.8333 s失稳、0.9333 s撞击代理、最终躺倒 |
| 后推网格最低点 | −0.032139 m，在原−0.05 m门槛内；不是“零穿透” |
| 接触通道 | physx_contact_report；肢体归因有效 |
| tensor接触力视图 | 不可用，保留SKIP；未把代理撞击时刻叫实测力峰值 |

前推与控制失效长试验最低点分别约 −0.1064 / −0.0974 m，被剔除。它们没有变成训练样本。若干额外反馈控制试验产生滑移或接触归因中断；这些反馈未进入最终实现。所有失败日志与临时配置存档在 artifacts/review_20260923。

同配置 human verify 通过：关节/DOF/CPU FK、PD、驱动关闭负对照、抬升回落、不发散和末态胶囊支撑均通过。回落0.8711 m，末段200 ms漂移0.023 mm。它证明动力学在跑，不是完整跌倒生物力学验证。

## 公寓 → Sionna → 复数 CIR

模块：src/sim2sense_fall/sionna/apartment.py、channel.py。
入口：scripts/sionna/import_fall_mesh.py。

- 从 USD 使用的同一 ScenePlan 转换，世界米制Z-up；全部231个几何部件（屋顶、墙、地面、家具）均保留。
- box精确三角化；圆柱32段离散；父级家具旋转/平移在CPU解析。转换后核对对象数。
- 动态家具目前使用初始配置位姿，未同步被人体撞动后的家具；不宣称通用动态环境同步。
- Paths.a 按 real + i·imag 组合；CIR 为复数 sinc taps，共用0–2 μs绝对时延网格、间隔10 ns（100 MHz）。
- 源帧等间隔抽取；--frames 12 是目标最大值，601帧实际取11帧、间隔55/120 s，采样率2.1818 Hz，属于低速 smoke，不是50 Hz数据集。
- 明确随机seed=42、ray count=300000、max_depth=4；保存输入文件哈希、SMPL/试验元数据、场景配置哈希、来源标签、收发位置、材质、逐帧路径实虚部及延迟。
- invalid源直接拒绝。physics_trial 必须在已完成 trials_index 中 gates.usable=True。kinematic来源不再根据名称标fall。
- 统计量为 Σ|a_p|² 的非相干信道增益（无量纲），不是接收瓦特，也不是相干接收功率。

最终产物：artifacts/review_20260923/final_physics_channel/ 中的 .cir.npz 与 .import.json。
源为最终后推物理跌倒。完整公寓、单天线V极化、3.5 GHz、同一Tx/Rx下有人/无人对照通过；静止重复数值误差远小于运动差异。修复后相对无人基线范围约 −11.74 至 −0.013 dB。该结果与旧内置floor_wall场景、旧材质和错误复数计算不能直接比较。

人体材质实测来源记录见 human-em-material.md：Muscle在3.5 GHz的 εr=51.4442299518、σ=2.5575182495 S/m。全身均匀肌肉、厚度0.02 m仍是代理假设。

## 图像与网格

最终截图：artifacts/review_20260923/captures_final/contact_sheet.jpg，帧0/48/96/144，1600×900原图。逐帧平均像素差1.078/5.428/4.762。实际查看确认头脚完整、人体可见，不能仅靠像素差断言。此图来自 scripted kinematic replay；物理跌倒证据是上面的 trial 输出。

独立 audit_mesh_geometry.py 复核四段旧采集：
V=6890、F=13776、E=20664、Euler=2，边界/非流形/不一致边方向均为0。前三段面积约1.99116 m²；reach_then_topple为1.98070–1.99116 m²。自交未测。
侧倒网格实际最低顶点 −0.29750 m；旧报告的 −0.219 m来自另一种采样体点，不应混用。侧倒保持invalid且拒绝进无线样本。

## AMASS 状态纠正

本机 data/humans/amass_raw 有 CMU 2088 + Transitions_mocap 110 = 2198 个npz，旧“CMU未下载”已过时。
--write-slice 已对真实目录运行：2146个唯一stem软链接、52个重复stem未纳入切片，原文件未删除。
在切片中实际导入3个CMU序列，来源哈希/重定向通过，0个跌倒候选。这是3条smoke，不是2198条完整筛选或许可重新审计；去重切片也不代表可以丢弃重复stem所对应的不同人物。

## 验证记录与边界

- compileall、ruff、git diff --check通过；系统pytest：246 passed / 8 skipped。
- bundled OpenUSD补跑：9 passed，覆盖相机头脚在视锥内和场景USD结构。
- scene build/verify：231几何/231碰撞体，椅子抬升0.25 m后回落0.25 m，原场景文件不变。
- human stable verify、5秒站立/后推simulate、GPU公寓物理人体CIR均通过。
- CPU scene/human/radio dry-run和旧mesh独立契约核查通过。
- 环境错误已记录：沙箱GPU不可见，宿主提权后可用；代理127.0.0.1:7897拒绝连接，直连IFAC成功；uv snap的DBus错误，使用已安装ruff；Isaac Python无pytest，系统Python加载bundled USD及其bin路径完成9项。
- 当前只有固定场景静态站立与一类后推物理跌倒闭环。自然行走/恢复、多方向有效跌倒、真实人体EM校准、50 Hz规模数据、模型训练/报警性能尚未完成。

## 复现入口

    python3 scripts/humans/simulate.py --dry-run --config configs/humans/human_smpl_stable.yaml --motions configs/humans/standing_validation.yaml --trial stand_neutral:none --out artifacts/review_20260923/stable_dry
    ~/isaacsim/python.sh scripts/humans/simulate.py --headless --config configs/humans/human_smpl_stable.yaml --motions configs/humans/standing_validation.yaml --trial stand_neutral:none --trial stand_neutral:push_backward --out artifacts/review_20260923/final_physics
    python3 scripts/sionna/import_fall_mesh.py --dry-run --trial-json artifacts/review_20260923/final_physics/smpl_neutral_standing__stand_neutral__push_backward.trial.json
    /home/gsh/.local/opt/sionna/bin/python scripts/sionna/import_fall_mesh.py --frames 12 --trial-json artifacts/review_20260923/final_physics/smpl_neutral_standing__stand_neutral__push_backward.trial.json --out artifacts/review_20260923/final_physics_channel
    ~/isaacsim/python.sh scripts/humans/view_amass.py --motion fall_forward_reference

所有试验采用physics seed=20260922、RT seed=42、固定公寓smoke划分，不建立train/test。配置/模型/数据哈希随产物保存，汇总见 artifacts/review_20260923/verification_summary.json。
