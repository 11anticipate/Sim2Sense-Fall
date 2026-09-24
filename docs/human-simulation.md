# SMPL 人体、动作与跌倒仿真

维护日期：2026-09-24。[文档索引](README.md) · [当前计划](../task_plan.md) · [执行记录](progress.md)。
阶段 7 部分完成：人体与物理链路可运行，键盘动作质量仍未全部通过。

## 控制链路与入口

```text
AMASS 原始动作 → 坐标适配/重定向 → 关节位置与速度目标
键盘输入 → 步态、速度、朝向、启停 → 有界根辅助控制
                         ↓
PhysX 关节 PD + 外力 + 重力 + 碰撞 → 实际连杆姿态 → SMPL 蒙皮
```

PhysX 是物理引擎，动作选择和目标生成在 Python 控制器中执行。
SMPL 提供骨架与显示网格；分段刚体承担碰撞，显示皮肤不直接等于碰撞表面。
根外力/力矩辅助仍在使用，不能将当前键盘模式称为自主平衡。

| 入口 | 用途 | 证据边界 |
| --- | --- | --- |
| `scripts/humans/keyboard.py` | 人工键盘及自动 demo，实时实际姿态蒙皮 | 辅助物理；支持移动、转向、停止、复位 |
| `scripts/humans/view_amass.py` | 原始/脚本参考预览 | 每帧设置根/关节位姿，不能用来验收碰撞响应 |
| `scripts/humans/simulate.py` | 配置驱动的物理试验和真值导出 | 检查控制模式、根辅助和传送记录；GUI 可回放实际记录 |
| `scripts/humans/render_recording.py` | 渲染已保存实际网格或参考网格 | 离线渲染，必须标注输入来源 |
| `scripts/humans/verify.py` | CPU/USD/PhysX 基础验收 | 不替代完整动作质量验收 |

## 使用

以下命令在仓库根目录执行；首次使用先按 [场景指南](indoor-scene.md) 导出公寓 USD。

```bash
# CPU：不启动 Isaac
python3 scripts/humans/keyboard.py --dry-run --out artifacts/humans/keyboard_dry

# GUI：默认 W/S、A/D、空格、R、Esc
~/isaacsim/python.sh scripts/humans/keyboard.py --out artifacts/humans/keyboard_manual

# 自动键盘事件 + 实时截图
~/isaacsim/python.sh scripts/humans/keyboard.py --demo --capture \
  --out artifacts/humans/keyboard_demo

# 原始参考预览，不是物理动作验收
~/isaacsim/python.sh scripts/humans/view_amass.py \
  --config configs/humans/human_smpl_multiaxis.yaml \
  --amass-root data/humans/amass_raw/Transitions_mocap/mazen_c3d \
  --motion amass__walkbackwards_stand_poses
```

完整键位、输出文件与实测截图见 [键盘控制](keyboard-control.md)。
蹲下、起立、摔倒尚未开放为键盘动作；新需求与验收任务只登记在计划中。
本轮“起立”按蹲姿到站姿规划，R 是传送复位，不是地面起身。
行走路线由用户选择，不规划主动避障；墙/家具碰撞响应仍属于必须验收的能力。

## 资产、坐标与配置

- 本机 SMPL v1.1.0 neutral 已加载：6890 顶点、13776 面、24 关节。
  male/female 文件也存在，不能据此推断所有体型已经过物理验收。
- 本地 AMASS 为 CMU 2088 条 + Transitions 110 条，共 2198 条；并非全部通过动作筛选。
  原始资产与授权数据不入 Git，片段使用源文件哈希、人物、序列和时间区间追溯。
- AMASS SMPL-H 身体关节映射到 SMPL；手指不保留，SMPL 末端 hand 关节无直接对应项。
- 模型局部基适配与世界坐标分开：局部旋转 `B R B.T`，根旋转 `R_source @ B.T`；
  本地 CMU/Transitions 世界平移为 Z-up，不再用模型基旋转平移。
  根归一化只消除首帧世界偏航并一致变换平移，保留倾斜；落地使用恒定竖直偏移。
- 键盘采用 `human_smpl_multiaxis.yaml`：57 DOF / 62 连杆 / 22 碰撞体。
  `human_smpl_stable.yaml` 是另一个静态站立/后推试验配置，不能混用结果。
- `keyboard.yaml` 指定 AMASS 裁剪、周期化、速度、相机和阈值；派生周期步态不是原片段严格重放。
  `root_assist.yaml` 明确有限根力/力矩；人体质量和惯量仍包含工程近似，自碰撞未验收。

## 实测状态

| 测试 | 结论 | 依据 |
| --- | --- | --- |
| 多轴刚体基线 | PD 最大误差 11.462°；重力下降 1.0176 m；22 碰撞体 | [AMASS 审计](amass-physics-audit-2026-09-24.md) |
| 完整侧行/后退 | 有限根辅助下分别 8.147° / 10.793°，通过该轮跟踪门槛 | 同上；不代表无辅助行走 |
| 键盘 GUI | 10.3 s、1236 步/回调；实时蒙皮及 R 正常 | [键盘实测](keyboard-control.md) |
| 键盘动作质量 | 关节 13.787° 通过；穿地 9.86 mm、滑速 p95=0.751 m/s 不通过 | `gui_tilt/report.json` |
| 挡墙 | 根被阻挡；皮肤局部越墙约 51.2 mm，未完全通过 | `barrier/report.json` |
| 无辅助行走 | 侧行失败；键盘仍依赖显著根辅助 | AMASS 审计与键盘报告 |
| 静态站立/后推跌倒 | 特定配置的 5 s 站立和单类跌倒个例通过 | [2026-09-23 复核](verification-2026-09-23.md) |
| 右臂摆动 | 用户手动前进时观察到异常，尚未复现定位 | [计划 P0-A](../task_plan.md#p0-a-右臂摆动异常) |

以上为已有实测，本次文档整理没有新跑 Isaac。人体基础导入成功、单动作跟踪通过、整体物理质量通过是不同结论。
GPU/GUI 不再沿用 2026-09-22 的不可见状态；2026-09-24 已有 RTX 4060 实际运行与截图。

## 接触、跌倒与真值

PhysX contact report 的句柄用 `PhysicsSchemaTools.intToSdfPath` 解码，记录实际碰撞对象、
接触点、法向和冲量，包含 pelvis 及 head/hand 末端碰撞体。
本机 tensor 接触力视图仍不可用，不以空数组或零值冒充实测接触力。

外力和控制能力下降已有试验实现；`support_loss` 仍未完成。
参考动作名、按键触发和 `fall` 标签必须分开：最终标签依据实际轨迹、撞击和末态。
现有启发式判据仍可能误判主动降低身体，不能把 AMASS 下坐或下蹲直接当成跌倒。
新的摔倒状态需明确停用/减弱哪些辅助，避免“触发摔倒”仍被根控制拉回站姿。

物理试验导出统一时间轴的实际根/关节/连杆状态、参考值、接触、辅助和固定拓扑 SMPL 网格；
键盘记录格式与试验导出不同，见 [网格契约](mesh-export.md)。
当前信道 smoke 为固定公寓，不存在已完成的训练/测试划分；`split_key` 联合键不能独自保证人物隔离，
正式划分需按带数据集命名空间的人物/来源分组并验证集合交集。

## 文档与实现位置

- `src/sim2sense_fall/humans/`：资产、旋转、rig、PhysX 适配、根控制、键盘目标、事件和导出。
- `scripts/humans/`：CPU 规划、构建、查看、仿真、键盘、渲染与验收入口。
- `configs/humans/`：人体、AMASS 来源、参考动作、站立测试、键盘与辅助配置。
- `tests/humans/`：CPU 契约、控制与回归；不能替代 Isaac 接触和视觉实测。
- 历史方案及旧试验表保留在 [原人体指南](history/human-simulation-before-2026-09-24-review.md)，当前待办统一维护在 [计划](../task_plan.md)。
