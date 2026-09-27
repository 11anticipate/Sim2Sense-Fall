# 行走拖滑与原地转向滑冰：归因、修复与 A/B

维护日期：2026-09-25。工具：`audit_slip_attribution.py`、`screen_turn_clips.py`、
`verify_turn_clips.py`（常驻、CPU、只读）；修复在 `teleop.py::bake_swing_clearance`
与 `TeleopController`（turning 模式）。
证据：`artifacts/humans/limp_ab/slip_attribution.json`、`artifacts/humans/slip_lift_ab/`、
`artifacts/humans/turn_screen/`、`artifacts/humans/turn_ab/`。
[全部文档](README.md) · [当前计划](../task_plan.md)。

## 用户缺陷（09-25）

1. **行走时脚一直在滑动**；
2. **原地按 A/D 不抬脚**，双脚贴地像滑冰一样转向。

> 09-25 后续专项更正：下节“目标侧脚速减指令速度”的公式错误，0.40 m/s归因证据撤回。
> 原地A/D目前已禁用；抬脚单项未解决滑步。现行实现与Isaac通过证据见
> [滑步专项修复](slip-resolution-2026-09-25.md)。下文保留历史调查过程，不代表当前结论。

## 一、旧滑速归因（公式及主因论证已撤回）

滑速门槛（p95 ≤ 0.15 m/s）从未通过，但"没通过"不可行动。新工具把接触帧上
踝点的世界速度分解为三项（对记录状态做 FK 差分，`new_fwd` 会话）：

| 项 | p50 | p95 | 归属 |
| --- | ---: | ---: | --- |
| 目标侧脚速 − 指令速度（**参考拖行**） | **0.40** | 0.56–0.97 | AMASS 窗口/重定向 |
| 根实际速度 − 指令速度 | 0.02 | 0.06–0.14 | 根辅助弹簧 |
| 实际关节 − 目标关节的脚速差 | 0.03 | 0.19–0.94 | PD/限速/支撑 IK |

物理接触反而在"刹车"：实际滑速 p95（0.22/0.50）**低于**目标命令的滑速——参考在脚
贴地时命令它以最高 ~1 m/s 前移，地面摩擦顶住了一部分。

机制（逐相位 FK）：**摆动相抬脚太低**。右脚摆动抬脚中位仅 ~15-20 mm，25/158 个相位帧
脚底贴地（<1 cm）同时以前进方向移动；且运行时 `StanceFootController` 的抬脚 IK
每步从**新鲜参考**重新出发，0.087 rad 的单步预算永远累积不成 ~20° 的真抬脚。

## 二、修复：把抬脚烤进参考（`bake_swing_clearance`）

`load_gait` 在构建周期参考后按配置 `swing_lift_m: 0.05` 做**离线有界 IK**：
摆动帧的目标脚底高度走平滑台阶剖面（前 30% 平滑上升、平顶、后 20% 平滑下降——
触地/离地高度与斜率皆为零），支撑帧的穿地（最深 −31 mm）清零。
端点差与周期接缝守卫：帧 0 与帧 −1 是同一周期时刻，两侧抬脚增量取平均，
端点差严格不变（前进 1.54°、后退 3.33° 保持）。

剖面三迭代（每步都有实跑证据）：

- 正弦剖面：触地瞬间下落速度最大（~0.22 m/s 硬砸地面）——后退会话顶满 1500 N
  根辅助上限、collar 甩出 37° 误差；
- sin² 剖面：软着陆但中段弧线太低（p50 只 24-31 mm），拖行窗口回来；
- 平滑台阶：软着陆 + 平顶，两者兼得（最终采用）。

**前后向冻结（落地零速）实测否证**：让脚在下降段保持落点前后位置以零世界速度着陆，
前进会话关节误差 3.91→8.21° 而滑速 p95 只改善 0.02——撤回，仅抬脚+穿地清除保留。

### lift5 A/B 判定（`slip_lift_ab/`，同 demo 同 seed，与 `limp_ab/new_*` 基线比）

| 方向 | 指标 | 无抬脚 | 有抬脚 | 判定 |
| --- | --- | ---: | ---: | --- |
| 后退 | 行走滑速 p50/p95 (m/s) | 0.081 / 1.178 | **0.068 / 0.991** | 改善 |
| 后退 | 关节误差 (deg) | 11.39 | **7.86** | 改善 |
| 前进 | 关节误差 (deg) | 4.59 | **3.91** | 改善 |
| 前进 | 行走滑速 p50/p95 (m/s) | 0.021 / 0.343 | 0.028 / 0.449 | 混合 |

前进 p95 上升的机制：抬脚把"连续蹭地"变成"更少但更有冲击的触地瞬态"，p95 统计
集中在触地帧。两种配置都在 0.15 门外——滑速门槛的最终归因仍是 P0-B（根辅助抬载 +
参考自身滑步），抬脚不解决全部但消除了可见的贴地拖行，且后退/前进跟踪都更好。

过程记录（诚实）：第一版后退 A/B 的"失稳"（1500 N、37°）其实是我生成配置时把 W 键
复制进了后退 demo，角色朝 −Y 撞墙——lift1–3 的后退数据作废，修正配置后重跑。

## 三、原地抬脚转向

`screen_turn_clips.py` 全库筛查 2198 条（根旋转大 + 位移小）→ 126 候选 →
`verify_turn_clips.py` 重定向 + FK 抬步验证（排除空中转体/单脚旋转）→
**CMU/83/83_56**：177°、4.17 s、六步交替（左 3 + 右 3）、rig 可表达。
窗口切四步周期（1.625 s 起、1.6667 s），接缝同相位。
（筛查第一版 yaw 公式 `atan2(x, z)` 错误——Z-up 世界里那是横滚不是偏航，已修。）

接线：

- `keyboard.yaml::turn`：`support_mask_from: foot_height`（转向 clip 根几乎不平移，
  travel 掩码不适用；脚底 <5 mm 记支撑）+ `swing_lift_m: 0.05`；
- `TeleopController`：根停 + 指令转速 > `turn_step_min_deg_s`(8°/s) 时进入
  `turning` 模式——按参考自身节奏播放步进（相位随 `turn.duration_s` 推进，方向随
  指令转向与素材 yaw 符号匹配），heading 仍随指令连续旋转，weight 随进/出渐变；
- `keyboard.py`：turning 期间**禁用支撑锚**（锚会把一只脚钉死在旋转的身体上——
  这正是要消除的滑转）；
- 松开 A/D 后转速衰减穿过阈值，当前步完成即回 `stand`（有回归测试钉住）。

A/B（`turn_ab/`，hold-A 4 s ×3 循环，判据=转向期双脚离地帧占比、yaw 跟随、关节门）：

| 配置 | A 段双脚离地帧 | yaw 跟随 | 关节误差 | 皮肤穿地 |
| --- | ---: | ---: | ---: | ---: |
| off（滑转基线，无 turn 参考） | **0 / 480（0%）** | −268° | 2.48° | 3.7 mm |
| on（抬脚转向，1.0× 步频） | 250 / 480（52%） | −269° | 22.83° ✗ | 2.6 mm |
| on（**0.6× 减速**，出厂） | 241 / 480（50%） | −267° | **13.05° ✓** | 2.7 mm |

基线 0% 离地正是用户看到的滑转；抬脚转向下身体真正通过踏步完成旋转，yaw 跟随不变，
减速到 0.6× 后关节误差回到 15° 门内。50% 离地率是轻快小碎步的特征（转身期单脚
法向冲量低），视觉复核留待用户 GUI 实测。

## 复现

```bash
# 滑速归因（CPU、只读）
python3 scripts/humans/audit_slip_attribution.py --run artifacts/humans/limp_ab/new_fwd
# 转向素材筛查 + 验证（CPU、只读）
python3 scripts/humans/screen_turn_clips.py
python3 scripts/humans/verify_turn_clips.py --candidates artifacts/humans/turn_screen/turn_candidates_cmu.json
# 抬脚/转向实跑（GPU）
~/isaacsim/python.sh scripts/humans/keyboard.py --config artifacts/humans/slip_lift_ab/config_lift_fwd.yaml --out artifacts/humans/slip_lift_ab/lift5_fwd --headless --demo
~/isaacsim/python.sh scripts/humans/keyboard.py --config artifacts/humans/turn_ab/config_turn_on.yaml --out artifacts/humans/turn_ab/turn_on --headless --demo
```
