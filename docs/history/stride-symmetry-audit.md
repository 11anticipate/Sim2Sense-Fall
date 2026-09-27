# 走路一瘸一拐（物理侧判定）：换整步幅窗口后跛行消失

维护日期：2026-09-24。工具 `scripts/humans/audit_stride_symmetry.py`（常驻、CPU、只读），
门槛 `configs/humans/stride_symmetry_gate.yaml`（预登记），
证据 `artifacts/humans/limp_ab/`（4 次 GPU Isaac 实跑）。
参考侧根因与修复见 [步幅窗口拟合](gait-window-fit.md)。
[全部文档](README.md) · [当前计划](../task_plan.md)。

## 本文回答的问题

[窗口拟合](gait-window-fit.md)把 forward/backward 窗口改到整步幅，但那篇里的全部数字
都是**参考侧**的，并明确写了「出数前不得声称一瘸一拐已在物理上消失」。本文是那次
A/B 的物理侧判定：**旧窗口在 PhysX 里确实跛，新窗口在 PhysX 里确实不跛**。

## 方法

### 测什么：体轴前后错位，逐整周期

跛行的可见症状是「一条腿一直伸在前面」——即两脚**平均前后位置的恒定差**
（stagger），而不是步幅比值。旧窗口前进侧的步幅比本来就是 0.989（近乎对称），
比值永远抓不住这个症状；错位才抓得住。

- 前后轴 = 根旋转矩阵第一列（面朝方向），与 `travel.py` 同一约定。
  第一版审计误用了第二列（左右轴），量出 160→193 mm 的假「未修复」——
  那是自然站距 + 相位快照，不是跛行。该错误已在工具注释与测试里钉死。
- 每只脚的前后轨迹来自**记录的 PhysX 关节状态**（`control.npz::joints`）的正运动学，
  同时对 `joint_target`（控制目标）做同一计算：目标与实际错位之差 = 跟踪层的贡献。
- **只对整周期打分**： Walking 段按 gait_phase 回绕切周期，跨度不足一周期的不计。
  A/B demo 的 S 段只有 0.2–0.33 个周期，它的「错位」只是相位快照（测到过 716 mm
  的假读数）；整周期规则的拒绝行为有回归测试钉住。
- 后退播放时相位**递减**，回绕判定是「原始相位跳变 |Δ|>0.5」，
  不是 `Δ<0`（那是后退的每一步）——第一版犯过此错，已修并有测试。

### 门槛（预登记于 `stride_symmetry_gate.yaml`）

- `max_stagger_m: 0.05`——主判据。旧窗口物理侧 232 mm 必须判失败，
  修复后必须压在 50 mm（约步幅的 4%）以下。
- `min/max_stride_ratio: 0.90/1.11`——辅助判据，只防 gros 不对称。

### 试验协议（`artifacts/humans/limp_ab/config_*.yaml`）

demo 内置 R 复位循环：`[2 s 站立, 10 s W, 1 s 制动, 1.5 s, R] × 3`（45 s，每方向 3 段行走）。
前进 heading 90（+Y 走廊净空 5.03 m，走 4 m）；后退 heading −90（S 同样沿 +Y，
绕开 −Y 方向仅 1.33 m 的墙）。四个 run：{old, new} 窗口 × {前进, 后退}，
其余配置逐位相同。GPU（RTX 4060）实跑，`errors: []`，RTF 与键盘会话同量级。

## 结果（`stride_symmetry.json`）

| run | 方向 | 整周期数 | 步幅 L/R (mm) | 比值 | **错位 mean (worst)** | 判定 |
| --- | --- | ---: | --- | ---: | --- | --- |
| old_fwd | forward | 6 | 599 / 606 | 0.989 | **+231.6 mm** (232.0) | ❌ |
| new_fwd | forward | 6 | 694 / 640 | 1.084 | **+4.6 mm** (5.0) | ✅ |
| old_bwd | backward | 6 | 716 / 644 | 1.111 | **+113.8 mm** (114.7) | ❌ |
| new_bwd | backward | 3 | 826 / 788 | 1.049 | **−1.1 mm** (1.1) | ✅ |

- **跛行消失**：前进错位 231.6 → 4.6 mm（50 倍），后退 113.8 → −1.1 mm（100 倍）。
- **跟踪不是共犯**：目标侧错位与实际侧差 ≤ 2.4 mm（前进 3.2→4.6、后退 2.0→−1.1），
  与 P0-A 的「PhysX 忠实执行目标」结论一致。错位完全来自参考窗口，也只被窗口修复。
- **步幅比值在旧配置下是假阴性指标**：old 前进 0.989「看起来对称」而人眼看到跛；
  修复后比值 1.084 反而比旧的大——比值与可见跛行无关，这就是门槛以错位为主的原因。
- 视觉对照（同相位 0.25/0.5/0.75，跟随相机，俯视/侧视）：
  `montage_top_forward.png`、`montage_side_forward.png`、`montage_top_backward.png`、
  `montage_side_backward.png`。侧视图受室内家具遮挡，俯视图可见 old 行单腿拖后、
  new 行两腿对称；画面判读以数值为准。
- 顺带的同协议 A/B 副产物（同一批 run 的 `report.json`）：
  脚底滑速 p95 前进 0.610 → **0.261 m/s**（减 57%），锚点释放 1575 → 636 次，
  关节误差 6.086° → 4.589°——新窗口的脚真的更「踩得住」。后退侧滑速 0.789 → 0.783
  基本不变。滑速仍全部 > 0.15 门槛，仍归 P0-B（根辅助抬着身体、参考自身滑步）。

## 残留与未做

- 步幅比 new 前进 1.084（L 比 R 长 8%）：过门槛、非可见跛行症状（错位 4.6 mm），
  未深挖；若日后要收窄，从支撑 IK 与接触不对称入手，不回退窗口。
- 判定基于 4 条 45 s 会话（每方向 2–6 个整周期）、单一公寓单一速度 0.4 m/s，
  无训练/测试划分；未做转向中的步幅判定（转向不推进步态相位，见 P0-A 记录）。
- 本次会话的两处分析缺陷（前后轴选错、后退回绕误判）都先产生过**看起来可信的假读数**
  （「new 193 mm 未修复」「backward 716 mm」），均由整周期/体轴规则修正并有回归测试；
  引用本文件数字时以 `stride_symmetry.json` 为准。

## 复现

```bash
# 判定（CPU、只读，消费任一实跑目录）
python3 scripts/humans/audit_stride_symmetry.py --run artifacts/humans/limp_ab/old_fwd \
  --run artifacts/humans/limp_ab/new_fwd --run artifacts/humans/limp_ab/old_bwd \
  --run artifacts/humans/limp_ab/new_bwd --out artifacts/humans/limp_ab/stride_symmetry.json

# 长时实跑（每方向 ~2 min GPU）
~/isaacsim/python.sh scripts/humans/keyboard.py --config artifacts/humans/limp_ab/config_new_fwd.yaml \
  --out artifacts/humans/limp_ab/new_fwd --headless --demo

# 同相位对照图
~/isaacsim/python.sh scripts/humans/render_recording.py \
  --record artifacts/humans/limp_ab/new_fwd/recording.npz \
  --out artifacts/humans/limp_ab/new_fwd/top \
  --times 5.86 6.58 7.3 --azimuth-deg 90 --elevation-deg 85 --follow
```
