# 摔倒坍缩观感整改：关节驱动释放 + 被动关节黏性（2026-09-27，两轮）

## 用户判定与机制定位

用户实测判定摔倒过程"像雕像"，并给出机制方向：**摔倒之后各个关节 PD 控制器不应该发力**。

机制定位与该方向一致。F 触发时位置驱动已归零（`fall_control_scale: 0`），但驱动**阻尼**
按 `fall_damping_scale: 0.15` 保留（9–22.5 Nm·s/rad），且 falling 期间速度目标被强制写 0，
阻尼项因此成为 `damping × (0 − 实际速度)` 的纯速度刹车：在摔倒达到的关节角速度下制动力矩
达 150–390 Nm，远超各关节的重力矩。全阻尼（60–150 Nm·s/rad）时终端膝速只有 ~19 deg/s
（`HumanRuntime.set_control_scale` 文档记录的实测），0.15 缩放在高速段仍是硬刹车——
这就是"雕像感"的来源。根辅助在 fall 时已被 `fall_force` 完全替换（60 N × 0.25 s 方向偏置），
不构成拖拽。

零阻尼的既定顾虑是实测求解器 NaN 域：完全无阻尼的 57 DOF ragdoll 弹道式折叠
（实测坍缩中 2864 deg/s），触地冲击使 PhysX 报 `non-finite world position for link 'pelvis'`。
本轮用**逐关节 `maxJointVelocity` 限速**封住该域，而不是靠残余阻尼：限速钳住关节速度峰值，
从而钳住触地动能，零阻尼因此变得求解器安全。

## 实施

- `usd_human.author_human` / `build_human_stage` 新增 `joint_velocity_limit_rad_s`：
  在每个转动关节 author `PhysxJointAPI.maxJointVelocity`。**单位是 USD 角度单位（deg/s），
  配置值 rad/s 在 authoring 时换算**（见下文单位缺陷）。
- `ActionConfig.fall_damping_scale` 下界 0.05 → 0：0 = 完全无驱动（位置项与阻尼项都为零）。
  校验注释与 `set_control_scale` 文档写明依赖关系：**零阻尼仅在 rig 带限速时求解器安全**；
  无限速的 stage 必须保持正值。
- `load_keyboard_config` 新增顶层 `joint_velocity_limit_rad_s`（默认 None = 旧行为），
  校验有限非正。四份配置（keyboard / acceptance_matrix / locomotion_acceptance /
  keyboard_amass_hybrid）同步 `fall_damping_scale: 0.0` + `joint_velocity_limit_rad_s: 20.0`。
- `keyboard.py` 记录行新增 `joint_speeds`（`joint_velocities_rad_s`），control.npz 同步——
  本实验的判定量此前无记录字段。
- `keyboard.main` 接受可注入 argv，供实验工具复用同一物理入口。

## A/B/C 单因子实验（`scripts/humans/fall_release_ab.py`）

每档一次 Isaac 会话（RTX 4060，固定公寓，`--demo --headless --capture`），同一 18.2 s
时间线内两次完全相同的 F 摔倒（间隔 R 复位）。三档仅差目标因子，其余逐位来自
`configs/humans/keyboard.yaml`：

| 档 | fall_damping_scale | maxJointVelocity | 设计 |
| --- | --- | --- | --- |
| A_damped015 | 0.15 | 无 | 出厂基线 |
| B_damped005 | 0.05 | 无 | 旧下界 |
| C_release0 | 0.0 | 20 rad/s | 完全无驱动 + 限速 |

实测（`artifacts/humans/fall_release_ab/analysis.json`，两摔取均值）：

| 档 | 峰值关节速度 | 倒至 60% 站高 | 膝折叠角 | NaN 行 | 限速断言 |
| --- | --- | --- | --- | --- | --- |
| A (0.15) | 2137–3017 deg/s | 1.54 s | 21° | 0 | —（无限速） |
| B (0.05) | 542–705 deg/s | 1.51 s | 32° | 0 | —（无限速） |
| **C (0+限速)** | **1146 deg/s（=20 rad/s 钳位）** | **0.30 s** | **134–141°** | **0** | 57/57 关节 =20 |

- A 档低速折叠（21°）即"雕像"：阻尼刹车压制膝盖打弯，身体要等质心被推出支撑面才整体倾倒。
- B 档介于两者，膝折叠 32°，仍偏刚。
- C 档膝立刻被重力压弯直接瘫下去：坍缩 0.30 s（A 的 1/5）、膝折叠 134–141°（完全瘫折）、
  峰值速度正好钳在限速值上（限速在起作用的直接证据），两次摔倒无 NaN——**限速确实封住了
  旧实测的 NaN 域**。
- 渲染对比（`renders/`，同相机同刻）：A 档触发后 1.6 s 仍僵在半站姿；C 档 0.4 s 已瘫坐
  撑地、1.2 s 倒向侧面、2.4 s 趴地躺平，手臂自然摊开，无驱动痕迹。

## 过程缺陷（已修，如实记录）

第一次 C 档跑出全局关节误差 39.7°、站立段 p95 速度恰为 0.35 rad/s：USD/PhysX 的
`maxJointVelocity` 角速度单位是 **deg/s**，按 rad/s 数值直接写入等于把驱动钳在
20 deg/s，全身爬行（`artifacts/humans/fall_release_ab/invalid_run_units_bug/` 保留全套装
产物含该次回归会话）。修复：authoring 时 `np.rad2deg` 换算；测试断言更新为换算期望；
分析脚本 clamp 校验同步。该缺陷由"站立段速度分位恰好等于钳位值"这一一致性检查暴露，
属配置-单位契约错误，非机制错误。

## 第二轮：用户复看 C 档"像一摊水"——补回被动关节黏性

用户复看后判定：零阻尼的人"像一摊水，连基本人的样子都看不出来"，并提出**是否需要
一定阻尼像真实人体关节**。数据完全支持这个判断：

- C（0）请求→触地仅 0.36 s，比躯干自由落体（0.9 m 约 0.43 s）还干脆；膝折 134–141°
  （小腿顶死大腿），躺平根高 0.184 m——这就是"水"：无任何关节阻力。
- A/B（0.05–0.15）1.5 s，是"雕像"。
- 真人失稳到触地约 0.6–1.0 s，介于两者之间；真实关节存在被动黏性
  （肌肉张力、关节囊、韧带阻力），**完全零阻尼只适合无意识布娃娃，不适合模拟摔倒**。

### D–G 细扫（限速 20 rad/s 常开）

| 档 | damping_scale | 请求→触地 | 膝/髋/肘折叠 | 读取 |
| --- | --- | --- | --- | --- |
| C | 0 | 0.36 s | 134–141° / 60–90° / 7–29° | 水 |
| G | **0.005** | **0.94 s** | **112–115° / 11–21° / 7–10°** | **平躺屈膝，真人节奏** |
| E | 0.01 | 1.30 s | 50–78° / 19° / 2–9° | 躺姿像人但节奏偏慢 |
| D | 0.02 | 1.39–1.48 s | 24–51° / 7–11° | 偏雕像 |
| F | 0.04 | — | 13–35° / 2–7° | 接近雕像 |
| A | 0.15（无限速） | 1.61 s | 21° | 出厂基线（雕像） |

- **G（0.005）采纳**：触地 0.94 s 正中真人窗口；髋 11–21°（躯干平贴）+ 膝 112–115°
  （平躺屈膝）+ 肘 ~8°（手臂伸直）= 典型摔倒躺姿，保持人形；峰值关节速度 866 deg/s
  **连限速都没碰到**（限速在该档是纯 NaN 保险）。0.005 × 60–150 Nm·s/rad =
  0.3–0.75 Nm·s/rad，恰在文献人体被动关节黏性的低段量级。
- 两轮之间的陡峭过渡（0→0.94 s→1.30 s→1.42 s）说明该参数是"悬崖"不是"斜坡"：
  0.005 与 0.01 之间观感差异巨大，调参需细步进。
- E/D/F 渲染保留在各自 `renders/` 目录供 GUI 复看。

### 采纳与回归（第二轮后最终值）

- 四配置（keyboard / acceptance_matrix / locomotion_acceptance / keyboard_amass_hybrid）
  终值：`fall_damping_scale: 0.005` + `joint_velocity_limit_rad_s: 20.0`。
- 限速不约束正常动作的量化（零阻尼轮实测，限速机制未变仍适用）：默认 demo 正常动作
  关节速度峰值 8.41 rad/s，为限速 20 的 42%（余量 2.4 倍）；站立段 p95 仅 0.35 rad/s。
- 默认配置标准 demo 回归（`artifacts/humans/keyboard_fall_release_regression/`，
  0.005 终值重跑）：errors=[]、关节误差 max 5.527°（≤15° 门，与基线 5.8° 同级）、
  滑速 p95 0.019 m/s（≤0.15 门）——被动黏性回调不影响常规动作门。
- 全套 CPU 452 passed / 12 skipped（新增：限速 authoring 3 条 + 配置契约 1 条 +
  damping_scale=0 合法性 1 条替换旧负例）；compileall、Ruff 全过。
- 局限（如实）：侧向/行走中摔倒未复测；0.005 由两次确定性重复支持，非统计；
  限速 20 rad/s 先验选取未扫描；"真人 0.6–1.0 s"为文献常识量级引用，未做实测标定。
