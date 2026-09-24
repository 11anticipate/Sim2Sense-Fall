# AMASS 动作和物理交互实证复核

> 日期证据：本轮早于 [键盘实时显示](keyboard-control.md)，下文“实时皮肤静态”仅描述当时试验入口。
> 后续键盘已有运行期间截图，但滑步/穿透仍未通过。最新待办见 [当前计划](../task_plan.md)：
> 新增动作仅蹲下、起立、摔倒；路径由用户键盘控制，拳击、坐椅、抓握及主动避障不纳入本轮。
> [全部文档](README.md)。

## 目标与证据边界

以实际 AMASS、SMPL、PhysX 状态及渲染检查动作是否完成。参考预览不作为物理完成证据。
保留现有工作树改动。此文件记录本轮证据，历史文档结论需要重新实测。

## 初步发现

- 实际 AMASS 原始文件 2198 条。RTX 4060 Laptop 8 GB，driver 595.91.07。
- 原导入器以模型局部 basis 同时旋转世界平移；真实 CMU/Transitions 为 Z-up 世界。
  sit_stand 原始 trans 峰峰值 XYZ=[0.240, 0.385, 0.802] m；原转换将 Z 变成 X。
  10_05 原始根旋转作用于模型 +Y 后为 [0.068, 0.018, 0.998]，说明原始根已将模型直立。
- 原 normalize_root_motion 用 R[k] @ R[0].T 抹去全部初始姿态，且不旋转平移，
  不构成一致的世界坐标变换。应仅绕重力轴变换朝向，保留倾斜和高度变化。
- simulate 的自由模式只驱动关节；没有根轨迹/平衡闭环。--pin-root 每步传送骨盆，
  只能作为辅助调试，不能证明无辅助动作完成。
- 历史 trials_amass_anchored.log 在 2298/5202 帧因几何接触归属失败退出，
  只有 human_trial.usda，没有完成的 trial 记录。
- 默认视口皮肤没有随实际关节逐帧变形；已有结束后播放函数需验证时间线确实停止。

## 已修复的实际缺陷

1. 区分 SMPL 模型局部 Y-up 基和 AMASS 世界 Z-up 基。局部关节用 `B R B.T`，
   根旋转用 `R_source @ B.T`，世界平移保持原值。归一化只消除首帧偏航，
   对根旋转和平移施加同一世界变换，保留初始倾斜；首帧落地仅用恒定竖直偏移。
2. 接触句柄用 `PhysicsSchemaTools.intToSdfPath` 解码，两侧 collider 路径均存入
   `.contacts.npz`。补上根刚体下的 pelvis collider，接触法向/冲量随人体侧统一换向。
   不再依靠 2 cm 几何邻近猜归属，避免长动作中途报错。
3. 补 head、hand 末端碰撞体，共 22 个；皮肤与碰撞代理仍非完全重合。
4. PD 同时输入目标位置与参考速度；新增关节限位预检，无法表达的动作提前拒绝。
5. 新增显式有限根外力控制及 `.control.npz`。辅助不是自主平衡，不能混入无辅助结果。
   力/力矩上限分别为 1500 N / 400 Nm，重力补偿比例 0.7，全部在 YAML 和产物中记录。
6. 新增 `render_recording.py`，在物理停止后渲染实际保存的逐帧 SMPL 世界坐标网格。
   `captures.json` 保存输入哈希、帧号、时间和相机。参考和实测共用相机。

## 本轮实测结果

以下都是本机真实 Isaac 运行；单一固定公寓，无训练/测试集划分，不能推断跨场景泛化。
统一 seed=20260922，SMPL neutral v1.1.0，1.7 m / 72 kg，57 DOF / 62 连杆，
物理 120 Hz，导出信道网格 50 Hz。AMASS 数据为本地 Transitions_mocap/mazen_c3d。
实际模型、动作源、配置、场景 SHA-256 与源时间区间见每个 `.trial.json`；
重定向版本 `retarget_version=2`。运行环境 Isaac Sim 6.0.1-rc.7，RTX 4060 Laptop。

指标：关节误差为全时段全部 DOF 的最大绝对目标误差（门槛 15°）；根位置为最大
欧氏误差（门槛 0.15 m），根旋转为最大测地角（门槛 20°）；皮肤最低 Z 相对地板 Z=0。
通过这些门槛仅表示当前辅助跟踪可用，不等于人体生物力学正确或完全无穿透。

| 试验目录 | 时长 | 最大关节误差 | 最大根位置误差 | 最大根角误差 | 最低皮肤 Z | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sidewalk_full` | 7.275 s | 8.147° | 24.6 mm | 4.382° | -3.15 mm | 完整侧行，辅助通过 |
| `backwalk_clear` | 7.867 s | 10.793° | 27.4 mm | 3.895° | -3.70 mm | 完整后退，辅助通过 |
| `sidewalk_free` | 3 s | 37.304° | 2.110 m | 104.92° | -65.6 mm | 无辅助失败 |
| `sidewalk_barrier` | 3 s | 38.088° | 1.132 m | 11.311° | 0 mm | 动作跟踪失败；挡墙响应成立 |
| `boxing_final` | 3 s | 33.34° | 32.6 mm | 3.526° | -3.14 mm | 速度前馈与 collar 调整后仍失败 |
| `sit_assisted_limits2` | 3 s | 45.75° | 103.4 mm | 4.25° | -28.7 mm | 速度前馈加入前失败，非最终控制器结论 |
| `sit_final_3s` | 3 s | 38.434° | 59.1 mm | 3.709° | -27.95 mm | 末帧已坐地，全程跟踪仍失败 |
| `crouch_full` | 8.717 s 源片段 | 未运行物理 | - | - | - | 左膝副轴超限 8.535°，预检拒绝 |
| `sit_final` | 8.158 s 源片段 | 未运行物理 | - | - | - | 当前配置左髋副轴超限 2.992°，预检拒绝 |

`backwalk_full` 从默认出生点撞到卧室东墙（1822 个墙接触点），根误差 0.866 m。
`backwalk_clear` 将出生点移到 (12, 1.455)，相同动作全长通过。
这说明需要场景感知的轨迹/支撑面安排，不能将与家具冲突的参考强制执行。

### 截图与碰撞对照

产物根目录：`artifacts/amass_audit_20260924/`（被 Git 忽略）。

![同视角参考/实测及挡墙对照](../artifacts/amass_audit_20260924/evidence_comparison.png)

上排：全长侧行第 436 帧（3.633 s），左为 AMASS 参考，右为实际 PhysX 状态蒙皮。
下排：同一 3 s 动作、同一控制器、同一相机，左为无墙，右为有墙。
已实际查看 PNG，而非仅检查文件存在。截图是实际保存状态的后渲染，不是仿真运行中的实时皮肤。
完整 5 帧序列分别在 `sidewalk_full/captures`、`sidewalk_full_reference/captures`、
`backwalk_clear/captures`、`backwalk_reference/captures`。
全长动作的 20 张参考/实际截图汇总为 `full_motion_sequences.png`，已逐行查看。

挡墙为独立场景 `barrier.usda`，中心 (10.5, 2.55, 1.0)，尺寸 (3, 0.12, 2) m。
178 帧有 919 个实际挡墙接触点，峰值冲量 26.207 Ns；根最大 Y=2.343 m，
墙近侧 Y=2.49 m，人体根被阻止继续前进。皮肤最大 Y=2.515 m，
约 25.2 mm 的局部皮肤越过墙表面，说明碰撞代理仍需拟合；不宣称零穿墙。
地板接触与碰撞对象 ID 来自 PhysX report。Tensor 接触力视图不受本机构建支持，
因此保存的是接触点/冲量，`has_contact_forces=false` 如实保留。

早期自由 sit_stand 失败已直接查看 `sit_physics/captures_top/frame_00360.png`：
实际人体前栽、头部没入地板，与 `sit_reference/captures/frame_00360.png` 的坐地姿态不同。
对应旧配置最低皮肤 -128 mm，不能用参考预览冒充坐地动作完成。
最终控制器复测前 3 s，关节误差由 45.75° 改善到 38.434°，仍不通过，
皮肤最低 -27.95 mm；当前截图在 `sit_final_3s/captures`，与坐地参考共用相机。
已查看末帧：坐地姿态接近参考，但不能以末帧外观替代全过程误差及穿透检查。
启发式标签输出 `fall`，这里仅记录分类器输出，不视为 AMASS 片段的真实意图标签。

### 最终基础回归

- CPU：277 passed / 9 skipped，compileall 与 Ruff 通过。
- 场景 CPU dry-run、最终人体 CPU dry-run 通过。
- 场景重新导出到独立 `scene_rebuild/` 并通过实际物理验证：动态椅子抬高 0.25 m
  后下落 0.25 m，回到原始位置；场景哈希未被验证过程改写。
- 最终人体 Isaac 验证 `final_isaac/human_verify.json` 通过：57 DOF，22 碰撞体；
  隔离碰撞后的最大 PD 误差 11.462°，驱动关闭负对照 85°；抬高后实际下降 1.0176 m，
  最后 200 ms 水平漂移 0.037 mm，最终碰撞几何未穿地。它验证基础刚体链，
  不替代动作全程皮肤/接触验收。
- 曾遇审批 HTTP 503，现权限阻塞已解除；Ruff 首次因旧代理端口拒绝连接失败，
  清除该命令代理环境后 `uv tool run ruff check .` 成功。

## 当轮剩余能力（历史建议，当前范围以计划为准）

1. **自主平衡和动作控制**：根外力峰值接近 1 kN、力矩可达 400 Nm，属于显著外部辅助。
   下一阶段需要接触感知全身控制或物理模仿策略，逐步撤去辅助后验证行走/恢复。
2. **接触几何和无滑动验证**：头手已补碰撞，但脚/躯干代理与 SMPL 皮肤仍不贴合；
   需按接触点速度而非踝关节速度测滑动，补接触持续时间、穿透分布和冲量一致性。
   当前自碰撞关闭，不应声称肢体相互作用正常。
3. **坐地起立、拳击、蹲行**：快速/大幅动作未通过；不能无条件放宽全部关节限位或门槛。
   需要关节轴/限位复核、惯量与增益标定、支撑切换控制。
4. **场景适配**：路径不得穿家具，坐椅动作需要对齐座面；当前已实测地板与挡墙，
   尚未验收坐椅、搬物、抓握等任务。
5. **体型一致性**：AMASS betas 可读但当前物理人体仍为 neutral 统一缩放，
   没有按源人物 betas 构建质量分布与碰撞体；不能宣称源人体严格重建。
6. **标签和显示**：启发式跌倒标签可能把主动坐地当跌倒；AMASS 意图不能由高度阈值决定。
   物理期间实时皮肤仍静态，当前可验证路径为仿真后实际记录回放。

阶段 7 保持未完成。本轮完成了两种真实动作的辅助物理跟踪、挡墙对照和可追溯截图证据。

## 复现命令

```bash
# 完整侧向行走；默认 seed 在配置中固定
~/isaacsim/python.sh scripts/humans/simulate.py --headless \
  --config configs/humans/human_smpl_multiaxis.yaml \
  --amass-file data/humans/amass_raw/Transitions_mocap/mazen_c3d/walksideways_stand_poses.npz \
  --root-assist configs/humans/root_assist.yaml \
  --out artifacts/amass_audit_20260924/sidewalk_full

# 后退：同一命令改为 walkbackwards_stand_poses.npz，并加
# --spawn-x 12 --spawn-y 1.455，输出换到 backwalk_clear

# 实际记录的截图；参考截图另传 .mesh.npz 并共用 --camera-from
~/isaacsim/python.sh scripts/humans/render_recording.py \
  --record artifacts/amass_audit_20260924/sidewalk_full/smpl_amass_multiaxis__amass__walksideways_stand_poses__none.npz \
  --out artifacts/amass_audit_20260924/sidewalk_full/captures

# 独立挡墙场景；加 --dry-run 可先在 CPU 校验
~/isaacsim/python.sh scripts/humans/build_contact_barrier.py \
  --out artifacts/amass_audit_20260924/barrier.usda \
  --center 10.5 2.55 1.0 --size 3 0.12 2
# 侧行命令加 --amass-duration 3 --scene artifacts/amass_audit_20260924/barrier.usda
# 使用 sidewalk_barrier 输出；此动作跟踪预期失败，接触记录用于检验墙的阻挡响应
```
