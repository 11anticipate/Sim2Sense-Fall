# 数据集卡：train01（正式训练前的第一个分组感知批量包）

生成日期：2026-09-27。批次根目录 `artifacts/batches/train01/`，
划分文件 `artifacts/batches/train01/splits.json`（seed 20260927）。
复现入口见 [项目 README](../README.md) 的「训练前数据流水线」；本卡只描述这一批的事实与边界。

## 组成

8 个 Isaac 键盘会话（120 Hz 原生蒙皮）→ 46 个分段样本 → **31 个通过分段级准入**并
完成 Sionna RT 复数 CIR 的样本。被拒收的 15 段全部因为该活动自己的实测门未过
（起身、坐/蹲/弯腰过渡、起立交接），不是标注缺失。

| 标签 | 样本数 | 说明 |
| --- | ---: | --- |
| `stand` | 16 | 会话首尾的站立段 |
| `walk` | 7 | 前进/后退/转向中的行走段 |
| `fall` | 6 | 站立摔、行走中摔、转向中摔、后退中摔（均含实测冲击与躺倒验证） |
| `crouch` | 1 | 深蹲保持（`adl_postures`） |
| `sit` | 1 | 地面坐保持（`adl_postures`，恢复期低权威辅助剖面后才过门） |

- 通过准入的蒙皮真值合计 **72.6 s @ 120 Hz**（`physics_pre_step` 采样，非插值）。
- 信道侧**全率 120 Hz**：31 样本共追踪 **8714 帧** CIR（每样本 61–552 帧，中位 240），
  `sample_rate_hz` 全部 120.0，`target_rate_achieved` 等七项检查全过，`failures` 全空。
  第一次以 `--frames 24` 产出的 5–40 Hz 版本保留在 `sionna_stride_5to40hz/` 供对照。
  全率下 `static_body_repeat` 出现 3 例瞬时失败，按幂等重试（3→1→0）解决，**未放宽判定**。
- 全部样本 `seed = 42`（RT 前端），场景为固定公寓 `indoor_apartment.usda`。

## 划分

分组单位 = **物理会话**（同一会话的分段共享连续轨迹、接触历史与控制器状态），
绝不按样本切分；`splits.json` 的 `leakage_check.groups_spanning_splits` 为空。

| split | 样本 | 标签构成 |
| --- | ---: | --- |
| train | 16 | stand 8, fall 4, walk 2, crouch 1, sit 1 |
| val | 11 | stand 6, walk 4, fall 1 |
| test | 4 | stand 2, walk 1, fall 1（整会话 `fall_walking`） |

`crouch`、`sit` 各只有 1 个组 → 只能落在单一 split，已在
`report.insufficient_groups` 里显式登记，不得声称它们被跨组评测过。

## 标签来源与失衡起点

标签来自会话的 `fall.events` 与控制器模式时间线（`export_session_mesh.py`），
**不来自文件名**；`fall` 必须同时有实测冲击与实测低姿+倾角，否则拒收。

键盘会话没有试验那样的显式失衡起点。本批的 `imbalance_onset_s` 由
`detection_data.session_event_times()` 从实测根高导出：取冲击前「骨盆仍在摔倒前高度
5 cm 以内」的最后一帧（高度口径而非速度口径，避免接触回弹把标签抹掉），
并要求总降幅 ≥ 0.15 m，否则不给正样本（fail-closed）。
`ChannelSample.onset_source` 恒为 `derived:root_descent_to_impact`，
不得当作人工标注的失衡起点。实测该导出点落在按键请求后 0.15–0.43 s、
冲击前 0.17–0.52 s——也就是说**请求时间确实不等于失衡起点**，这个差现在是量出来的。

## 已知边界（正式训练前必须知道）

1. **信道时间分辨率：已按预登记口径重跑为 120 Hz。** 本批第一次 RT 阶段实测只有
   5–40 Hz，原因不是算力，而是 `import_fall_mesh.py --frames` 是**步长上限**而不是目标
   采样率（492 帧的 120 Hz 片段要 96 帧，实得 82 帧 = 20 Hz），我又不该把批次墙钟里
   混着的 Isaac 会话时间算成 RT 成本（因此一度错报「全批约 10 h」）。
   实测真实成本：固定开销约 2 s + 每帧 0.062 s（RTX 4060 笔记本），
   4.1 s 摔倒片段全率 492 帧 = **33 s/样本**，整批 31 样本 ≈ 10 min。
   新增 `--target-hz`（并拒绝超过蒙皮节率的声明），批量计划改用 `rt_target_hz: 120`。
   **仍未达的另一支预登记口径**：摔倒段 ≥200 Hz 无混叠——蒙皮真值节拍由
   `physics_dt_s = 1/120` 封顶，要 200 Hz 需把物理步长降到 1/240 重跑会话，
   按实测 RTF 0.26 推算会话墙钟时间约翻倍。属用户决策项。
2. 单一受试者（SMPL neutral）、单一场景（一套公寓）、单一 RT seed →
   跨人/跨房间/跨链路泛化**未被本批测量**。
3. 本批 8 个会话由**两个代码状态**产出：5 个在恢复期辅助降权之前，3 个在之后
   （`adl_postures`、`fall_getup_standing`、`fall_getup_walking`）。差异记录在各自
   `report.json::controller_sources_sha256` 与 `root_assist_sha256`；被拒收的
   `*_v1` 版本仍留在批次目录里供对照。
4. 起身（`get_up`）与起立交接（`stand_up`）样本全部未过门 → 本批**不含**起身类训练样本。
   见 [起身悬空专项](getup-float-2026-09-27.md)。
5. 训练入口仍把报告标为 `flow_smoke_only`：120 Hz 版是 31 样本 / 427 训练窗 / 66 评测窗
   （其中 165 个摔倒正窗）。整会话留出 `fall_walking` 后窗口准确率 0.879、
   速度头 MAE 7.44 Hz——n=1 会话留出，**不构成性能结论**。
   同一 seed 下划分与升级前**逐样本完全一致**（已断言），所以两个版本可直接对比。

## 复现命令

```bash
PYTHONPATH=src python3 scripts/humans/audit_spawn_clearance.py --human-radius-m 0.9 \
    --pick 8 --out artifacts/humans/spawn_clearance_activity.json
PYTHONPATH=src python3 scripts/humans/make_dataset_configs.py
PYTHONPATH=src python3 scripts/sionna/batch_generate.py --plan configs/sionna/batch_train01.yaml \
    --out artifacts/batches/train01 && bash artifacts/batches/train01/run_batch.sh
PYTHONPATH=src python3 scripts/sionna/assign_splits.py --batch artifacts/batches/train01 \
    --seed 20260927 --fractions 0.6,0.2,0.2
~/.local/opt/sionna/bin/python scripts/sionna/train_baseline.py \
    --sionna-dir artifacts/batches/train01/sionna \
    --splits artifacts/batches/train01/splits.json --eval-split test \
    --velocity-weight 0.5 --device cuda --out artifacts/batches/train01/train_velocity
```
