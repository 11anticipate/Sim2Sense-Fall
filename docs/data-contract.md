# 数据契约

每个无线样本以一个 `ChannelSample` 表示。数组维度必须与 `channel_representation` 一致，所有时间都使用秒，且同一 `sample_id` 下的运动、网格和无线观测共享时间基准。

| 字段 | 类型 | 约束 |
|---|---|---|
| `sample_id` | string | 全局唯一、稳定 |
| `scene_id` | string | 房间/布局/材质配置标识 |
| `subject_id` | string | 仿真人体或受试者匿名标识 |
| `hardware_profile` | string | 发射机、接收机、天线和频段配置 |
| `activity` | enum | `fall`、`adl`、`unknown` |
| `timestamp_s` | float array | 严格递增、有限值 |
| `channel` | complex array | 与时间轴长度匹配 |
| `channel_representation` | enum | `csi` 或 `cir` |
| `sample_rate_hz` | float | 大于 0 |
| `simulator_version` | string | 生成工具和版本 |
| `metadata` | mapping | 随机种子、材质、姿态、链路等可追溯信息 |

部署模型只消费 `channel` 及允许公开的元数据；人体骨架、动态网格和视频属于训练/验证辅助资产，不能默认进入部署数据流。

## 划分要求

禁止先随机切窗再划分。应先按 `scene_id`、`subject_id` 或 `hardware_profile` 划分域，再在各域内切窗，避免相邻窗口泄漏到不同集合。
