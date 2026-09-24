# 系统架构

[文档索引](README.md) · [当前计划](../task_plan.md)。下图描述目标链路，实际完成状态以计划及试验报告为准。

```text
场景资产（CPU 可复现）
  ├─ configs/scenes/*.yaml       房间尺寸、墙体归属、门窗开口、家具、材质覆盖
  ├─ scenes/spec.py               schema + 快速失败校验
  ├─ scenes/materials.py          视觉 / 力学 / 电磁 三类属性
  ├─ scenes/furniture.py          参数化家具配方
  └─ scenes/planner.py            墙体分段、开口、家具 → ScenePrim 列表 + 清单
          │ ScenePlan（纯数据，无 Isaac 依赖）
          ▼
scenes/usd.py → /World USD
  ├─ 几何：Cube / Cylinder，Z-up，单位米
  ├─ 碰撞：UsdPhysics.CollisionAPI（全量）
  ├─ 刚体：RigidBodyAPI + MassAPI（仅动态家具，组合刚体）
  ├─ 材质：视觉 UsdShade + 按摩擦三元组去重的物理材质
  └─ sim2sense:* 自定义属性（房间、语义、质量、可移动、标签）
          │ USD 场景
          ▼
Isaac Sim
  ├─ 物理约束人体轨迹
  ├─ 动态 mesh / 关节真值
  └─ 场景、材质、链路元数据
          │ 统一时间戳
          ▼
Sionna RT
  ├─ 多径路径与传播时延
  ├─ CIR / CSI
  └─ 频段、天线、噪声与硬件随机化
          │ ChannelSample
          ▼
预处理与窗口化
  ├─ 复数幅相/实虚部规范化
  ├─ 时频表示或时序编码
  └─ 按域划分、质量检查
          ▼
检测模型
  ├─ 监督分类：fall / adl
  ├─ 异常检测：学习正常活动分布
  └─ 域泛化：未见场景、个体、硬件
          ▼
报警聚合
  ├─ 连续窗口阈值
  ├─ 报警冷却与去抖
  └─ 置信度、延迟、审计日志
```

## 组件边界

- `schema.py` 和 `validation.py` 不依赖 Isaac Sim/Sionna，作为跨运行时的稳定接口。
- `simulators.py` 只定义适配器协议和 dry-run；真实仿真实现应放到独立模块，记录确切版本和场景配置。
- `windowing.py` 是 CPU 可运行的报警基线，不代表最终模型性能。
- 后续模型代码必须接收 `ChannelSample` 或其派生张量，不直接读取 USD、视频或任意目录中的隐式文件。
- 场景侧同样是「规划 / 落地」分离：`scenes/spec.py`、`scenes/materials.py`、
  `scenes/furniture.py`、`scenes/planner.py` 是纯 Python，可在没有 Isaac Sim 的机器上单测和评审；
  `scenes/geometry.py` 和 `scenes/numbers.py` 提供 CPU 校验；
  `scenes/usd.py`、`scenes/view.py` 和 `scenes/verification.py` 的运行时操作需要 USD/Isaac，`pxr` 均惰性导入。
- 场景实现集中在 `src/sim2sense_fall/scenes/`；`scripts/scenes/` 负责命令行和应用生命周期，
  `tests/scenes/` 对应场景回归。配置仍在 `configs/scenes/`，产物仍在 `artifacts/scenes/`。
- 场景的消费边界是 `ScenePlan`（不可变数据）和导出的 USD；下游不应该反过来依赖
  `scenes.planner` 的内部函数。

## 人体控制与真实运行时

- `humans/amass.py` 负责动作读取和坐标适配，`humans/rig.py` 将参考旋转分解为配置关节目标。
- `humans/teleop.py` 处理键盘意图、周期步态、限速和平滑；`humans/root_control.py` 生成有界根辅助力/力矩。
- `humans/usd_human.py` 连接 PhysX 驱动与接触报告，读取实际姿态；SMPL 显示消费实际姿态。
- `scripts/humans/keyboard.py` 管理窗口、物理回调、实时蒙皮和记录。现有键盘模式为辅助物理控制。
  新增蹲下/起立/摔倒状态机尚待实现；行走路线由用户选择，不包含主动避障或导航模块。
- `sionna/apartment.py` 转换固定公寓几何，`sionna/mesh_import.py` 导入人体，`sionna/channel.py` 处理复数信道。
  公寓 + 物理后推跌倒的 CIR smoke 已完成；动态家具同步、数据集和训练仍待完成。

运动学参考预览、物理仿真和记录后渲染是三条不同路径。完整边界见 [人体指南](human-simulation.md)。

## 关键风险

1. 仿真信道与真实硬件的幅度、相位、噪声和同步误差不一致。
2. 单人模拟跌倒不能代表老人真实跌倒、缓慢滑落和遮挡情况。
3. 多径变化可能来自家具或其他人，而非跌倒主体。
4. CSI 仍可能暴露活动、位置和身份属性，需要最小化存储与访问控制。
5. 场景是单一固定布局，若不随机化房间、材质与家具，模型会把「这个房间」学成特征，
   而不是学「跌倒」这一类事件的物理变化。
6. 电磁材质中有一部分是 ITU-R P.2040 的代理值（瓷砖、洁具、地毯、织物），如果直接
   把它们当作真值，仿真信道会带上系统性偏差。
7. 家具几何是图元近似，粗糙的接触形状会影响跌倒时的接触动力学与遮挡，需要在加入
   人体后重新评估。
