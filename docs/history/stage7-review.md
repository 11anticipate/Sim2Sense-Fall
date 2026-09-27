# 阶段 7 批判性验收（2026-09-22）

> 历史快照：下文旧资产扫描、单轴 rig、GPU 和未完成项不能作为当前状态。
> 后续真实 AMASS 已导入，多轴 PhysX 与键盘 GUI 已运行；当前边界见 [人体指南](human-simulation.md)，
> 证据见 [文档索引](README.md)。保留本页的失败样例和数据划分风险供回归。

> 本文保留第一次复审的缺陷证据。二次验收发现当时资产扫描处于旧状态：当前 `data/humans/smpl/` 已有并实际加载 SMPL v1.1.0 neutral/male/female（3/5 模型可用），neutral CPU 静止蒙皮已通过；AMASS 原始序列仍未发现。逐帧网格整改与最新边界见 [`docs/mesh-export.md`](mesh-export.md) 和 [`docs/progress.md`](progress.md)。因此下文“SMPL 0/5、所有产物均为代理”的句子只代表第一次复审快照，不代表当前资产状态。

## 结论与资产（第一次复审快照）

**第一次复审完整验收不通过；程序骨架与胶囊代理原型部分通过。** 本轮快照未修改实现代码。二次验收已修正资产扫描并补上逐帧网格导出，但 Isaac 稳定性和 AMASS 仍未闭环。

本节的旧扫描记录为：配置资产根被判定不存在、SMPL 审计 0/5。二次验收已确认该结论过时：当前 `data/humans/smpl/` 有 3 个 v1.1.0 模型，neutral 已实际加载；项目仍没有 AMASS 原始序列，现有动作均为 `motion_kind: scripted`。详见本文件开头的更新说明。

## 待整改问题

### R1 / P1：文件存在被错误当成真实模型导入

位置：`scripts/humans/plan.py:116`、`:339`；`scripts/humans/common.py:114`；`scripts/humans/build.py:116`。

`--require-model` 只检查路径；摘要根据任意登记模型的存在数量切换到 `smpl_skin_mesh`。实际构建/模拟/验收仍调用默认程序骨架，没有接入模型加载和蒙皮；`allow_procedural_skeleton: false` 也未执行。

CPU 负例：临时目录用内容为 `not a SMPL model` 的文本冒充登记文件，`plan.py --require-model` **退出 0**，摘要为 `smpl_skin_mesh`，刚体清单却仍为程序骨架。关闭降级选项后 `build.py --dry-run` 同样退出 0。

整改：统一解析并加载选定资产，验证内容、静止姿态、坐标、蒙皮；禁止降级时快速失败。表示类型由实际几何决定。**下载本身不会自动补齐真实 SMPL 链路。**

### R2 / P1：PD 验收目标重复转换单位

位置：`scripts/humans/verify.py:294`。

`dof_limits_rad()` 已提供弧度，取半并裁剪后又调用 `np.radians()`。膝关节中程本应 **75 度**，实际目标仅 **1.309 度**，小于 15 度验收容差。历史约 2 度误差不能证明中程跟踪合格。

整改：统一单位、检查目标幅度，加入驱动关闭应失败的负对照。

### R3 / P1：撞击判据混合不同体点、不同方向速度

位置：`src/sim2sense_fall/humans/events.py:389`。

近地/下降取全体点最低高度，速度却取任意体点的最大三维速度，并非同一个近地体点的向下速度。构造性 CPU 反例：最低点位于离地约 0.1 m，以 0.001 m/s 缓慢下降；另一点在 0.5 m 高处以 2 m/s 水平移动，骨盆保持 0.4 m。无体点触地，仍输出 `fall`，`first_impact_s=0.008333...`。此反例同时暴露初始已低位窗口未被排除。

整改：对同一体点计算向下速度与支撑面距离，区分接触代理和实际接触，补摆臂、慢下降、初始躺倒和主动躺下负例。现存两批接触力均缺失，不能称为测得的接触真值。

### R4 / P2：批次索引把频率写成步长

位置：`scripts/humans/simulate.py:352`、`:454`。

变量取 `1 / dt` 后写入 `physics_dt_s`。两个现存批次索引均为 **120.0 s**，单条 trial 和 NPZ 时间差则为 **1/120 s**，相差 14400 倍。下游依据索引做同步会出错。

整改：区分 Hz 与秒，并交叉校验 NPZ、单条记录、批次索引；重新生成或迁移旧索引。

### R5 / P2：联合键不保证人物隔离

位置：`src/sim2sense_fall/humans/export.py:179`；`docs/human-simulation.md:194`。

`subject=person1|sequence=clip1` 与 `subject=person1|sequence=clip2` 是两个组，可被分到两侧，不能保证文档宣称的同一人物不跨训练/测试。当前没有训练划分，属于后续泄漏风险。

整改：跨人物评测按带数据集命名空间的人物 ID 分组；若需同时隔离多个关联标识，采用连通分量，并检查最终集合交集。

### R6 / P2：受控动作超容差仍被计为可用

位置：`scripts/humans/simulate.py:405`、`:427`、`:773`。

可用性只依据标签非 `invalid`，跟踪是否达标仅写 metadata。已有辅助弯腰、坐下、原地走误差分别为 **71.883、69.124、35.383 度**，均超过 15 度容差却计为可用，不能据动作名称宣称对应日常动作验证完成。

整改：分别报告物理数值有效、参考跟踪合格与标签可信；受控日常动作执行跟踪门禁，控制失效/被动跌倒使用独立协议。

## 验证记录

| 检查 | 本轮结果 |
| --- | --- |
| compileall | 通过 |
| pytest | **134 passed / 8 skipped**，8 项依赖 pxr，本轮未用 bundled pytest 补跑 |
| 本机 Ruff | 通过 |
| `uv tool run ruff check .` | 退出 46，DBus `Process 2 is a kernel thread, refusing.` |
| plan、build dry-run、simulate dry-run | 通过，仅验证代理/参考链路 |
| 真实目录 `plan.py --require-model` | 退出 1，模型缺失 |
| 伪模型、禁止降级反例 | 错误地通过，见 R1 |
| Isaac 人体验收 | **退出 1**，历史全部 PASS 未复现 |

Isaac 命令：`DISPLAY= WAYLAND_DISPLAY= timeout 90s ~/isaacsim/python.sh scripts/humans/verify.py --out /tmp/stage7-review-isaac`。USD 结构、步长、极限与前 3 个姿态位置比较通过，随后 `ValueError: quaternion contains a non-finite value`，伴随 `Invalid PhysX transform`。日志另有 `NVML_ERROR_DRIVER_NOT_LOADED`、接触视图 `AttributeError: 'NoneType' object has no attribute 'check'`，尚不能确定发散根因。

完整日志保存在 `artifacts/humans/stage7-review-isaac.log`。历史 `human_verify.json` 保留，不能当成本轮通过证据。GPU/GUI 未验证。首次误加 `plan.py --dry-run`、`verify.py --headless` 被拒绝，属于审查调用错误，已改正，不列为代码缺陷。CPU 脚本重写了默认规划/参考标签产物，原有物理试验 NPZ 未重跑。

固定场景 `apartment_cn_two_bedroom`，seed=20260922，程序骨架 24 连杆/14 DOF/19 胶囊体。无新训练、数据划分、模型权重或检测性能指标。配置 SHA-256：`4d6c39b81123fe1c151ff0a032755dce5007c64121539c33ece53a04f01306e5`；场景 SHA-256：`38f062c3ca9d54e5ab8f2b2cc7428ce06cb85c7933d74530a3c3452958403b86`。

## 完整验收条件

1. 修复 R1–R6，增加能够揭露上述错误的负例。
2. 取得 SMPL 并实际接入骨架、体型、蒙皮与逐帧/USD 几何，不能只修改表示类型。
3. 取得少量 AMASS，记录文件哈希、子集/人物/序列、坐标和帧率，完成重定向和回放核验。
4. 重跑稳定可复现的 Isaac 验收，分别报告映射、驱动、数值有效性和接触能力。
5. 补齐有效的易混淆动作与跌倒样例，接触缺失时保持代理事件语义。
