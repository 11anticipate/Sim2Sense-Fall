# 室内 USD 与代码批判性验收

> 历史问题报告（2026-09-21），不作为现行待办。现行 [场景指南](indoor-scene.md)、[计划](../task_plan.md) 与 [文档索引](README.md)。

> 目录迁移说明（2026-09-21）：场景实现现位于 `src/sim2sense_fall/scenes/`，入口位于 `scripts/scenes/`。下文历史缺陷的路径、行号和哈希保留为当时证据；当前命令见 [场景指南](indoor-scene.md)。

**整改更新：R1–R6 已修复并复验，见 [整改记录](indoor-scene-remediation.md)。下文保留修复前的验收事实、旧源码行号和旧资产哈希，不能当作当前缺陷状态。**

日期：2026-09-21。结论：**需整改，不建议将现状标记为研究仿真场景整体验收通过。**

当前资产能重复构建、包含室内家具、可进行单椅刚体仿真。问题集中在默认不可视、几何交叠、输入校验和验收脚本的失败状态传播。本轮仅补齐独立查看入口；以下标为“待修”的缺陷仍存在于场景生成/验证代码中。

## 1. “看不到内部”的直接原因及处理

实际 USD 有六块不透明吊顶，覆盖所有房间，且没有任何 Camera prim。原查看脚本只打开文件，没有隐藏屋顶或设置室内相机；从默认外部视角看见封闭外壳符合当前资产定义。家具没有丢失，实际包含 127 个家具几何、18 个灯具几何。

新增 `src/sim2sense_fall/scene_view.py`，独立查看脚本默认使用去顶俯视，可用：

```bash
~/isaacsim/python.sh scripts/scenes/view.py --view top
~/isaacsim/python.sh scripts/scenes/view.py --view roofless
~/isaacsim/python.sh scripts/scenes/view.py --view exterior
```

相机和吊顶/灯具可见性仅写在临时 session layer；隐藏渲染表面不会删除碰撞体。原始 USD 保持封闭完整，避免把观察用去顶场景误作为后续无线传播输入。`build_indoor_scene.py --gui` 和直接 File > Open 仍打开原始封闭资产。

- [原始外壳与去顶内部对比](../artifacts/acceptance/usd_before_after.png)
- [室内平面及问题标注](../artifacts/acceptance/usd_layout_findings.png)

这两张图来自实际 USD 的世界变换、Cube/Cylinder 几何与 displayColor。三维图使用 CPU 深度缓冲绘制，平面图隐藏吊顶/灯具并略去底部高于 1.2 m 的墙/门窗构件。它们是几何诊断预览，**不是 Isaac RTX 截图**，不能判断真实灯光、玻璃透明度、阴影和实时帧率。

## 2. 按优先级排列的待修发现

### R1 · P1 · 验证失败仍返回成功退出码

位置：`scripts/verify_indoor_scene.py:267–274`，尤其 `app.close()` 在最终返回之前调用。

复现：

```bash
~/isaacsim/python.sh scripts/scenes/verify.py \
  --manifest artifacts/acceptance/intentionally_missing.scene.json
```

本轮实际输出 `[FAIL] plan manifest available` 和 `verification: FAILED`，但命令退出码为 **0**。证据：`artifacts/acceptance/isaac_failure_exit_probe.log`。

本机 Isaac `SimulationApp` 默认 `fast_shutdown=True`，`close()` 默认使用 `exit_code=0` 终止解释器，后面的 `return exit_code or ...` 无法生效。自动化验收因此会把失败当成功。构建入口也采用在 `finally` 中 `app.close()`、之后再返回错误码的相同模式，应一起修。

修复方向：先计算最终状态，再按本机支持的接口调用 `app.close(exit_code=final_status)`；以子进程方式分别测试成功、故意失败和启动异常，不能只断言打印的 PASS/FAIL。

### R2 · P1 · 地基与六块地板体积交叠，顶面重合

位置：`src/sim2sense_fall/scene_planner.py:664–676`，以及 `plan_scene` 的地板生成。

实际几何：地基 z=[-0.30, 0]；所有地板 z=[-0.12, 0]。地板整体嵌在混凝土地基中，二者顶面重合；浮点尺度误差仅约 7e-9 m，不构成设计间隙。

这会产生共面表面渲染争用风险，并使同一落地点存在地板与混凝土两种接触材质，妨碍解释湿滑瓷砖与木地板的差异。后续 RT 转换也需要避免对重合表面/重叠体积赋予含糊介质。**几何交叠已证实；具体摩擦响应偏差和 RT 偏差尚未量化。** 单椅垂直落地不能证明摩擦正确。

修复方向：根据地板底面定义地基上表面，建立明确的层叠结构；新增不交叠检查与已知初速度滑动对照。多种地板厚度时要定义支撑层策略，不能只固定下移一个适用于当前房间的数值。

### R3 · P2 · 三块地毯穿墙，走廊地毯跨房间且越出外墙

位置：`configs/scenes/indoor_apartment.yaml:73`、`:135`、`:175`；默认 rug 尺寸为 2 × 3 × 0.012 m。

| 对象 | 实际范围/现象 | 影响 |
|---|---|---|
| 走廊地毯 | x=[6.5,8.5]，y=[2.1,5.1]；走廊仅 x=[0,8.4]，y=[3,4.2] | 跨入客厅、次卧，越出东外墙 0.1 m；房间语义与实际材料分布不一致 |
| 主卧地毯 | y=[0.05,3.05] | 穿过南北墙，北端越房间边界 0.05 m |
| 客厅地毯 | y=[0,3]，覆盖房间完整深度 | 与南北墙体交叠，未扣除内缩墙厚 |

目前 RoomSpec 仅校验房间彼此重叠和家具 id；没有检查家具在房间内的旋转后实际边界，也没有检验家具与墙的实体交集。测试只针对墙与墙、墙与洞口，捕获不了上述情况。

修复方向：为每块地毯明确合理尺寸与位置，并检查实例化后的家具几何是否越界/穿墙。家具放在地毯上属于可允许接触，不能把所有包围盒交集一概判错。

### R4 · P2 · 冰箱实体穿入厨房北墙

位置：`configs/scenes/indoor_apartment.yaml:287–291`。

冰箱置于房间局部 y=2.30 m，默认深度 0.70 m，部分前面板/把手超出名义深度；厨房内缩北墙占 y=[6.88,7.00]。实际 USD 中柜体和把手分别穿入墙内约 0.01 m、0.03 m。证据见 `geometry_findings.txt`。

修复方向：按完整家具零件的变换后外廓调整位置及朝向，并把把手、面板凸出计入净空。当前碰撞体为静态物体，物理步进不会自动推开重叠的冰箱和墙。

### R5 · P2 · 不同 id 规范化后会覆盖同一个 USD 路径

位置：`src/sim2sense_fall/scene_planner.py:71–77`，以及家具/房间路径拼接；`validate_prim_paths()` 只验证合法字符。

在同一房间添加 id 为 `same-name` 和 `same_name` 的两件家具，配置检查接受原始不同 id，但 `_safe()` 都生成 `same_name`。实测计划有 244 个图元，却只有 238 个唯一路径。USD `Define` 会复用已有路径，使两件家具合并或后写覆盖，清单和场景可能不一致。

修复方向：在完整计划输出时检查规范化后的路径唯一性并失败，或采用可逆且不碰撞的编码。只给当前默认配置写一个唯一性测试，不能保护其他配置。

### R6 · P2 · 数值与派生尺寸校验不完整

位置：`src/sim2sense_fall/scene_spec.py:52–80`、`FurnitureSpec.__post_init__` 与 `ScenePrim.__post_init__`，以及家具配方。

本轮构造的反例，均使用原始配置的副本：

- 家具位置设为 NaN，schema 和 planner 均接受，JSON 清单直接含 `NaN`，不再是严格 JSON，也不能作为有效空间真值。
- 床的 size=[0.05,0.05,0.05] 通过“正数”检查，但配方生成床垫 size=(-0.03,-0.11,0.22)，枕头宽度 -0.115 m。非法派生尺寸没有被阻止。

修复方向：所有空间/物理数值执行有限性检查；配方声明最小尺寸约束或按比例建模；在 ScenePrim 输出处再统一验证几何尺寸正且有限，序列化时拒绝 NaN/Infinity。

## 3. 研究用途与机器人救援的验收边界

以下属于尚未完成的能力，不作为“新增回归缺陷”计数：

- 入户门与卫生间门是常闭静态碰撞体，没有铰链或开门控制。机器人从室外进入、进入卫生间的路径不能按“已有门洞”认定可通行。
- 没有机器人尺寸、膨胀半径、转弯空间、人体跌倒占地和救援接近位置的通行性验证。仅凭俯视图不能给出“机器人可达全部房间”的结论。
- 家具是参数化盒体/圆柱体，适合作为粗几何基线，不是写实室内资产；人体、骨架、跌倒轨迹和 CSI/CIR 尚未接入。
- seed=20260921 目前只记录在清单中，没有驱动布局随机化；不存在训练/验证/测试场景域划分。
- `verify_indoor_scene.py` 把 344 prim、231 几何、8.4 × 7.0 m 写成固定预期，并主要比较数量。合法场景变化会被拒绝；同数量的错位家具/错误材质可能通过。后续应按清单逐路径验证变换、尺寸、语义、材质和碰撞设置。
- 电磁材质是映射元数据，Sionna 的实际场景导入、材质匹配、传播结果均未验证。

## 4. 本轮验证与可复现信息

| 检查 | 本轮结果 | 能证明什么 |
|---|---|---|
| 原始测试 | 27 passed | 现有 schema/planner 基线 |
| 修改后系统 Python 测试 | 28 passed，1 skipped | 新参数边界检查通过；系统 Python 无 pxr |
| bundled USD 下查看模块测试 | 2 passed | 视图切换、源图层不变、碰撞保留、相机覆盖地板四角 |
| `python3 -m compileall src tests scripts` | 通过 | 语法可编译 |
| `uv tool run ruff check .` | snap 权限错误，未完成 | 该启动路径不可用 |
| 已安装的 `/home/gsh/.local/bin/ruff check .` | 通过 | 同一项目 lint 检查通过 |
| CPU dry-run | 通过，另存 acceptance/cpu | 232 规划图元 / 231 几何 / 6 房间 |
| Isaac headless 重新构建 | 成功 | 新生成 USD 与原 USD 的 SHA-256 完全相同 |
| Isaac 物理 smoke test | 14 项日志 PASS，CPU 回退 | 椅子从抬高 0.25 m 回落；原文件不变 |
| 故意缺少清单的负例 | 日志 FAILED，退出 0 | 证实 R1，现有退出码不可作为验收门禁 |
| 新查看模式 Isaac 无界面集成 | 三种模式通过，231 碰撞体保留、源文件哈希不变 | 视口正确选中检查相机，吊顶可见性符合模式 |
| GPU 渲染/GUI 视觉 | 未验证 | NVML/GPU/显示环境不可用 |

物理指标定义：初始静止位置最大轴向漂移 ≤0.02 m；抬高 0.25 m 后下降至少 0.23 m；最终位置最大轴向残差 ≤0.02 m；每段仿真 3 s。当前椅子测得漂移/残差均显示为 0.0000 m，落差 0.2500 m。它不是学习模型性能实验。

版本与输入：

- 源代码基线：`a3f3e391c4c9aebb807da9cfd48d54bd0e7e9eab`，本轮在此基础上新增查看功能和测试。
- 场景：`apartment_cn_two_bedroom`；seed=20260921；频率 2.4 GHz；固定单场景，无数据集划分、无学习模型版本。
- OpenUSD：0.25.11；日志中的 Isaac-Sim Python 为 6.0；Isaac 使用本机 `/home/gsh/isaacsim` 安装，运行时日志保存在 acceptance 目录。
- USD SHA-256：`2e5fb8ec7523e972b5ef3ffb7f545a1078ffbf43480fc208e5f801c783cf86cb`。
- 配置 SHA-256：`89bea5a9ba601754fd9d2e8aa3ee5eceb8cead0f6d8d30dd3a7e7b09dc5265c4`。
- 既有 manifest SHA-256：`349c4b137f08ba2d29387a63da41704689237d51cefdcc70bc113ab3ea439398`。

证据文件均在 `artifacts/acceptance/`，沿用生成产物不入 Git 的规则：`usd_geometry.json`、`geometry_findings.txt`、`config_probes.txt`、`isaac_verify.log`、`isaac_failure_exit_probe.log`、`isaac_build.log` 和两张预览图。

代码结构上，CPU 规划与 USD 落地分离、不可变 dataclass、动态家具使用单一复合刚体、物理验证包含自由落体正向对照，这些设计值得保留。当前短板是验收断言的范围和错误传播，不能靠继续增加固定图元数量断言解决。

## 5. 整改顺序与通过条件

1. 先修 R1，保证故意失败一定返回非零退出码。
2. 修正地板/地基层叠、地毯尺寸和冰箱位置，重新导出，并自动检查实际几何交集。
3. 补齐 id 唯一性、有限数值和配方尺寸检查，使用本报告反例做回归。
4. 在有 GPU 与显示的环境验收去顶俯视、斜视、室内视角，以及地板是否闪烁、玻璃和灯光效果。
5. 根据机器人与人体规格明确门状态及通行标准，再执行可达性与接触/摩擦实验。

本轮没有改动既有场景布局或原始 USD，也没有提交/推送。查看问题已有代码改进；R1–R6 仍需整改。
