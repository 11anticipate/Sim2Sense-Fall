# 室内场景：构建、导出与查看

[文档索引](README.md) · [当前计划](../task_plan.md)。2026-09-24 整理：
公寓已用于人体 GUI 与物理 CIR smoke，第 6/8 节为 09-21 历史证据，不再代表 GPU 被阻塞。
键盘路线由用户控制；本轮不做主动避障，但保留人体对墙、家具和地板的接触验收。

本文件说明阶段 5 的室内三维场景资产：它由哪个配置文件描述、如何导出为 USD、如何用
Isaac Sim 查看 GUI 画面，以及实际跑通的验证结果。

对应实现：

| 文件 | 作用 |
|---|---|
| `configs/scenes/indoor_apartment.yaml` | 场景声明：房间尺寸、墙体归属、门窗开口、家具与材质覆盖 |
| `src/sim2sense_fall/scenes/materials.py` | 材质库：视觉 + 力学 + 电磁三套属性 |
| `src/sim2sense_fall/scenes/spec.py` | 场景 schema、YAML 加载与快速失败校验 |
| `src/sim2sense_fall/scenes/furniture.py` | 参数化家具配方（盒体/圆柱体拼装） |
| `src/sim2sense_fall/scenes/planner.py` | CPU 几何规划：墙体分段、开口、家具、清单 |
| `src/sim2sense_fall/scenes/usd.py` | USD 落地：几何、碰撞、刚体、材质、灯光 |
| `src/sim2sense_fall/scenes/numbers.py` | 有限数值与正尺寸校验 |
| `src/sim2sense_fall/scenes/geometry.py` | 世界变换、家具越界与穿墙检查 |
| `src/sim2sense_fall/scenes/verification.py` | 实际 USD 与清单逐图元/材质核对 |
| `src/sim2sense_fall/scenes/view.py` | 会话层相机与临时去顶显示 |
| `scripts/scenes/build.py` | 构建入口（dry-run / headless / GUI） |
| `scripts/scenes/view.py` | 打开已导出场景查看 GUI |
| `scripts/scenes/verify.py` | 场景与物理 smoke test |

## 1. 场景内容

配置的基础平面为 8.4 m × 7.0 m 的两室一厅一厨一卫加走廊住宅，Z 轴向上，单位 1 m；当前
`layout_scale_xy=2.20`，实际导出的水平 footprint 为 18.48 m × 15.40 m，墙高保持不变。
下方示意图使用未缩放的基础坐标，便于对照配置文件。

```text
y=7.0 ┌──────────────┬──────────────┬──────────────┐
      │   bathroom   │   kitchen    │  bedroom_2   │
y=4.2 ├──────────────┼──────────────┼──────────────┤
      │            corridor (1.2 m 宽)              │  ← 入户门在东端
y=3.0 ├───────────────────────────┬────────────────┤
      │         bedroom           │   living_room  │
y=0.0 └───────────────────────────┴────────────────┘
     x=0.0                      x=4.0            x=8.4
```

| 房间 | 起点 (x, y) | 尺寸 (m) | 地面材质 | 门 | 窗 | 家具件数 |
|---|---|---|---|---|---|---|
| `bedroom` | 0.0, 0.0 | 4.0 × 3.0 | `wood_floor` | 北墙 0.90 m | 南墙 1.60 m | 5 |
| `living_room` | 4.0, 0.0 | 4.4 × 3.0 | `wood_floor` | 北墙 1.00 m | 南墙 2.00 m | 7 |
| `corridor` | 0.0, 3.0 | 8.4 × 1.2 | `tile_floor` | 东墙 1.00 m（入户门，常闭） | — | 4 |
| `bathroom` | 0.0, 4.2 | 2.4 × 2.8 | `tile_floor` | 南墙 0.75 m（常闭） | 北墙 0.70 m | 6 |
| `kitchen` | 2.4, 4.2 | 3.0 × 2.8 | `tile_floor` | 南墙 0.80 m | 北墙 1.20 m | 5 |
| `bedroom_2` | 5.4, 4.2 | 3.0 × 2.8 | `wood_floor` | 南墙 0.80 m | 北墙 1.20 m | 6 |

墙高 2.7 m，墙厚 0.12 m，楼板厚 0.12 m，吊顶厚 0.15 m。共享墙只由一侧房间生成，
避免重复几何（约定见 YAML 头部注释）。

## 2. 材质属性（三类属性一次定义）

同一材质同时给出**渲染**（`UsdPreviewSurface`）、**力学**（PhysX 摩擦/恢复系数/密度）
和**电磁**（ITU-R P.2040-3 幂律系数，供 Sionna RT 使用）参数，避免渲染场景与无线电
场景悄悄描述成两套房子。

| 材质 | 用途 | 静/动摩擦 | 恢复 | 密度 kg/m³ | 厚 m | Sionna 材质 |
|---|---|---|---|---|---|---|
| `painted_drywall` | 涂料隔墙 | 0.60 / 0.50 | 0.02 | 700 | 0.12 | `itu_plasterboard` |
| `concrete_slab` | 结构楼板/地基 | 0.75 / 0.65 | 0.02 | 2300 | 0.30 | `itu_concrete` |
| `gypsum_ceiling` | 石膏吊顶 | 0.70 / 0.60 | 0.01 | 700 | 0.15 | `itu_ceiling_board` |
| `wood_floor` | 木地板 | 0.65 / 0.55 | 0.05 | 700 | 0.02 | `itu_floorboard` |
| `tile_floor` | 瓷砖（厨卫/走廊） | 0.45 / 0.35 | 0.03 | 2000 | 0.01 | `itu_marble`（近似） |
| `carpet` | 地毯 | 0.85 / 0.80 | 0.00 | 200 | 0.01 | `itu_floorboard`（近似） |
| `wood_furniture` | 实木框架 | 0.60 / 0.50 | 0.05 | 600 | 0.02 | `itu_wood` |
| `painted_mdf` | 烤漆 MDF 板件 | 0.55 / 0.45 | 0.05 | 700 | 0.02 | `itu_chipboard` |
| `upholstery` | 沙发/床垫织物 | 0.90 / 0.85 | 0.00 | 100 | 0.03 | `itu_ceiling_board`（近似） |
| `ceramic_sanitary` | 洁具陶瓷 | 0.50 / 0.42 | 0.05 | 2300 | 0.02 | `itu_marble`（近似） |
| `metal_fixture` | 五金/家电外壳 | 0.35 / 0.30 | 0.10 | 7800 | 0.005 | `itu_metal` |
| `glass_window` | 玻璃 | 0.20 / 0.18 | 0.20 | 2500 | 0.006 | `itu_glass` |
| `wood_door` | 门扇 | 0.55 / 0.45 | 0.05 | 600 | 0.04 | `itu_wood` |
| `mirror_glass` | 镜面 | 0.20 / 0.18 | 0.20 | 2500 | 0.004 | `itu_glass` |

电磁参数以 **幂律系数** 存储并按场景频率（默认 2.4 GHz）求值，换到 5 GHz 不会静默
复用旧数值。标注「近似」的条目表示 ITU-R P.2040 没有对应条目，使用了最接近的代理
材质——**这些代理需要在接入 Sionna 后与已安装版本的 `itu_*` 材质复核**。

## 3. 家具与物理属性

家具由盒体/圆柱体参数化拼装（不依赖任何在线素材库，保证可复现与可 diff）。
每件家具可配置：`size`、`material`、`physics`（`static`/`dynamic`）、`density_kg_m3`、
`elevation_m`、摩擦/恢复系数覆盖、`movable`、`tags`。

组合刚体：动态家具不会「按零件散架」。规划阶段把动态家具拆成
`furniture_body`（承载 `PhysicsRigidBodyAPI` + 单一质量）与其局部坐标系下的子碰撞体，
行为上等价于一个复合刚体。当前场景只有 `bedroom_2 / chair` 是动态物体，质量 13.56 kg。

静态家具（床、衣柜、沙发、洁具、橱柜等）作为世界坐标系下的静态碰撞体，带上各自的
物理材质——跌倒主体撞上它们时接触参数是有区别的。

## 4. 运行方式

### 4.1 CPU dry-run（不需要 Isaac Sim，CI 可跑）

```bash
cd /home/gsh/Documents/室内摔倒与机器人救援
python3 scripts/scenes/build.py --dry-run
```

校验场景文件并写出可复现清单 `artifacts/scenes/indoor_apartment.scene.json`。

### 4.2 Headless 导出 USD

```bash
cd /home/gsh/Documents/室内摔倒与机器人救援
~/isaacsim/python.sh scripts/scenes/build.py --headless
```

### 4.3 物理 smoke test

```bash
cd /home/gsh/Documents/室内摔倒与机器人救援
~/isaacsim/python.sh scripts/scenes/verify.py --settle-seconds 3
```

退出码为 0 才表示全部检查通过。

## 5. 用 Isaac Sim 查看 GUI 画面

> 前置条件：需要 GPU 与显示环境。本机为 NVIDIA RTX 4060 Laptop + Wayland（`DISPLAY=:0`），
> 早期版本曾在本机运行 A / B / C；目录迁移后的命令已更新，本轮未复验 GPU/GUI 画面。

### 方案 A（推荐）— 一条命令打开已导出场景并启用物理

```bash
cd /home/gsh/Documents/室内摔倒与机器人救援
~/isaacsim/python.sh scripts/scenes/view.py --activate-physics --settle-seconds 2
```

- `--activate-physics` 是必要的：独立运行时默认不会把 PhysX 挂到 USD stage 上，
  不加这个参数画面是「静止场景」，重力不会作用。
- `--settle-seconds 2` 表示打开后先步进 2 秒物理，让动态家具落稳。
- 关闭窗口或按 `Ctrl-C` 退出。
- 想在无人值守环境下验证这条路径，加 `--exit-after-seconds 20` 让它自动退出。

### 方案 B — 先用官方启动器启动 GUI，再手动打开

```bash
cd ~/isaacsim && ./isaac-sim.sh
```

启动后在 GUI 中 `File > Open`，选择：

```text
/home/gsh/Documents/室内摔倒与机器人救援/artifacts/scenes/indoor_apartment.usda
```

也可以直接把该 `.usda` 文件拖进 Isaac Sim 窗口。`/World` 是默认 prim，`Content` 面板里
可按 `rooms/<房间名>/walls|furniture|openings` 逐层展开检查。

### 方案 C — 重新构建并保持 GUI 打开

```bash
cd /home/gsh/Documents/室内摔倒与机器人救援
~/isaacsim/python.sh scripts/scenes/build.py --gui --settle-seconds 2
```

会重新规划、导出 `.usda`，然后打开 GUI 并步进物理。加 `--exit-after-seconds 20` 可
自动退出。

### 无显示环境（SSH / 远程）

Isaac Sim 提供流式启动器，本仓库没有在本机验证过这条路径，仅作为参考：

```bash
~/isaacsim/isaac-sim.streaming.sh
```

### 场景里能看到什么

- `/World/rooms/<room>/walls/*`：按门窗开口切分后的墙段（含门楣、窗下墙）。
- `/World/rooms/<room>/openings/*`：门扇、玻璃、窗框。
- `/World/rooms/<room>/furniture/<id>/*`：家具零件；动态家具下面是刚体 Xform。
- `/World/rooms/<room>/floor`、`ceiling`：楼板与吊顶。
- `/World/Materials/*`：视觉材质，带 `sim2sense:em_*` 电磁属性。
- `/World/PhysicsMaterials/*`：按摩擦/恢复系数去重后的接触材质。
- `/World/lighting/*`：穹顶光、太阳光与每个房间的吸顶补光。

每个几何 prim 上都有 `sim2sense:roomId`、`sim2sense:category`、`sim2sense:semantic`、
`sim2sense:physicsMode`、`sim2sense:massKg`、`sim2sense:movable`、`sim2sense:tags`
自定义属性，便于后续按语义查询和筛选。

注意：Isaac Sim 打开场景时会自行加上 `/Render`（渲染设置）和 `/OmniverseKit_Persp` 等
视口相机，这些是运行时的东西，不在导出的文件里。想看文件里到底有什么，直接用
`grep` 或 `Usd.Stage.Open` 读 `.usda` 本身（344 个 prim）。

## 6. 验证结果（2026-09-21 本机实际运行）

构建输入：`configs/scenes/indoor_apartment.yaml`，种子 `20260921`，频率 2.4 GHz。

| 指标 | 数值 |
|---|---|
| 地面面积 | 58.80 m²（6 个房间） |
| 墙面面积 | 77.19 m² |
| 规划图元 | 232（231 几何 + 1 刚体根） |
| 墙体段 | 46 |
| 门窗构件 | 27 |
| 家具/灯具零件 | 145 |
| 材质 | 14 视觉材质，12 接触材质 |
| 灯光 | 8（穹顶 + 太阳 + 6 个房间灯） |
| 刚体 | 1（`bedroom_2/chair`，13.56 kg） |
| USD 文件 | `indoor_apartment.usda`，335 KB，344 个 prim |

`scripts/scenes/verify.py --settle-seconds 3` 的 14 项检查全部 PASS：

```text
[PASS] scene opens -- 344 prims
[PASS] exported file has the expected primitive count -- 344 vs 344
[PASS] Z-up with metres as the unit -- up=Z m/unit=1.0
[PASS] colliders cover the geometry -- 231 colliders / 231 shapes
[PASS] rigid bodies authored
[PASS] authored geometry matches the plan -- 231 vs 231
[PASS] authored rigid bodies match the plan -- 1 vs 1
[PASS] floor footprint matches the plan -- 8.40 m x 7.00 m
[PASS] physics scene activated -- /World/PhysicsScene on physx
[PASS] dynamic bodies reported -- {"/World/rooms/bedroom_2/furniture/chair": [6.9, 5.8, 0.0]}
[PASS] .../chair stays put at rest -- drift 0.0000 m
[PASS] .../chair falls back under gravity -- fell 0.2500 m from 0.2500 m to 0.0000 m
[PASS] .../chair comes to rest where it started -- residual 0.0000 m
[PASS] verification left the exported scene untouched
verification: PASSED
```

两点值得说明：

- 「抬高再落下」是**正向对照**。只测「漂移为 0」无法区分「场景稳定」与「物理根本没跑」，
  抬高 0.25 m 后必须落回原位，才能证明重力、碰撞和接触材质确实生效。
- 验证跑在 `.usda` 的**临时副本**上，并比对原文件 SHA-256。Isaac Sim 的 stage-open 钩子
  会往内存图层里加 `/Render` 和视口相机（`/OmniverseKit_*`），如果直接验证原文件，
  报告里的图元数会虚高 7 个，且可能在保存时被写回文件。

CPU 侧：`python -m pytest -q` → 27 passed；`python -m compileall src tests scripts` 通过。

## 7. 已知限制与待办

1. **电磁材质需要复核**：`tile_floor`、`ceramic_sanitary`、`carpet`、`upholstery`、
   `mirror_glass` 使用了 ITU-R P.2040 中没有精确对应项的代理材质，接入 Sionna RT 后
   需与安装版本的 `itu_*` 材质数值对齐。
2. **未做领域随机化**：当前是单一固定布局，配置里的 `seed` 已写入清单但还没有驱动任何
   随机化。DGSense 路线要求随机化房间布局、材质、人体参数与链路，属于后续工作。
3. **人体质量仍需整改**：基础公寓不内嵌人体，人体由独立入口加入；阶段 7 已有键盘/物理试验，
   阶段 8 已有公寓 CIR smoke，滑步/皮肤穿透仍见 [键盘指南](keyboard-control.md)。
4. **墙体归属是手工约定**：共享墙由哪一侧生成写在 YAML 注释里，尚无自动校验；
   `tests/scenes/test_scenes.py::test_shared_walls_are_not_built_twice` 只保证跨房间不重叠。
5. **动态物体只有一把椅子**：足以验证复合刚体路径，但不足以支撑拖拽、碰撞链等更复杂
   的接触场景。
6. **玻璃与渲染外观**：玻璃用了 `opacity = 0.35`，实际透明度取决于渲染器设置；材质
   本身没有贴图，属于刻意的「几何 + 参数」风格。

## 8. 本轮验收与内部查看（2026-09-21）

**当前状态：验收发现的 R1–R6 已修复并通过 CPU/实际 USD/物理复验。**
详见 [`indoor-scene-remediation.md`](indoor-scene-remediation.md)；当时 GPU 视觉尚未验证，
后续人体实时截图与 CIR smoke 分别见 [键盘指南](keyboard-control.md) 和 [独立复核](verification-2026-09-23.md)。
以上第 6 节保留首次构建的历史验证记录。

`scripts/scenes/view.py` 现在默认使用去顶正交俯视相机，打开后可以检查六个房间内部：

```bash
~/isaacsim/python.sh scripts/scenes/view.py --view top
~/isaacsim/python.sh scripts/scenes/view.py --view roofless
~/isaacsim/python.sh scripts/scenes/view.py --view exterior
```

- `top`：隐藏吊顶和吸顶灯具几何，按地板范围自动构图；这是默认模式。
- `roofless`：相同的隐藏规则，采用斜上方透视相机。
- `exterior`：显示完整屋顶和灯具，查看建筑外壳。
- 相机与可见性仅写入 USD session layer；碰撞体仍存在，原始 USD 文件不变。
- 当前只修改独立查看入口；`scripts/scenes/build.py --gui` 和官方 GUI 的直接 File > Open
  仍显示原始封闭资产。建议先导出，再用上述查看入口。
- 本轮已验证 USD 可见性切换、相机覆盖四角、源图层不变和碰撞体保留；由于当前环境
  无 GPU/显示，**新查看入口的实际 GUI/RTX 画面尚未验收**。
- `artifacts/acceptance/usd_before_after.png` 与 `usd_layout_findings.png` 是从实际 USD 世界几何
  生成的 CPU 预览，不是 Isaac RTX 截图，也不用于判断光照、阴影、玻璃透明度。

## 9. 后续场景修改的检查规则

配置现支持 `foundation_thickness_m`（默认 0.30 m）。地基位于最厚地板下，较薄地板通过找平层连续支撑；所有行走面为 z=0。家具与墙的交集、房间外廓、非有限数值、负派生尺寸和路径碰撞均在 CPU 规划阶段拒绝。

验证器现按清单逐图元及材质比较，容差可通过 `--position-tolerance` 与 `--drop-tolerance` 指定。`--dry-run` 仅检查清单；`--static-only` 明确跳过动力学，其结果不能当作动态测试通过。
