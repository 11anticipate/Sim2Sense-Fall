# 把人体 Mesh 导入 Sionna RT

[文档索引](README.md) · [数据契约](data-contract.md) · [当前计划](../task_plan.md)。

## 当前公寓入口（2026-09-24 整理）

固定公寓 + 实际后推跌倒 → 复数 CIR 已实跑，证据是 [2026-09-23 独立复核](verification-2026-09-23.md)。
这是单场景、单类物理跌倒 smoke，不是 50 Hz 训练数据集，也未同步受人体撞动后的家具。
人体材料使用 [来源记录](human-em-material.md) 的 3.5 GHz 参数；参数模型不等于真人测量。

已有该物理试验产物时，从仓库根目录执行：

```bash
python3 scripts/sionna/import_fall_mesh.py --dry-run \
  --trial-json artifacts/review_20260923/final_physics/smpl_neutral_standing__stand_neutral__push_backward.trial.json
/home/gsh/.local/opt/sionna/bin/python scripts/sionna/import_fall_mesh.py --frames 12 \
  --trial-json artifacts/review_20260923/final_physics/smpl_neutral_standing__stand_neutral__push_backward.trial.json \
  --out artifacts/sionna/apartment_physics_review
```

输入不存在时，先按独立复核报告生成试验；`keyboard.py` 的 `recording.npz` 不是此入口要求的 trial 格式。

## 历史导入记录说明

以下环境/API 和首次 `floor_wall` 试验保留历史语境。旧功率 dB 值因丢弃虚部已撤回，
旧材料 εr=51、σ=2.16 S/m 已替换，不能引用下文旧输出为当前研究结果。


本轮（2026-09-23）把阶段 7 导出的逐帧人体网格真正接进 Sionna RT，并验证导入成功。
本文记录**本机环境**、**API 事实**（Sionna RT 2.1 的几何导入入口与直觉不符，写错会静默失败）
和**实测结果**。

## 本机环境（实测）

Sionna 不在系统 Python 里，也不在 Isaac Sim 的 Python 里，而在一个独立 venv：

```
/home/gsh/.local/opt/sionna/bin/python
```

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13（由 uv 的 cpython 创建） |
| sionna-rt | **2.1.0** |
| mitsuba | 3.9.1 |
| torch | 2.11.0+cu130 |
| numpy | 2.4.4 |
| drjit | 1.5.0 |

`pyvenv.cfg` 记录它由 `/home/gsh/IsaacLab/env_isaaclab/bin/python -m venv` 创建。
`mitsuba.variant()` 在 `import sionna.rt` **之后**才是 `cuda_ad_mono_polarized`；
只 import mitsuba 时是 `None`。

环境自检（已实跑，全通过）：

```bash
/home/gsh/.local/opt/sionna/bin/python ~/.local/opt/sionna/verify_sionna.py
```

```
[1] import sionna.rt OK | mitsuba 后端: cuda_ad_mono_polarized | torch: 2.11.0+cu130
[2] floor_wall 求解 OK | 路径 2 条, 时延(ns): [10.01, 14.15]
[3] CIR -> torch: (1, 8, 1, 8, 2, 1) torch.complex64 @ cuda:0, max|h| = 2.936e-04
[4] street_canyon (5 个物体) 求解 OK | 8 条有效路径 | 相对接收功率 -71.0 dB
[5] 街道 MIMO 信道 (1, 1, 1, 1, 8, 64) @ cuda:0
[6] 城市级场景 munich 可载入: 11 个物体
[7] headless GPU 渲染 OK
=== ALL CHECKS PASSED ===
```

## 怎么导入（Sionna RT 2.1 的 API 事实）

四条**实测**结论，写错前两条都会静默出错：

1. **几何不能用 `Scene.add()` 加。** `Scene.add` 只接受 `Transmitter` / `Receiver` /
   `RadioMaterialBase`，传 `SceneObject` 直接抛
   `Cannot add object of type SceneObject to the scene`。几何要走
   **`Scene.edit(add=...)`**（或对裸 Mitsuba 场景用
   `sionna.rt.scene_utils.extend_scene_with_mesh`）。
   我第一次写的探针建了对象**却没加进场景**，路径求解照常返回 —— 一个"成功"的空场景。
   所以 `place_mesh_in_scene()` 加完会**从 `scene.objects` 读回来核对**，不信返回值。
2. **`SceneObject` 可以直接吃内存里的 Mitsuba mesh**：`SceneObject(mi_mesh=...)`。
   不必把 145 帧写成 145 个 OBJ 文件。`mitsuba.Mesh` 通过 `mitsuba.traverse` 填：
   `vertex_positions`（展平 float）+ `faces`（展平 uint32）。
3. **`RadioMaterial` 是场景级资源，`Scene.edit(remove=...)` 不会把它摘掉。**
   每帧新建同名材质会在第二帧抛 `Name 'human_tissue' is already used by another item`。
   正确做法是注册一次、逐帧复用（见 `scene_radio_material`）。
4. **视口截图的返回值不是 awaitable。** `omni.kit.viewport.utility.capture_viewport_to_file`
   的 docstring 说返回 "future-like object"，但直接 `await` 会抛
   `object MultiAOVFileCapture can't be used in 'await' expression`。
   要等的是 delegate 的 **`wait_for_result()`**。

## 人体电磁材质是一个**建模假设**，不是测量

Sionna 的 ITU-R P.2040 表里只有 19 种**建筑材料**（concrete / brick / plasterboard /
glass / wood / 各种 ground / `vacuum` …），**没有人体组织**。所以人体必须显式给一个
`RadioMaterial`，并且必须把假设写进产物。

首次试验曾取 `ε_r = 51.0`、`σ = 2.16 S/m`、厚度 `0.02 m`（旧参数，已替换）。
`HumanMaterial.as_dict()` 会带上
`provenance: "modelling assumption, not measured"` 与 `source` 字符串，
现已补来源并将 3.5 GHz 参数改为 εr=51.4442299518、σ=2.5575182495 S/m，
详见 [人体电磁材料](human-em-material.md)；均匀材料与 0.02 m 厚度仍是代理假设。

另一个"不需要新假设"的选择是 `vacuum`（ε_r=1、σ=0），但那是错的：它让电磁意义上
**不存在人体**，会悄悄删掉这一阶段要测的交互本身。

## 历史首次导入输出（旧 dB 数值已撤回）

```bash
DISPLAY=:0 /home/gsh/.local/opt/sionna/bin/python scripts/sionna/import_fall_mesh.py \
    --sample fall_forward_reference --frames 12
```

```
[PASS] the sample is a licensed skin mesh -- smpl_skin_mesh, 6890 vertices
[PASS] the mesh sequence matches its manifest -- (145, 6890, 3), (13776, 3), 145 frames
[PASS] the body was re-centred on its own spawn -- 最大 |xy| = 0.952 m
[PASS] the scene solves without the body -- 2 paths, -64.51 dB
[PASS] the body was admitted to the scene -- human_f0000 with 6890 vertices
[PASS] the body changes the channel (positive control) -- 最大变化 +6.03 dB
[PASS] the channel changes as the body falls -- 全程散布 10.15 dB
[PASS] every frame produced a finite CIR of one shape -- (12, 8)
[PASS] the ChannelSample satisfies the data contract -- sionna_floor_wall__fall_forward_reference
```

**正向对照是关键**：同一场景、同一收发，分别求「有人体」与「无人体」。
导入若静默失败，信道不会有任何变化；只报告「解出 N 条路径」是发现不了的。
上方 **+6.03 dB / 10.15 dB 为已撤回的旧计算输出**，仅保留排错记录。
正确复数计算及新公寓对照见独立复核报告，不混合场景、材料或计算版本比较。

产物：`artifacts/sionna/fall_import/`

- `<sample>.cir.npz`：`timestamp_s` / `cir` / `frame_index`
- `<sample>.import.json`：环境、几何、材质、逐帧路径数与功率、基线、失败清单

## 首次 floor_wall 试验边界（历史）

- 场景是 Sionna **内置的 `floor_wall`**（地面 + 一面墙），**不是本项目的公寓**。
  把公寓转成 Sionna 场景是另一件事；混在一起会让失败无法区分是"人体导入不了"
  还是"场景转换错了"。因此产物的 JSON 里 `scene_note` 显式说明这一点。
- 人体被**重新居中到自己的出生点**（`body_recentring_offset_xy` 记录偏移量），
  因为采集的网格是在公寓世界坐标下的，而内置场景以原点为中心。
- 人体几何是 **`kinematic_replay`** 导出，传播是真的 Sionna RT ——
  也就是信道是**非物理轨迹**的函数。这一条写在 `fidelity_note` 里。
- 每个采样点只有 12 帧、`max_depth=4`。这是**导入验证**，不是数据集生成。

## CPU 侧覆盖

`tests/test_sionna_import.py`（11 项，不需要 Sionna 运行时）：

- 包在**没有 Sionna 的机器上必须能 import**（防止有人加模块级 `import mitsuba`）；
- 顶点/面数组契约：越界面索引、负索引、非整数面、非有限顶点、退化形状全部拒绝；
- 材质模型：默认**不是 vacuum**，携带 provenance，参数逐个校验；
- 缺运行时的报错必须**指名环境**，而不是裸 `ImportError`。

阵列契约用 `float32` / `uint32` 且 C 连续，因为 Mitsuba 的射线求交按索引读三角形，
越界索引在那里是**越界读**而不是渲染瑕疵。
