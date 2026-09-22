# 逐帧人体网格导出契约

本轮验收把人体几何从“按物理姿态采样的点云”补成了可供 NVIDIA Sionna RT 消费的固定拓扑三角网格。实现位于 [`mesh_sequence.py`](../src/sim2sense_fall/humans/mesh_sequence.py)，并已接入 `scripts/humans/simulate.py` 的 CPU dry-run 与 Isaac 记录路径。

每个物理帧写入：

- `mesh_vertices_xyz`：`(N, V, 3)`，世界坐标、米制、Z-up，列顺序为 `(x, y, z)`；
- `mesh_faces`：`(F, 3)` 的整数索引，所有帧共享同一拓扑；
- `channel_mesh_vertices_xyz`：在 `time_channel_s` 上对顶点坐标做线性插值，形状为 `(Nc, V, 3)`；
- JSON 元数据：`mesh_representation`、顶点/面数量、拓扑 SHA-256、坐标系、单位和固定拓扑标志。

当前有两条表示路径：

1. `smpl_skin_mesh`：从本机已取得并加载的 SMPL v1.1.0 neutral pickle 读取 6890 个顶点、13776 个三角面和 24 关节蒙皮权重；每个物理帧用实际 link pose 做线性蒙皮。
2. `capsule_proxy_mesh`：没有可用 SMPL 时，由物理胶囊生成闭合的确定性三角网格。它只用于 CPU/Sionna 接口 smoke test，不能当作真实人体网格。

两条路径都拒绝非有限顶点、非递增时间、非法面索引和跨帧顶点数变化。代理网格的 faces 和 SMPL faces 都不会随时间变化；变化的只有世界坐标顶点。

验收命令：

```bash
python3 -m compileall -q src tests scripts
python3 -m pytest -q
python3 scripts/humans/plan.py --out /tmp/sim2sense-plan
python3 scripts/humans/simulate.py --dry-run --out /tmp/sim2sense-dry
```

当前 dry-run 已实际加载 SMPL neutral，并为每个参考动作写出 `<motion>.mesh.npz`：例如 `stand_neutral.mesh.npz` 为 `(121, 6890, 3)` 顶点和 `(13776, 3)` 面索引，全部坐标有限且时间从 0 秒递增到 1 秒。Isaac 运行仍需单独复验物理数值稳定性；AMASS 动作序列尚未取得，因此 dry-run 动作仍是 scripted。
