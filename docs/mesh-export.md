# 逐帧人体网格导出契约

[文档索引](README.md) · [人体指南](human-simulation.md)。本契约描述 `simulate.py` 输出；键盘记录格式单列在末尾。

人体几何使用可供 Sionna RT 消费的固定拓扑三角网格。实现位于 [`mesh_sequence.py`](../src/sim2sense_fall/humans/mesh_sequence.py)，已接入 `scripts/humans/simulate.py` 的 CPU dry-run 与 Isaac 记录路径。
dry-run 使用参考正运动学姿态，真实物理试验使用实际连杆姿态，二者不能混称为物理真值。

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

SMPL neutral 与真实 AMASS 已用于实际记录，详见 [AMASS 审计](amass-physics-audit-2026-09-24.md)。
历史 scripted dry-run 的 `stand_neutral.mesh.npz` 为 `(121, 6890, 3)`，该形状只对应其 1 s 参考片段。
数据来源以每个产物的 provenance 为准，不能从文件名推断 scripted、AMASS 或实际物理。

## 键盘记录

`keyboard.py` 输出 `recording.npz`：`time_s`、`mesh_vertices_xyz`、`mesh_faces`，采样率由显示更新决定，
不含 `time_channel_s` 或重采样后的信道网格。它不是可直接替代 `.trial.json` + 试验 NPZ 的无线输入。
`control.npz` 单独保存物理时间上的目标/实际状态/外力，`contacts.json` 保存接触，`report.json` 保存来源和门槛。
记录采用有界窗口，必须读保留时间范围；完整格式与边界见 [键盘指南](keyboard-control.md)。
