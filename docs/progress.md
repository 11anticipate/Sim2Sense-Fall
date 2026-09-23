# 阶段进度

## 2026-09-22 — Transitions 录屏动作表现复核

- 复核用户提供的 `/home/gsh/Videos/Screencasts/Screencast from 2026-09-22 23-30-10.mp4`（约 7.05 s）：画面从站立开始，中段下坐并后仰、腿部抬起，后段恢复站立。
- 该表现与 Transitions `amass__sit_stand_poses` 的坐下/起立参考动作一致；它是动作预览证据，不是跌倒标签，也不代表已经完成 PhysX 接触响应验收。
- `view_amass.py` 当前每帧直接写入 DOF 和根部位姿并清零速度，属于运动学回放。墙体和地面碰撞代理仍写入 stage，但接触反作用会被下一帧传送覆盖；要验收碰撞应使用独立物理入口或后续 physics replay mode。
- `DISPLAY= WAYLAND_DISPLAY= ~/isaacsim/python.sh scripts/humans/verify.py` 的最新实际结果：19 个胶囊碰撞代理恢复，重力回落 `1.3051 m`，最低人体点 `-0.0000 m`，地面碰撞已通过。墙体代理已写入场景，但当前验收未覆盖墙体接触响应，不能把它概括为墙/地面碰撞均已完成。

## 2026-09-22 — AMASS 查看器启动路径与碰撞边界核对

- 指定 `--motion amass__sit_stand_poses` 时，查看器现在按动作 ID 直接定位单个 `.npz`，不再先解析 Transitions 的全部 110 个序列；未指定动作时仍保持完整库扫描和确定性排序。
- 实测隔离会话从进程启动到 headless 完成约 18 s，其中 Isaac Sim/Kit 扩展初始化约 13.4 s；日志显示当前隔离会话没有可用 CUDA/NVML，RTX 初始化失败后回退 CPU，这部分不能用仓库代码进一步压缩。首次启动还会建立 shader/材质缓存。
- `view_amass.py` 的 19 个胶囊代理确实带有 `UsdPhysics.CollisionAPI`，SMPL `/Skin` 明确是 visual-only。该入口是运动学回放：每帧写入 DOF/root 位姿并清零速度，所以墙/地面的 PhysX 反作用会被下一帧传送覆盖，不能把它当作碰撞响应实验。真实碰撞正向对照仍由 `scripts/humans/verify.py` 的重力回落和地面穿透检查承担。
- 重新运行 `DISPLAY= WAYLAND_DISPLAY= ~/isaacsim/python.sh scripts/humans/verify.py`：实际 Isaac CPU PhysX 通过，19 个碰撞代理恢复，抬高后骨盆下降 1.3051 m，最终最低人体点 `-0.0000 m`；地面碰撞链路有效，墙体代理已写入但本次检查未覆盖墙体接触响应。日志仍记录当前隔离环境的 CUDA fallback。

## 2026-09-22 — Transitions sit-stand 预览根位姿修复

- 用户复现 `view_amass.py --motion amass__sit_stand_poses` 时，人物曾因直接使用 AMASS 序列的全局 `trans`/root rotation 而出现在房间外并横向倾倒；蒙皮又没有使用同一根旋转，导致视觉皮肤与 PhysX 胶囊代理错位、穿墙。
- 新增 `normalize_root_motion()`：以首帧为局部锚点，使用 `t[k] - t[0]` 和 `R[k] @ R[0].T`；查看器的 `forward_kinematics()`、SMPL skin 和 `HumanRuntime.set_root_pose()` 现在消费同一帧归一化根位姿。
- 根旋转连续性检查改用旋转测地距离，避免跨越 180° 时主值轴角向量被误判为 359° 跳变。
- CPU 回归：`201 passed / 8 skipped`，Ruff 和 compileall 通过。真实 Transitions `amass__sit_stand_poses`：980 帧、120 Hz、8.158 s；Isaac headless CPU PhysX 播放和 USD/JSON 导出通过，`root motion anchored` 检查通过。
- 当前隔离执行仍记录 `NVML_ERROR_DRIVER_NOT_LOADED` / `cuInit failed (100)`，所以这次 headless 结果是 CPU PhysX fallback；宿主机 GUI 需要用用户已恢复的 NVIDIA 会话重新运行命令观察画面。

## 2026-09-22 — 人体 GUI 姿态、碰撞代理与取景修复

- 用户截图中的三项视觉问题已定位并修复：SMPL neutral 的水平 rest pose 造成手臂穿墙；胶囊碰撞体被错误显示造成皮肤与内部结构分离；整屋包围盒取景使人体显得过小。
- `configs/humans/human_smpl_neutral.yaml` 新增配置驱动的 `visualization.default_pose_rad`：左右肩分别为 `+pi/2`、`-pi/2`，左右肘为 `-0.20` rad；左肩限位扩展到 110°。CPU 正运动学实测左右上臂向下约 0.27 m，SMPL 6890 顶点姿态全部有限。
- `scripts/humans/build.py` 用上述 DOF 姿态生成 posed SMPL 网格，并在 GUI 启动时对同一组 PhysX DOF 写入 position/target；`--view human` 按 `/World/Human` 边界取景并隐藏屋顶，现为默认视角。`roofless`、`top`、`exterior` 仍可选。
- `usd_human.py` 将 19 个胶囊设置为不可见，同时保留 `UsdPhysics.CollisionAPI`；headless USD 检查确认 `19/19` 隐藏、`19/19` 仍是碰撞体，SMPL 网格为 6890 顶点 / 13776 三角面。
- 验证：`python3 -m pytest -q` 为 `199 passed / 8 skipped`；compileall 与 Ruff 通过；`DISPLAY= WAYLAND_DISPLAY= timeout 180 ~/isaacsim/python.sh scripts/humans/build.py --headless --name human_display_repair` 通过。Isaac 日志仍记录隔离环境不可见宿主机 GPU（NVML/CUDA CPU fallback），不把它写成 GPU 渲染通过。
- 复核补充：新增显示姿态回归后完整测试为 `200 passed / 8 skipped`，本机 `/home/gsh/.local/bin/ruff check .` 通过；`uv tool run ruff check .` 仍被运行环境的 DBus transient scope 错误阻断（`Process 2 is a kernel thread, refusing`），不影响已完成的本机 Ruff 检查。
- 用户在宿主机 GUI 首次运行时报告 `AssertionError: Instance's physics tensor entity is not valid. Play the simulation/timeline to re-initialize it`。根因是 GUI 分支在 `runtime.play()` 后立即写 DOF，Kit 尚未完成 tensor articulation 初始化；已补上 4 次 `app.update()`，与 `verify.py`/`simulate.py` 的已验证顺序一致。CPU 回归与 headless 构建复验继续通过；当前隔离环境没有显示会话，GUI 需在宿主机重新运行确认画面。

## 2026-09-22 — 人体 GUI 与真实蒙皮显示修复

- 复现了用户截图：`scripts/humans/build.py --gui` 未接入场景查看器的 roofless 相机，因此完整屋顶遮挡室内；同时该入口虽然加载 SMPL，却没有将网格传入 `build_human_stage()`，生成的 USD 只含胶囊碰撞体。
- 已修复 `scripts/humans/build.py`：默认 `--view roofless`，支持 `top` / `roofless` / `exterior`，通过临时 session layer 隐藏屋顶并设置相机；`--view exterior` 可恢复完整外观。
- 已修复 `build.py`、`verify.py`、`simulate.py` 的 USD 导出参数：真实 SMPL 网格传入 `/World/Human/Skin`。在 headless 复验中实际输出 `6890 skin verts`、`13776` 三角面、24 links、19 colliders，最终 `human build: PASSED`；基础场景哈希检查通过。
- 复验：`python3 scripts/humans/build.py --dry-run` 通过；`python3 -m pytest -q` 为 `199 passed / 8 skipped`；本机 Ruff 和 compileall 通过。当前隔离会话的 Isaac 日志仍因没有宿主机 GPU/显示而 CPU fallback，但 USD 结构与蒙皮计数已验证。

## 2026-09-22 — AMASS 导入与 GPU/CUDA 复核补充

- 已新增 `src/sim2sense_fall/humans/amass.py` 和 `scripts/humans/import_amass.py`：本地 AMASS `.npz` 扫描、字段校验、源 SHA-256、SMPL-H 52→SMPL 24 重定向、Y-up→Z-up、来源元数据和摔倒候选筛选；`scripts/humans/simulate.py` 可通过 `--amass-root` / `--amass-fall-only` 接入 dry-run/仿真选择。
- 新增 `tests/humans/test_amass_import.py`，覆盖字段缺失快速失败、来源哈希、坐标转换、候选摔倒和确定性排序。用 `/tmp/synth-amass-import` 合成 fixture 实跑 `import_amass.py`：`2 sequences scanned, 1 fall candidates`；合成数据仅验证管线，不能当作真实 AMASS。
- 当前资产根扫描未发现真实 AMASS `.npz`，因此真实子集、人物、序列、许可和物理复核仍待用户提供已授权数据；项目不会自动下载注册制数据集。
- GPU/CUDA 复核结果：PCI 设备 `01:00.0` 为 RTX 4060 Max-Q，内核 `nvidia` 驱动 `595.91.07` 已绑定，`nvcc 12.0`、`libcuda.so.1`、`libcudart.so.12` 和 `libnvidia-ml.so.1` 存在，`systemd-udevd` 与 `nvidia-persistenced` 均运行；但 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 均不存在，`/sys/class/misc` 也没有 NVIDIA 节点，`nvidia-smi` 返回 `couldn't communicate with the NVIDIA driver`。`udevadm test` 显示规则会调用 `/sbin/ub-device-create`，但该工具在当前受限会话无权限创建节点；`sudo` 被 `no new privileges` 阻断。结论是宿主机 udev/设备节点或容器权限问题，非仓库代码缺少 CUDA toolkit。
- 因此 Isaac 日志中的 `NVML_ERROR_DRIVER_NOT_LOADED` / `No usable CUDA device present` 仍属环境限制；本轮没有手工 `mknod`、卸载/重载驱动或修改系统服务，GPU/RTX 渲染不能写成已通过。宿主机管理员需在真实系统会话修复 udev 设备节点后再复验 `nvidia-smi`、CUDA sample 和 Isaac GPU backend。

### 宿主机修复后的复验（用户终端，2026-09-22）

- 用户在真实宿主机执行 udev 规则重载、`udevadm trigger` 和 `nvidia-persistenced` 重启后，`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`、`/dev/nvidia-modeset` 和 `nvidia-uvm-tools` 已恢复。
- 用户提供的 `nvidia-smi` 已成功返回：RTX 4060 Laptop GPU，驱动 `595.91.07`，驱动报告 CUDA `13.2`，显存 `1720 MiB / 8188 MiB`，GPU 利用率 `40%`。这证明宿主机 NVML 和字符设备链路已经恢复；此前的 GPU 故障已解决。
- 本仓库的受限执行会话仍看不到宿主机 `/dev/nvidia*`，所以不能从该隔离会话代替用户运行 Isaac GPU smoke test。真实宿主机下一步运行 `~/isaacsim/python.sh scripts/humans/verify.py` 和 `~/isaacsim/python.sh scripts/humans/build.py --headless`，检查日志中不再出现 `NVML_ERROR_DRIVER_NOT_LOADED` / `No usable CUDA device`，并用 `nvidia-smi` 观察 Isaac 进程显存占用。
- `nvcc` 工具链仍是本机 CUDA 12.0，而 NVIDIA 驱动向后兼容并报告 CUDA 13.2；这是驱动支持版本与本地编译工具包版本不同，不是当前设备故障。

## 2026-09-22 — 房屋扩大与 AMASS 预览复验

- `configs/scenes/indoor_apartment.yaml` 新增 `layout_scale_xy: 1.10`。缩放同时作用于房间 origin/size、门窗宽度与偏移、家具位置和水平尺寸，墙高保持原值；CPU manifest 和 Isaac headless USD 实测 footprint 为 `9.24 m × 7.70 m`，面积为 `71.148 m²`。
- 新增 `scripts/humans/view_amass.py`。它只读取已授权的原始 AMASS `.npz`，在同一帧更新 SMPL skin、PhysX articulation DOF 和 root pose，并调用 `configure_inspection_view(mode="human")`；不把导入产物 NPZ 当作原始 AMASS 输入。
- 用 `/tmp/synth-amass-import/Subject1/fall.npz` 播放 `amass__fall`：Isaac headless 启动、PhysX 激活、241 帧播放和 USD 预览导出全部通过，退出码 0；产物为 `artifacts/humans/amass_preview.usda`。合成 fixture 不能作为真实 AMASS 实验结果。
- 首次预览失败是 `Prim` 直接调用 `GetPointsAttr()`，已改为 `UsdGeom.Mesh(prim)` 后复验通过。误把 `/tmp/amass-import-result`（retargeted NPZ）作为原始输入会明确失败，这是输入边界而非静默降级。
- 本次隔离 Isaac 日志仍报告 `NVML_ERROR_DRIVER_NOT_LOADED`、`cuInit failed (100)` 并回退 CPU PhysX；宿主机 `nvidia-smi` 已由用户复验正常，GPU Isaac smoke test 仍需在宿主机显示/设备会话运行。
- 官方 AMASS 列表访问已实际尝试：凭证 dry-run 通过并识别 4 个配置项，但带现有代理访问返回 `Connection refused`；去掉代理后返回 TLS `SSLV3_ALERT_HANDSHAKE_FAILURE`。因此本轮没有把在线下载写成完成，待网络/代理恢复后可直接重跑 `fetch_assets.py --site amass --list`。

## 2026-09-22 — Transitions 本地下载核验

- `/home/gsh/Downloads/Transitions.tar.bz2` 已通过 `bzip2 -tv`，归档内有 110 个 `.npz`，字段实测包含 `poses (N,156)`、`trans (N,3)`、`mocap_framerate`、`gender` 和 `betas`；解压后占用约 261 MB，存于 `data/humans/amass_raw/Transitions_mocap/`。
- `scripts/humans/import_amass.py --root data/humans/amass_raw --out artifacts/humans/amass_transitions_import` 已通过：110 sequences scanned，全部完成 SMPL-H→SMPL 重定向；候选筛选为 `0 fall candidates`。这批数据可用于 `sit_stand`、`walk`、`crawl`、`run` 等易混淆动作，不能写成已取得跌倒样本。
- 真实 Transitions 动作 `amass__sit_stand_poses` 已在 Isaac headless 中播放 980 帧并通过，产物和元数据写入 `artifacts/humans/amass_preview.usda/.json`；本次运行仍是当前隔离环境的 CPU PhysX fallback。
- CMU 下载状态：此前看到的 `.part` 临时文件和 0 字节目标文件目前均已从 Downloads 消失，当前没有可校验的 CMU 压缩包；需要官网重新下载或由浏览器对仍存在的任务执行续传。

## 2026-09-22 — 房屋再次扩大

- 用户要求把当前房屋再扩大到 2 倍。`layout_scale_xy` 已从 `1.10` 调整为 `2.20`，因此当前实际导出 footprint 从 `9.24 × 7.70 m` 变为 `18.48 × 15.40 m`；房间高度、墙厚和竖向家具尺寸保持不变。
- 房间 origin/size、门窗宽度/偏移、家具位置和水平尺寸继续由同一个缩放锚点统一变换，避免房间与家具各自缩放造成坐标错位；CPU 测试期望和 USD manifest 将同步刷新。

## 2026-09-22 — 阶段 7 二次验收与逐帧网格导出（当前）

### 修复后复验（repair6）

- 修复 `scripts/humans/verify.py` 的 PD 验收：每个 DOF 单独跟踪，传送后清零根部/关节速度，并在控制试验期间暂时关闭 19 个**人体自身**碰撞体，避免固定室内家具/墙体把控制误差污染成关节跟踪失败；重力和地面检查前恢复全部碰撞体。
- 新增 `HumanRuntime.reset_velocities()` 与 `set_body_collisions_enabled()`，并将逐 DOF 的 target/reached/error 写入 `human_verify.json`，保留可审计证据。
- 最新 headless Isaac：[`artifacts/humans/human_verify_repair6.json`](../artifacts/humans/human_verify_repair6.json)，**PASSED**；PD 最大误差 `1.540°`（容限 `15°`），逐 DOF target/reached/error 及 stiffness/damping 均写入 JSON，驱动关闭负对照、19 个碰撞体恢复、抬高后重力回落、地面穿透和基础场景哈希均通过。运行仍记录无 NVIDIA/CUDA 设备，因此这是 CPU PhysX 回退，不是 GPU 渲染验收。
- 新增 `scripts/humans/migrate_trials_index.py` 并迁移两个旧批次索引：`physics_dt_s` 从错误的 `120.0` 修为 `0.008333333333333333`，同时加入 `physics_hz: 120.0`；迁移脚本会读取被引用 trial JSON 的 provenance 做一致性校验。

- 资产结论已纠正：`data/humans/smpl/` 中存在并可加载 SMPL v1.1.0 neutral、male、female 三个 pickle；审计为 **3/5 个声明模型可用**。neutral 实测 6890 顶点、13776 三角面、24 关节、300 个 shape directions，静止姿态线性蒙皮最大漂移 `2.22e-16 m`。因此“SMPL 尚未下载”不成立。
- AMASS 仍未取得：在配置资产根及项目数据目录中未发现 AMASS 原始 `.npz` 动作序列；当前动作仍为 `scripted`。不能把现有动作称为 AMASS。
- 新增 [`docs/mesh-export.md`](mesh-export.md) 和 `humans/mesh_sequence.py`：固定拓扑、逐帧世界坐标 `(x,y,z)`、米制、Z-up、物理时间轴与信道时间轴均有校验。
- `scripts/humans/build.py` 与 `simulate.py` 现在共享同一实际 SMPL 资产和尺度拟合；CPU 规划的 `ground_offset_m` 已从此前程序骨架的 `0.9911 m` 对齐到 SMPL 规划的 `1.2462 m`。
- `GroundTruth`/NPZ 现在写入 `mesh_vertices_xyz`、`mesh_faces`、`channel_mesh_vertices_xyz`、顶点归属和拓扑哈希。无真实模型时使用明确标记的 `capsule_proxy_mesh`，不会伪称 SMPL。
- CPU 验收：`compileall` 通过；`196 passed / 8 skipped`；本机 Ruff 通过；`uv tool run ruff check .` 仍被环境 DBus 错误阻断（`Process 2 is a kernel thread, refusing`）。`scripts/humans/plan.py` 通过并验证真实 SMPL；`scripts/humans/simulate.py --dry-run` 通过，并实际写出每个动作的 `<motion>.mesh.npz`（`stand_neutral`: `(121, 6890, 3)` 顶点、`(13776, 3)` 面、全部有限），代理网格无零面积三角形。Isaac 数值稳定性曾在 repair3 失败，已由 repair6 复验修复。
- 下一步：在已通过的 Isaac 分层验收上继续检查每帧 link pose→mesh 的有限性、拓扑一致性和 USD/NPZ 时间对齐；取得 AMASS 后再做动作筛选与重定向；随后进入 Sionna RT 首条 CIR/CSI smoke test。
- Isaac repair3 的失败记录保留为历史证据：[stage7-final-isaac.log](../artifacts/humans/stage7-final-isaac.log)。其后的 repair6 已通过；失败原因是控制试验未隔离人体碰撞且复用传送后的速度状态，已在代码中修复并由逐 DOF 证据复验。

## 2026-09-22 — 阶段 7 第一次批判性复审（历史快照）

- 完整验收不通过，阶段 7 恢复未完成状态；程序骨架/CPU 契约部分可用。详见 [`stage7-review.md`](stage7-review.md)。该快照的资产扫描结论已被二次验收纠正。
- 当时误报两个配置资产根不存在、SMPL 审计 0/5；二次验收确认项目内已有 3 个 SMPL v1.1.0 pickle，并通过 neutral CPU 加载与静止蒙皮复验。AMASS 原始序列仍未发现。
- CPU：compileall、134 passed / 8 skipped、本机 Ruff 通过；plan/build/simulate CPU 路径通过。`uv tool run ruff check .` 退出 46，DBus `Process 2 is a kernel thread, refusing.`。
- 六项待整改：伪模型导致真实蒙皮假通过及禁止降级无效；PD 目标重复转弧度；体点/方向速度混用导致假撞击；索引把 120 Hz 写成 120 s；联合键不保证人物隔离；日常动作超跟踪容差仍标可用。
- Isaac 结构与前 3 个姿态比较通过后报 `ValueError: quaternion contains a non-finite value`，退出 1；同时有 `NVML_ERROR_DRIVER_NOT_LOADED`、接触视图 `AttributeError: 'NoneType' object has no attribute 'check'`。历史全部 PASS 未复现，根因未定位，不能只归为 GPU 缺失。
- 日志：`artifacts/humans/stage7-review-isaac.log`。固定场景 `apartment_cn_two_bedroom`、seed=20260922、程序胶囊代理；无新训练/数据划分，指标与哈希见验收报告；GPU/GUI 未验证。

## 2026-09-21

### 阶段 1：环境与资料核对 — 已完成

- 项目目录原为空，仅包含外部提供的只读 `.git` 目录。
- 本地 Zotero 数据库可读，发现 187 个条目；相关全文位于 `/home/gsh/Zotero/storage/` 和 WorkBuddy 导出目录。
- 已检查本地无线跌倒检测文献和 PDF 文本层。

### 阶段 2：文献证据 — 已完成

- 完成 SiFall、DGSense、CSI-Bench、Chu et al.、TED-Net/DGNN 和数字孪生雷达路线的证据记录。
- 明确项目评测必须覆盖域偏移、连续流式数据、仿真到真实差距和报警代价。
- 证据边界记录在根目录 `notes.md`，没有把预印本或摘要外推成已验证事实。

### 阶段 3：仓库骨架与代码 — 已完成

- 建立 `src/sim2sense_fall` 包、schema、验证器、窗口化报警基线和 Isaac/Sionna 适配器协议。
- 加入 `pyproject.toml`、`configs/baseline.yaml`、`.gitignore` 与核心测试。

### 阶段 4：文档与管理约束 — 已完成

- 建立 `AGENTS.md`，要求每阶段更新计划、进度和证据文档。
- 完成 README、架构说明、数据契约和远程同步说明。
- 首次测试发现系统环境没有自动把 `src/` 加入导入路径，已在 `pyproject.toml` 固定 `pytest` 路径；同时修正了 `slots` dataclass 测试构造方式。

### 阶段 4 验证记录

- `python -m compileall -q src tests`：通过。
- `PYTHONPATH=src python -m sim2sense_fall.windowing --duration-s 10`：通过，生成 33 个窗口。
- `python -m pytest -q`：通过，4 passed。
- Python 3.10 兼容性检查：将 `StrEnum` 替换为 `str, Enum`，兼容 smoke test 通过。
- 清理了测试缓存；生成数据、模型和仿真缓存均由 `.gitignore` 排除。
- 首次 `pytest`：因导入路径和测试构造问题失败，已修复，待复跑。
- `ruff check .`：当前环境未安装 `ruff`，未将其写成通过。

### 阶段 5：Isaac Sim 室内场景 — 已完成

目标：搭建包含卧室、客厅、卫生间等典型室内空间的三维场景，配置墙体、地板与家具属性，
导出 USD，并给出 GUI 查看方式。

- 环境核对：Isaac Sim `6.0.1-rc.7` 位于 `/home/gsh/isaacsim`（`isaac-sim.sh` / `python.sh`），
  GPU 为 RTX 4060 Laptop（8 GiB），CUDA 13.2，显示环境 `DISPLAY=:0`。
  注意 Isaac Sim 6 使用 `isaacsim.*` 命名空间（`omni.isaac.core` 已不存在），且 `pxr`
  只在 Kit 运行时启动后才可导入。
- 新增场景管线：`scene_materials.py`（材质库）、`scene_spec.py`（schema + YAML）、
  `scene_furniture.py`（参数化家具配方）、`scene_planner.py`（CPU 几何规划）、
  `isaac_scene.py`（USD 落地）。
- 新增配置 `configs/scenes/indoor_apartment.yaml`：8.4 m × 7.0 m，两室一厅一厨一卫加走廊，
  6 个房间、46 段墙、11 处门窗开口、34 件家具/灯具。
- 新增脚本：`scripts/build_indoor_scene.py`（`--dry-run` / `--headless` / `--gui`）、
  `scripts/view_indoor_scene.py`（GUI 查看）、`scripts/verify_indoor_scene.py`（物理 smoke test）。
- 新增文档 [`indoor-scene.md`](indoor-scene.md)：场景说明、材质与家具属性表、运行方式、
  GUI 查看指令和验证记录。

#### 阶段 5 关键实现决定

- **三类材质属性一次定义**：视觉（`UsdPreviewSurface`）、力学（PhysX 摩擦/恢复/密度）、
  电磁（ITU-R P.2040-3 幂律系数 + Sionna 材质名）。电磁参数按频率求值而不是硬编码，
  换频段不会静默复用旧数值。
- **规划与落地分离**：几何规划是纯 Python，可在无 Isaac Sim 的机器上单测与评审；
  `isaac_scene.py` 只消费规划结果。`pxr` 采用惰性导入，CPU 上导入该模块不会失败。
- **组合刚体**：动态家具不能「按零件各成一个刚体」，否则首次接触就散架。规划阶段把它
  拆成刚体根 Xform（承载 `PhysicsRigidBodyAPI` + 单一质量）与局部坐标子碰撞体。
- **不做在线素材依赖**：家具全部用盒体/圆柱体参数化拼装，保证离线可复现、可 diff。

#### 阶段 5 验证记录

- `python -m compileall -q src tests scripts`：通过。
- `python -m pytest -q`：通过，27 passed（新增 `tests/test_scenes.py` 23 项，覆盖材质校验、
  场景校验、墙体开口分段、共享墙不重复、组合刚体、清单可序列化）。
- `uv tool run ruff check .`：通过，All checks passed（隔离运行，未改动 Isaac Sim 环境）。
  过程中用 `ruff check --fix` + `ruff format` 清理了新文件的导入顺序、废弃导入与行宽，
  并确认格式化前后场景清单逐字段一致（几何未被改变）。
- `python3 scripts/build_indoor_scene.py --dry-run`：通过。
- `~/isaacsim/python.sh scripts/build_indoor_scene.py --headless`：通过。
  产出 `artifacts/scenes/indoor_apartment.usda`（335 KB，344 prim，231 几何，231 碰撞体，
  1 刚体/13.56 kg，14 视觉材质，12 接触材质，8 灯光）与
  `indoor_apartment.scene.json`（180 KB 清单）。
- `~/isaacsim/python.sh scripts/verify_indoor_scene.py --settle-seconds 3`：14 项检查全部 PASS，
  含正向对照「刚体抬高 0.25 m 后落回原位，残差 0.0000 m」，并比对原文件 SHA-256
  确认验证过程不改动导出产物。
- `~/isaacsim/python.sh scripts/view_indoor_scene.py --activate-physics --settle-seconds 2`：
  GUI 实际打开并驻留，物理步进正常。

#### 阶段 5 踩到的坑（已修复，记录以免重复）

1. **`Sdf.Layer.Export` 导到自己的路径会静默留空文件**：用 `Usd.Stage.CreateNew(path)`
   建 stage 时必须 `Save()`，只有继承来的匿名 stage 才用 `Export()`。已在 `build_stage`
   中按来源分支处理，并加了「导出后文件不得为空层」的断言。
2. **USD prim 名不能以数字开头**：家具零件原命名 `00_frame` 会触发
   `Path must be an absolute path: <>` 这种完全看不出原因的报错。已改名为 `p00_frame`，
   并在 `build_stage` 前用 `Sdf.Path.IsValidIdentifier` 预校验整份规划。
3. **独立运行的 Isaac Sim 不会把 PhysX 挂到 USD stage**：时间线能播、物理步数在涨，
   但没有任何物体会动，`omni.physx.tensors` 报
   `Failed to get a valid attached USD stage id`。修复方式是
   `SimulationManager.enable_all_default_callbacks()` + `setup_simulation()`（封装为
   `activate_physics()`）。这也是为什么验证脚本必须做「抬高再落下」的正向对照——
   只测漂移会把这个故障判成通过。
4. **Kit 会接管 `sys.stdout`**：headless 下 `print` 的输出会被吞掉。构建报告改为同时写
   `sys.__stdout__` 和 `artifacts/scenes/*.build_report.txt`。
5. **`SimulationApp.close()` 会直接终止解释器**：`finally` 里必须先写报告再 `close()`。
6. **YAML 1.1 浮点指数必须带符号**：`2.4e9` 会被解析成字符串，必须写 `2.4e+9`。
   已在解析器里加了针对性错误提示。
7. **Kit 会把未知命令行参数转发给自己**：脚本在启动 `SimulationApp` 前需要清空
   `sys.argv`。
8. **Isaac Sim 打开场景会往内存图层塞 prim**：`omni.usd.get_context().open_stage()`
   之后，stage 上多出 `/Render` 和 4 个 `/OmniverseKit_*` 视口相机（共 7 个 prim），
   而 `Usd.Stage.Open` 走图层缓存也会读到这份被改过的图层——于是同一份 `.usda`
   会报出 344（文件真实值）和 351（进程内被改写后的值）两个数。验证脚本已改为在
   `.usda` 的临时副本上运行，并比对原文件 SHA-256，保证产物不被验证过程污染。

### 阶段 6：GitHub 上游同步 — 已完成（未推送）

- 目标：`git@github.com:11anticipate/Sim2Sense-Fall.git`
- 早前错误：`ssh: Could not resolve hostname github.com`，且 `.git` 只读。这两条**已解除**：
  本轮 `git ls-remote --heads origin` 与 `git fetch origin` 均成功，`.git` 可写。
- 当前事实：本地 `main` = `origin/main` = `c77a37d`
  （*chore: scaffold Sim2Sense fall sensing project*），0 ahead / 0 behind，顶层目录树一致。
  也就是说上游目前只有脚手架提交，没有需要合并的内容。
- 许可证核查：上游**没有 LICENSE / COPYING 文件**，默认即「保留所有权利」。在拿到明确许可前，
  不应假设代码可对外分发。
- 本轮动作：把场景工作提交到 `feature/indoor-scene` 分支（3 个 Conventional Commits），
  `main` 保持与上游一致、未被改动。
- 未做：**没有 push**，也没有开 PR。推送属于对外发布动作，等用户确认。
- 附注：`.workbuddy/`（Agent 工作记忆）目前是未跟踪状态，是否纳入版本管理由用户决定。

### 阶段 6 验证记录

- `git ls-remote --heads origin`：`c77a37dc…  refs/heads/main`，退出码 0。
- `git fetch origin`：成功，无新对象。
- `git rev-list --left-right --count origin/main...HEAD`：`0  0`。
- `git ls-tree --name-only origin/main` 与本地已跟踪的顶层条目逐项一致。
- `git status --short`：仅剩未跟踪的 `.workbuddy/`，无未提交的源码改动。

## 下一步

1. 确认是否把 `feature/indoor-scene` 推到远端并开 PR；同时决定 `.workbuddy/` 是否纳入版本管理。
2. 用场景配置里的 `seed` 驱动房间布局、材质与家具的随机化，形成训练域族（对应 DGSense 路线）。
3. 在场景中加入人体（刚体或骨架）并导出与场景同时间基准的运动真值。
4. 接入 Sionna RT，把 `/World` 几何与 `sim2sense:em_*` 材质映射为传播场景，生成首条
   `ChannelSample` 并完成 CPU schema 校验。
5. 复核代理电磁材质与已安装 Sionna 版本 `itu_*` 数值的一致性。
6. 上游没有 LICENSE，先与仓库所有者确认许可范围，再考虑任何对外分发。

## 2026-09-21 — 室内 USD 验收：证据采集

- 已读取计划、配置、场景规划/导出/查看/验证代码和现有 27 项测试。
- `python` 不存在（`command not found`），改用 `python3`：27 tests passed，compileall 通过；CPU dry-run 输出另存 `artifacts/acceptance/cpu/`，未覆盖既有产物。
- `uv tool run ruff check .` 被 snap 环境阻止：`required permitted capability cap_dac_override not found`。使用已安装的 `/home/gsh/.local/bin/ruff check .`：通过。
- GPU 读取报 `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`；提权的只读 `nvidia-smi` 未执行，自动审批服务报 `503 Service Unavailable`，属于审批服务故障而非安全拒绝。
- Isaac 原验证脚本实际运行成功，14 项 PASS（`artifacts/acceptance/isaac_verify.log`），PhysX 使用 CPU 回退；日志同时报 `NVML_ERROR_DRIVER_NOT_LOADED`、`No device could be created`、`Failed to open display`。这能证明本轮物理 smoke test 通过，不能证明 GPU 渲染或 GUI 正常。
- 直接导入 pxr 首次报缺包，其后报 `libusd_tf.so` 缺失；显式配置本机 bundled USD Python 和动态库路径后，OpenUSD 0.25.11 成功读取实际 USD：344 prim / 231 geometry / 0 camera。
- 实际几何坐标确认六块吊顶覆盖全部房间；发现地基地面重合、地毯穿墙/越界、冰箱穿墙。验收结论尚待汇总。

### 验收阶段 2 完成：问题复现与内部查看

- 已用实际 USD 包围盒验证越界/穿墙；当前全部家具仅有轴对齐或 90° 倍数旋转，因此所报告盒体与墙的交集为真实盒体体积交集，不是任意旋转 AABB 的误报。
- 故意传入不存在的 manifest，Isaac 验证日志明确 `verification: FAILED`，但命令退出码仍为 **0**。`SimulationApp.close()` 默认 fast shutdown 且默认 `exit_code=0`，吞掉了后续返回值。这一验收门禁漏洞尚未修复。
- 配置反例均被错误接受：家具位置 NaN（manifest 含 NaN）、`same-name` 与 `same_name` 清洗后路径冲突（244 规划图元只有 238 个唯一路径）、0.05 m 床尺寸生成负床垫/枕头尺寸。
- 新增 `scene_view.py` 与查看器 `--view top|roofless|exterior`；默认去顶俯视，临时 session layer 隐藏屋顶与灯具，保留原文件和碰撞体。
- 系统 Python：28 passed / 1 skipped（缺少 pxr）；Isaac bundled Python 追加本机 pytest 所在目录后，2 项查看模块测试 passed，覆盖图层隔离、碰撞体保留、模式切换和相机包含地板四角。首次 bundled Python 测试报 `No module named pytest`，已用本机既有测试包完成，无下载。
- compileall 和本地 ruff 通过。新增查看器的 GPU/GUI 尚未验证。

### 验收阶段 3 完成：报告与最终验证

- 完成 `docs/indoor-scene-review.md`：6 项可复现待修缺陷、研究用途边界、证据位置与整改顺序。
- 重新 headless 导出到 `artifacts/acceptance/rebuilt/` 成功，USD SHA-256 与原件完全相同：`2e5fb8ec7523e972b5ef3ffb7f545a1078ffbf43480fc208e5f801c783cf86cb`。
- 三种新查看模式在 Isaac 无界面下完成视口集成：相机切换正确；top/roofless 隐藏 6 块吊顶，exterior 恢复；231 碰撞体保留；源文件哈希不变。见 `inspection_view_checks.json` / `inspection_view.log`。没有 GPU 画面输出，GUI/RTX 视觉仍待验证。
- 从实际 USD 生成并检查了对比图与布局标注图；三维预览采用 CPU 深度缓冲，图上明确标注不是 RTX 截图。
- `task_plan.md` 当前状态已从“场景任务完成”改为“基础构建完成、整体验收需整改”。生成资产未加入 Git，未提交/推送；用户已有 `.workbuddy/` 保持未跟踪。

## 室内验收整改 — 实现与 CPU 回归

- R1：构建/验证/查看入口把最终错误码传给 `SimulationApp.close(exit_code=...)`；启动失败明确报错。验证器先检查清单，并拒绝 0/NaN 时长及无效落差对照。
- R2：地基上表面位于最厚地板底面，厚度改由 `foundation_thickness_m` 配置；较薄地板下增加混凝土找平层，所有行走表面仍为 z=0。
- R3/R4：三块地毯明确适配房间的尺寸，冰箱旋转 180° 使门/把手朝室内。增加 CPU 世界几何校验，处理动态家具父变换、旋转盒体与圆柱体，拒绝越界和穿墙；允许接触以及地毯与家具的合理叠放。
- R5/R6：完整计划检查规范化路径唯一性；配置和派生零件检查有限数值与正尺寸；JSON 禁止 NaN/Infinity。
- 验证器移除 344/231 等固定计数，以清单逐图元比较世界变换、尺寸、标签、碰撞、刚体质量以及渲染/物理/电磁材质。
- 实现后的第一轮 CPU 回归：70 passed / 1 skipped；ruff 通过。新增子进程测试模拟 Kit 立即退出，确认构建/验证的启动异常与运行期异常均返回 1。

### USD 与运行时复验

- 真实 OpenUSD 回归 9 项通过：完整导出匹配、地板/地基分离、6 种同计数错误资产被拒绝，以及此前查看模式测试。
- 完整系统测试：70 passed / 8 skipped（8 项需要 pxr 的测试已在 bundled USD 环境实际执行）；compileall 通过。
- `uv tool run ruff check .` 本轮报 DBus：`Process 2 is a kernel thread, refusing.`；已安装 `/home/gsh/.local/bin/ruff check .` 通过。
- 首次 Isaac headless 构建因无 GPU 触发图形错误弹窗，60 s 超时（退出 124）；其后验证因 USD 尚不存在而非零退出。改为 `DISPLAY= WAYLAND_DISPLAY=` 后 CPU 回退构建成功。
- GPU 提权只读检查仍被自动审批服务 503 阻止，未执行；没有绕过审批或修改系统驱动。
- 修复后的 USD 在 Isaac CPU 回退下通过 10 项检查，含逐图元/材质匹配、椅子稳定、0.25 m 抬升回落、源文件不变；退出码 0。

### 整改完成与默认资产更新

- 实际 Isaac 运行期负例（旧 USD / 新 manifest）返回 1，准确识别地基、三块地毯、冰箱的变换差异，确认不再出现失败返回成功。
- 已把通过验证的 USD/清单更新到默认 `artifacts/scenes/`；旧版本备份在 `artifacts/remediation/before/`。新 USD SHA-256 为 `38f062c3ca9d54e5ab8f2b2cc7428ce06cb85c7933d74530a3c3452958403b86`。
- 更新后实际 USD 几何审计：家具越界、家具/墙体体积交集均为空；地基顶面 -0.12 m，与地板底面接触，行走面仍为 0 m。
- 已生成并逐张检查修复后的 CPU 立体预览和平面图；未把 CPU 预览写成 RTX 截图。
- `docs/indoor-scene-remediation.md` 汇总六项闭环、测试结果、资产/源码哈希和后续边界；`task_plan.md`、研究笔记与场景文档已同步。未提交或推送。

## 场景目录整理 — 迁移完成

- 可复用实现迁入 `src/sim2sense_fall/scenes/`，三个入口迁入 `scripts/scenes/`，对应测试迁入 `tests/scenes/`。
- 同步模块导入、脚本仓库定位、测试子进程与当前文档；旧入口不保留兼容副本。
- 首轮 CPU 回归 70 passed / 8 skipped；下一步复验实际 USD 与物理行为。

### 目录整理 — 回归完成

- CPU 测试 70 passed / 8 skipped；bundled OpenUSD 测试 9 passed（包含被跳过的 8 项）。compileall、已安装 Ruff、diff 空白检查通过。
- `uv tool run ruff check .` 退出 46，实际错误仍为 DBus `Process 2 is a kernel thread, refusing.`，使用 `/home/gsh/.local/bin/ruff check .` 完成检查。
- 新入口 CPU 构建/清单验证通过；从 `/tmp` 使用绝对路径构建及查看帮助成功，验证迁移后的仓库定位。包发现包含 `sim2sense_fall.scenes`。
- Isaac 构建和验证均退出 0，物理/资产 10 项 PASS；日志有 `NVML_ERROR_DRIVER_NOT_LOADED`、`No usable CUDA device`，实际使用 CPU 回退，GPU/GUI 视觉未复验。
- USD SHA-256 仍为 `38f062c3ca9d54e5ab8f2b2cc7428ce06cb85c7933d74530a3c3452958403b86`，与迁移前逐字节一致。清单仅 generator 改为 `sim2sense_fall.scenes.planner`，默认清单已同步。
- YAML 仅材质模块路径注释变化，当前 SHA-256 为 `34109c0df85b82ad86d7fc367cc3beaca0b730361444a5e93814407df0e59d83`；历史验收报告的旧配置/清单哈希保留。
- 本轮为固定场景回归，无数据集划分、无模型；场景 `apartment_cn_two_bedroom`，seed=20260921，2.4 GHz。物理指标沿用整改记录：静置/回落各 3 s，抬升 0.25 m，最大轴向误差容限 0.02 m。
- 日志和含源码/资产哈希的摘要在 `artifacts/reorganization/`。场景配置与产物目录保持清晰边界，文档和协作指令已同步。

## 2026-09-22 — 阶段 7：SMPL 人体、动作与跌倒仿真

实现与验收细节见 [`human-simulation.md`](human-simulation.md)。本节只记录实际跑过的东西与失败教训。

### 交付概览

- 新增 `src/sim2sense_fall/humans/`（skeleton / assets / rotations / motion / config / rig /
  events / export / usd_human）、`scripts/humans/`（common / plan / build / simulate / verify）、
  `configs/humans/`（assets / human_smpl_neutral / motions）、`tests/humans/test_humans.py`（52 项）。
- `python3 scripts/humans/plan.py`：全部 PASS。
- `python3 -m pytest -q`：122 passed / 8 skipped（8 项需 pxr，在 bundled USD 环境另跑）。
- `/home/gsh/.local/bin/ruff check .`：All checks passed。
- `python3 -m compileall -q src tests scripts`：通过。

### Isaac Sim 实际运行

- 早期 `~/isaacsim/python.sh scripts/humans/verify.py` 记录为**全部 PASS**，但该记录不能代表当前状态；后续使用实际 SMPL 规划重跑后未复现。
  物理姿态与 CPU 正运动学在 6 组姿态、24 连杆上误差 **0.00 mm**；
  关节极限往返最大偏差 7×10⁻⁶ 度；抬高 0.25 m 后骨盆下落 0.279 m；最低体点 0.0000 m；
  PD 跟踪最大误差 2.0 度（预先登记容限 15 度）；物理步长 1/120 s 生效。
  产物 `artifacts/humans/human_verify.json`。
- `~/isaacsim/python.sh scripts/humans/simulate.py --headless`（自由根，7 项）：
  `fall=2`（`control_loss`、`walk_in_place`）、`no_fall=5`（无扰动站立与四向推动）。
- `~/isaacsim/python.sh scripts/humans/simulate.py --headless --pin-root`（带记录的骨盆钉定，6 项）：
  `no_fall=4`（站立、弯腰、坐下、原地行走），2 项被有效性门禁排除（下蹲、后倒参考）。
- `~/isaacsim/python.sh scripts/humans/build.py --headless`：导出带动画学人体的 USD。

### 最新复验状态（2026-09-22）

- `python3 -m compileall -q src tests scripts`：通过；`python3 -m pytest -q`：**196 passed / 8 skipped**；本机 `/home/gsh/.local/bin/ruff check .`：通过；`uv tool run ruff check .` 仍被 DBus 环境错误阻断。
- `scripts/humans/plan.py`：真实 SMPL neutral 资产审计和 CPU 规划通过；`scripts/humans/simulate.py --dry-run`：通过，`stand_neutral.mesh.npz` 为 `(121, 6890, 3)`、`(13776, 3)`，数值有限。
- repair3 的失败结果（PD 57.064°、非有限残差、回落 0.1127 m）已被 repair6 复验替代；不要将该历史快照当作当前状态。当前 CPU/USD/Isaac 分层验收通过，AMASS 动作筛选和自由站立/行走控制仍未完成。

### 踩到的坑（均已修复，记录以免重复）

1. **`UsdPhysics.RigidBodyAPI` 没有阻尼属性**：`CreateLinearDampingAttr` 属于
   `PhysxSchema.PhysxRigidBodyAPI`（属性名为 `physxRigidBody:linearDamping`）。已单独封装，
   schema 缺失时记录「未写入阻尼」而不是静默跳过。
2. **`Articulation.get_world_poses()` 只返回根连杆的位姿**（双连杆也返回一行）。逐连杆世界位姿
   必须用 `RigidPrim(path).get_world_poses()` 读。
3. **连杆名与骨骼关节名不同**：根连杆的 prim 名取自 articulation 路径（`Human`），不是 `pelvis`。
   运行时的名字索引改为按规划路径构造，并在名字对不上时直接报错。
4. **接触视图必须在构造时创建**：`RigidPrim(..., contact_filter_paths=..., max_contact_count=...)`；
   否则 `get_net_contact_forces` 抛断言。当前实现仍取不到（记录为能力缺失与原因，不伪造零值）。
5. **物理张量视图只在时间轴播放时有效**：未 `play()` 就读关节位置会抛
   `Instance's physics tensor entity is not valid`。运行封装新增 `play()`/`pause()`。
6. **出生点必须在房间地板上**：公寓是逐房间铺地板，硬编码 `(0, 0)` 落在房间之外，
   人体直接穿过世界下落 40 m。改为从场景配置推导（最大房间内网格采样，取离家具最远且避开墙厚的点）。
7. **空中对照的高度不能穿过天花板**：抬到 +2.0 m 会撞 2.7 m 天花板，实测误差 317 mm。
   改到 +0.9 m 后误差为 0。检查代码里写下了这段经历。
8. **比对参照量必须是引擎自报的关节角**：比 `指令角` 时驱动器正把关节拉回零位，差值被误算成映射误差；
   改成用实测关节角做正运动学，测的才是映射本身。
9. **物理步长与物理速率的混淆**：`execute_trial` 把「速率」当「步长」用，导致 1 s 试验的时间轴导出成
   14400 s，稳定期被压缩成一步，于是所有试验都从倒地的身体开始。已修正并加入时长自检断言。
10. **骨盆高度门限的量纲**：`pelvis_height_fraction` 原先乘的是**身高**，0.55×1.70 m 相当于直立骨盆高度的
    94%，于是重力下正常的 6 cm 下沉被判成「姿态丧失」。改为乘**站立骨盆高度**。
11. **世界锚定（`root_mode: anchored`）不可用**：PhysX 对 `body0 = 空` 的固定关节返回非有限四元数。
    现在模拟入口显式拒绝该模式并给出替代路径，而不是产出 NaN 产物。
12. **参考动作必须与关节折叠自洽**：跌倒参考的根高度按「半个刚体绕足翻倒律」下降，
    否则身体横躺后其他胶囊体会扫到地板以下；下蹲参考的骨盆下降量与腿部折叠不匹配时，
    双脚会穿过地板并被有效性门禁排除（这是门禁在正常工作，不是门禁出错）。

### 明确未验证 / 未达标

- **该历史记录中的“真实 SMPL 蒙皮网格与 AMASS 序列均未取得”已过时。** 当前 SMPL neutral 已实际加载并可导出 `smpl_skin_mesh`；AMASS 序列仍未取得。无模型时才使用明确标记的 `capsule_proxy_mesh`。
- **自由站立与行走控制未达标**（阶段计划允许不以此作为导入成功条件）。
- 多轴关节链未在 Isaac Sim 中运行；`support_loss` 扰动未实现。

---

## 2026-09-23 物理交互 P0 修复（阶段 7 之后）

对照 `docs/physics-interaction-audit.md` 的根因清单，本轮完成 P0-1 与 P0-2，并在修复过程中
发现并修掉了第三条更隐蔽的缺陷。全部结论均有 GPU 实跑证据。

### P0-1 出生/站立高度两种约定混用 —— 已修复

`rig.py` 把 `ground_offset_m`（体坐标系量）当 `spawn_root_position`（根连杆世界坐标）用，
导致出生点系统性偏移。现已统一到同一约定，`verify.py` 双侧确认：

- `[PASS] CPU model puts the lowest capsule on the floor -- lowest capsule point at +0.000000 m`
- `[PASS] spawn height and ground offset use one convention -- spawn root z 1.012637 m, ground offset 1.012637 m`

> 更正：早先审计里写的「出生悬空 0.2336 m」是把 `|pelvis.rest_position[2]|` 误读成悬空量。
> 两份约定的真实差值是 **−0.044589 m**，方向是**下沉**而非悬空。已在审计文档中更正。

### P0-2 接触通道 —— 已修复并改为几何归因

`omni.physics.tensors` 的接触视图在本机不可用（插件报
`Pattern '/World/Human' did not match any rigid contact for filters`，随后
`rigid_prim.py:1924` 抛 `AttributeError: 'NoneType' object has no attribute 'check'`）。
可用的替代是 `omni.physx` 的 `get_contact_report()`，但它返回的是**一对不透明数值句柄**，
且实测 `PhysxContactReportAPI` 是**按 actor 而非按 collider** 生效的（摘掉全部 19 个人体胶囊，
报告仍是同样 6 对；摘掉环境侧才清空）。因此**逐连杆归因不可能从句柄做出来**。

改成**几何包含**判定：拿报告里的接触点去和实时世界坐标胶囊体做包含测试。GPU 实测：
`attribution=geometry, limbs=['left_ankle', 'right_ankle']`，单次试验 632 行接触覆盖 121 帧。

同时消除一处**静默降级**：原先三处 `except ContactSourceUnavailable` 会把「通道坏了」和
「没碰到东西」混为一谈，一次异常就能把整段接触历史抹成空而所有检查仍报 PASS。现在：
逐帧中途失效 → 抛错拒绝出结果；稳定期可读但归因不到连杆 → 抛错；稳定期确实零接触 → 单独报错。

### 新发现并修复：支撑面高度被 bbox API 静默算错

`contact_is_support` 依赖「支撑面高度」标量，这条链路连踩三层：

1. `UsdGeom.BBoxCache.ComputeWorldBound` 返回**错的参考系**——不带 `xformOp:scale` 也不带
   prim 自身 translate。地板世界顶面应为 `0.00`，它返回 `−0.0`；其上方的天花板返回同一个值
   （应为 2.85）。错值恰好压线通过容差，所以从未暴露。
2. 改手算时 `BBox3d`/`Range3d` 在本构建里**都没有 `TransformBy`**（`hasattr → False`），
   按 USD 文档写的调用直接抛 `AttributeError`，又被上面的 `try/except` 吞掉。
3. `ComputeLocalBound` 本身也错：返回**已缩放但仍被 prim translate 偏移**的盒
   （地板实测 `z ∈ [−0.12, 0.0]`，几何真值 `[−0.06, +0.06]`），再叠一次 translate 就成了 `−0.06`。
   它对 `render`/`proxy`/`guide` 还返回反向无穷哨兵盒。

**独立交叉验证**（不依赖任何 bbox API）：实测接触点全部落在 `z = 0.00000`、`normal_z = +1.0000`，
与场景 authord 的 `−0.06 + 0.06 = 0.00` 吻合；`docs/progress.md` 早先独立记录的
「地基顶面 −0.12 m，行走面仍为 0 m」也一致。

**修法**：绕开 `BBoxCache`，从 prim 自身 `size`/`radius`/`height` 构造居中几何盒再变换 8 个角点。
本场景碰撞体全是 `Cube`(197) 或 `Cylinder`(34) 且都带 `size`，此法完备；未知类型返回 `None`。

**效果**：`support_surface_height_m` 从 `0.0`（错值压线）变为 `−1.34e−9`（浮点零，真值）；
`contact_is_support` 从 **True=0 / False=632** —— 一个躺在地板上的身体却判「无任何支撑接触」——
变为 **True=456 / False=176**：456 条全部 `z = 0.00000`、`normal_z = +1.0000`（躺在地面上），
176 条是躯干/肩/肘撞立面的冲击（高度最高 1.45 m）。回归测试 CPU 可跑、不依赖 `pxr`。

### 本轮验收

```bash
python -m compileall -q src tests scripts   # OK
uv tool run ruff check .                    # All checks passed!
python -m pytest -q                         # 213 passed, 8 skipped
DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/verify.py                  # 31 PASS / 0 FAIL
DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/simulate.py --trial stand_neutral:none
```

`verify.py` 关键行：

```
[PASS] contact channel is readable -- 2 pairs on the probe step, source physx_contact_report
[PASS] reported contact points resolve to body limbs -- attribution=geometry,
       limbs=['left_ankle', 'right_ankle']
[PASS] support surface height is readable -- -1.341104505225843e-09 m
[PASS] raised body falls back under gravity -- pelvis descended 0.9338 m from 1.2289 m
[PASS] no body geometry is pushed through the floor after settling -- lowest body point -0.0000 m
```

### 仍未做（诚实记录）

- **P0-3**：稳定期仍每步 `set_root_pose` 传送骨盆。「站着」不是因为地面支撑，而是因为每步都在传送；
  `settle_used_root_support: true` 是诚实记录，但首帧从不处于力学平衡。这是目前最大的剩余 P0。
- **P0-4**：`view_amass.py` 的 physics replay mode 未实现。
- 后果之一：`stand_neutral`（一个**站立**片段）仍被标成 `fall`（`peak_descent_speed 6.20 m/s`、
  `final_trunk_angle 95.9°`、最低点 `−0.0406 m` 穿地 41 mm）。根因就是 P0-3。

---

## 2026-09-23 摔倒 Mesh 采集（分支 `feature/fall-mesh-capture`）

用户指令：P0 改完后提交并开新分支，采集**摔倒动作**的人体 3D Mesh + 三维坐标序列
`(x, y, z)`，为导入 Sionna RT 做准备；**只采集摔倒**；要求「多方面确认」准确性。

### 修掉的缺陷：SMPL 导入被偏航 90°（真缺陷）

详见 [`mesh-orientation-defect.md`](mesh-orientation-defect.md)。摘要：

- **文件帧 ≠ 管线帧**。授权 pkl 是 `X=横向(左右) Y=上 Z=前`；管线（`RestSkeleton` 文档）
  是 `X=前 Y=左 Z=上`。两者差一个绕垂直轴的 90°。
- 旧代码用 `up_axis_conversion("y","z")`，它**只保证 up 不变**（docstring 的前提「两帧共享
  +X 前向」对 SMPL 文件不成立），于是把「文件 X(横向)」映射到「管线 X(前)」——人体侧着站，
  而 up 仍朝上，所以它自己的检查全过。
- 修为 `body_frame_conversion(up=, forward=, left=)`（完整解剖三元组，强制 `det=+1` 拒绝镜像）
  + `_derive_body_frame()` 从关节实测三元组并交叉验证 + `_check_standing_frame()`。
  证据记入 `SmplModel.source_frame`（现报 `up=y forward=z left=x`）。
  `_dominant_axis` 删除；`up_axis_conversion` 保留给只需要 up 的 AMASS retarget 路径。
- 实测：模板 extent 由 `[1.5341, 0.1546, 1.4963]` → `[0.2905, 1.7451, 1.7174]`；
  与程序化骨架的方向余弦 ≥ 0.9965。

### 顺带修掉的隐患：`fit_mesh_to_rest_joints` 的关节配对

原实现按行号裸 `argmin` 配对，而 rig 缩放到配置身高、pkl 保持原身高，**正确配对**也会差
几十毫米（踝 53 mm、脚 91 mm）。实测 ≥1.75 m 时 argmin 变成**非双射**，静默返回
scale `1.0794` 而非真实的 `1.0459`。现改为**按共用关节名取对应** + 用**关节间距离**确认
（绝对坐标受绕原点缩放影响会把正确配对判错；骨向量方向分不开共线脊柱关节）。
另断言拟合残差是纯相似：实测 **0 µm**。

### 导出闸门：补上「旋转看不掉」的检查

**原有检查全部绕垂直轴旋转不变**（stature 范围、FK 对独立 CPU 链、身高拟合收敛、胶囊包含、
「mesh 会动」），所以偏航 90° 全过 —— 这正是缺陷能活下来的原因。新增两条，
**比较 mesh 与 links**（links 对 links 在坏导出上也过）：

| 检查 | 钉住 | 实测 |
| --- | --- | --- |
| `the body stands up in the first frame` | 俯仰 | 头面顶点 z `+1.5955` > 踝面 z `+0.0907` |
| `the mesh and the links agree on the body's facing` | 偏航（朝向） | 两者横向轴都是 `y`（1.8252 / 0.3038 m） |

**正向对照已做**：把导出网格绕 Z 偏航 90°（extent 正好复现 `[1.8252, 0.3038, 1.7963]`）→
两条都 `FAIL`，报 `mesh lateral axis 'x'` vs `links lateral axis 'y'`。
不会失败的检查不算证据。另把 `stand_neutral` 会误判 FAIL 的「mesh 必须动」改成
「骨架动时 mesh 必须跟着动」（静态参考不动是正确结果）。

### 采集产物

```bash
PYTHONPATH=src python3 scripts/humans/collect_fall_mesh.py --fall-only --overwrite
PYTHONPATH=src python3 scripts/humans/verify_fall_collection.py      # 独立第二实现
PYTHONPATH=src python3 scripts/humans/preview_fall_meshes.py         # 目视确认
```

`artifacts/humans/fall_mesh/`（gitignore，不入库）：

| sample | 帧 | 顶点 | 面 |
| --- | --- | --- | --- |
| `fall_forward_reference` | 145 | 6890 | 13776 |
| `fall_backward_reference` | 145 | 6890 | 13776 |
| `fall_lateral_reference` | 145 | 6890 | 13776 |
| `reach_then_topple` | 121 | 6890 | 13776 |

每样本两个文件（`<id>.mesh.npz` 几何 + `<id>.points.npz` 坐标序列）**共享一条 `time_s`**；
顶点数/面数跨帧不变，单一拓扑 sha256 `19710f11eb74fe51…`。
`coordinate_system: world_z_up_xyz`、`units: m`、`axis_order: xyz`。
`fidelity: kinematic_replay` —— **纯 FK 回放，无重力/无接触**，与 `simulate.py` 的
`physics_trial` 严格区分，不得混入同一清单。

清单新增 `body_model` 块（授权文件 sha256 / 身高 / 顶点面数 / 实测 `source_frame`）与
`points.link_names` / `joint_names`，满足 `AGENTS.md` 的模型版本要求并让 `(N,24,3)` 可定位。
`hashes.mesh_source_sha256` 曾误灌**动作**的 provenance（脚本动作恒 null），
已拆为 `motion_source_sha256` + `body_model_sha256`。

### 本轮验收（CPU，全部实跑）

```bash
python -m compileall -q src tests scripts   # OK
uv tool run ruff check .                    # All checks passed!
python -m pytest -q                         # 227 passed, 8 skipped
PYTHONPATH=src python3 scripts/humans/collect_fall_mesh.py --fall-only --overwrite
                                            # 全部 [PASS]，含新增两条解剖检查
PYTHONPATH=src python3 scripts/humans/verify_fall_collection.py
                                            # ALL CHECKS PASS (4 samples)，独立实现
```

目视确认（`fall_preview.png`）：首帧为张臂站立、皮肤与连杆重合、三个片段各自向
前/侧/后倒至平躺。

### 仍未做（诚实记录）

- 本轮采集**只有 `kinematic_replay`**。真正带动力学的摔倒轨迹需 `simulate.py` 的
  `physics_trial`，而它仍被 **P0-3** 卡住（稳定期每步 `set_root_pose`，PD 撑不住 → 自由落体）。
- AMASS 侧：`--write-slice` 已实现但**未对真实树跑过**（2198 个 npz，stem 重名）。
- `fall_lateral_reference` 的事件标签是 `invalid`（穿地 −0.219 m）—— 纯 FK 回放不查地板，
  这是预期结果而非新 bug；**不能**为了「全绿」放宽 `penetration_limit_m`。

---

## 2026-09-23 查看动作与 P0-3 的边界（分支 `feature/fall-mesh-capture` 续）

用户问：是否必须先做完 P0-3 才能在 Isaac Sim 里看到人物动作？

**不是。** 两件事正交：

- **看得到** 靠 **kinematic replay**：查看器每帧直接写 DOF 与根位姿并清零速度，不跑控制环。
- **P0-3 管的是「物理自己能不能站住」**：`simulate.py` 稳定期每步 `set_root_pose` 传送骨盆
  撑着人体，释放后 PD 撑不住就塌。这只影响自由根试验（`physics_trial`）。

### 修掉的真实缺口：查看器只认 AMASS

`view_amass.py` 的 `--amass-root` 原本是 `required=True`，因此在没有授权 AMASS 数据的机器上
**根本无法看任何动作**——而内置动作库就在旁边，回放循环本来也不关心里来源。改为可选：

```bash
~/isaacsim/python.sh scripts/humans/view_amass.py --motion fall_forward_reference
~/isaacsim/python.sh scripts/humans/view_amass.py --fall-only
```

`--fall-only` 判据随来源改变（AMASS 筛选器 / 库自身的 `fall_reference` 标签）；输出文件名与
报告横幅同样分开，不允许报告写「AMASS preview」却在播脚本动作。文件未改名
（8+ 处文档引用它）。

### 顺带修掉：`verify.py` 的「不发散」判据原来在测缺陷

原判据是无驱动自由落体后骨盆水平位移 ≤ **固定 1.0 m**。修复前它 PASS 是**巧合**——人体被
偏航 90°，倒向 0.52 m 外那面墙被挡住（0.4263 m）。修复后人体按应有方向倒，位移 1.3855 m。

探针实测（逐帧骨盆轨迹）：

```
direction = -2.0 deg from +X        ← 正好是管线前向 +X
horizontal displacement = 1.3855 m  (dx +1.3846, dy -0.0495)
final pelvis height     = 0.3029 m  ← 躺平
```

**这是帧修复正确的独立佐证**，不是新 bug。判据改为断言它名字真正声称的两件事：会停下来
（末段 200 ms 爬行 **0.005 mm**）+ 位移不超过自身体型的可达范围
（`2.0 ×` 站立骨盆高度 = 2.0253 m，覆盖倒伏弧 + 撞地滑移）。界由实测站立高度推出，不是照观测值调。

### Isaac 侧复验（本分支首次跑 GPU）

`DISPLAY=:0 ~/isaacsim/python.sh scripts/humans/verify.py` → **`human verify: PASSED`**

```
[PASS] physics pose matches the CPU model with left_knee at +55 deg -- largest link error 0.00 mm over 24 links
[PASS] joint limits round-trip through USD degrees and Isaac radians -- largest limit mismatch 0.000007 degrees
[PASS] the body does not diverge across the room -- pelvis moved 1.3855 m horizontally and crept 0.005 mm over the final 200 ms (limit 2.0253 m = 2 x the 1.0126 m standing pelvis height)
[PASS] no body geometry is pushed through the floor after settling -- lowest body point -0.0000 m
```

### 新定位的开放项（属于审计 P2-1）

`scene_spawn_point` 选的是「离家具最远」的点，结果落在客厅 `(9.320, 0.520)`，而客厅 bounds 为
`x 8.800..18.480 / y 0.000..6.600` → **离 −X 与 −Y 墙各只有 0.52 m**。向前(+X)倒有 9.16 m 空间，
倒向 ±Y 会在 0.5 m 处撞墙。**需要空间的动作与试验（摔倒采集尤其）都受它影响**，建议优先于 P0-3 处理。
