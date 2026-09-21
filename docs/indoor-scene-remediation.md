# 室内场景整改记录

> 目录迁移说明（2026-09-21）：场景实现现位于 `src/sim2sense_fall/scenes/`，入口位于 `scripts/scenes/`。下文历史缺陷的路径、行号和哈希保留为当时证据；当前命令见 [场景指南](indoor-scene.md)。

原验收报告中的 R1–R6 已修复，默认场景已更新为通过复验的新 USD。当前可以作为固定场景的人体接入、场景变体和 Sionna 适配的代码基础；这不等于人体、无线传播或机器人救援链路已验证。

## 修复与防回归

| 问题 | 修复 | 复验 |
|---|---|---|
| R1：失败返回成功 | 构建、验证和查看入口把最终退出码传给 Isaac 关闭接口；启动错误返回非零 | 独立子进程模拟立即退出；实际 Isaac 旧 USD / 新清单负例返回 1，正确资产返回 0 |
| R2：地板/地基重叠 | 地基顶面下移至最厚地板底面，较薄地板补找平层；地基厚度可配置 | 混合厚度 CPU 测试；实际 USD 地基 z=[-0.42,-0.12]，地板 z=[-0.12,0]，体积不交叠（容差 1e-6 m） |
| R3：地毯穿墙 | 主卧 1.8×2.5 m、客厅 2.0×2.5 m、走廊 1.4×0.8 m | 恢复旧默认尺寸的三个负例均拒绝；实际几何无越界/穿墙 |
| R4：冰箱穿墙 | 旋转 180°，门及把手朝室内 | 旧朝向负例拒绝；实际几何与北墙无交集 |
| R5：路径碰撞 | 完整 ScenePlan 检查规范化后路径唯一性 | 房间和家具两类 `same-name` / `same_name` 冲突均拒绝 |
| R6：无效数值/尺寸 | 配置及派生零件检查有限数值与正尺寸，JSON 拒绝 NaN/Infinity | NaN、±Infinity、布尔/字符串坐标、微小床配方、直接 dataclass 构造反例均验证 |

`src/sim2sense_fall/scenes/geometry.py` 按实际零件处理旋转盒体、竖直圆柱和动态家具父变换，检查房间外廓、墙体交集、地板/支撑板体积交叠。它允许接触与家具放在地毯上的合理叠放，不能代替机器人通行净空计算。

`src/sim2sense_fall/scenes/verification.py` 把 USD 与 manifest 逐项比较：实际图元路径、世界变换、形状尺寸、房间/语义、碰撞开关、刚体质量、视觉绑定及颜色、摩擦/恢复系数和电磁属性。旧有 344/231 固定计数不再是验证标准。六种“数量相同但内容错误”的资产变异都有真实 USD 测试。

## 当前复验结果

- 系统 Python：**70 passed，8 skipped**。跳过项需要 pxr；另外在本机 bundled USD 中执行了 **9 passed**（包含上述 8 项及一个共同的参数检查）。
- `python3 -m compileall src tests scripts`：通过。
- 本地已安装 `ruff check .`：通过。`uv tool run ruff check .` 因 snap/DBus 环境报 `Process 2 is a kernel thread, refusing.`，此启动路径未成功。
- Isaac CPU 回退构建成功；实际物理与清单验证 **10 项 PASS，退出 0**。椅子静止漂移显示 0.0000 m，抬高 0.25 m 后落回，残差显示 0.0000 m。
- 负例：旧 USD 搭配新 manifest，检查器准确定位地基、地毯、冰箱差异，日志 FAILED，退出 **1**。
- 实际 USD 世界几何审计：家具越界和家具/墙体交集均为空。排除合理地毯叠放后，当前轴对齐家具也未发现跨家具体积交集。
- 已检查 [修复后几何预览](../artifacts/remediation/usd_fixed.png) 和 [修复后平面图](../artifacts/remediation/usd_layout_fixed.png)，均来自更新后的 USD。

物理指标：每段仿真 3 s；静止/回落最大轴向位置误差 ≤0.02 m；0.25 m 抬高后的下降量至少 0.23 m。容差现可通过验证 CLI 配置；无效时长或失效的正向对照参数会提前失败。

## 使用

```bash
# CPU 规划与几何预检
python3 scripts/scenes/build.py --dry-run
# CPU 仅验证清单；明确不检查实际 USD 或物理
python3 scripts/scenes/verify.py --dry-run
# 重新生成，并验收实际资产
~/isaacsim/python.sh scripts/scenes/build.py --headless
~/isaacsim/python.sh scripts/scenes/verify.py
# 去顶俯视 / 去顶斜视
~/isaacsim/python.sh scripts/scenes/view.py --view top
~/isaacsim/python.sh scripts/scenes/view.py --view roofless
```

只想验证静态 USD 可使用 `--static-only`；日志明确标出 dynamics SKIP。默认验证要求真实动态正向对照，不能把零时长当作物理通过。

本环境无 GPU，但默认 DISPLAY 会触发阻塞错误弹窗；本轮运行 Isaac CPU 回退时仅对单次命令设置 `DISPLAY= WAYLAND_DISPLAY=`，避免弹窗。没有修改系统环境、驱动或 Isaac 安装。需要真实 GUI 时不要清空显示变量。

## 版本与证据

- 场景：`apartment_cn_two_bedroom`；seed=20260921；2.4 GHz；单一固定场景，无训练/测试划分、无学习模型。
- 代码基线：`a3f3e391c4c9aebb807da9cfd48d54bd0e7e9eab` 加当前未提交修复；源码摘要记录在 `artifacts/remediation/validation_results.json`。
- OpenUSD 0.25.11；本机 Isaac-Sim Python 6.0。
- 新 USD SHA-256：`38f062c3ca9d54e5ab8f2b2cc7428ce06cb85c7933d74530a3c3452958403b86`。
- 配置 SHA-256：`17e0df6ef116a603676def6de8939d20ec31c7814f75fb419d1ee6092dce9b06`。
- 清单 SHA-256：`f6880ca86a92967c59c4b6743e9b034ce51bdfa248794eddb19b55e43711424e`。

默认输出 `artifacts/scenes/indoor_apartment.usda` 已替换为经过验证的候选资产，其旧版本备份在 `artifacts/remediation/before/`。日志、几何坐标、复验摘要和预览保存在 `artifacts/remediation/`；这些生成产物不入 Git。

## 后续工作边界

1. **GPU/GUI 真实渲染待复验**：本环境 NVML 无法创建 GPU；提权 GPU 检查因自动审批服务 503 未执行。当前图片是 CPU 几何预览，不验证 RTX 灯光和玻璃效果。
2. **机器人可达性待建模**：入户门及卫生间门仍为固定关闭状态，需要先定义机器人尺寸、门状态和救援接近距离，再验证可达性。
3. **材质数值响应待标定**：已确认碰撞绑定与层叠几何，尚未用滑动实验标定不同材质摩擦响应；电磁映射也需和实际 Sionna 版本复核。
4. **人体、CSI/CIR 与域随机化属于下一阶段**：当前 seed 尚未驱动随机化。后续布局变体会先经过本轮新增的 CPU 几何校验。
