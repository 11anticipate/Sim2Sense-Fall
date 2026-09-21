# Sim2Sense-Fall 项目计划

## 目标

建立一个可复现、可扩展的室内无线信道跌倒检测研究仓库：以 Isaac Sim 生成受物理约束的人体运动与场景状态，以 Sionna RT 生成 CIR/CSI 等无线观测，再训练可解释、可评测的跌倒检测模型，并保留从仿真到真实数据的验证边界。

## 阶段

- [x] 阶段 1：检查项目目录、协作规范、Git 状态和本地 Zotero 文献
- [x] 阶段 2：核对核心文献，形成项目相关的证据记录和研究路线
- [x] 阶段 3：建立仓库目录、Python 工程配置、数据契约和最小代码骨架
- [x] 阶段 4：编写 README、架构说明、路线图和阶段进度文档
- [x] 阶段 5：搭建 Isaac Sim 室内场景（卧室/客厅/卫生间/厨房/次卧/走廊），配置墙体、地板、家具属性并导出 USD
- [x] 阶段 6：同步 `git@github.com:11anticipate/Sim2Sense-Fall.git` 并核对上游内容
- [ ] 阶段 7：接入 Sionna RT，完成首个端到端 smoke test（场景 → CIR/CSI 样本）
- [ ] 阶段 8：用场景 `seed` 驱动房间布局/材质/家具随机化，形成训练域族

## 关键问题

1. 仿真人体轨迹、动态网格与无线信道样本如何用统一时间戳对齐？
2. 如何区分跌倒检测能力与场景、硬件、个体的域偏移？
3. 哪些指标能够覆盖漏报、误报、报警延迟和仿真到真实的性能变化？
4. 生成数据和真实采集数据如何在隐私、许可和可复现性上分开管理？

## 已做决定

- 本地 Zotero 和随附 PDF 是当前文献事实源；网络元数据在网络恢复前不作为已核实事实。
- 代码采用 `src/` 布局、类型标注、`ruff`、`pytest` 和配置文件驱动。
- 原始数据、生成数据、模型权重和渲染缓存默认不入 Git；仓库只保留小型示例、元数据和生成说明。
- 先实现稳定的数据契约与 dry-run，再接入重量级 Isaac Sim / Sionna 运行时。
- 场景几何用参数化图元拼装，不依赖在线素材库；同一材质同时定义渲染、力学和电磁三套属性。
- 场景规划（CPU）与 USD 落地（Isaac Sim）分离：规划可脱离 Isaac Sim 单测和评审，USD 只消费规划结果。

## 错误与阻塞

- 2026-09-21：**已解除** — 早前 `github.com` DNS 解析失败；现在 `git ls-remote origin` 正常，
  `origin/main` 停在 `c77a37d`，与本地 `main` 完全一致（0 ahead / 0 behind）。
- 2026-09-21：**已解除** — 早前工作树 `.git` 是只读空目录；现在 `.git` 可写，存在 `main`
  分支与提交 `c77a37d`，且 `origin` 已指向目标仓库。场景工作已提交到
  `feature/indoor-scene` 分支，`main` 保持不动。详见 [`docs/progress.md`](docs/progress.md)。
- 2026-09-21：独立运行的 Isaac Sim 不会把 PhysX 挂到 USD stage 上，物理步数在涨但没有任何物体运动。已封装 `activate_physics()`（`enable_all_default_callbacks` + `setup_simulation`）修复，并加入「抬高后落回」的正向对照测试，避免把「物理没跑」误判成「场景稳定」。
- 2026-09-21：上游仓库**没有 LICENSE 文件**，默认即「保留所有权利」。在明确许可前不应假设
  代码可以对外分发或复用。
- 2026-09-21：**未做**推送。`feature/indoor-scene` 只存在于本地，是否推送与开 PR 待用户确认。

## 当前状态

**室内场景 R1–R6 整改完成，已通过 CPU、实际 USD 和物理复验**：`configs/scenes/indoor_apartment.yaml` 描述的 6 房间住宅可导出为
`artifacts/scenes/indoor_apartment.usda`（232 图元 / 231 碰撞体 / 14 材质），
当前验收器 10 项检查通过（CPU 回退），失败负例实际返回非零；详见 `docs/indoor-scene-remediation.md`。构建与 GUI 查看指令见 [`docs/indoor-scene.md`](docs/indoor-scene.md)。

## 室内场景子任务

- [x] 创建卧室、客厅、卫生间、厨房、次卧及走廊配置，设置静态碰撞、材质与摩擦属性。
- [x] 创建实际 Python 文件、CPU dry-run 和 USD 导出入口。
- [x] 运行验证并记录结果，提供本机 GUI 启动指令。

## 下一步

1. 用 `configs/scenes/indoor_apartment.yaml` 的 `seed` 驱动房间布局/材质随机化，形成训练域族。
2. 在场景中加入人体（刚体或骨架），导出与场景同时间基准的运动真值。
3. 接入 Sionna RT，把 `/World` 下的几何与 `sim2sense:em_*` 材质映射成传播场景，生成首条 `ChannelSample`。
4. 复核 `tile_floor`、`ceramic_sanitary`、`carpet`、`upholstery` 等代理电磁材质与实际
   Sionna 版本 `itu_*` 数值的一致性。

## 2026-09-21 室内 USD 批判性验收

- [x] 核对现有 USD、配置、查看入口和测试覆盖，复现内部不可见问题。
- [x] 检查布局、物理与参数边界，运行 CPU 检查和可用的 Isaac 验证。
- [x] 交付验收报告与可查看的内部证据，更新进度和研究笔记。

首次验收结论为需整改；现 R1–R6 已修复，查看入口提供临时去顶俯视/斜视；GPU/GUI 视觉待有显示环境复验。

上述整改已完成，原始缺陷证据保留在 `docs/indoor-scene-review.md`；当前修复与复验见 `docs/indoor-scene-remediation.md`。

## 验收整改 R1–R6

- [x] 修复运行入口错误退出码，增加进程级负例；让验证依据 manifest 而非固定计数。
- [x] 修复地基层叠与家具布局，增加 CPU 几何越界/穿墙检查。
- [x] 严格校验有限数值、派生尺寸和规范化路径，加入回归测试。
- [x] 重新构建 USD、执行 CPU/Isaac 验证、刷新预览及整改记录。

范围：先完成当前固定室内场景可靠性基础；人体、Sionna、门铰链和机器人导航仍为后续阶段。

## 场景代码目录整理

- [x] 将场景实现集中到 `src/sim2sense_fall/scenes/`，入口集中到 `scripts/scenes/`，测试集中到 `tests/scenes/`。
- [x] 同步导入、仓库路径定位及当前使用文档，保留历史验收记录的证据语义。
- [x] 执行 CPU、OpenUSD 和 Isaac 回归，并确认目录迁移没有改变场景几何。

目录整理已完成：CPU 70 passed / 8 skipped，bundled USD 9 passed，Isaac CPU 回退 10 项 PASS；新旧 USD 字节一致。当前入口统一见 `scripts/scenes/`，详细证据见 `docs/progress.md`。
