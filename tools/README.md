# Bench2Drive → DiffusionDrive V1 工具

> **建立前提：** 既有工程的接口、设计或有限验证记录；版本和样本以正文为准。
> **解决问题：** 说明Bench2Drive → DiffusionDrive V1 工具的结论、依据和适用边界。
> **适用范围：** 工具与数据转换。
> **不包含：** 重复维护根目录快速启动。
> **当前状态：** 当前维护。
> **相关代码：** tools/
> **最后更新：** 2026-10-24（代码固化与历史原件的集中归档入口）。

当前主线为原版 V1 完整监督式模仿学习。原始模型位于 `../repositories/DiffusionDrive`；`../repositories/DiffusionDriveV2` 完整保留，但不参与当前训练。原版 V1 中的 `V2TransfuserModel` 属于 V1。

操作入口只在 [快速启动](../快速启动.md) 维护；训练合同、配置索引和历史实验见 [训练主文档](training/README.md)；当前问题见 [问题记录](../docs/问题记录.md)；维护规则见 [AGENTS](../AGENTS.md)。

## 目录职责

| 目录 | 当前职责 | 是否当前训练入口 |
|---|---|---|
| `b2d_adapter/` | Route/Frame 读取、坐标与时间转换、Camera/LiDAR/Status、轨迹/检测/七类 BEV 监督 | 数据合同，不是启动入口 |
| `b2d_adapter/static_layers/` | M5 地图边界、RoadMark 与 Crosswalk 独立语义层 | 未接入正式基线 |
| `training/diffusiondrive_v1/` | 训练运行链、数据管线、冻结配置、历史验证/恢复工具 | 仅 `fit.py` 等当前运行链 |
| `validation/b2d_adapter/` | 共享几何/合同单测、V1 回归、历史检查与可视化 | 长期测试保留；`check_*`/`validate_*` 不是当前训练入口 |
| `validation/b2d/` | 继承的时钟、语义与 CARLA 几何来源验证；含在线认证链（`leaderboard_session_capture`、`navigation_semantics_check`、`session_continuous_audit`、`certified_overlay_v4`）与在线 Status 重放（`online_status_parity`） | 需独立明确运行 |

## 模块职责索引

### 当前训练运行链

`training/diffusiondrive_v1/` 下：

`fit.py`、`runtime.py`、`paired_dataset.py`、`cache_only_reader.py`、`single_pass_cache_reader.py`、`procfs_mount_reader.py`、`sampling.py`、`training_progress.py`、`training_health.py`、`lr_experiment.py`、`bf16_policy.py`、`evaluate_checkpoint.py`、`official_v1_audit.py`、`features_cache.py`、`target_cache.py`、`target_storage.py`。

这些文件承载当前正式训练、恢复、评估或运行时合同检查。修改前必须核对 `execution_source_sha256`、Checkpoint 合同和当前进程状态。

### 数据生成与恢复管线

`feature_jobs.py`、`feature_service.py`、`feature_migration.py`、`feature_storage.py`、`feature_shared_storage.py`、`target_jobs.py`、`target_service.py`、`target_recovery.py`、`target_retry_audit.py`、`recovery_paired.py`、`system_feature_relocation.py`。

这些是 Features/Targets 生成、迁移、审计和历史恢复工具，不是当前正式训练的启动入口。具体状态见 [训练主文档：模块职责索引](training/README.md#模块职责索引)。

### 历史验证/一次性入口

`check_labels.py`、`check_m1.py`、`preflight.py`、`prepare.py`、`train.py`、`baseline_setup.py`、`full_recovery_rules.py`、`gravity6_recovery.py`。

保留用于历史证据和复现；不要把它们当作当前训练命令。旧路径、旧 PID 和旧 control 只代表历史快照。

### 旧控制器

`training_control.py`、`training_monitor.py` 只管理历史 Batch8 控制状态。当前正式训练通过实际输出目录的 `training_progress.json` 和 `pause.request` 监控；旧 `control.json` 不对应当前正式进程。

### 长期测试

`../tests/agent/`、`../tests/training/`、`validation/b2d_adapter/test_*.py` 是长期回归，原路径保留。不要仅凭 `check`/`audit` 命名删除历史脚本；先检查导入、文档和证据引用。

## 配置与冻结边界

- 配置状态索引见 [训练主文档：配置索引](training/README.md#配置索引)。
- `analysis/**/v1_validation/` 是冻结证据区，包含按 SHA、路径、SQLite/Manifest 注册的机器数据；不按扩展名或目录名搬移。
- `camera_original_compatibility.json`、`m4_original_compatibility.json`、`v1_target_contract.json` 是历史 producer/相机/目标合同桥，不是可随手删除的“兼容垃圾”。
- `../repositories/`、`../outputs/diffusiondrive_v1/assets/`、`../analysis/bndp3_original_il/sources/` 是官方仓库或运行/来源资源，禁止目录美化迁移。
- 已替代文档及两份未引用配置的原件已压缩归档，独立的 `undo/` 工作目录已清理；原路径、SHA 与恢复说明见 [开发留痕](../outputs/project_history/README.md)。

## 维护分类（保留）

| 类别 | 处理 |
|---|---|
| 当前核心 | Adapter、官方 V1、fit/paired Dataset、冻结配置与索引，保持原路径 |
| 有效辅助 | Agent 框架、地图层、存储调度、可视化导出、独立源码审计，按职责保留 |
| 文档合并 | 当前操作入口只在快速启动；训练设计/历史/证据在训练主文档；问题统一记录 |
| 长期回归 | `tests/agent/` 与 `validation/b2d_adapter/test_*.py` 原路径保留 |
| 一次性验证 | 新临时脚本放 `/tmp`；不按任务轮次新建永久测试 |
| 运行结果 | 证据放 `outputs/` 或既有证据区；不写入源码目录 |
| 已替代 | 核对依赖后归档原件，删除记录与恢复依据集中保存在 `outputs/project_history/` |
| 无法确认/受保护 | 官方仓库、冻结证据、配置/缓存身份、历史迁移原件，保留并记入问题记录 |
