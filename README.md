# Bench2Drive → DiffusionDrive V1

> **建立前提：** 工程进入官方V1训练与独立Agent开发阶段。
> **解决问题：** 提供项目主线、目录职责与少量核心入口。
> **适用范围：** 所有进入本工程的开发者。
> **不包含：** 详细设计和重复启动命令。
> **当前状态：** 当前维护。
> **相关代码：** tools/、agent/、tests/
> **最后更新：** 2026-10-25（当前 Route split v2 正式实验状态入口；运行状态由 `TRAINING_STATUS.md` 维护）。

当前主线：Bench2Drive V0.0.4 的 Camera/LiDAR/Status 缓存与原版七类 Targets 配对，训练官方 DiffusionDrive V1 Baseline；CARLA Agent 是独立的未完成部署分支。V1 官方网络的类名就是 `V2TransfuserModel`；独立的 DiffusionDriveV2 RL/SEL 不参与本次训练。

**从 [快速启动](快速启动.md) 开始**。Status 离线专家路线实验与原 Baseline/Scratch 保留各自的数据和恢复合同；当前配置、检查与运行状态见快速启动及实际实验目录，历史 PID、步数和旧 `control.json` 不作为实时状态。设计、验收和效果限制见 [训练主文档](tools/training/README.md)。

| 要做什么 | 唯一入口 |
|---|---|
| 当前训练状态与边界 | [TRAINING_STATUS.md](TRAINING_STATUS.md) |
| 当前状态、环境、磁盘检查、暂停、评估与 CPU 回归 | [快速启动.md](快速启动.md) |
| 训练合同、配置状态索引、模块职责索引与历史实验 | [训练主文档](tools/training/README.md) |
| 当前运行 / 数据管线 / 历史验证 / 冻结证据索引 | [工具与维护导航](tools/README.md) |
| 数据读取、坐标、Camera/LiDAR 与监督 | [Adapter](tools/b2d_adapter/README.md) |
| 模型、架构与迁移历史 | [架构导航](analysis/diffusiondrive_b2d_architecture/README.md) |
| Agent 架构、接口与实现边界 | [Agent](agent/README.md) · [长期回归](docs/agent_test_report.md) |
| 活跃问题、监控项与已解决问题 | [问题记录](docs/问题记录.md) |
| 报告图片 | [Camera / LiDAR](analysis/diffusiondrive_b2d_architecture/v1_validation/report_sensor_examples/README.md) |
| 文件维护规则 | [AGENTS.md](AGENTS.md) |
| 代码版本、删除清单与历史恢复 | [开发留痕](outputs/project_history/README.md) |

目录职责：`tools/` 为 Adapter、训练与验证代码/配置；`agent/` 为部署框架；`tests/` 为长期 Agent 回归，历史 Adapter 回归保留在 `tools/validation/`；`docs/` 为长期设计与问题记录；`analysis/` 为既有研究及冻结验收证据；`outputs/` 为运行产物；`repositories/` 为独立官方仓库，不做目录美化迁移。

**当前/冻结边界：** `analysis/**/v1_validation/`、`analysis/data_interface_analysis/`、历史配置/Checkpoint/Manifest/SQLite 和 `outputs/diffusiondrive_v1/assets/` 是冻结证据或运行资源，不按“旧文件”批量删除或改写。配置状态与模块职责见训练主文档；历史实验只作为复现依据，不是当前操作入口。
