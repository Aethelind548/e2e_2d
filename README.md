# Bench2Drive → DiffusionDrive V1

> **建立前提：** 原版V1与独立Status实验需要可审阅的代码发布边界。
> **解决问题：** 导航到操作入口、数据合同、Agent边界和发布验收。
> **适用范围：** 第一方源码、长期回归与固定上游依赖。
> **不包含：** 数据/模型资源分发、历史实验重写或已完成闭环的声明。
> **当前状态：** 发布重构；服务器恢复与Agent闭环尚未验收。
> **相关代码：** tools/、agent/、tests/、repositories/DiffusionDrive/。
> **最后更新：** 2026-10-28。

模型主线为官方DiffusionDrive V1，网络类名仍是V2TransfuserModel。Baseline使用fit.py；R2使用独立Status入口，包含专家路线Status、EMA、参数组LR与BEV权重实验，保持各自数据/源码/Checkpoint合同。Agent真实模型初始化与在线特征仍未完成。

| 内容 | 入口 |
| --- | --- |
| 环境、实际验证过的CLI与CPU回归 | [快速启动](快速启动.md) |
| 训练状态记录及实时状态边界 | [TRAINING_STATUS.md](TRAINING_STATUS.md) |
| 训练/恢复合同与实验索引 | [训练主文档](tools/training/README.md) |
| Adapter输入、坐标与标签合同 | [Adapter](tools/b2d_adapter/README.md) |
| Agent接口与未完成能力 | [Agent](agent/README.md) · [长期测试说明](docs/agent_test_report.md) |
| 问题与复现 | [问题记录](docs/问题记录.md) |
| 本次发布范围与外部资产 | [发布清单](analysis/release_refactor/release_file_manifest.md) · [资源Manifest](tools/release/resources.json) |
| 验收结果与合并限制 | [发布就绪结论](analysis/release_refactor/release_readiness.md) |
| 当前代码RC与独立克隆验证 | [RC结论](analysis/release_refactor/phase3/release_candidate_summary.md) · [克隆测试](analysis/release_refactor/phase3/clean_clone_test_report.md) |
| 文件维护规则 | [AGENTS.md](AGENTS.md) |

官方V1按.gitmodules和Git树中的固定提交管理，已有scheduler兼容补丁独立保存在tools/release/patches/，不覆盖或重新克隆当前本地仓库。V2历史源码继续本地保留，来源提交及既有Python补丁记录在资源Manifest中。

研究文本、冻结analysis证据和outputs运行资产留在本地，不进入普通代码发布范围。outputs包含必要权重、地图、解码环境、注册与索引，排除Git不表示可删除。历史架构/来源/开发留痕仍保留原路径，其本地索引不作为远程代码包已携带的承诺。
