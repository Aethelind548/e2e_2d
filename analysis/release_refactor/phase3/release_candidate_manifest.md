# Release Candidate发布清单

> **建立前提：** 用户授权Git RC收尾与第二轮精简。
> **解决问题：** Release Candidate发布清单。
> **适用范围：** refactor/release-v1；基线6d7b709；2026-10-28。
> **不包含：** 训练恢复、GPU/真实数据/CARLA验收、修改冻结资产或推送合并。
> **当前状态：** 代码RC审查；外部集成另行验收。
> **相关代码：** tools/、agent/、tests/、tools/release/。
> **最后更新：** 2026-10-28。

基线树：335普通文件、31,341,138字节，另1个官方V1 gitlink。53项排除共7,147,709字节（6.817MiB）；原件仍在磁盘，Git历史未缩小。净文件/字节统计以本轮最终Git树差异为准，见RC摘要。

发布包含：第一方Adapter、Baseline/R2独立训练与恢复/评估、Agent未完成边界、长期tests及tools/validation/b2d_adapter/test_*.py、必要CLI/冻结来源合同、requirements、资源Manifest、已有Python补丁、MIT全文、导航与发布审查。R1父版本和R2配置、scratch及仍存在合同关联的旧配置保留，避免静默切断恢复/来源链。

排除项的完整逐路径清单见second_cleanup_review.md。数据/Feature/Target缓存、Checkpoint、日志、LAZ环境、第三方V2工作树和非部署研究证据均留本地、排除普通Git。outputs/diffusiondrive_v1/assets含运行资源，不能删除。43项外部资产继续由tools/release/resources.json描述。

官方V1 gitlink固定9b52ed0ec06b073d82d6f392ab084c7b301c8681，GitHub URL在.gitmodules；已有scheduler补丁应用后才是当前验证工作树。历史V2提交1cd12a1e155c34dcc471261835444c8d5587580b仅来源/补丁记录，非运行子模块。两者保留MIT全文；nuplan等依赖按其来源许可证分发。第一方公开授权许可证尚未由所有者选定，不擅自授予MIT。

发布集合检查不采用git add -A；历史文件只从索引排除、保持原路径/SHA。临时克隆与venv位于/tmp；机器证据位于outputs/release_refactor/phase3且不发布。
