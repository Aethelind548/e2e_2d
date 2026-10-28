# Release Candidate审查结论

> **建立前提：** 用户授权Git RC收尾与第二轮精简。
> **解决问题：** Release Candidate审查结论。
> **适用范围：** refactor/release-v1；基线6d7b709；2026-10-28。
> **不包含：** 训练恢复、GPU/真实数据/CARLA验收、修改冻结资产或推送合并。
> **当前状态：** 代码RC审查；外部集成另行验收。
> **相关代码：** tools/、agent/、tests/、tools/release/。
> **最后更新：** 2026-10-28。

**A. 结论：满足本轮“可供代码审查的Release Candidate”标准；不认定正式远程训练恢复或Agent闭环部署已就绪。** 分支refactor/release-v1，代码提交`67f95ea`；完整CPU新克隆验证通过。公开分发许可证及完整空Ubuntu依赖安装仍须独立确认，不自动合并/推送。

**B. 精简量：** 53项历史文件从Git发布集合排除，合计7,147,709字节（6.817MiB）；全部原件留在本地，独立tar逐SHA核验。新增7文件（6交付报告+1长期CPU测试），因此净减少46份普通文件。净字节与最终树统计见下方；这是发布树体积减少，不是删除本地资产或缩小Git历史。

**C. 仍保留的历史接口：** prepared/paired/cache-only/recovery及其身份验证、旧M4/相机兼容注册、恢复规则与Feature/Target队列迁移、被长期测试调用的Route/Status审计函数、status_route_eval公开CLI转发、Level2 opt-in测试。保留原因分别是当前间接引用、Checkpoint/producer合同、正式回归与来源链，或缺少可证明等价替代；不凭文件名删除。full Status预检含历史样本数量/版本约束，保留并不表示它已适用于R2。R1父版本和R2独立配置继续保留，模型/权重合同未混用。

**D. 新克隆可独立完成无外部数据CPU测试。** 从已提交代码全新克隆、GitHub初始化固定子模块、应用已有补丁后，702 passed、28 skipped、17 xfailed，包含真实官方V1合成CPU前后向及2-worker测试。使用已有模型软件依赖+临时venv，不携带原outputs/权重/数据；因此不是全新操作系统依赖安装验收。227 Python、31 JSON、18核心/官方导入、12帮助入口通过。两个真实资产检查为显式opt-in，缺失记UNVERIFIED。

**E. 正式远程恢复尚缺：** 目标Ubuntu/GPU驱动及完整依赖安装；43项外部资源、地图、codec、Feature/Target载荷/索引/SQLite及注册owner/UUID；实际R2 raw Checkpoint与launch/progress/源码合同；旧fit/lr预期源码pin对应原实现定位或独立迁移合同；GPU/真实数据恢复一致性；Agent真实在线特征、模型初始化/断点绑定与CARLA闭环。以上均不靠修改冻结JSON或放宽SHA解决。

本轮唯一公共实现收敛为完全相同的sha256重复函数体，沿用已有模块及公开名字；未改模型、Loss、Scheduler或Sampler。原3438项审计/输出资产元数据一致，35项源码身份核验一致。第一方干净可交付；官方原仓库保留既有补丁，因此root可显示子模块dirty，不能用reset清除。

发布分支与main可达历史没有超大blob，最大3,368,692字节。另有内部refs/codex快照包含大资产，不在两分支历史；保留原状，只能发布审核过的普通分支，不能mirror整个.git。敏感模式未检出但不构成绝对保证。

交付六份报告位于本目录；完整逐项排除/保留依据见second_cleanup_review.md，补丁影响见interface_cleanup_diff.md，实际复现和skip边界见clean_clone_test_report.md。历史阶段报告不覆盖，新结论以本phase3样本边界为准。

## 最终发布树统计

基线335普通文件 / 31,341,138字节；当前289普通文件 / 24,279,377字节，另保留1个V1 gitlink。净减少46文件 / 7,061,761字节（6.735MiB，22.53%）。

本轮完成后停止，等待Release Candidate最终人工审核；未合并main、未推送GitHub、未改写Git历史。
