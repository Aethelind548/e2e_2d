# 验证接口修改与影响

> **建立前提：** 用户授权Git RC收尾与第二轮精简。
> **解决问题：** 验证接口修改与影响。
> **适用范围：** refactor/release-v1；基线6d7b709；2026-10-28。
> **不包含：** 训练恢复、GPU/真实数据/CARLA验收、修改冻结资产或推送合并。
> **当前状态：** 代码RC审查；外部集成另行验收。
> **相关代码：** tools/、agent/、tests/、tools/release/。
> **最后更新：** 2026-10-28。

| 原文件/接口 | 原因 | 新接口/行为 | 有效调用方 | 验证方式与结果 |
|---|---|---|---|---|
| tools/validation/b2d/full_route_preview_audit.py::sha256 | 与route_truncation_root_cause.sha256完全相同，已有依赖关系 | 删除重复函数体，直接导入公共sha256；公开名称不变 | snapshot、同模块审计/所有外部导入 | 37项小批CPU回归通过；新增snapshot验证内容hash与mtime/size，见克隆结果 |
| tests/agent/test_config_lifecycle.py::test_local_leaderboard_signatures_and_evaluator_order | 硬编码原工作区外的Leaderboard源码 | B2D_LEADERBOARD_SOURCE显式opt-in；未提供skip，提供错误路径/签名仍失败或明确不可用 | 可选外部源码一致性测试 | fake宿主桥接口测试仍执行；真实源未提供UNVERIFIED |
| tests/b2d_adapter/test_candidate_quality_diagnostic.py::test_official_anchor_file_reports_no_zero_displacement_prior | 基线克隆失败，默认依赖未发布Anchor | B2D_OFFICIAL_ANCHOR显式opt-in；保留官方SHA、shape和数值断言 | 可选原资产测试 | 新克隆默认skip；不是原资产PASS，指定不存在文件仍失败 |
| tests/release/test_cpu_model.py::test_official_v1_cpu_forward_backward_without_external_assets | 发布包需有不依赖原权重/数据的真实CPU模型检查 | 临时随机ResNet权重+合成Anchor；真实官方V1结构/Loss前后向；0 optimizer step | 长期CPU发布测试 | 小批回归PASS；不证明官方预训练/R2恢复 |
| .gitignore、README、快速启动、TRAINING_STATUS、tools/training/README、tools/b2d_adapter/README | 区分Git发布与本地历史；入口不依赖原codec目录 | 精确53项忽略、RC导航及真实已验命令 | 维护者/独立克隆 | index集合、命令与文档复核见清单 |

网络/Loss/Scheduler/Sampler与所有producer/正式训练恢复实现未改。公共hash采用已有模块、同样1MiB流式读取、返回同一hexdigest，没有削弱身份检查。

53项只从Git集合排除，不删除其函数；本轮未删除其他未知接口。旧status_route_eval公共main转发、prepared/paired/cache-only/recovery验证、相机兼容注册、M4规则兼容和Level2 opt-in保持原有路径。外部数据/GPU预检不为本次整理执行。
