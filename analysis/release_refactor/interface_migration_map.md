# 接口迁移与调用方影响

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 接口迁移与调用方影响。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 分批实施，未验证项目明确标注。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 当前映射

| 原文件/接口 | 新接口 | 调用方 | 保留的检查 | 当前结果 |
| --- | --- | --- | --- | --- |
| status_route_eval.legacy_main | status_route_eval_core.main（公开main原已使用） | CLI/python模块main | 现有experiment-config、LABEL=PATH、评估指标与数据合同 | 公开入口委托与help验证通过；未读取Checkpoint或数据 |

未更改任何生产producer、网络、Loss、Scheduler、Sampler或Checkpoint恢复校验。所有配置pin保持原样。
