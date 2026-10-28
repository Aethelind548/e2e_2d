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

## 批次二

| 原文件/接口 | 新接口 | 原因/调用方 | 验证与结果 |
| --- | --- | --- | --- |
| audit_v1_target_rejections.interface(pairs,cfg,model) | 新增keyword-only output_path；显式目标已存在时拒绝，默认路径保持 | check_v1_recovered_target_interfaces.main原先先写基线再恢复，现直接写recovered报告 | 隔离临时目录验证baseline原字节不变、显式目标拒绝覆盖、默认路径兼容；网络用stub，仅验输出路由 |
| check_v1_recovered_target_interfaces.main | interface(...,output_path=recovered_report) | 移除基线read/write/restore副作用；必要原版网络/有限值检查不变 | 8项恢复/报告测试通过 |
| test_native_builders/test_engineering_builders模块导入 | 删除未使用的validate_native_builders.ROOT导入 | 该旧模块已不存在，ROOT在测试正文未使用 | 重新收集配置测试，原opt-in测试继续保留 |
| test_level2模块导入 | 启用历史Level2 suite时import_module加载，绑定类staticmethod | 历史依赖不存在，避免未启用时阻断所有CPU测试；显式启用仍报错，不伪造替代验证 | 默认套件明确skip；历史集成不能标通过 |

批次二首次验证：1 failed、11 passed、23 skipped。新增可收集的旧配置测试假定Camera policy只有两个，实际已有三个；改为遍历所有policy验证SHA各异且均未认证，随后重新验证。跳过为原有真实数据/Level2显式启用项，非本轮放松合同。

批次二修复后：12 passed、23 skipped。旧NativeConfig测试改为检查全部三个既有Camera policy的身份各异，未改生产配置或策略。全量套件首轮：692 passed、26 skipped、17 xfailed、2 failed；其中旧policy假设已修复，另一项为2-worker DataLoader队列/信号量错误，单独定位，不标通过。
