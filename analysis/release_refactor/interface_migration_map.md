# 接口迁移与调用方影响

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 接口迁移与调用方影响。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 受限发布审查完成；真实恢复/部署仍未验收。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 修改映射

| 原路径 | 原函数/接口 | 原因 | 新接口/替代 | 影响调用方 | 验证 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| tools/training/diffusiondrive_v1/status_route_eval.py | legacy_main及内部mean/旧导入 | 公开main已委托core；无有效调用 | status_route_eval_core.main；公开main保持 | 当前CLI；旧历史参数不属于现CLI | 9项评估/入口回归 | 移除未选用旧实现，原件保留 |
| tools/validation/b2d_adapter/audit_v1_target_rejections.py | interface | 固定输出使恢复验证临时覆盖基线 | keyword-only output_path；显式目标存在则拒绝 | 恢复验证main、默认历史调用 | 报告隔离/拒绝覆盖/默认兼容3项 | 必要前向/有限值检查不变 |
| tools/validation/b2d_adapter/check_v1_recovered_target_interfaces.py | main报告写入段 | 删除写基线后恢复的副作用 | interface(...,output_path=recovered_report) | 历史恢复验证CLI | 同一报告隔离回归与静态引用核对 | 不再临时覆盖冻结基线 |
| tools/validation/b2d_adapter/test_target_retry_audit.py | 新增InterfaceReportIsolation | 验证真实文件输出隔离，不替代模型验收 | 既有长期回归扩展3项 | pytest/unittest | 8项本文件测试通过 | 临时目录/stub只用于报告路由 |
| tools/validation/b2d_adapter/test_native_builders.py | 模块导入、test_version_hash_and_no_certification_override | 旧ROOT模块缺失且未使用；旧测试误以为仅2策略 | 删除无效导入；遍历现有3策略检查SHA各异/均未认证 | 原测试路径与opt-in类 | NativeConfig测试通过，真实项明确skip | 未改变生产策略 |
| tools/validation/b2d_adapter/test_engineering_builders.py | 模块导入 | 不存在的ROOT导入未被使用 | 删除无效导入，无替代代码 | 原测试路径 | 配置单测与收集通过 | 真实工程项保持原opt-in |
| tools/validation/b2d_adapter/test_level2.py | 模块导入、setUpClass、引用validate_* | 未启用的历史集成不应阻断CPU收集 | 启用时import_module并绑定原接口 | ActualBuilderTests | 默认明确skip；显式启用缺实现仍失败 | 历史验收仍UNVERIFIED，未伪造替代 |
| tests/training/test_status_route_eval_core.py | 新增公开main与help回归 | 保护当前CLI委托和参数契约 | 保留原指标测试并扩展2项 | pytest | 9项通过 | 不读取模型或数据 |
| tools/release/preflight.py | small_text_sha256 / inspect_profile / main | 建立长期只读发布资源检查职责 | 独立工具；不接管训练或改冻结配置 | 部署检查CLI、tests/release | 8项回归、3种profile真实运行 | 缺资源/继承pin/未完成Agent明确BLOCKED或UNVERIFIED |
| tools/release/__init__.py | 包入口 | 独立发布工具职责 | tools.release包 | python -m tools.release.preflight | 语法/导入通过 | 不导入模型 |
| tests/release/test_preflight.py | 8个长期回归 | 防止误读大型资产、源码身份不符与越界来源 | 新发布检查套件 | pytest | 8项通过，纳入最终701项 | 断点载荷不被读取 |

## 合同不迁移

fit.py、lr_experiment.py、status_experimental_train.py、全部Adapter/producer源码与旧训练JSON保持原字节。没有统一改写SHA/shape/标签/几何/挂载/Checkpoint校验。R2配置继承的fit源pin与当前不符；10月24日独立副本和source_snapshot_files.jsonl只对应当前版本，未找到旧pin实现；实际R2 launch/raw断点不可见，维持未验证。

Baseline入口fit、数据Baseline-only、raw完整恢复合同，与R2专家Status/EMA/BEV实验严格分列。新资源Manifest只描述外部资产；--location和其他位置参数只用于预检，不影响实际reader或旧配置，不能作为迁移完成证明。
