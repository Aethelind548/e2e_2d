# 发布回归与未验证项目

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 发布回归与未验证项目。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 受限发布审查完成；真实恢复/部署仍未验收。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 实际结果

| 检查 | 结果 | 证据与边界 |
| --- | --- | --- |
| 原始全量基线 | 收集失败4项 | 缺laspy、两处未使用缺失ROOT模块、历史validate_level2实现缺失；baseline_tests.log/XML |
| 批次一评估入口 | 9 passed | batch1_tests.log/XML；保留原数值指标并测CLI委托/help |
| 批次二报告/旧测试 | 12 passed、23 skipped | batch2_tests.log/XML；包括报告路由stub检查，不当作真实模型验收 |
| 最终完整CPU收集 | 701 passed、26 skipped、17 xfailed、1 deselected | final_tests.log/XML；原有真实集成关闭、Agent pending保持xfail |
| 2-worker DataLoader独立复测 | 1 passed（解除沙箱本地IPC限制后） | worker_unrestricted_tests.log/XML；受限环境PermissionError在worker_tests.log保留 |
| 语法 | 268份通过 | AST检查，不写pyc、不修改源码 |
| 核心模块导入 | 12/12通过 | import_config_checks.json；仅类定义/接口导入，没有启动模型/训练 |
| JSON配置加载 | 41份通过 | 同上，包括新资源描述；不代表数据/Checkpoint合同成立 |
| 官方V1身份 | 12项检查通过，只有原有scheduler兼容补丁 | official_v1_audit.frozen_sources；没有修改网络/损失源码 |
| 实际官方模型前向/反向 | CPU合成单batch通过；546个梯度张量有限 | model_cpu_smoke.json/log；原版V1架构与Loss，0 optimizer step，不是实数据/R2验收 |
| 恢复协议单测 | 最终套件中通过 | 既有累积边界、状态/EMA、resume目录/显式SHA合同回归；不等于实际R2断点恢复 |
| 实际R2/历史Baseline完整断点恢复 | UNVERIFIED | 当前R2 raw checkpoint/launch不可见，旧fit源码pin不匹配；未用其他历史断点冒充 |
| GPU/CARLA/全量数据/新Ubuntu环境安装 | UNVERIFIED | 未运行GPU或CARLA；没有启动训练/大规模预处理、下载或安装依赖 |
| 发布预检 | 真实返回BLOCKED/UNVERIFIED（exit2） | baseline/r2/agent_preflight.json；缺资源/旧pin/未完成Agent均如实展示 |

## 环境与复现

模型Python3.9.23、Torch2.7.1+cu128、Lightning2.2.1；CUDA_VISIBLE_DEVICES为空。主环境缺codec，本轮测试进程只在sys.path末尾追加已有LAZ环境的site-packages（laspy2.6.1、lazrs0.7.0），未拼接替换模型numpy、未安装包。pytest禁止cacheprovider、PYTHONDONTWRITEBYTECODE=1，结果只写outputs，临时fixture在/tmp。2-worker错误来自沙箱禁止本地multiprocessing套接字，解除限制仅重跑该测试通过，未放松生产检查。

验证过的帮助/只读/回归命令维护在快速启动.md；真实集成skip、Agent xfail、实际恢复/GPU/闭环未验证分别统计，不合并为通过。

## 最终静态、发布与保护检查

发布选择范围335个路径，Python AST检查268份通过；第一方绝对静态import悬空0处，历史Level2动态缺失依赖单独保留为未验证阻塞。legacy_main当前源码不再有调用；旧ROOT导入已移除。高置信私钥/Token/URL凭据模式0处；66份文件仍含历史本机路径，涉及431行，这些是冻结合同/历史文档的迁移风险，未静默脱敏改写SHA。扫描覆盖全部拟发布文本，非任意格式口令保证。

全部普通Git文件小于5 MiB，最大为冻结训练配置约3.37 MB；无数据/Checkpoint/数组/SQLite/环境目录。官方来源作为gitlink登记；原MIT原文与两个已有Python补丁随包，补丁reverse-check通过。原outputs/analysis资产元数据、官方仓库与producer/训练源码保护检查均无变化，详情在protected_checks.json。
