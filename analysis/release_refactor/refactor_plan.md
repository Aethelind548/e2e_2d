# 发布重构计划与安全基线

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 发布重构计划与安全基线。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 分批实施，未验证项目明确标注。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 安全基线

根仓库原为main未提交，现有工作先归档到outputs/project_history/release_v1_baseline_20261028.tar.gz，对730份来源文件逐项核验SHA；配套JSON保存清单。该备份覆盖本轮可能修改的源码/配置/文档与已有补丁，不宣称已备份全部数据/模型资产。318份第一方源码/配置/测试/文档已建立初始提交4defa3c，随后创建refactor/release-v1。Git作者使用明确的Release Automation本地身份，未设置全局用户或远程。根Git写入需沙箱提权，已按任务授权完成。

外盘R2 output/training_progress/last.ckpt及/tmp监督脚本当前不可见，可见进程中没有对应入口；这不认证宿主无训练。因此不修改任何现有生产/训练源码、实验JSON、producer、注册、索引、权重、官方仓库及原补丁。复用八份workspace_audit，不重复递归扫描全资产。

## 批次与验收

1. 先完成函数级静态调用、动态入口、CLI/配置与冻结身份影响分析；结果见deprecated_interface_inventory.md和outputs/release_refactor/function_references.json。
2. 移除公开Status评估包装器中未选用的legacy_main；原实现保存在基线提交和独立源码备份，当前CLI继续委托status_route_eval_core.main。旧文档命令明确标为历史，不假装兼容旧参数。
3. 为历史interface验证增加显式输出路径，迁移恢复检查调用，去掉临时覆盖/恢复基线报告的写法；补既有长期回归。
4. 按代码/资源分层准备发布边界、V1固定提交与补丁、外部资源Manifest、受限依赖说明和只读预检；保留全部历史研究资产，不做没有证据的批量删除。
5. 分批运行静态、导入、配置、CPU数据合同、Agent和恢复协议回归；真实模型/GPU/完整Checkpoint按环境与显式资产可用性单独判定，不以合成结果代替。
6. 每批提供独立提交与diff；最终只给合并建议，不合并main或推送。

## pin问题处理边界

base配置pin的fit.py与lr_experiment.py与当前文件不同；10月24日发布副本及source_snapshot_files.jsonl均对应当前版本，不能提供匹配的原始实现。R2实际外盘Checkpoint/launch不可见，故恢复身份保持阻塞。严禁更新旧pin或放松校验解决本问题；本轮不修改这两个文件。
