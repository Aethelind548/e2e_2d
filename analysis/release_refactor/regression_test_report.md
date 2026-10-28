# 发布回归与未验证项目

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 发布回归与未验证项目。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 分批实施，未验证项目明确标注。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 基线与批次一

原始全量CPU套件在收集阶段失败：test_adapter缺主环境laspy；test_engineering_builders/test_native_builders导入已不存在的validate_native_builders.ROOT（两个ROOT导入均未实际使用）；test_level2依赖已不存在的validate_level2。均为修改前已存在的问题。证据outputs/release_refactor/baseline_tests.log及XML，不能算全量通过。

批次一：test_status_route_eval_core.py为9 passed，含原有指标与新增公开入口/help回归。模型/GPU/完整Checkpoint/外盘资产仍未验证。
