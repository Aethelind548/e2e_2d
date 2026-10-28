# 移除与归档代码记录

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 移除与归档代码记录。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 分批实施，未验证项目明确标注。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 批次一

| 原路径 | 原接口 | 原因 | 替代 | 调用方影响 | 验证 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| tools/training/diffusiondrive_v1/status_route_eval.py | legacy_main及旧依赖导入 | 公开main早已委托新评估器；rg仅命中本文件定义/说明，未发现有效调用 | status_route_eval_core.main | 公开main及当前CLI行为保持；旧route-only参数原本已不被main接受 | 既有test_status_route_eval_core追加入口委托/真实help合同测试 | 9项通过；原代码在4defa3c与730文件源码归档中保留 |

没有删除任何整个历史研究目录、长期测试、训练配置或模型资源。其他相似检查器存在producer/恢复/研究合同，不因重名删除。
