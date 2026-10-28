# Git发布检查清单

> **建立前提：** 用户授权Git RC收尾与第二轮精简。
> **解决问题：** Git发布检查清单。
> **适用范围：** refactor/release-v1；基线6d7b709；2026-10-28。
> **不包含：** 训练恢复、GPU/真实数据/CARLA验收、修改冻结资产或推送合并。
> **当前状态：** 代码RC审查；外部集成另行验收。
> **相关代码：** tools/、agent/、tests/、tools/release/。
> **最后更新：** 2026-10-28。

| 检查 | 结果 | 证据/限制 |
|---|---|---|
| 分支/基线 | PASS | refactor/release-v1；基线6d7b709；代码RC 67f95ea；main仍4defa3c |
| 修改前安全状态 | PASS | 第一方干净、已有提交/源码tar/Git bundle；官方子仓库已知补丁保留 |
| 发布集合 | PASS | 精确53项索引排除、原件SHA一致；tests长期路径保留；没有批量git add -A |
| .gitignore/未跟踪必需内容 | PASS（代码范围） | 根树无outputs/数据/Checkpoint/SQLite/数组载荷；独立克隆227源码+31 JSON+接口测试通过 |
| 官方V1提交/可达性 | PASS | GitHub ls-remote与子模块下载成功；固定9b52ed0；dirty原件未复制 |
| 上游既有补丁 | PASS | 干净V1/V2提交apply --check+实际apply，补丁SHA与范围一致 |
| 第三方许可证 | PASS（所载源码） | MIT全文及来源提交保留；nuplan等依赖仍按其上游许可证 |
| 第一方公开分发许可证 | 待所有者决定 | 未擅自给用户源码授予许可证；代码审查不等于公开授权完成 |
| 发布分支/主分支历史大文件 | PASS | 两分支可达历史最大3,368,692字节，无>5MiB对象 |
| 全部内部refs大对象 | 注意 | refs/codex工具快照含3个>100MiB资产对象；不在main/发布分支历史，不属于普通分支push；不得使用--mirror/推送内部refs；未改写或删除 |
| 敏感模式扫描 | 未检出 | 扫根仓库可达小型blob的私钥/GitHub/OpenAI/AWS特征；发布分支所有blob均在扫描大小范围；不是全面秘密不存在的证明 |
| 配置绝对路径 | 已披露，部署仍阻塞 | 冻结配置/原producer身份保持原样，资源Manifest描述外部资产；新机器须独立位置/合同验收，不能改旧SHA |
| CPU发布回归 | PASS | 全新克隆702 passed；28 skip/17 xfail不计为通过 |
| 真实环境完整安装 | UNVERIFIED | 本轮复用已安装软件依赖；requirements未在空Ubuntu完整安装 |
| 原资产保护 | PASS（可见范围） | 53本地原件逐SHA核验；35项已固定源码身份一致；3438项原analysis/outputs元数据无差异 |
| 实时实验状态 | UNVERIFIED | R2外盘/控制在本命名空间不可见；未启动/停止训练，无GPU/CARLA操作 |
| merge/push/history rewrite | 未执行 | root无远程；main未动；内部refs保留 |

内部大对象只查对象类型、路径与体积，未读取Checkpoint/SQLite载荷或对大型数据全量哈希。模式扫描仅适用受大小限制的小型文本范围。机器结果见本地history_scopes.json、static_checks.json、asset_metadata_check.json、clone_validation.json；无需将数值研究资产加入Git。

准予准备审查RC；完整服务器恢复/Agent部署不能放行。最终人工验收后再单独决定远程目标与公开许可、合并/发布范围；本轮到此停止。
