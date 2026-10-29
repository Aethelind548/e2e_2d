# SHA 固定运行资产

> **建立前提：** Git 迁移需要携带小型固定资产，并明确其余 SHA 文件的获取边界。
> **解决问题：** 提供原样 Anchor 和必要源码、配置、外部资产的 SHA 索引。
> **适用范围：** manifest.json 指定的发布提交、Baseline/R1/R2 与 Agent 配置；本地 R3 不在本次上传范围。
> **不包含：** 数据集、骨干权重、Feature/Target、SQLite/JSONL 索引、训练断点及机器注册实体。
> **当前状态：** Anchor 与本地冻结预期一致；远程 GPU 训练和实际断点恢复仍未验收。
> **相关代码：** tools/release/resources.json、tools/training/diffusiondrive_v1/runtime.py、tools/training/diffusiondrive_v1/fit.py。
> **最后更新：** 2026-10-29。

| 内容 | 用途 |
| --- | --- |
| official_kmeans_navsim_traj_20.npy | 2,688 字节、float64、20×8×2；V1 轨迹先验，保持原始字节 |
| manifest.json | 必要源码和配置的原路径、现发布字节 SHA、原合同预期 SHA、外部资源及获取边界 |
| SHA256SUMS | 本目录三份内容文件的传输完整性清单，路径相对于项目根目录 |

唯一操作入口是[快速启动中的资产校验与还原](../快速启动.md#6-sha-固定资产校验与还原)。只还原 Anchor；不改训练配置的 plan_anchor_path，不覆盖已有不同内容，也不新增运行时回退。新目录与 requirements.txt 同级。

源码、冻结配置、补丁和许可证继续使用原 Git 路径，不再复制一套。官方 V1 由子模块提交及现有补丁重建；资料见[官方训练说明](https://github.com/hustvl/DiffusionDrive/blob/9b52ed0ec06b073d82d6f392ab084c7b301c8681/docs/train_eval.md)和[现有许可证](../tools/release/licenses/HUST_Vision_Lab.LICENSE)。本次验证 Anchor 与工程冻结指纹一致，没有重新生成聚类结果。

骨干权重、训练数据、缓存、索引、注册及实际 raw Checkpoint 仍按[外部资源 Manifest](../tools/release/resources.json)独立迁移。这里列出其要求，不声称已携带或通过真实载荷验收。位置模板不会自动重映射训练配置中的冻结路径。

manifest.json 的 release_sha256 是对发布字节的观察，不能替代原 expected_sha256。继承的 route-split 基础配置仍有 fit.py、lr_experiment.py、BF16 策略源码指纹差异，逐条保留为 match_at_release=false；不修改旧值、不绕过检查。R2 独立入口的恢复合同需要结合实际断点另行验收，不能把继承字段的差异直接当作 R2 已恢复成功或必然失败。

合同是否放松留待后续讨论。本次没有改变模型、Loss、Scheduler、Sampler、冻结配置或恢复校验。
