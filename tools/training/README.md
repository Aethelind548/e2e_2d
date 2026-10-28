# DiffusionDrive V1 训练与缓存

> **建立前提：** 官方V1 Baseline已配对并完成当前精度/读取验收。
> **解决问题：** 维护训练合同、实现边界、恢复机制与验收依据。
> **适用范围：** V1训练与缓存接口。
> **不包含：** 实时PID/吞吐快照及与快速启动重复的操作指南。
> **当前状态：** 当前维护。
> **相关代码：** tools/training/diffusiondrive_v1/
> **最后更新：** 2026-10-26（记录 armA→contA→cos50→cos50b 四阶段短程结果：余弦退火无效、确认过拟合；三灰 Feature 生成规则与解压容量入文；启动 R1-EMA-BEVLR 双变量验证）。

发布说明（2026-10-28）：下表同时记录本地历史配置；不表示所有配置随代码RC分发。53项历史工具/实验配置从Git集合排除、原路径保留；具体路径及来源提交见[二轮清单](../../analysis/release_refactor/phase3/second_cleanup_review.md)。保留R1/R2、恢复读取器与必要合同；旧Status subset预检已有full版本替代，历史结果不作为当前验收。

## 当前合同与启动入口

操作命令只在根目录 [快速启动.md](../../快速启动.md) 维护；本文负责设计、合同、验收和限制。所有命令从工程根目录运行。

- 官方仓库 `repositories/DiffusionDrive` 固定参考 `9b52ed0ec06b073d82d6f392ab084c7b301c8681`，使用官方 V1 `V2TransfuserModel`、`TransfuserAgent`、`transfuser_loss`、Lightning training_step 与 get_optimizers。
- 官方 AdamW、WarmupCosLR、ResNet34 和 20×8×2 Anchor 保持原机制；模型与 Loss 源文件不因缓存/精度适配而改变。既有 scheduler 的 verbose API 兼容补丁保留。
- Baseline-only 配对 218622 帧：训练 **210776**、Town05 验证 **7846**，RouteGroup 隔离。恢复/扩展 producer 和 gravity6 不混入当前数据。
- 当前独立实验为 [baseline_bs8_acc8_lr1e4_scratch.json](diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json)：micro8×累积8、8 Worker、prefetch2、100 Epoch/329400更新、LR1e-4；不含 `warm_start_source`，从官方 ResNet34 ImageNet 图像编码器和固定 Anchor 初始化，其他模块随机初始化。历史 [baseline_bs8_acc8_lr1e4_formal.json](diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal.json) 是 Best26347 Warm Start 冻结父配置，不能当作 Scratch。
- **当前运行状态：** 旧 Warm Start 恢复实例已在 step 6538 合作暂停，输出 `/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_formal_resume_wrapper_20261021_194833`，完整暂停断点为 `pause-step000006538.ckpt`；不要覆盖该目录。Scratch 当前输出为 `/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_scratch_20261021_210751`，PID/命令/初始化审计见该目录 `launch.json`、`initialization_audit.json`。父运行最新完整断点仍为 step 2250。P007 的累积边界保存缺陷已修复；旧断点使用 [baseline_bs8_acc8_lr1e4_formal_resume_fix.json](diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal_resume_fix.json) 和 `--allow-checkpoint-boundary-fix` 显式迁移，Scratch 新断点不再需要该标志，见 [问题记录 P007](../../docs/问题记录.md#p007-累积-checkpoint-边界与恢复阻断)。Scratch 当前按 `training_progress.json` 为 `paused / step 58650 / epoch 18（零基 17）`，暂停断点 `pause-step000058650.ckpt`；本轮只读归因没有恢复或改变该状态。
- 历史 [BF16 control.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/bf16_training_20261020/control.json)不管理新正式进程。新分支通过实际输出目录的`training_progress.json`及`pause.request`查看/合作暂停；配置/producer/Manifest/随机种子冻结，不在读取期间迁移缓存。

## 配置索引

状态含义：`FROZEN-PARENT` 是冻结父合同，不能编辑；`CURRENT-CONTRACT` 是旧 Warm Start 断点恢复合同；`CURRENT-SCRATCH` 是历史从头训练合同；`CURRENT-STATUS-V2` 是当前 Route split v2 正式实验；`HISTORICAL` 只用于复现；`DATA-PIPELINE` 是数据生成/存储状态；`DESIGN-ONLY` 不能直接运行；`ARCHIVED` 已移入 `undo/`，可按归档记录原样恢复；`UNUSED-CANDIDATE` 尚未发现运行/恢复引用，归档前仍需核对 PSSD 元数据。

| 配置 | 状态 | 用途/边界 |
|---|---|---|
| `baseline_bs8_acc8_lr1e4_formal.json` | `FROZEN-PARENT` | 冻结父配置；micro8×累积8、LR1e-4、100 Epoch/329400 更新；SHA `42ac6c65401f90d91e9570b4633b6387584aeaf743ceb093e9fd7ab9c6bac172` |
| `baseline_bs8_acc8_lr1e4_formal_resume_fix.json` | `CURRENT-CONTRACT` | 旧 Warm Start 断点恢复合同；仅更新 `fit.py` 与验证源码 SHA，数值/数据/预算不变；SHA `248b4a7202a5adf7853a626768d5b7e03181e4e49b2c9959362da45ebdb62197` |
| `baseline_bs8_acc8_lr1e4_scratch.json` | `HISTORICAL-SCRATCH` | 历史从头训练合同；移除 `warm_start_source`，保留官方图像预训练与随机初始化；SHA `5b5ab97000a0450cdb47d5da87007e592da5c91fcf89342cb451ddd69bbb66f3` |
| `baseline_bs8_acc8_lr6e4_official_route_split.json` | `CURRENT-STATUS-V2-BASE` | 当前 Status v2 的 base config；lr 6e-4、seed 0、五来源 paired manifest、historical28 registration、无旧 `data`/warm-start；SHA `c4e17981c3538e05a55d76624e2095e902bd6a1fc15b44650da90f944a7ba431` |
| `experiments/status_routepreview_v2_split801010_seed0_20261025_012809.json` | `CURRENT-STATUS-V2-PAUSED` | 当前正式实验；step 43,650 已暂停；100 epochs、full validation、val/loss Checkpoint、micro16×accum4、BF16 工程策略；Test 冻结不加载 |
| `experiments/status_agentbox_shortA_lr3e4_lrscale_clip1000.json` | `AGENTBOX-SHORT-A` | Agent box 修复短程眂 A；5 epoch、best5008 仅权重、lr_scale 生效、base 3e-4、clip 1000 |
| `experiments/status_agentbox_shortB_lr6e4_lrscale_clip1000.json` | `AGENTBOX-SHORT-B` | 短程眂 B；同 A 但 base 6e-4，分离 lr_scale 与降峰值 |
| `experiments/status_agentbox_shortC_lr3e4_flatlr_clip1000.json` | `AGENTBOX-SHORT-C` | 短程诊断臂 C；base 3e-4 且 `paramwise_lr_scale=false`，复现抹平行为，不晋级长训 |
| `experiments/status_agentbox_contA_lr3e4_lrscale_clip1000_20261025.json` | `AGENTBOX-CONT-A` | 臂 A 续训；从 step 12,520 last.ckpt full-resume 到 25 epoch，`--resume-sha256` 固定 `601f6699…`；已完成（state=completed，best 仍为 ep10） |
| `experiments/status_agentbox_paperval_lr3e4_clip1000_20261026.json` | `AGENTBOX-PAPERVAL` | v4 入口；验证也计算训练口径目标（2 步 focal+L1），monitor 改为 `val/paper_loss`；从 contA 断点 full-resume 到 30 epoch。**未使用：`max_epochs=30` 仍配 100-epoch 余弦，已由 `AGENTBOX-COS50` 取代** |
| `experiments/status_agentbox_cos50_lr3e4_clip1000_20261026.json` | `AGENTBOX-COS50` | v5 入口；从 contA ep25 last.ckpt full-resume 到 50 epoch，`scheduler_epochs=50` 使余弦退火真正走到 `min_lr`；monitor `val/paper_loss`、`checkpoint_keep=10`；峰值 LR、batch、clip、数据均与 contA 相同。**已暂停于 ep32/step 77,749；作为 cos50b 父合同** |
| `experiments/status_r1_ema_bevlr5e1_lr3e4_20261026.json` | `R1-EMA-BEVLR` | v6 入口；**从头 25 epoch**（`policy=official-scratch`，不 resume）。相对 cos50 **只改两项**：① `ema_enabled=true`、`ema_decay=0.999`；② 顶层 `opt_paramwise_cfg` 追加 `bev_semantic_head.lr_mult=0.5`。LR 峰值 3e-4、`scheduler_epochs=25`、batch 16×acc4、clip 1000、loss 权重、数据划分全部与 cos50 一致，`val/paper_loss` 可跨臂比较 |
| `experiments/status_r2_cos100_ema5ep_bevw3_20261027.json` | `R2-COS100-EMA5EP-BEVW3`（当前） | 从 R1 的 `last.ckpt`（epoch 2 / step 7,512）**同目录 resume**。三项改动：① `max_epochs`/`scheduler_epochs` 25 → **100**；② `ema_decay` 0.999 → **0.99992**（12,520 步 = 4.99 epoch 窗口）；③ 新增 `loss_weights.bev_semantic_weight` 14 → **42**。因第③项，`val/paper_loss` 绝对值**不再与 R1/cos50 可比**，跨版本只能比较去权重后的分项；EMA 构造器从配置取 decay，不从 checkpoint 恢复 |
| `baseline_bs8_acc8_lr1e4_warmstart.json` | `HISTORICAL-SMOKE` | Batch64 权重 Warm Start 三步 Smoke；不是正式长训 |
| `baseline_bs8_acc8_lr1e4_warmstart_pssd_targets.json` | `CURRENT-PSSD-TARGETS` | 与 `baseline_bs8_acc8_lr1e4_warmstart.json` 逐字段相同，仅新增 `target_storage_registration` / `allowed_target_storage_registration_sha256` / `target_location_reader_sha256` 三个 relocation 字段；Targets 从 PSSD 解析；SHA `c4d651edb5c3a2bba24836867fc6d6a3f0864191b1d849b51d385b117b527127` |
| `baseline_original_7class_bs8_bf16_20261020.json` | `HISTORICAL-FROZEN` | BF16 历史正式训练；旧 control/证据引用 |
| `baseline_original_7class_20261019.json` | `HISTORICAL-PREFLIGHT` | 早期正式训练/预检；历史脚本引用 |
| `baseline_original_7class_bs8_fresh.json` | `FROZEN-CONTROL-REFERENCE` | cache-only 冻结 control/transition 引用 |
| `baseline_original_7class_bs4_fresh.json` | `ARCHIVED` | 归档前工作区/PSSD 元数据文本引用 0；已移入 `undo/configs/`，恢复命令见 [undo README](../../undo/README.md#2026-10-21-安全归档候选) |
| `baseline_bs8_acc4_lr1e4_finetune.json` | `ARCHIVED` | 归档前工作区/PSSD 元数据文本引用 0；已移入 `undo/configs/`，恢复命令见 [undo README](../../undo/README.md#2026-10-21-安全归档候选) |
| `pilot_original.json`、`pilot_proportional.json`、`m1_subset.json`、`m2_subset.json`、`m4_subset.json`、`camera_proportional_check.json`、`paired_check.json`、`baseline.json` | `HISTORICAL-VALIDATION` | Pilot/子集/相机/配对验证；被测试或历史脚本引用 |
| `features_cache.json`、`features_remaining_20261015.json`、`features_jobs.json`、`features_shared_storage.json`、`features_system_storage.json` | `DATA-PIPELINE` | Features 生成、任务集合和存储计划 |
| `features_manifests.json` | `DATA-PIPELINE`（生成） | 由 `feature_jobs` 导出；不手工编辑 |
| `targets_full.json`、`targets_recovery_irrelevant_v1.json` | `DATA-PIPELINE` | Targets 全量/历史恢复任务 |
| `v1_target_contract.json`、`camera_original_compatibility.json`、`m4_original_compatibility.json` | `RUNTIME/FROZEN CONTRACT` | 目标/相机/历史 producer 合同桥；不要删除 |
| `formal_training_proposal.json`、`experiments/m5_abc_design.json` | `DESIGN-ONLY` | 设计稿；`execute_training=false` 或 `design_only=true`，不能直接当 fit 配置 |

**维护规则：** 冻结配置不能为了“减少重复”而改写 `data` 列表；其 SHA/路径可能进入 Checkpoint、Manifest 或历史证据。未来新实验应使用小型 overlay 配置（本轮暂不实现代码），避免再复制整份约 22.4 万行 JSON。

## 模块职责索引

| 模块组 | 文件 | 当前角色 |
|---|---|---|
| 当前训练运行链 | `fit.py`、`runtime.py`、`paired_dataset.py`、`cache_only_reader.py`、`single_pass_cache_reader.py`、`procfs_mount_reader.py`、`sampling.py`、`training_progress.py`、`training_health.py`、`lr_experiment.py`、`bf16_policy.py`、`evaluate_checkpoint.py`、`official_v1_audit.py`、`features_cache.py`、`target_cache.py`、`target_storage.py` | 当前正式训练、恢复、评估或运行时合同检查 |
| Status v2 运行链 | `status_experimental_train.py`、`experimental_status_reader.py`、`route_split.py`、`status_training_preflight_full.py`、`status_training_monitor.py` | Route split 8:1:1、Step0 官方初始化、全量 validation、val/loss Checkpoint、Test 隔离与只读监控 |
| 数据生成/恢复管线 | `feature_jobs.py`、`feature_service.py`、`feature_migration.py`、`feature_storage.py`、`feature_shared_storage.py`、`target_jobs.py`、`target_service.py`、`target_recovery.py`、`target_retry_audit.py`、`recovery_paired.py`、`system_feature_relocation.py` | 数据生成、迁移、审计和历史恢复；不是当前训练启动入口 |
| 历史验证/一次性入口 | `check_labels.py`、`check_m1.py`、`preflight.py`、`prepare.py`、`train.py`、`baseline_setup.py`、`full_recovery_rules.py`、`gravity6_recovery.py` | 保留为冻结工具；旧 PID/旧 control 只代表历史快照 |
| 旧控制器 | `training_control.py`、`training_monitor.py` | 历史 Batch8 控制；不管理当前正式进程 |
| 长期测试 | `tests/agent/`、`tests/training/`、`tools/validation/b2d_adapter/test_*.py` | 原路径保留，不按文件名删除 |
| 设计稿/不可执行配置 | `formal_training_proposal.json`、`experiments/m5_abc_design.json` | 只作设计依据，不能直接启动 |

## 当前模式与历史模式

| 维度 | 当前正式运行 | 历史/复现模式 |
|---|---|---|
| 数据入口 | `paired_manifest` + `cache-only` | `prepared_manifest`、raw Dataset |
| 文件读取 | `verified-single-pass` | `legacy`（多次打开） |
| 挂载检查 | `procfs` | `findmnt` |
| 精度 | 保守 Conv2d BF16 + FP32 融合/Head/Loss/验证 | FP32、全网络 AMP 对照 |
| 更新 | micro-batch8 × 累积8 | batch8、batch2、batch4 |
| 验证/诊断 | 每 epoch 7846 帧；每 250 更新健康面板 | 短周期/固定小面板 |
| 控制 | 实际输出目录 `training_progress.json` + `pause.request` | 旧 `control.json` / `training_control` |

这些历史分支用于复现旧 producer/Checkpoint，不是“可随手删除的兼容垃圾”；不要在整理目录时删除或重写。

## 输入与监督

| 字典 | 张量 | 单样本 shape / dtype | 大小 |
|---|---|---|---|
| features | `camera_feature` | `(3,256,1024)` float32 / RGB [0,1] | 3.146 MB |
| features | `lidar_feature` | `(1,256,256)` float32 / 归一化计数 | 0.262 MB |
| features | `status_feature` | `(8,)` float32，命令4 + vx/vy/ax/ay | ~0 |
| targets | `trajectory` | `(8,3)` float32，4 秒真实未来 | 96 B |
| targets | `agent_states` / `agent_labels` | `(30,5)` float32 / `(30,)` bool | 630 B |
| targets | `bev_semantic_map` | `(128,256)` int64，七类 0–6（磁盘 uint8 无损编码） | 32 KB |

整帧 Feature 3.42 MB、Target 58.5 KB（其中纯张量 33.5 KB，余下为 sources/map/provenance 等元数据）。解压比 Feature 0.996、Target 0.572 —— `torch.save` 不压缩数据，Feature 磁盘≈内存；Target 元数据占 43%，加载后只取 `targets` 反而更小。

### `camera_feature`：三路前视拼接

规则 `b2d-proportional-crop-v1`，实现 `tools/b2d_adapter/camera_adapter.py:8-26`：

```python
stitched = np.concatenate((left[23:-23, 346:-346],
                           front[23:-23, :],
                           right[23:-23, 346:-346]), axis=1)
ToTensor()(cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_LINEAR))
```

| 步骤 | 参数 |
|---|---|
| 输入 | `cam_l0` / `cam_f0` / `cam_r0`，Bench2Drive 原始 **1600×900 RGB** |
| 上下裁剪 | 各 23 px（三路相同） |
| 左右裁剪 | 左/右相机各 346 px；中相机不裁 |
| 横向拼接 | **3416 × 854** |
| 缩放 | `cv2.INTER_LINEAR` → **1024 × 256** |
| 归一化 | `ToTensor()` 一次 `/255` → [0,1] |

**与官方 NAVSIM 的差异**：官方 `transfuser_features.py:55-74` 用上下 28 px、左右 416 px，拼接 844×3136。本工程按相同的最终 4:1 宽高比换算成 Bench2Drive 的 1600×900 输入，因此数值不同但不改变语义。旧 `navsim-pixel-crop-v1` 只服务历史复现，不用于当前训练。

### `lidar_feature`：BEV 占据直方图（单层）

实现 `tools/b2d_adapter/lidar_adapter.py:15-40` + `config.py:24-38`：

1. 读当前帧 LAZ → N×3 XYZ。**LAZ 已是 Actor ego 坐标，不做 world2lidar 反变换**；
2. 后轴对齐：`x -= axle_dx(-1.388632877)`、`y = -(y - axle_dy(0.0))`、`z += lidar_z_shift(0.0)`；
3. 高度过滤：仅保留 `0.2 < z < 100` m —— **只有 above 层，地面层不使用**（`use_ground_plane=False`，故输出 1 通道而非 2）；
4. 2D 直方图：范围 `[-32, +32]` m、`4 px/m` → **256 × 256 bin**（行=x，列=y，bin 边界递增）；
5. 截断并归一化：`min(count, 5) / 5` → [0,1] float32。

**未认证项**：高度原点（`lidar_z_shift_m=0.0` 是显式占位，未经验证）；「LAZ 已是 Actor 坐标」依据既有成对样本推断，未做独立标定。

### `status_feature`：命令 4 + 运动 4

实现 `tools/b2d_adapter/native_builders.py:52-58`：

```python
np.concatenate((command, state.velocity, state.acceleration)).astype(np.float32)
```

| 维度 | 内容 | 来源与约束 |
|---|---|---|
| 0–3 | Navigation4 one-hot（Left / Straight / Right / Unknown） | `near_route_command`：按 near 路线目标相对工程后轴的 bearing 编码；**不使用原始 RoadOption，不做整数映射，不用 Unknown 兜底** |
| 4–5 | `(vx, vy)` | 因果后轴位置差分，滞后 `dt/2 = 0.05 s` |
| 6–7 | `(ax, ay)` | 因果差分，滞后 `dt = 0.1 s`；**不使用 IMU 原始加速度或标量速度补零** |

**未认证项**：`formal_semantics = "unverified"`。该编码是工程近似，不等价于 NAVSIM 的 20 m near 规则，也不代表采集端的真实导航语义。

类别为 Background/Road/Walkway/Centerline/Static/Vehicle/Pedestrian。缓存编码可将 BEV 无损保存为 uint8，读取后恢复 int64。原数据、标注和来源限制见 [Adapter](../b2d_adapter/README.md)；坐标与时间对齐的证据见 [coordinate_validation.md](../../analysis/data_interface_analysis/coordinate_validation.md) 与 [temporal_alignment.md](../../analysis/data_interface_analysis/temporal_alignment.md)。

## 缓存读取

2026-10-24，Targets 源盘恢复后开始复制，持续读取时再次 EIO；PSSD 上已有 106,191 份（约 5.82 GiB）载荷通过冻结 SHA 核验，剩余数据未完成、源文件删除 0，训练地址未切换。完整状态见 [迁移状态](../../outputs/diffusiondrive_v1/targets_to_pssd_20261024/migration_status.json)。新增 `target_relocation.py` 的 `TargetLocationResolver`，由 `CacheOnlyPairedDataset`（含 Status/评估基础读取）、`RecoveryPairedDataset` 和 `PairedDataset` 可选调用；未配置存储注册时沿用原读取路径。启用时配置必须固定 `target_storage_registration` 的路径/SHA、`allowed_target_storage_registration_sha256` 和 `target_location_reader_sha256`，并校验原根及 producer、源 SQLite 元数据、PSSD 根 owner 标记及读取期间身份变化。36 项相关 CPU 单测通过，**正式存储注册和启用配置均未发布，真实 PSSD DataLoader 验收未执行**。数值 producer、原始配置、配对索引与 Manifest 保持不变；全量复制校验和 train/validation 批读取通过后才能切换和删除源文件。不得直接替换冻结 `target_root.path` 或放松 Checkpoint 恢复合同。

`CacheOnlyPairedDataset` 由显式 producer-aware Manifest 选择配对，Worker 复用 SQLite `mode=ro/query_only` 连接，不保持长事务。冻结配对索引、注册文件和根目录 owner 身份受检查；不逐帧扫描原始 Route、JPG、LAZ 或标注。它信任已冻结的数据版本及记录的来源 SHA，原始文件变化复审仍属于 audited 路径。

当前执行参数为 `--cache-read-mode cache-only --mount-check-mode procfs --cache-file-read-mode verified-single-pass`。

- procfs 后端逐次核对内核 mountinfo、UUID 对应块设备、可见 st_dev、文件系统和异常只读状态；不使用跳过检查的TTL。
- 单次读取把每份完整文件读入缓冲，计算 SHA256 后由现有冻结加载器反序列化**同一份字节**。大小/SHA、sample_id、producer、来源、坐标、地图、shape/dtype、范围/有限值检查仍保留；读取和解码前后检查文件身份变化。
- Target 无损 codec 继续复用。作用域在单线程 Dataset worker 中运行，异常也恢复模块绑定、释放单个缓冲；不把全数据集复制到内存。
- 只读执行策略不改变 Tensor、采样选择或数值训练合同。旧 `features_cache.py` / `target_cache.py` / `target_storage.py` 保持冻结，不能为整理目录改动 producer。

必要实现位于 [cache_only_reader.py](diffusiondrive_v1/cache_only_reader.py)、[single_pass_cache_reader.py](diffusiondrive_v1/single_pass_cache_reader.py)、[procfs_mount_reader.py](diffusiondrive_v1/procfs_mount_reader.py)。

## BF16 边界与合同

最终采用 `bf16-conv-fp32-fusion-heads-loss-eval-v1`，仅训练时 Backbone Conv2d 使用 BF16，输出转回FP32。归一化、融合、全部 Head、几何、内部轨迹 Loss、外部多任务 Loss、验证、参数、梯度和AdamW动量保持FP32。通过工程精度边界调用官方公式，不改模块类型/state_dict键。数值策略标识保持冻结；`bf16_policy.py` 仅增加旧断点恢复校验，当前源码 SHA 为 `bc6a55c962327ac55975cb0bb23678e1d0836d08f611f5256296ee92e2dfd146`，旧断点通过显式迁移接受，不改变训练数值。

全网络 AMP 的梯度方向余弦约0.941，整体Backbone AMP的同模式轨迹最大坐标差约0.56m，均未放行。保守卷积策略在16帧/20步对照中：总Loss相对差0.0166%，梯度余弦0.999936，解码器同模式最大XY差约3.17cm，选中轨迹约0.89cm，没有模式切换；同权重同噪声的FP32验证推理逐位一致。有限对照不是长期收敛或物理真值认证。

首次 FP32→BF16 必须独立实验，并显式使用受控的 `--allow-fp32-to-bf16`：只允许指定精度及其策略/SHA变化，保留父Checkpoint和完整Optimizer/Scheduler/RNG/Sampler/epoch/global_step。普通后续续跑不再带这个标志。同合同恢复保持严格SHA和状态检查；不能任意改batch、监督或学习率后借此绕过。

## Loss与官方V1对齐核查

2026-10-20只读核查官方固定提交`9b52ed0ec06b073d82d6f392ab084c7b301c8681`：12份核心Python源码中，模型、Agent、transfuser_loss、multimodal_loss与Lightning等保持逐字一致；唯一既有差异是scheduler移除新版PyTorch不支持的verbose构造参数，AST核对调度公式相同。`runtime.model_configuration`仅替换本地Anchor/Backbone资源路径，不重写Loss权重。实际恢复运行报告确认继承官方training_step/configure_optimizers，使用官方V1类、AdamW和WarmupCosLR；无BNDP/V2 RL/SEL、自定义Planning Loss、Dice/IoU Loss或额外梯度目标。

总目标为`12*L_trajectory + 20*L_diffusion + 10*L_agent_class + 1*L_agent_box + 14*L_bev`，与官方相同。

| 项 | 真实实现及聚合 | 权重/一致性 |
|---|---|---|
| 轨迹模式分类 | 原始20个Anchor与GT的8点平均XY距离选择正模式；sigmoid focal，gamma=2、alpha=.25 | 每decoder层10，保持原版 |
| 轨迹回归 | 对GT最近Anchor对应模式的8×3 `[x,y,heading]` 执行L1 mean | 每层8，保持原版 |
| 轨迹汇总 | 两个decoder层的加权分类+回归Loss相加，外层再乘trajectory_weight | 外层12，非自定义重复加权 |
| Agent分类 | 官方Hungarian匹配后的binary_cross_entropy_with_logits，含无对象槽位 | 10，保持原版 |
| Agent框 | 匹配后`[x,y,heading,length,width]`的L1和，只对有效GT累加，除以整个batch有效GT数（空GT分母保护为1） | 1，保持原版；匹配cost只使用XY L1及分类cost |
| BEV | 七类cross_entropy，对全部像素mean，无额外类权重/ignore/dice | 14，保持原版 |
| diffusion_loss | 主Loss支持该键，但本V1轨迹Head只返回trajectory_loss及分层Loss，没有独立diffusion_loss键 | 权重20保留，实际该项0与官方一致，不能据此说扩散未训练 |

训练只对官方`loss_dict['loss']`反向，不把记录用的trajectory_loss_0/1再次加进总Loss。日志trajectory_loss已乘外层12，分层trajectory_loss_0/1包含内部10/8但尚未乘外层12。当前运行的464001步例子：`12*(1.1590956+1.0719042)≈26.7720`；再加Agent分类4.6474、框26.3937、BEV2.2552，得到总Loss60.0684。不能把这些数值当米制轨迹误差。

验证由eval的官方forward_test按预测分类分数选模式，不用GT选择最佳轨迹；没有trajectory_loss键时，官方主Loss对推理轨迹与GT的8×3 L1乘12。ADE/FDE则只计算XY米制误差，新增指标不进入训练梯度。训练的GT最近Anchor模式监督与推理的argmax选择口径不同是官方设计，不是本Adapter新增的Oracle评估。

**明确保留的工程差异：** 当前batch8/单GPU/8Worker/保守Conv2d BF16，而固定官方default_training.yaml为每进程batch64/4Worker、DDP策略、16-mixed默认精度；官方总batch还取决于实际启动设备数，不能凭默认yaml假定具体GPU数。当前Loss/Head和验证保持FP32，但来自BF16卷积的训练特征会影响数值轨迹，因此不声称整段训练与官方默认精度逐位一致。B2D来源、七类标签构建、配对缓存、Town05验证、种子和样本分布也与NAVSIM不同。AdamW参数组和WarmupCosLR仍使用官方创建逻辑，当前epoch17的两个组实际学习率均为0.0005697360、weight_decay=0.0001；这不是手工把Backbone组单独调成相同学习率，官方Scheduler在组无lr_scale时本来就返回相同lr。

已有日志聚合修正（按实际batch样本数而非字典键数）、ADE/FDE、SHA读取、暂停/完整恢复属于包装层；不改变Loss公式。Loss已对齐，不为“对齐”再更换权重或补噪声MSE。若后续比较官方默认batch/FP16或其他训练配方，应另建配置及实验合同，先做数值/效果对照，不原地修改活动任务或篡改Checkpoint合同。效果退化的模块定位见下文只读归因；不能未经证据归因于Loss被替换。

## 断点、暂停与 ETA

`CommittedSampler` 保存已提交样本而不是预取游标；恢复核对首批sample_id，完整比较546份AdamW状态及Scheduler、RNG。Lightning在batch-end保存时可能出现completed落后1，只修正已加载的循环记账，不修改存盘Checkpoint。epoch-end replay保护防止重复调度；旧实验的Checkpoint不被新目录清理。

**P007 修复（2026-10-21）：** `PreservedCheckpoint.on_train_batch_end` 在 `trainer.global_step <= _resume_step` 时跳过保存，避免恢复后的第一个累积窗口在 `pending_micro>0` 时触发 `ModelCheckpoint`；`on_train_epoch_end` 同步跳过父 epoch 的 replay 保存。旧断点通过 `validate_resume(..., allow_checkpoint_boundary_fix=True)` 显式迁移，只允许 `execution_source_sha256` 中 `fit.py` 改变和 `precision_policy_source_sha256` 更新为当前验证源码；Optimizer/Scheduler/RNG/Sampler 与 global_step 保留。新合同配置为 `baseline_bs8_acc8_lr1e4_formal_resume_fix.json`，父配置不改。

合作暂停在批次提交后结束并原子保存 `pause-step*.ckpt`，验证结束前的暂停可能等待当前验证完成。status确认进程退出且paused后才resume，不能只依据旧JSON的running/paused文字。新resume在PSSD创建独立输出目录并继承最新读取/Worker覆盖；训练控制会拒绝重复fit进程、错误UUID、SHA不符和输出盘低于100GiB保留。

进度每约5秒更新；吞吐按最近约300秒活跃训练估算，含数据等待/正常保存开销，不计暂停。**含验证 = 剩余训练 + 剩余验证**，最近三次完整验证平均耗时用于预测；无验证观测时不伪造总ETA。PID、步数、温度、IO与速度均是运行状态，不在本文反复追加快照。

## 验收依据

| 事项 | 冻结证据与限度 |
|---|---|
| 官方网络与优化机制 | [official_v1_training_audit.md](../../analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/official_v1_training_audit.md)；运行时inspect/源码SHA/实际对象/参数组 |
| 原版预检 | [preflight_result.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/preflight_result.json)、[epoch_boundary_resume_validation.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/epoch_boundary_resume_validation.json) |
| 缓存直读 | [cache_only_reader_validation_20261020.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/cache_only_reader_validation_20261020.json)：20帧逐位一致、0/2 Worker、禁止raw访问 |
| 挂载后端 | [procfs_reader_validation_20261020.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/procfs_reader_validation_20261020.json)：UUID/FS/可见设备及故障拒绝 |
| BF16数值与恢复 | [numeric_validation.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/bf16_training_20261020/numeric_validation.json)、[preflight_summary.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/bf16_training_20261020/preflight_summary.json)：7步迁移+8步再次恢复，32帧Town05固定验证面板 |
| 单次读取+SHA | [validation.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/single_pass_reader_20261020/validation.json)：16训练+4验证帧，Tensor逐位一致；每对样本从4次文件打开降至2次；0/2 Worker及错误SHA/身份/IO拒绝 |
| 训练实际启用 | [running_snapshot.json](../../analysis/diffusiondrive_b2d_architecture/v1_validation/single_pass_reader_20261020/running_snapshot.json)：从332853步恢复，数值合同相同且真实优化继续；更新状态以control为准 |

驻GPU卷积BF16对照约提速19%；单次读取暖页小样本约55→61 samples/s。它们均不能直接当作端到端加速比例；实际step/s与ETA读取控制状态。新测试临时输出使用 `/tmp`，需保留的证据放 `outputs/`，不再按每次任务新建同主题Markdown。

### Status-only 微调准入（2026-10-23，NO-GO）

全量 Route Preview 与 Status Overlay v2 已就绪并完成 CPU 预检，但**不满足启动条件**。判定与证据集中在 `outputs/diffusiondrive_v1/route_preview_v11/20261023_status_integration/`：导航审计（`navigation/`）、218,622 行 overlay 校验（`overlay/`）、独立读取层契约与测试（`reader/`）、预热微调草案与预检（`finetune/`）。

- 预检 18 项全 PASS：9 项冻结身份、Status 布局与顺序、summary 一致性、读取器版本、准入探针、新输出目录、与暂停运行隔离、`fit.py` 无 overlay 引用、CPU checkpoint 元数据（`global_step=9882`、763 项 state_dict）；第 19 项为准入判定本身，结果为 `NO_GO`。
- 阻断项：导航认证 0 行（224/1,100 Route 仅有离线 XML 代理，可比子集一致 85.1%，改写行新 Status 与 XML 一致 113/124）；12 条缺陷 Route 已在版本化目录修复通过但仍需新 Overlay 合同，当前训练侧必领排除口径 **204,976**（train 198,568 / validation 6,408）；`fit.py` 未接入读取层；物理载荷全量哈希未跑（Target 1,930/1,930、Feature 2,610/2,610 抽样均通过）。Target 载荷盘 `/data/datasets` 已恢复可读（顶层 `/dev/sde1`，UUID 正确），但旧 `emergency_ro` 挂载层与持久化配置仍需处理。
- 微调合同已准备好但未启动：权重-only 预热自 `best-epoch002-step009882.ckpt`（`global_step=9882`，SHA `f5bb31bd…7f68a7b2`），Optimizer/Scheduler/Sampler/global_step 全部重置；父 Scratch 运行停在 **step 58,650**（`last.ckpt` SHA `0be42e5f…204e83c1`）且**不 resume、不覆盖、不作起点**；新输出目录 `/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/status_full_overlay_warmstart_20261023` 尚未创建。
- 在线认证轮（2026-10-23，第二轮）：会话扩展到 **9 条 Route / 2,394 连续帧**；统一 Navigation4 规则为“稠密 `set_global_plan` 世界坐标 + 20 m 弧长 + 2.0 m 侧向 + 5.0 m 投影”；8 项合同逐字段审核后生成 **`certified_status_overlay_v4`：993 行**（train 652 / validation 341，7/2 Route；Straight 603 / Left 205 / Right 185），读取层校验结构错误 0，仅剩三个审批门错误码，`training_admitted=false`。详见 `finetune/status_finetune_go_nogo_v5.md`、`certified_status_overlay_v4_report.md` 与 `navigation_semantics_unification.md`。
- 全 Route 分级验证（2026-10-23，独立 CPU 轮）：1,100 条全部建索引；230 条离线 CARLA 0.9.15 稠密代理计划，35,161 帧可比，166 条有主 XML 可靠代理关联，83 条整体通过候选。8 个孤立异常帧、195 段连续不一致均保留；可组合 Status 候选 37,300，严格八项证据与当前几何交集为 928（train 599 / validation 329），不是新增训练准入。剩余 934 条来源未验证；9 条已有 leaderboard 捕获仍只是后补本地 XML 会话。合同审批门不变，训练准入 **0**，本轮无训练或新旧 Status 对照。详见 [认证报告](../../outputs/diffusiondrive_v1/route_preview_v11/20261023_route_validation_v1/route_certification_report.md)，命令见 [快速启动 6.5](../../快速启动.md#65-全量-route-导航索引与分级认证cpu离线地图)。
- 训练准备轮（2026-10-23）：新增 `status_training_manifest.py`、`experimental_status_reader.py`、`status_training_preflight_v2.py`、`status_experimental_train.py`；转向区间一致性 75 个区间 0 例相反分支；训练清单 **193,471 行**（train 187,913 / validation 5,558；1,021/23 Route），11 项预检含真实 GPU 前向/反向 **GO**（loss 67.55、梯度有限、预检阶段未执行 optimizer step）。详见 `20261023_status_training/status_warmstart_training_plan.md` 与 `experimental_status_training_preflight.md`。
- 短程 30 步（2026-10-23，已执行）：输出 `status_preview_warmstart_20261023`，两个指标版本各 30 条 train、0 非有限值；v0 `train/loss` 94.29→64.17（min 55.16），v1 92.61→93.59（min 47.38），val loss 末值约 20.0；`train_result.json` 为 `global_step=30`。该目录 checkpoint 名为旧模板 `stepstep=000030*.ckpt`。
- 长线 2,000 步（2026-10-23，已完成）：配置 `warmstart_longrun_config.json`，`val_check_interval=250`、`validation_limit=512`、`limit_val_batches=1.0`；新增 `val/ade_m`、`val/fde_m`、`val/heading_error_deg` 与分命令 ADE/FDE，checkpoint 命名 `step-XXXXXX.ckpt`、`checkpoint_keep=-1`。8 次验证全部有限：ADE 3.35→**2.88**（1,250 步）→3.01（2,000 步），FDE 7.44→**6.30**（1,750 步）→6.77，终点航向 13.4°→11.9°→12.4°，`val/loss` 31.51→29.01 区间波动；全表见训练方案 §2.2。首版 `save_top_k=2` 无 `monitor` 导致 `MisconfigurationException`（未产生任何 optimizer step 或输出），已改 `checkpoint_keep=-1`。
- 全量 100 epochs（2026-10-23，已暂停）：配置 `full_config.json`，从 `step-002000` 严格 weight-only 预热（763 张量逐位相等，Optimizer/Scheduler/Sampler/RNG/global_step 重置），共 **2,348,900** 步；每 epoch 用全部 5,558 行验证，只保留 `val/ade_m` top-2 `best-*.ckpt` + `last.ckpt`（无周期 step 快照），`checkpoint_index.json` 记录选择。`pause.request` 在 optimizer step 边界生成 `pause-step*.ckpt` 并正常退出，`--resume` 完整恢复继续。FP32 首轮暂停在 **11,711 步**，输出 `status_preview_full_20261023`。
- BF16 卷积版本（2026-10-23，已暂停）：配置 `full_bf16_config.json`，沿用冻结策略 `bf16-conv-fp32-fusion-heads-loss-eval-v1`（`install_fp32_losses`，仅 Backbone Conv2d BF16，融合/Head/Loss/验证 FP32），从 FP32 暂停快照 `pause-step011711.ckpt` strict weight-only 起步；100 epochs / 2,348,900 步。500 步实测 **7.09 it/s**（FP32 6.42 it/s，约 +10%），暂停在 **2,186 步**，输出 `status_preview_full_bf16_20261023`。
- batch16 + BF16（2026-10-24，已暂停在 **109,433 步**）：当前配置 **`full_bf16_bs16_val250_w8_pssd_targets_config.json`**（`num_workers=8`，Targets 从 PSSD 解析），旧 `full_bf16_bs16_val250_w8_config.json` 因源盘 payload 已删只能作历史。从 `pause-step002186.ckpt` strict weight-only 起步，每 epoch 11,744 步，100 epochs = **1,174,400** 步；每 250 步在固定前 1,024 行验证，best 每 20 次验证写一次。500 步实测 **3.84 it/s（61.5 samples/s，12 workers）**，比 BF16 batch8 约 +8%、比 FP32 约 +20%。已暂停/恢复多次：44,683 步用于 90° 转弯对照；**2026-10-24 04:13:53 在 100,200 步（epoch 8）被 `systemd-oomd` 连带杀掉**（杀掉整个 VS Code cgroup，45 个进程，日志无 Traceback，该步无断点），随后从 `best-epoch008-step095708.ckpt`（含 Optimizer/Loops）以 **8 workers** 在独立 `systemd-run --user --scope --unit=dd-train-status` 中恢复，scope 内存约 **11.4 GB**、实测 **3.33 it/s（53.3 samples/s）**，比 12 workers 低约 13%；**2026-10-24 09:42 合作暂停在 109,433 步**，断点 `pause-step109433.ckpt`；**18:51 因 `/data/datasets` USB 盘 EIO 中止在 112,850 步**（该步无断点）。worker 扩展实测 12/16/20 均 3.72–3.74 it/s，无提升；RAM 不足，不要再扩容。
- **Targets 迁移到 PSSD（2026-10-24，已完成）：** 230,446 个 payload（13.61 GB）从 `/data/datasets` 复制到 `/run/media/aethelind/PSSD`，`verification_result.json` 确认**全部源/目标 SHA256 与大小相等、全部 indexed SHA 与冻结索引一致**；`target_storage_registration.json`（SHA `cddaf09a70ee5c3385f560479ee80dd85636ea3d0d65d3cf1ac5c108b25ce7b4`）登记 4 个逻辑 root 的 old→new 映射；CPU 实读 `reader_acceptance.json` 通过（train/validation 各 8 样本均从 PSSD 解析且 SHA 匹配）。随后 `delete` 删除源盘 **230,441** 个 payload（13.61 GB），仅保留 5 个 owner marker，逐文件写 `source_deletions.jsonl`；删除后再次实读验收通过，源盘上已无 `.pt`。新 base config `baseline_bs8_acc8_lr1e4_warmstart_pssd_targets.json`（SHA `c4d651ed…`，仅新增 3 个 relocation 字段）与新训练配置 `full_bf16_bs16_val250_w8_pssd_targets_config.json`（SHA `62c96491…`）是当前唯一可用的 Target 读取路径；旧 base/w8 配置与 `/data/datasets` 目标树不可再用。证据均在 `outputs/diffusiondrive_v1/targets_to_pssd_20261024/`。
- 简短开环路线检查（step 43,207）：`status_route_eval.py` 在 3 条未参与监控子集的验证路线各 80 帧上得 ADE 4.39 / FDE 9.38 / 航向 11.9°；路口左转最好（1.67/3.61），变道最差（7.54/15.21）。只做开环预测，不是 CARLA 闭环。
- 90° 转弯专项对照（step 2,000 / 17,082 / 44,683 / 95,708）：`status_turn_analysis.py` 读取多个断点的 `samples.jsonl`，用专家目标 8 个路点的**累计转角**筛出转弯帧（首末夹角会把 90° 弯判小），并要求各断点 `sample_id` 完全一致。4 条未参与监控子集的转弯路线共 640 帧，累计转角 ≥60° 的 74 帧上 ADE 2.07→1.60→1.34→**1.14**、FDE 3.93→3.26→2.27→**2.06**、航向 10.8°→5.4°→7.0°→**3.8°**；右转路线 −56%/−54%，左转路线 −37%/−39%，无信号灯转弯 FDE −48%。说明约 90° 转弯已开始学习，但仍是开环预测，不代表 CARLA 闭环过弯。
- 全量验证开环评测（2026-10-24，`status_route_eval.py --all-routes`）：23 条验证路线 5,558 帧、FP32、batch1，指标定义与训练 `val/ade_m` 一致（epoch 0 校准：工具 1.989/4.328 vs 训练 1.982/4.321）。最新 best（95,708）全量 **3.826 / 8.194 / 6.0°**，比 epoch 0 的 1.989/4.328 **总体变差**；剔除 `SequentialLaneChange_Town05_Route24339` 后其余 4,000 帧由 2.34/5.02 改善到 **1.92/4.25**，而该路线 1,558 帧由 1.09/2.55 崩到 **8.72/18.33**——1,392 个 `vx<0.5 m/s` 停车帧专家目标终点约 1.8 m，模型却预测约 19.9 m。监控子集只有 14% 停车帧（全量 57%、训练 43%），所以子集 ADE 降到 1.856 的“历史最好”同时掩盖了该退化；checkpoint 选择应改用跨 23 条路线、覆盖停车/转弯的分层子集或定期全量验证。证据：`outputs/diffusiondrive_v1/route_preview_v11/20261024_fullval/`。
- 子集与“蠕行未学会”判别（2026-10-24）：监控子集**固定**（索引 `rows[:1024]` + `shuffle=False` + 索引 SHA 固定，`val/samples` 恒为 1,024），但 1,091 帧中**只有 4 帧**专家目标终点 <1 m（全量 2,606/5,558），子集 `vx<0.5` 的 141 帧专家终点平均 12.5 m（停一下就走），因此测不到“保持停车”。同一批子集帧上 epoch 0 为 2.715/6.049、最新 best 为 1.902/3.988（子集认为最新更好），全量却相反（1.989/4.328 vs 3.826/8.194）——**排序反转**证明子集口径在带偏 best 选择。但蠕行确实学会了：训练集 4,000 帧抽样中 30.4% 专家终点 <1 m；对 150 个训练停车帧，模型预测终点均值 0.179 m、中位 0.086 m、ADE 0.089 m，全部 ≤2 m；验证集 3 条路线的 973 个停车帧中位预测 0.24–1.50 m、`corr(vx,pred)=0.90–0.94`。真正的问题是特定场景模式塌缩（`Route24339` 预测中位 22.2 m、99% ≥10 m、`corr=0.36`；`Route27900` 预测 7.4 m），不是全局没学会停车。
- 分量 Loss 走势（2026-10-24，至 102,459 步）：训练侧全部可下降项单调下降——`loss` 66.59→37.20（epoch 0→8，斜率 −0.333/1k，t=−98）、`trajectory` 59.20→32.43、`class` 0.526→0.299、`box` 3.65→2.32、`bev` 3.22→2.15、两个轨迹头 2.58/2.35→1.40/1.31；95,708 步恢复无跳变。`diffusion_loss` 恒为 0 属预期：官方 `transfuser_loss()` 在 `predictions` 无 `diffusion_loss` 时置 0，本配置走确定性轨迹输出。验证侧（固定 1,024 行）：`val/loss` 24.72→23.83、`val/trajectory_loss` 10.69→10.22、`val/agent_box_loss` 4.85→4.32 在降；`val/agent_class_loss` 持平（0.876→0.868）；`val/bev_semantic_loss` **上升**（8.25→8.66，t=+11）。逐类诊断（256 行固定样本）：BF16 开始后 10,901 步 CE 仍为 0.4036（FP32 11,711 为 0.4064），到 44,683→0.4366、95,708→**0.4961**，是渐进漂移；三个速度档都变差、171/256 行变差，不是停车专属；背景/车辆改善（车辆 CE 2.48→2.11、IoU 0.10→0.17），road 0.58→0.70、centerline 1.38→1.71 轻微退化，**walkway CE 1.46→2.08 但 IoU 0.39→0.46、recall 0.43→0.52**——边界像素过度自信、校准变差；行人 recall 恒为 0（数据极稀疏）。该路径无数据增强、`use_ema=False`，BEV 权重 14 而轨迹占 loss 约 87%，判定主因是稠密 BEV 头过拟合与校准漂移，非精度或停车场景。跨子集不可比（epoch 0 全量 `val/loss` 26.04、`val/trajectory_loss` 9.34）。
- 用开环样本复算全量轨迹损失：epoch 0 9.37 → 95,708 步 **16.65**（+78%），而 1,091 行监控子集 12.70→**9.07**（−29%）——训练与子集 loss 正常下降，全量轨迹损失变差，与 `Route24339` 停车塌缩一致。
- BEV 张量归属（2026-10-24 核对）：语义图**不是模型输入**，而是监督目标 `targets['bev_semantic_map']`（int64 128×256，0.25 m/px，前方 0–32 m × 左右 ±32 m，类别 {0,1,2,3,5,6}）；输入只有 `camera_feature`(3,256,1024)、`lidar_feature`(1,256,256，±32 m 见方)、`status_feature`(8,)。Backbone 输出的 BEV 特征为 **64×64×64（约 1 m/px）**，语义头经卷积后 `Upsample` 到 7×128×256，交叉熵权重 14；轨迹分支使用同一 64×64 的 `cross_bev_feature`，不读语义图。因此**提高 BEV 标签分辨率（如 256×512）不会提升预测分辨率**（监督会比特征网格细 8 倍），反而抬高 CE/过度自信；且需重建约 19 万样本 Target 缓存、改变数据集 SHA。若要让规划用上地图，应改架构（地图输入通道 / `bev_down_sample_factor` 4→2 / 把覆盖扩展到前方 64 m）并重训。
- 训练监控（2026-10-24）：`status_training_monitor.py` 单行输出 state/epoch/step/百分比/loss/pid/最近 val/best，`--watch 10` 循环；吞吐用 5 分钟滑动窗口（`--window 300`、`--min-span 120`），避免 progress 每 50 步跳变造成假速率；`state=running` 但无进程时输出 `WARNING=state-running-without-process`。替代 6.6 8c 旧内联 heredoc（依赖未 export 的 `DD_PY/DD_BS16_OUT`，且把 tqdm 的 `it/s` 当吞吐）。测试 `tests/training/test_status_training_monitor.py` 5 项。
- 唯一变量是 Status 前四维（命令）来源；模型、Loss、Optimizer、相机/LiDAR/Target 缓存与 A0 Motion4 均不变。用户已显式授权本轮 Status 短程与长线实验；父 Scratch 恢复、CARLA 在线导航评估仍须另行授权。独立入口为 `tools/training/diffusiondrive_v1/status_finetune_entry.py`（`status-finetune-entry-v1`），入口两级准入在真实数据上均按预期拒绝；其测试见 `tests/training/test_status_finetune_entry.py`。

## Checkpoint效果与开环评估

新增独立 [evaluate_checkpoint.py](diffusiondrive_v1/evaluate_checkpoint.py)，只读验证冻结Checkpoint语义SHA、官方源码、Baseline Manifest/缓存身份，严格加载权重，执行FP32 `eval/inference_mode`与不传targets的官方Agent.forward。无Trainer、Optimizer或反向传播。代码路径及测试 [test_evaluate_checkpoint.py](../../tests/training/test_evaluate_checkpoint.py) 已由用户批准；13项CPU数值回归通过。测试覆盖样本加权、航向环绕、因果基线、检测一对一匹配、BEV混淆矩阵与文件atime变化。

2026-10-20 17:45完成每模型7846帧的Town05完整验证：Early(step2817) ADE/FDE为2.8608/5.5699m；Best(step26347)为1.8690/3.8885m；Latest启动快照(step375000)为4.9213/10.2674m。同样本CV为2.2754/4.9328m。三者无非有限轨迹，参数/缓冲版本未改变；这不代表驾驶效果合格。Best存在有限的开环增益，375000步快照反而劣于静止与恒速位置基线，且该工程检测门限下TP=0。训练Loss下降、异常轨迹门限触发率下降都不能替代准确性验证。

轨迹按有限样本/时间计数；检测累计TP/FP/FN后计算P/R/F1，门限0.5置信度+2m中心匹配仅为工程定义；BEV累计像素混淆矩阵后算七类IoU。每个Checkpoint样本顺序、seed137、batch1相同；与训练日志batch8的噪声分组不同，不能要求逐位相等。按场景/速度/Town分组；只有Town05验证，不能冒用训练Town充当验证。

完整 [效果审计](../../outputs/diffusiondrive_v1/checkpoint_evaluation/audit.md)、[指标](../../outputs/diffusiondrive_v1/checkpoint_evaluation/metrics.json)、[三模型汇总](../../outputs/diffusiondrive_v1/checkpoint_evaluation/comparison.csv)、[曲线](../../outputs/diffusiondrive_v1/checkpoint_evaluation/curves.png)、[同帧轨迹对照](../../outputs/diffusiondrive_v1/checkpoint_evaluation/trajectory_cases.png)及逐样本来源SHA留存在用户批准的输出目录。原周期权重可能由训练自身正常清理，本轮未另存权重副本；375000步结果不能归给后来的450000步。在线LiDAR/Status、真实Predictor、车辆控制及独立开发路线尚有阻碍，当前不准入CARLA驾驶验收。

### 严重偏差的只读归因

2026-10-20在Best26347、FP32后期203933及当前暂停465326断点上，使用相同的32训练/32验证帧检查全部候选、逐层融合注意力、对象Query及速度敏感性；原权重、模型/Loss和训练配置保持不变，无优化步骤。分层验证停车组含4个历史难例，以下是诊断证据，不是总体性能估计。

- 后期融合`_tf_decoder`的跨注意力数值饱和，30个Agent Query输出几乎相同，检测框重复；降低置信度至.01也无法恢复定位。不是仅检测阈值问题。
- 固定场景与噪声，只将vx增加5m/s，Best验证轨迹平均变化5.82m，当前仅约10⁻⁷m；结合后两层Status注意力为数值0，定位到状态条件通路失效。此反事实只检验敏感性，不作物理性能评分。
- 当前训练停车8帧ADE为.076m，验证停车8帧为8.042m；GT oracle候选最低ADE仍有1.552m。规划同时存在模式选择泛化与候选回归退化，不能把训练Loss下降解释为验证收敛。
- FP32后期已经出现同类故障，BF16不是必要条件；BN统计替换也未恢复。特征/投影尺度增长已观测，但学习率、小batch及多任务梯度谁是首要上游诱因尚未由受控优化实验确认，不能直接宣布唯一根因或擅改Loss。

详见批准的 [归因报告](../../outputs/diffusiondrive_v1/checkpoint_evaluation/audit.md#当前465326步的收敛归因实测)、[汇总证据](../../outputs/diffusiondrive_v1/checkpoint_evaluation/attribution.json)及 [逐帧观测](../../outputs/diffusiondrive_v1/checkpoint_evaluation/attribution_samples.jsonl)。建议从较健康Best另立单因素短对照，先检验学习率与有效batch；保持旧合同及暂停断点，不把这类改配方实验冒称同合同续训。本轮只做已批准的只读诊断，没有启动修复训练。

20:04的训练异常来自Target USB盘EIO；20:31重挂后的可见UUID、故障样本SHA以及450000步断点已核验。控制器仅支持正常paused状态的resume；用户要求暂不改代码，因此没有新增recover或篡改state。恢复入口见 [快速启动：启动与断点续训](../../快速启动.md#5-启动与断点续训)；旧control不会自动接管，本轮未自动重启。后续应继续关注USB/UAS链路稳定性。

## Batch64 + 1e-4 权重微调：已完成Smoke Test，未启动2000步

> **状态：历史实验（Smoke 已冻结）。** 当前正式实验见下一节；本节命令不用于当前训练，仅用于复现三步 Smoke。

独立配置为 [baseline_bs8_acc8_lr1e4_warmstart.json](diffusiondrive_v1/baseline_bs8_acc8_lr1e4_warmstart.json)。从Best26347只加载模型，763份参数/Buffer逐项相等；不恢复父Optimizer、Scheduler、Sampler、Global Step或RNG。源Checkpoint SHA为`619736355e7739dc7f6cc771d0fd91367cf79e48e144f3e7e9c69d46880e6899`，来源及执行源码SHA写入新合同。原Best、465326步权重以及本轮开始时的旧JSON SHA均核对未变；旧JSON已被此前手工改成1e-4，本轮没有将其当作原6e-4冻结合同来修改或续跑。

旧配置还存在根级`accumulate_grad_batches=4`，入口实际只读取`training.accumulate_grad_batches`。新JSON已清理这一继承的无效键，明确使用嵌套的8；原Smoke快照保留，验收证据记录清理前后文件SHA，并核对除此忽略键外所有解析配置相等、数值合同不变。没有把根级4当作已经生效的Batch32来汇报。

| 实际设置 | 值与边界 |
|---|---|
| 数据 | 同一Baseline Manifest SHA `10fc60447fd51b6155f52030dabeecf3222b63e0a3f69acd314bef00b3d4e22f`；210776训练/7846 Town05验证，RouteGroup不泄漏 |
| 更新 | 单GPU、micro-batch8、累积8，正常更新64样本；梯度裁剪/BN/采样规则未改 |
| Optimizer | 官方Agent创建AdamW，weight_decay=1e-4；Warm Start时状态为空，step从0开始 |
| Scheduler | 官方WarmupCosLR，100epochs、warmup3、min_lr=1e-6；新基准1e-4，重置进度 |
| 实际LR | epoch0两组均3.333333333e-5，epoch1为6.666666667e-5，epoch2与3为1e-4；未加lr_scale，原版Scheduler会覆盖Backbone初始半LR |
| 精度/模型 | 原V1模型、Anchor、Loss及其权重不变；既有Conv2d BF16，其余融合/Head/Loss/验证FP32 |
| 预算 | 配置上限2000个Optimizer Step；约3294更新/完整epoch，故首轮2000步仍处epoch0，实际LR不会到达基准1e-4 |
| 诊断 | 初始、每250更新、短验收结束点；同一32训练/32验证分层面板，其中验证停车包含4个历史难例，不能当总体性能 |
| 输出 | [PSSD实验根目录](/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_warmstart/)，与历史实验及缓存隔离；此分支没有登记到旧control.json |

实现位置：Warm Start在 [fit.py](diffusiondrive_v1/fit.py) 的Agent构造后、LightningModule构造前调用 [lr_experiment.py](diffusiondrive_v1/lr_experiment.py)，修复此前agent未定义的问题；`--init-checkpoint`与`--resume`互斥。累积规则进入semantic SHA，普通resume继续严格比较合同，不能改累计次数或LR后强行恢复。新实验自己的完整Checkpoint保留Optimizer、Scheduler、Sampler及随机数，并校验执行源码SHA。

[Evidence Callback](diffusiondrive_v1/fit.py) 在非更新Micro-batch只累计窗口和检查参数未变；Optimizer真正更新后，才提交Sampler、更新样本数、记录原版Loss窗口均值及实际LR。采样预取不进入已提交游标，Checkpoint只允许pending=0。暂停请求会等到窗口完成；非有限Loss/梯度在更新前受控退出，记录最近健康Checkpoint，不保存含异常状态的当前位置，不强杀进程。异常退出写入 `failed_recoverable` 和 `resume_from`（最近健康/暂停断点；启动阶段失败回退到父 `--resume`），不把它当成正常paused断点直接接续。

[TrainingProgress](diffusiondrive_v1/training_progress.py) 区分Micro-batch与Optimizer Step，按实际已提交样本计算吞吐。26347个Micro-batch对应3294次更新：3293个完整窗口及最后3个Micro-batch/24样本。尾窗口保持Lightning原生每Micro-loss除8的规则，尾贡献为3/8而不重新归一到1。CPU回归直接验证原生Lightning的8+2窗口缩放。**BN仍按8计算，Agent框Loss按每Micro-batch有效对象数归一；累积平均不等于物理Batch64的全局对象归一。** 未改原版Loss来制造数值等价。

[training_health.py](diffusiondrive_v1/training_health.py) 复用既有评估指标与获批归因面板，以观察hook统计第三层QK Std/熵、Query差异、Q/K投影RMS、BEV memory、Ego→Status注意力、速度敏感性、分组ADE/FDE、停车误动率和检测F1。保持eval/FP32，恢复CPU/CUDA及Python/NumPy RNG，核验参数、Buffer和现有梯度版本未改变。Train面板的Loss是原版**推理分支**Loss，真实训练目标另存`step_evidence.jsonl`与CSV，不混为同一口径。工程预警阈值明确冻结；Query坍塌/注意力严重饱和会请求安全暂停，规划连续退化或QK连续快速增长也会暂停，不能称作论文官方门限。

2026-10-21 Smoke结果见 [smoke_validation.json](/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_warmstart/smoke_validation.json)：

- 真实GPU先更新2步，再严格恢复1步；共24个Micro-batch、192个唯一样本，与独立重建的种子采样序列一致，cursor=0→128→192。21个非更新Micro-batch参数保持不变，三次更新均产生有限且非零参数变化。
- 第9个Micro-batch请求暂停，到第16个才完成第二次更新并保存。恢复时546份AdamW状态、Scheduler、RNG、Sampler全部通过检查；同权重恢复后的面板指标相同。
- 实际三个优化步的两个LR组均3.333333333e-5。CPU [回归](../../tests/training/test_lr_experiment.py)及既有评估回归合计24项通过，覆盖原版Warmup、尾窗口、状态合同和诊断的RNG/梯度/BN保护。
- 现有`evaluate_checkpoint.py`另完成新合同权重的2帧只读接口Smoke，含缓存身份、轨迹/检测/BEV聚合；结果在实验根的`smoke_evaluation/`，只证明接口可用，不当成完整验证或效果验收。
- 初始→第3步的诊断验证ADE为3.1632→3.0384m，QK Std为5.4272→5.4289，熵为.5055→.5088，Query Std为.3454→.3448，未触发预警。**只有3步且是分层诊断，不能据此宣称低LR/Batch64解决了退化。** 250/500/2000步和完整7846帧评估均未执行。
- 首次零步运行因诊断器未接受Python数值Loss而失败，已修复并保留`smoke_pause`证据；成功验收在`smoke_boundary`及`smoke_resume`。原构造器可能先输出Hub不可用/未提供Checkpoint的日志，随后严格加载全部Best权重的证据才决定Warm Start是否成功；未联网下载。

[训练Loss/LR曲线](/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_warmstart/smoke_training_curves.png)、[分组规划曲线](/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_warmstart/smoke_group_curves.png)及 [网络健康曲线](/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_warmstart/smoke_resume/health/health_curves.png)均来自上述真实记录。PSSD UUID/写权限、100GiB保留门限及两次32MiB fsync/SHA通过；短检查不认证USB长期稳定，旧数据盘断连历史仍需监控。

本节是历史 Smoke 的设计/证据记录，启动与暂停细节不再复制到快速启动；需要追溯时从 [快速启动：历史与冻结入口](../../快速启动.md#9-历史与冻结入口) 进入本页和冻结证据。本轮授权仅Smoke，不自动启动短周期或长训练。未来应比较相同固定面板与完整验证集，不把192帧训练样本冒充验证，也不把改变LR和有效Batch的组合结果单独归因于某个超参数。

## Batch64正式100 Epoch训练

> **状态：旧 Warm Start 实例已暂停在 step 6538；当前 Scratch 从头实验从 step 0 运行。** 两个实验的输出和断点分开保留；本节设计说明不替代快速启动的只读命令。

2026-10-21按用户授权启动独立正式分支，复用上述24项回归及3步GPU验收，不重跑Smoke。正式配置 [baseline_bs8_acc8_lr1e4_formal.json](diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal.json)及 [启动登记](/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_formal/launch.json)冻结路径、配置SHA、来源Best、实际命令和PID。运行状态以该目录实时文件为准，不把本段日期或历史吞吐当实时状态。

- 全量210776训练样本，每epoch26347个Micro-batch，约3294个Optimizer更新；`max_epochs=100`、`max_steps=329400`同时成立，已移除独立短实验的2000步预算。新合同冻结预算、完整验证频率、Best指标与断点保留策略；不接受CLI改horizon后混续。
- micro8、累积8、基准LR1e-4、AdamW/weight_decay1e-4、原WarmupCosLR/Anchor/全部Loss及保守Conv2d BF16保持验收设置。权重Warm Start重置旧优化状态，实际epoch0两组LR为1e-4/3；正式训练开始后按原Scheduler逐epoch推进。
- 每250个Optimizer更新执行既有64帧健康面板；每1个完整epoch执行7846帧Town05验证。正式Best按`val/ade_m`选择，FDE及多任务验证Loss同时记录；Attention/Query和持续验证退化依旧按原工程规则合作暂停，不因单批Train Loss波动停止。
- 长训若保留全部250步健康Checkpoint将接近900GiB，超过启动时443GiB可用空间。因此只在正式配置启用初始断点+最近4份`health/diagnostic-step*.ckpt`保留，所有数值JSON/曲线保留；`checkpoints/`的Best/last和`periodic/`最新断点独立保存。只清理该新运行目录内严格命名的旧诊断断点，不跨目录触及历史实验、初始权重或缓存。
- 仅为上述明确的长训容量和预算问题局部修改包装层，2项针对性CPU回归通过；未重测模型或改训练公式。旧包装层源码SHA快照保存在正式目录`inherited_sources/`，短实验JSON及其证据未改；其冻结源码SHA与当前实现不同，不能拿历史短实验命令绕过校验复用当前实现。

操作入口见 [快速启动：当前状态](../../快速启动.md#1-当前状态与唯一入口)。Checkpoint、健康指标和实际LR均在PSSD，至少保留100GiB。异常健康暂停和更新前非有限停止的区别沿用上节；完整恢复只使用本正式分支的同合同断点。

### Scratch 从头实验（当前）

独立配置 [baseline_bs8_acc8_lr1e4_scratch.json](diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json) 由当前正式合同复制，移除了 `warm_start_source`，SHA 为 `5b5ab97000a0450cdb47d5da87007e592da5c91fcf89342cb451ddd69bbb66f3`。实验输出为 `/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_scratch_20261021_210751`；启动命令、PID、配置SHA和旧实例暂停断点见该目录 `launch.json`，启动前检查见 `preflight.json`。

- **初始化：** `fit.py` 以 `checkpoint_path=None` 构造官方 Agent；Scratch 启动没有 `--resume`/`--init-checkpoint`，配置也没有 `warm_start_source`。官方 ResNet34 ImageNet 图像编码器由本地预训练文件加载，固定 Anchor 路径/SHA 校验；其他 Backbone、LiDAR encoder、Transformer decoder、Agent/轨迹/BEV Head 和 BEV projection 均由官方构造器随机初始化。`initialization_audit.json` 显示 step0 Checkpoint 的 Optimizer state 为空、无 `warm_start`/`resume` 记录，图像编码器 216 个张量与预训练权重逐位一致。`training.log` 可能先出现 timm/Hub 离线加载失败日志，随后官方代码回退到本地 `bkb_path`；以审计文件为准，不能只看该错误行。
- **预算与精度：** 全量 210776 训练帧、7846 Town05 验证帧；micro8×累积8、effective batch 64、基座LR1e-4、AdamW、官方WarmupCosLR 100 Epoch/329400 更新，Conv2d BF16、融合/Head/Loss/验证 FP32。Warmup 前 3 Epoch 的实际两组 LR 从 `3.333333e-5` 开始，epoch 3 起为 `1e-4`；不能把配置基准LR当成第一步实际LR。
- **保护与监控：** 继续每250个Optimizer Step健康面板、每Epoch完整Town05验证、periodic Best/last与健康断点；P007 的累积边界保存守卫对本次从头训练同样生效。Scratch 新断点不需要合同迁移标志；旧 Warm Start 断点仍需 `--allow-checkpoint-boundary-fix`。
- **比较计划：** 在相同训练数据、验证协议、Epoch/样本曝光量下，对完整 Town05 比较全量 ADE/FDE、停车/缓行/低速起步分组 ADE/FDE、Agent F1、BEV mIoU、QK Score Std、Attention Entropy、Agent Query Std 和真实训练 Loss。不得用不同实验的 Global Step 直接横比，也不得用固定 64 帧健康面板代替完整验证结论；健康面板只用于早期异常监测。

### Scratch 转弯/停车/低速起步只读归因（2026-10-22）

固定行为面板包含 Town05 验证 128 帧（8 类×16，23 条 Route）和训练 64 帧（8 类×8，58 条 Route），训练/验证 Route 无重叠；比较 Best（step 9882）与 late（step 58500，零基 epoch 17）。执行使用 CPU/FP32、同一 seed、不向 forward 传 target、20 个候选只读 hook 捕获，optimizer_steps=0；386/386 条 selected 轨迹与 score argmax 候选逐位一致。面板刻意行为均衡，不是自然频率总体，不能把这些 ADE 当完整 7846 帧验证结果。协议、逐样本和汇总在 [归因输出](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/)：[evaluation_protocol.json](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/evaluation_protocol.json)、[selected_oracle_metrics.csv](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/selected_oracle_metrics.csv)、[behavior_metrics.csv](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/behavior_metrics.csv)、[failure_cases.csv](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/failure_cases.csv)、[行为曲线](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/behavior_selected_oracle.png)、[Frame317 案例](../../outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/case_frame317.png)。

128 帧验证汇总（oracle 是在前向之后用目标 ADE 选候选的**可达性上界**，不是可部署选择器）：

| Checkpoint | selected ADE/FDE m | oracle ADE/FDE m（GT 上界） | 选择差 m | 纵向/横向 MAE m | 航向 MAE ° | 检测 F1 | BEV mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best 9882 | 3.114 / 6.877 | 1.086 / 2.281 | 2.028 | 2.664 / .917 | 7.04 | .250 | .298 |
| Late 58500 | 3.206 / 7.162 | 1.071 / 2.476 | 2.135 | 2.920 / .698 | 5.13 | .486 | .392 |

关键行为分组（验证集，selected/oracle）：

| 分组 | Checkpoint | ADE m | 选择差 m | 横向 MAE m | 航向 MAE ° | 行为观测/错误率 |
|---|---|---:|---:|---:|---:|---:|
| 停车 | Best | .791/.673 | .117 | .092 | .52 | 误动 43.75% |
| 停车 | Late | 2.068/.784 | 1.284 | .054 | .46 | 误动 56.25% |
| 接近停车 | Best | 5.263/1.158 | 4.106 | .473 | 2.58 | 终点 21.77 m，目标 10.76 m |
| 接近停车 | Late | 4.807/1.251 | 3.555 | .384 | 2.47 | 终点 21.93 m，目标 10.76 m |
| 低速起步 | Best | 2.398/1.090 | 1.308 | .191 | 2.11 | 漏起步 50.0% |
| 低速起步 | Late | 2.873/1.305 | 1.567 | .212 | 3.03 | 漏起步 56.25% |
| 左转 | Best | 5.107/1.370 | 3.737 | 2.952 | 23.16 | — |
| 左转 | Late | 4.561/1.287 | 3.274 | 2.450 | 12.96 | — |
| 右转 | Best | 2.937/1.446 | 1.491 | 1.184 | 9.93 | — |
| 右转 | Late | 2.695/1.111 | 1.584 | .803 | 8.35 | — |

- **训练面板拟合变好，但验证选择泛化变差。** 64 帧训练面板 selected/oracle 为 Best 1.789/.734 m（差 1.055）、late .384/.259 m（差 .124）；同权重验证为 3.114/1.086 m（差 2.028）和 3.206/1.071 m（差 2.135）。late 在训练样本上明显拟合更好，验证选择差却扩大，不能用训练面板改善证明驾驶能力恢复。
- **候选覆盖不是首要瓶颈，选择才是。** 386 条样本均有有限候选；验证集内 selected/oracle 差超过 2 m 的比例：接近停车 50%/50%（Best/late）、左转 50%/50%、右转 19%/38%、低速起步 25%/25%、直线 25%/31%、交互 25%/19%；其中左转和接近停车最集中。若候选整体缺失，oracle 也会同样差，但事实是 oracle ADE 始终约 1.07–1.09 m。
- **转弯失效以“转向不足/错误机动模式”为主，误差沿 4 s 时域累积。** 左转 Best 轨迹总航向变化均值约 −4.5°，目标是 27.5°，oracle 约 30.9°；late 为 16.7°、oracle 26.6°。右转 Best/late 为 −23.9°/−15.3°，目标 −34.4°，oracle −24.5°。验证左转 selected 横向误差 Best 2.95 m、late 2.45 m，oracle ADE 仍约 1.3 m；候选集中存在正确转弯，主要错误是评分选到了近似直行/不足转向的候选。
- **停车是候选生成与模式选择共同失败，越到后期选择越差。** 停车目标终点平均 0.07 m，Best/late 选中轨迹终点 1.88/5.24 m；即使按 ADE 选 oracle，仍有 43.75%/56.25% 样本终点超过 1 m，说明候选集本身也缺少可靠的零位移模式。接近停车组 selected 终点约为目标的两倍，oracle 也有 43.75%/62.5% 样本超出目标 2 m，但 selected 的超限率为 75%/87.5%，选择差仍是主要放大项。
- **低速起步主要是选择失败。** 起步组目标终点平均 10.53 m，selected 仅 4.43/5.11 m；漏起步率 50%/56.25%，而 oracle 仅 12.5%/18.75%，且候选集没有“所有候选都漏起步”的样本（0%）。即候选能给出起步轨迹，分类头没有选它。
- **Frame317 是同一机制的极端样例。** 目标路径仅 0.034 m；Best 选中 mode 18，ADE 10.56 m、FDE 23.01 m，oracle mode 8 为 0.96 m；late 选中 mode 2，ADE 4.61 m、FDE 12.41 m，oracle 为 1.43 m。两次 argmax 的 score margin 仅 .071/.046，late 甚至给错误的前向候选更高分；这是停车误动/排名问题，不是“没有候选”。
- **late 不是整体灾难性遗忘，但不能说更好。** 同一行为面板 selected ADE 从 3.114 增到 3.206（+2.9%），oracle 从 1.086 降到 1.071（略好），检测 F1 .250→.486、BEV mIoU .298→.392；转弯 selected 反而改善，但停车、低速起步、蠕行和交互分组变差。late 是 step 58500/epoch 18，不是 Epoch 12–15 的完整 Checkpoint；本轮没有做 7846 帧全量验证，不能据此宣称后期权重全面退化。
- **Adapter 不在本实验的零级因果链上。** 本轮读取冻结缓存并直接前向，未经过在线 B2D Adapter；因此这里数米级的选择差不能归给 Adapter。既有 64 帧审计仍未确认 FAIL，但保留状态后向差分约 50/100 ms 时延、差分加速度与 IMU xy MAE 约 1.3–1.9 m/s²、18/64 帧 near/far 方位分类不同、精确时钟/同步 NOT_VERIFIED 等风险；它们可能恶化闭环停车/起步，但不是本面板选择差的直接原因。
- **下一步最小训练实验（建议，尚未执行）：** 从 Best 9882 初始化，冻结 Backbone、候选回归和感知头，只训练/校准 20 路 mode scoring head，使用训练集 candidate ADE 排名目标（最小 ADE 候选为正，加入停车/起步/转弯 hard negatives）并做行为均衡采样；验收固定用同一 128+64 面板和完整 Town05，要求选择差、停车误动率、起步漏检率同时下降。不要再做全参数长训来期待选择问题自行消失；若更换 Loss/模型，应另立合同并先获得授权。
- **2026-10-22 Adapter 第一阶段只读原型：** A2V（speed 幅值融合速度 + A0 加速度）在验证集把速度 vMAE `.127→.051 m/s`、P95 `.448→.166`，但加速度没有改善；转弯/接近停车的速度抖动仍为 SUSPECT，不能直接更新正式 Feature。路线 Route ID 覆盖仅 `2/81`、dense path `0/81`，route-preview 仍被数据来源阻塞；停车/起步 36 个相似 Status 配对中 A=31、B=5，Camera/LiDAR 充分性仍是 NOT_VERIFIED。详见 [数据接口分析与证据](../../analysis/data_interface_analysis/README.md) 的「Adapter 输入语义第一阶段只读原型」节。
- **2026-10-22 Status 上游集成准入：** 独立 Tracker、A0 诊断和 Status Overlay 通过 75 项相关 CPU 回归；81 Route / 192 Frame 保留 190 个几何有效 Preview 和八维数值候选，但来源认证 Status 为 0。20 Route 端到端容量试运行处理 8806 帧，几何有效 5555、数值候选 5507，局部路径末端覆盖仍不足；全部 `training_admitted=false`。正式 A0、八维 `[command4,velocity2,acceleration2]`、Feature/Target Producer、Manifest 和 Checkpoint 均保持不变，Scratch paused/58650；没有接入正式训练 reader。全量生成与微调准入维持 **NO-GO**，依据见 [测试与准入报告](../../outputs/diffusiondrive_v1/route_preview_v11/20261022_status_upstream/status_builder_test_report.md)，命令仅见 [快速启动 6.3](../../快速启动.md#63-status-上游隔离审计cpu)。
- **2026-10-22 候选轨迹生成质量与 Anchor 覆盖只读诊断：** 新增 `tools/validation/b2d/candidate_quality_diagnostic.py`（anchors/evaluate/report 三个只读子命令，`verify_pins` 前置，不传 target、0 优化步）+ 9 项 CPU 回归，输出在 [候选质量诊断](../../outputs/diffusiondrive_v1/candidate_quality_diagnostic/REPORT_CN.md)。核心结论：20 条官方 Anchor **没有零位移模式**（最小终点 3.22 m，mode 8），而验证集 7846 帧中 48.9% 的 GT 路径 ≤0.1 m 且 **100% 落在 mode 8**；停车帧候选终点中位数 22.7 m、7/16 帧无任何 ≤1 m 候选，`原地不动`在 15/16 帧优于 20 候选最优者；但 `anchor_mode` 候选 ADE 1.189 m ≈ oracle 1.086 m（Anchor↔Oracle 0.812），而 Selected↔Anchor 仅 0.531、转弯合格覆盖 1.000 却选中欠转率 50%/50%。因此**停车是 Anchor 覆盖 + Decoder 压缩不足，转弯主要是 Mode 评分泛化**；Late 仅停车子集退化（mode 8 终点 1.58→2.32 m），非整体候选退化。ADE/Oracle/行为标签全部只写诊断目录，未进入训练、argmax、采样或 Checkpoint 选择；本轮未改官方源码、正式配置、缓存与 Manifest。

## Route 末端监督审计（2026-10-24）

全量 1,100 条原始 Route、44,000 个末尾帧：A=0、B=1、C=0、D=1,093、E=6；冻结 Target 队列中这 44,000 帧均为 `ineligible/insufficient_future`。当前可自动恢复样本为 **0**，不能将已完成状态或末帧低速等同于终点稳定停车。唯一 B 类末帧前刚重新加速；6 条 Failed 的具体终止原因没有记录，不能猜作超时。

现有 Builder 保留默认拒绝未来不足的行为；显式 `terminal_padding=True` 只允许重新读取原始完成证据、实测时间与稳定停车几何后认定的 A 类，仅重复真实终点后轴位姿。补齐仍使用原坐标转换和 `[8,3]` 合同，不生成未来传感器、Agent、BEV 或 Status8。当前标注没有逐帧时间戳，40 帧的名义首尾跨度是 3.9 秒，严格物理时长尚未验证。新增 `terminal_policy.py` 只负责资格检查，不建立第二套 Builder。

[末端审计与逐 Route CSV](../../outputs/diffusiondrive_v1/terminal_target_recovery/20261024/audit/route_terminal_audit.md) · [恢复策略、函数差异及测试结论](../../outputs/diffusiondrive_v1/terminal_target_recovery/20261024/audit/terminal_target_recovery_report.md)。62 项 CPU 测试通过，1 项原有传感器集成测试未启用；9 个真实样本均被正确拒绝，独立派生目录为空。此次源码具有独立 SHA，未重签旧 producer、缓存、Overlay、Pair Index 或配置；没有修改划分、网络或 Loss，没有启动训练。复现入口只在[快速启动](../../快速启动.md#7-cpu-回归)维护。

**2026-10-24 全量帧恢复审计：** 279,032 个唯一 Target 帧中，193,471 已训练、44,000 未来不足、41,561 其余待恢复；A=0、B=9,242、C=0、D=44,000、E=7,168、F=25,151。对 B 的 9,242 个已有 Feature+Target 的恢复配对重算 Status，8,455 帧通过原合同进入候选索引（train 8,405 / validation 50），原 193,471 行 0 改动；候选索引 201,926 行。historical28 的 28 个 Target 已复制到 PSSD 并加入新 registration `f02431f4...`，五个来源均可读取。完整账目、218,622/193,471/25,151/11,258/1,183/12,710/1,677 的解释和规则审计见 [frame_recovery_audit.md](../../outputs/diffusiondrive_v1/route_preview_v11/20261024_frame_recovery/frame_recovery_audit.md)。

## Route split v2 正式实验（2026-10-25，当前）

当前实验 `status_routepreview_v2_split801010_seed0_20261025_012809` 使用 201,926 行候选快照的 seed-0 Route 8:1:1 split：880/110/110 Route，train 160,266 / validation 17,191 / test 16,202，val/test 路线尾部 40 帧排除 8,267。派生索引、配对索引和 split manifest 位于 [split_seed0_801010](../../outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/split_seed0_801010/dataset_split_manifest.json)。Test 只在最终评估时读取，训练/调参/Checkpoint 选择均不加载。

训练语义恢复为官方 V1：`TransfuserAgent(checkpoint_path=None)` + 官方 timm ResNet34 ImageNet；AdamW、base lr 6e-4、weight_decay 1e-4、WarmupCosLR 100 epochs/warmup 3；100 epochs；无旧 Status init/resume；无 `validation_limit`/每250 step/`val/ade_m` Checkpoint。micro batch 16 × accumulation 4 = effective 64；BF16 仅 Backbone Conv2d 的工程策略保留，loss/eval 仍 FP32。Checkpoint 仅按全量 `val/loss` 保存 best + last；ADE/FDE/heading 仅作诊断。

正式审计与证据：[训练清理审计](../../outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/diffusiondrive_v1_training_cleanup_audit.md)、[污染审计](../../outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/checkpoint_dataset_contamination_audit.md)、[preflight](../../outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/training_preflight_v2.md)、[当前状态](../../TRAINING_STATUS.md)。配置为 `experiments/status_routepreview_v2_split801010_seed0_20261025_012809.json`。

**2026-10-25 暂停与 Best/Latest 评估：** 训练在 step 43,650 合作暂停，Best 为 step 5,008。17,191 行 Validation 的配对评估（FP32、batch16、GT 不进 forward、34,382 次预测）显示：Latest ADE 1.7808→1.8675 m、FDE 3.8733→4.2958 m；Agent box+class 占 Loss 增量 **92.32%**，Precision/Recall 从 0.531/0.514 降到 0.030/0.019；BEV mIoU 0.3538→0.3183。主要退化在 Agent 分类/召回支路，其次 BEV 多类特征；轨迹退化集中在正常行驶、左转和停车尾部。完整报告：[evaluation_report.md](../../outputs/diffusiondrive_v1/route_preview_v11/20261025_eval_routepreview_v2_best5008_latest43650/full/evaluation_report.md)。

### Agent box 退化定位与修复（2026-10-25）

Validation 曲线不是渐进过拟合：Agent box 在 epoch1 为 5.147（Best）、epoch2 6.273、**epoch3 阶跃到 22.698**，Agent class 1.42→3.00、BEV 5.04→10.48，之后未恢复。两个 checkpoint 的 scheduler `base_lrs=[6e-4,3e-4]`，但实际两个参数组 lr 都是 6e-4（Latest 5.697e-4），说明官方 `WarmupCosLR` 丢失了 `image_encoder lr_mult=0.5`；同期 Backbone 权重 L2 2930.8→4047.7、BN running_var max 10,063→93,801、Agent head L2 20.0→40.9、BEV head 9.7→30.1。诊断报告：[agent_box_degradation_diagnosis.md](../../outputs/diffusiondrive_v1/route_preview_v11/20261025_eval_routepreview_v2_best5008_latest43650/agent_box_degradation_diagnosis.md)。

官方机制缺陷：`transfuser_agent.get_coslr_optimizers` 只在新增参数组上写 `lr`，从不写 `lr_scale`；`WarmupCosLR.get_lr` 只有在 `param_groups[0]` 含 `lr_scale` 时才按组缩放，因此 yaml 声明的 `lr_mult` 被完全抹平。**官方仓库保持逐字未改**，适配全部落在本地入口 `status_experimental_train.py`（VERSION 升为 `status-experimental-train-v3`）：

| 修复 | 位置 | 行为 |
|---|---|---|
| 分组 LR | `install_paramwise_lr_scale` + `apply_paramwise_lr_scale` | 按 `opt_paramwise_cfg` 的 name 模式给每组写 `lr_scale`，重建同一官方 `WarmupCosLR`；image_encoder 峰值 = base×0.5 |
| 启动即证伪 | `optimizer_contract` | 训练前构建真实优化器，断言每组 `lr_scale`、warmup 首值、峰值；缺 `lr_mult` 直接 `SystemExit`，写进 `run_identity.json` |
| 续训回写 | `audit_lr_scale` + 回调 `on_train_start` | `Optimizer.load_state_dict` 会用旧断点覆盖 `param_groups`，因此启动时重写 `lr_scale` 并按 `scheduler.get_lr()` 重设 lr，逐次追加 `lr_schedule.jsonl` |
| 梯度保护 | `Trainer(gradient_clip_val, gradient_clip_algorithm='norm')` | 短程三臂统一 1000（实测正常总范数 608–645，见下）；其余实验默认 1.0 只适用于旧入口 |
| 静默失败防线 | `grouped_gradient_norms` + 回调 `on_before_optimizer_step` | 每 N 步写 `grad_norms.jsonl`（backbone/bev_head/agent_head/trajectory_head/other）；非有限 loss 或 grad norm 立即停止，不留到下一 epoch |
| 数据盘保护 | `main` 输出目录创建 | 输出根的上两级不存在（外置盘未挂载）时拒绝 `mkdir`，避免在系统盘上遮住挂载点 |
| 父断点 | `--init-weights/--init-weights-sha256` | 只加载 763 个权重张量（复用 `status_route_eval_core.load_weights_only`），不恢复 optimizer/scheduler/global_step；与全量 `--resume`（仍限制在实验目录内）严格分离 |

真实 `TransfuserAgent` 离线校验：组 0 为 439 个张量（lidar encoder、transformers、各 head）`lr_scale=1.0`，组 1 为 108 个 `_backbone.image_encoder.*` 张量 `lr_scale=0.5`，scheduler 均为 `WarmupCosLR`。回归测试 `tests/training/test_lr_scale_guard.py`（8 项）覆盖：分组写回、warmup/峰值、续训回写、混合分组拒绝、分支 grad norm、非有限检测、A/C 合同、官方源码未被改动。

短程对照配置（5 epoch、从 Best 5,008 只加载权重、`grad_norms.jsonl` 每 10 步）：

**裁剪阈值为什么是 1000 而不是 1.0：** 首次启动实测（step 0 / 10）总梯度范数为 **608.5 / 645.2**，其中 trajectory_head 占 607.1 / 640.9，backbone 24.3 / 34.6，agent_head 24.6 / 63.3，bev_head 1.7 / 1.8。若用 clip 1.0，每一步都会被等比缩小约 640 倍（AdamW 可大体抵消，但小梯度参数会被 eps 显著抑制），等于把 A 臂变成另一个实验；因此三臂统一改为 **1000**（约正常值 1.5–1.6 倍），只截断真正的尖峰。首次那版 clip=1.0 的 10 余步试探已停止并删除，不开留作实验目录；阈值改完后 `grad_norms.jsonl` 新增 `clip_scale` 字段，可直接看出每步是否被裁。

| 臂 | 配置 | base lr | 实际峰值（其余组 / image_encoder） | 作用 |
|---|---|---|---:|---|
| A | `experiments/status_agentbox_shortA_lr3e4_lrscale_clip1000.json` | 3e-4 | 3e-4 / 1.5e-4 | 最保守修复 |
| B | `experiments/status_agentbox_shortB_lr6e4_lrscale_clip1000.json` | 6e-4 | 6e-4 / 3e-4 | 分离 lr_scale 与降峰值 |
| C | `experiments/status_agentbox_shortC_lr3e4_flatlr_clip1000.json` | 3e-4 | 3e-4 / 3e-4 | 诊断对照，复现抹平行为，不晋级长训 |

启动前必须先用 `--dry-run` 确认 `optimizer_contract`；通过条件：Agent box 不再在 epoch3 阶跃（相邻 epoch 增幅 <2×）、Agent Precision/Recall ≥ Best 的 90%、BEV 不继续上升。BN 冻结（原 D 臂）尚未实现，仅在 A/B/C 仍不稳时再加。

**当前阻塞：** `/run/media/aethelind/PSSD` 与 YMTC 特征盘在诊断时未挂载，`/data/datasets` 处于 `emergency_ro` 且有 I/O error，因此 A/B/C 只完成静态校验、真实 Agent 优化器合同校验和 CPU 回归，**尚未启动任何训练**；恢复前需重新挂载外置盘并核对 Best 断点 SHA `ea499a04...`。

### 验证口径修正（2026-10-26，v4）

诊断发现 `val/loss` 与训练目标不同口径：`TrajectoryHead.forward` 在 eval 走 `forward_test`，预测里没有 `trajectory_loss` 键，`transfuser_loss` 回退成 `12 × 单步 plain L1`，实测量级只有真实目标的 **1/4.5–1/4.7**（32 个验证样本离线复算：83.96/61.82/50.28 vs 18.71/13.11/11.22）。因此旧 monitor 实际被 agent + BEV 主导，轨迹项被压到约 21% 权重。

v4 入口的修正（训练数学、梯度、优化器**均不变**，只改评估与选点）：

| 项 | 实现 |
|---|---|
| `training_form_loss` | 仅把 `_trajectory_head` 切到 `forward_train` 分支，冻结全部 Dropout，按 `paper_seed = epoch*100003+batch_idx` 固定扩散采样，结束后恢复分支/Dropout/RNG |
| 日志 | 新增 `val/paper_loss` 与 `val/paper_{trajectory,agent_class,agent_box,bev_semantic}_loss`；ADE/FDE 仍由 `forward_test` 计算，指标口径不变 |
| Monitor | `checkpoint_monitor` 配置生效（之前写死 `val/loss`），可选 `val/loss` / `val/paper_loss`，并记录 `checkpoint_policy.validation_objective` |
| 代价 | 每个验证 batch 多一次 2 步扩散前向，验证时间约翻倍（≈9 → ≈18 min/epoch） |

真实 `TransfuserAgent` 冒烟：`val/paper_loss=62.39`（traj 44.53 + cls 1.35 + box 2.18 + bev 14.33），`head.training` 与 RNG 均已恢复，同 seed 两次结果完全一致（delta 0.0）。

### 可接入训练的数据规模（2026-10-26）

口径：一帧计为**可接入**需同时满足 ① 在 `split_seed0_801010` 配对索引中 ② Feature 索引 `complete` 且有 `bytes` ③ Target 索引 `complete` 且有 `bytes`。空间按成对帧自身大小累加。`excluded_route_tail` 属于划分策略产物、不进入任何可用于训练或评估的集合，因此**不计入下表**。数据源为只读查询，未改动任何索引、缓存或 Checkpoint。

| 数据划分 | 划分帧数 | 可接入帧 | 对应 Route | Feature | Target | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| Train（训练集） | 160,266 | **160,213** | 847 | 548.23 GB | 9.42 GB | 557.66 GB |
| Validation（验证集） | 17,191 | **17,191** | 102 | 58.83 GB | 1.01 GB | 59.84 GB |
| Test（测试集，保留至最终评估） | 16,202 | 16,201 | 105 | 55.44 GB | 0.95 GB | 56.39 GB |
| **合计** | **193,659** | **193,605** | **1,054** | **662.50 GB** | **11.38 GB** | **673.88 GB** |
| 其中 train + validation（实际训练与选点） | 177,404 | 177,404 | 949 | 607.06 GB | 10.43 GB | **617.49 GB** |

Route 数三集互不重叠（847 + 102 + 105 = 1,054）。候选 Route 全集 1,100 条中有 35 条完全没有张量，划分按 880/110/110 分配后，实际有可用帧的是 847/102/105。

Target 缺口共 54 帧（train 53 + test 1），集中在 `AccidentTwoWays_Town13_Route30356_Weather10_02-10-03-31-*` 一条 Route，其 Target 位于 `historical26`/`historical28` producer，未进入当前主索引，不影响训练运行。

另需注意：`route_split.py` 会把 validation/test 每条 Route 的末尾 40 帧切出并标为 `excluded_route_tail`（共 8,267 帧 / 28.78 GB，分属 218 条 Route），train 的 Route 尾部则完整保留。该集合**既不参与训练也不参与评估**，且**不能加回 train**——这些帧物理上属于 validation/test 的 Route，回填会造成 Route 级数据泄漏。实测 `excluded ∩ train = 0`。

**实际参与训练与选点：train + validation = 177,404 帧 / 949 Route / 617.49 GB**。Test 的 16,201 帧不参与训练、调参或 Checkpoint 选择。

| split | 可接入帧 | Left | Right | Straight |
| --- | --- | --- | --- | --- |
| train | 160,213 | 30,621 (19.1%) | 29,693 (18.5%) | 99,899 (62.3%) |
| validation | 17,191 | 4,828 (28.1%) | 1,814 (10.6%) | 10,549 (61.4%) |
| test | 16,201 | 2,859 (17.6%) | 1,979 (12.2%) | 11,363 (70.1%) |
| **train + validation** | **177,404** | **35,449 (20.0%)** | **31,507 (17.8%)** | **110,448 (62.3%)** |

读取 100% 命中：Feature 索引覆盖全部 201,926 帧，Target 缺口仅 54 帧（train 53 + test 1），全部集中在 `AccidentTwoWays_Town13_Route30356_Weather10_02-10-03-31-*` 一条 Route，其 Target 位于 `historical26`/`historical28` producer，未进入当前主索引，不影响训练运行。

**存储分布（`du --apparent-size` 实测，与索引逐项一致）：**

| 类别 | 位置 | 占用 |
| --- | --- | --- |
| Feature | YMTC `diffusiondrive_v1_tensors_YMTC/features` | 482.85 GB（51.6%） |
| Feature | PSSD `diffusiondrive_v1_tensors_PSSD/features` | 452.06 GB（48.4%） |
| Target | PSSD `diffusiondrive_v1_targets_cache`（baseline） | 13.02 GB |
| Target | PSSD `…targets_recovery_20261017_r2`（projected + compatible） | 0.54 GB |
| Target | PSSD `…targets_recovery_20261017`（历史） | 0.05 GB |

单帧均值 Feature 3.42 MB、Target 58.8 KB。Feature 索引登记 273,212 帧 / 934.91 GB 与两块盘实测完全吻合，说明 2026-10-19 从系统盘迁出的 78,392 帧已全部落到 PSSD，无重复占用。

**两类张量覆盖不一致是当前主要存储冗余：** Feature 覆盖 273,212 帧而 Target 只 230,401 帧，**71,286 帧（243.93 GB，占 Feature 总空间 26%）有特征但未进入任何划分**，来源为异常区间排除（12,712 帧）、12 条缺陷 Route 及未选用的 Route 帧。这部分不是损坏数据，而是当前 split 主动排除的样本；是否回收需先确认是否有补做 Target 或重划分计划，不能按目录直接删除。

候选 Route 全集 1,100 条，实际有张量的是 1,065 条（35 条无帧），故划分按 880/110/110 分配但可接入 Route 为 847/102/105。

### 余弦退火跨度修正（2026-10-26，v5）

诊断：contA 在 25 个 epoch 内 `val/loss` 于 15.51–16.77 之间震荡，**best 始终停在 ep10（15.5080）**，最后一个 epoch 反而恶化 8.2%。根因不是超参不合适，而是**官方 `WarmupCosLR` 的 `epochs=100` 被硬编码**（`transfuser_agent.py:175`），而实验只跑 25 个 epoch：

| | 名义 | 实际 |
| --- | --- | --- |
| 退火区间 | ep3 → ep100 | ep3 → ep26 就结束 |
| 走完的余弦角度 | 180° | **33.4°** |
| LR 降幅 | −99.7% | **−13.2%**（3.0e-4 → 2.59e-4） |

从 checkpoint 读到的 `lr_schedulers` 状态证实：ep20 时 `last_epoch=21`、`_last_lr=[2.7531e-4, 1.3765e-4]`——调度器在推进，只是**曲线太扁平**，等于全程恒定 LR 训练。

**v5 修正（官方仓库零改动）：** 新增配置项 `scheduler_epochs`，默认等于 `max_epochs`，构建调度器时传入；**续训时必须在 `_enforce_lr_scale` 里重新注入**——`WarmupCosLR.state_dict` 会保存 `epochs`，`load_state_dict` 会把 checkpoint 里的 100 原样恢复，否则修正会被静默覆盖。注入后按新跨度重算 `get_lr()` 并写回各参数组。

真实 checkpoint 验证（contA `last.ckpt`，`last_epoch=25`）：

| 场景 | scheduler_epochs | 组0 LR | 组1 LR |
| --- | --- | --- | --- |
| 注入前（恢复成 100） | 100 | 2.6363e-4 | 2.6363e-4 |
| 注入后 | 50 | **1.6546e-4** | 1.6546e-4 |
| ep50 末段 | 50 | 1.0e-6 | 1.0e-6 |

同时修复 `training_progress.json` 的 LR 字段：此前只在 `on_train_start` 写一次快照，导致 ep21 仍显示 ep5 的值；现改为每次写入时从 `trainer.optimizers[0].param_groups` 现读（`live_learning_rates`）。

新增回归 3 项（`tests/training/test_lr_scale_guard.py`）：跨度覆盖并重算 LR、重复注入幂等、进度 LR 现读。`tests/training` 全量 **144 passed**。

**注意首步降幅：** 从 100 改为 50 后，ep25→ep26 的 LR 由 2.60e-4 降到 1.56e-4（−37%），这是把此前漏掉的退火一次性补齐，不是新的超参变更。`AGENTBOX-PAPERVAL`（30 epoch + 100-epoch 余弦）因此未启动，由 `AGENTBOX-COS50` 取代。

### EMA 与 BEV head 降 LR（2026-10-26，v6）

**背景：** cos50b 已把学习率这一变量排除干净——ep25→ep31 LR 从 1.65e-4 降到 1.05e-4（−36%），`val/paper_loss` 反而从 76.49 升到 79.39（+3.8%），同期 train 从 20.53 降到 15.50（−24.5%）。退火**加剧而非缓解过拟合**。同期 `val/bev_loss` 是唯一持续恶化的分项（5.68 → 6.09–6.22，约 +8%），而 `bev_head` 只占 0.55% 梯度范数（2.35 / 425.7），说明它既没被轨迹头保护、又缺少独立正则。

**R1 只改两项**，loss 公式、数据划分、batch、clip、峰值 LR 全部不动，保证 `val/paper_loss` 与 cos50 跨臂可比：

| 变量 | cos50 | R1 | 目的 |
| --- | --- | --- | --- |
| EMA | 无 | `ema_enabled=true`、`ema_decay=0.999` | 降低 `val/*` 权重抖动 |
| BEV head LR | 1.0× base | `bev_semantic_head.lr_mult=0.5` | 阻止 BEV 分项中后期恶化 |

**checkpoint 语义（关键，勿混淆）：**

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `best-*.ckpt` / `last.ckpt` | **raw 权重**（与优化器、LR 调度器状态严格对应） | 断点续训、最终交付 |
| `ema-*.ckpt`（每 epoch 一个） | 仅 `status-ema-weights-v1` 平滑权重 + `decay`/`num_updates` | 单独评估 EMA 效果，**不能 resume** |

顺序是硬约束：`on_train_batch_end` 更新 shadow → `on_validation_start` 备份 raw 并 `copy_to` EMA → 验证在 EMA 权重上跑 → `on_validation_end` **先写 `ema-*.ckpt`，再恢复 raw**。因此 `ModelCheckpoint` 拿到的永远是 raw 权重，`val/paper_loss` 反映的是 EMA 权重。累积梯度窗口内（`global_step` 未变）跳过更新，避免同一批参数被平滑多次。

**实现要点：**

- `ModelEMA` 与 `EMACallback` 是纯逻辑类，**模块级不 import torch / lightning**（`torch` 在函数内导入），便于 CPU 单元测试。
- Lightning 的 `_validate_callbacks_list` 要求 callback 是 `pl.Callback` 真子类，`EMACallback` 直接注册会报 `ValueError: Expected a parent`。因此提供 `make_ema_lightning_callback()`，内部定义 `_LightningEMA(pl.Callback)` 委托给 `EMACallback`，真实逻辑仍留在可测试的纯类里。
- `num_updates < 100` 时用 warmup 衰减 `min(decay, (1+n)/(10+n))`，避免早期 shadow 被初始随机权重长期拖住。
- callback 注册顺序 `[ema, checkpoint, control]`：EMA 必须先于 `ModelCheckpoint`，否则 best 可能存到平滑权重。
- `opt_paramwise_cfg` 用**顶层配置键**覆盖（在 `model_configuration()` 之后、`TransfuserAgent(...)` 之前），不写进嵌套的实验说明字段——嵌套位置 `config.get` 读不到，会导致优化器只分 2 组。

**dry-run 实测（3 组优化器）：**

```
param_group_lr_scale : [1.0, 0.5, 0.5]
param_group_params   : [435, 108, 4]      # 其余 / image_encoder / bev_semantic_head
param_group_peak_lr  : [3e-4, 1.5e-4, 1.5e-4]
```

**端到端 smoke**（真实组件、`limit_train_batches=3`、`limit_val_batches=2`、2 epoch）：EMA 建立且 `num_updates=5`；生成 `ema-000-000003.ckpt`、`ema-001-000006.ckpt`；`best-001-000006.ckpt` 为 raw；训练后权重 vs 初始 L2 = 2.534（在更新），vs EMA = 0.273（验证后已恢复 raw）。

**新增回归：** `tests/training` 由 146 增至 **151 passed**（EMA 5 项：warmup 衰减、`copy_to` 精确恢复、`state_dict` 往返、验证期换入并恢复、无优化器步时跳过）。

### 指标命名对照（2026-10-26）

**`val/paper_loss` 就是训练目标在验证集上的值，是唯一的「真 loss」。`val/loss` 不是同一种量，不能与之比较。**

| 用途 | 指标名 | 公式 | 判读 |
| --- | --- | --- | --- |
| 训练实际优化 | `train/loss_epoch` | $12\sum_{k=0}^{1}[10\,\mathrm{Focal}_k+8\,\mathrm{L1}_k]+10\mathcal{L}_{cls}+\mathcal{L}_{box}+14\mathcal{L}_{bev}$ | 真值基准 |
| **验证（正确口径）** | **`val/paper_loss`** | **完全同一公式**，仅换数据集 | **可与 `train/loss_epoch` 直接比较** |
| 验证（旧代理） | `val/loss` | $12\cdot\mathrm{L1}_{plain}+10\mathcal{L}_{cls}+\mathcal{L}_{box}+14\mathcal{L}_{bev}$ | **量纲不同，不可比较** |

`val/loss` 的轨迹项只有 `12 × 单步 plain L1`：单步（只用最后一个解码步）、无 focal 分类项、无锚点匹配，且 `L1` 是对 `[8,3]` 全元素平均。辅助项（`agent_class` / `agent_box` / `bev`）三者完全相同。

**判别证据（ratio 是否跨过 1.0）：** 同一公式下验证集必然比训练集难，故 `val / train > 1`；若出现 `val / train < 1`，两个数就不是一回事。

| epoch | `train/loss_epoch` | `val/paper_loss` | paper ÷ train | `val/loss` | legacy ÷ train |
| --- | --- | --- | --- | --- | --- |
| 20 | 26.63 | — | — | 15.63 | **0.59** |
| 24 | 24.31 | — | — | 16.77 | **0.69** |
| 25 | 20.53 | 76.49 | **3.73** | 16.22 | 0.79 |
| 26 | 19.38 | 76.77 | **3.96** | 16.20 | 0.84 |

`legacy ÷ train` 全程 **0.59–0.84**（验证低于训练，数学上不可能同源），且该比值随训练单调漂移，说明旧指标与真实目标的偏离幅度本身在变——这正是它无法稳定排序 Checkpoint 的原因。`paper ÷ train` 为 3.73–3.96，符合「验证集更难」的正常关系。

**使用约定：**
- 判断收敛或过拟合，只看 `train/loss_epoch` 与 `val/paper_loss` 的间距；
- `val/loss` 与 `val/trajectory_loss` 仅作历史连续性参考，**不用于选点、不用于跨实验比较、不用于设定阈值**；
- `val/ade_m` / `val/fde_m` 是 `forward_test` 诊断量，不在任何 loss 公式内；
- 跨样本集的绝对值不可比（见 [P011](../../docs/问题记录.md)）：全量 17,191 帧与 2048 帧子集是两个尺度。

对照图：`outputs/diffusiondrive_v1/route_preview_v11/20261026_cos50_loss_curves/metric_correspondence.png`。

**同批数据三模式实测（2026-10-26，`cos50 best-026`，FP32/GPU，3 个 batch × 16 样本）：**

| 模式 | 总计 | 轨迹项 | agent_class | agent_box | BEV |
| --- | --- | --- | --- | --- | --- |
| A 训练路径（`agent.train()`） | 49.009 | 41.393 | 0.449 | 1.981 | 5.186 |
| **B 验证正确口径（`training_form_loss`）** | **48.411** | **43.288** | 0.220 | 1.402 | 3.501 |
| C 旧代理（`forward_test` fallback） | 7.745 | 2.622 | 0.220 | 1.402 | 3.501 |

- **B ÷ A = 0.988** —— 验证正式口径与训练路径相差 1.2%，量级与分项构成一致；
- **C ÷ B = 0.160** —— 旧代理只有正确口径的 16%；
- **B 与 C 的 aux 三项逐位相同**，差异完全落在轨迹项，与公式推导一致；
- A 的 aux 高于 B，源于 `train()` 模式 Dropout 生效 + BatchNorm 使用 batch 统计量，属标准 train/eval 差异，不是口径错误；
- 轨迹头只用 `LayerNorm`（无 BatchNorm），因此 `training_form_loss` 把 `head.training=True` 不会让验证结果依赖 batch 组成。

**结论：`val/paper_loss` 可以作为训练过程参照**，它与梯度实际优化的目标同源。但需遵守两条边界：
1. 它与 `train/loss_epoch` 的差距（当前约 4 倍）是**训练集与验证集的泛化差距**，不是口径差异，不能当作异常；
2. 不得与 2048 帧子集值混用（P011），跨样本集只比趋势不比绝对值。

## 短程训练结果：余弦退火未能打破验证平台（2026-10-26）

四个阶段连续拼接成同一条训练轨迹（armA 5 + contA 20 + cos50 2 + cos50b 4），累计 30 个 epoch。所有阶段共用同一数据划分、同一 loss 公式、同一 batch 配置，仅改变一个变量。

| 阶段 | epoch | 改动 | 目的 |
| --- | --- | --- | --- |
| `AGENTBOX-SHORT-A` | 0–4 | 修复 `lr_scale`、base 3e-4、clip 1000 | 判断 agent_box 退化是否由 lr_mult 被抹平引起 |
| `AGENTBOX-CONT-A` | 5–24 | 续训至 25 epoch（余弦仍为 100） | 观察峰值 LR 下的退化是否持续 |
| `AGENTBOX-COS50` | 25–27 | `scheduler_epochs=50`、monitor `val/paper_loss` | 让余弦退火真正走到 `min_lr` |
| `cos50b` | 27–30 | 同上，从 pause 断点续训 | 继续观察退火段 |

### 结果

| epoch | train/loss | `val/loss`（旧） | `val/paper_loss` | paper traj | paper BEV | ADE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 59.46 | 17.682 | — | — | — | 1.596 |
| 4 | 49.08 | 16.156 | — | — | — | 1.432 |
| 10 | 36.23 | **15.508** | — | — | — | **1.350** |
| 24 | 24.31 | 16.775 | — | — | — | 1.603 |
| 25 | 20.53 | 16.219 | 76.49 | 66.78 | 5.68 | 1.415 |
| 26 | 19.38 | 16.204 | 76.77 | 67.08 | 5.56 | 1.409 |
| 27 | 18.61 | 16.512 | 77.06 | 67.24 | 5.75 | 1.455 |
| 28 | 17.60 | 16.607 | **75.89** | **65.75** | 6.00 | 1.411 |
| 29 | 16.91 | 17.455 | 77.17 | 66.79 | **6.22** | 1.521 |
| 30 | 16.31 | 17.563 | 78.26 | 67.81 | 6.20 | 1.526 |
| 31 | **15.50** | 17.010 | **79.39** | **69.08** | 6.09 | 1.454 |

LR 在 ep25→ep29 从 1.65e-4 降到 1.16e-4（−30%），到 ep50 会到 1e-6。

### 结论

1. **余弦退火不仅无效，反而加剧过拟合。** `val/paper_loss` 从 76.49 单调升到 **79.39（+3.8%）**，同期 train 从 20.53 降到 15.50（−24.5%）。ep28 的 −1.16 是噪声，随后连续两轮反弹。学习率变量已彻底排除。
2. **训练仍在下降、验证不动 = 过拟合。** train 从 59.46 单调降到 16.91（**−71.6%**），从未出现平台；同期 `val/paper_loss` 零改善，train/val 原始口径 gap 达 3.9 倍并在扩大。
3. **轨迹项在退火段同步恶化。** 66.78 → 69.08（+3.4%）；从 ep10 起 19 个 epoch 未突破。低 LR 没有带来精细收敛，只让模型更深地拟合训练集。
4. **BEV 是唯一持续恶化的分支**：5.68 → 6.22（**+9.5%**），并直接抵消轨迹改善——ep28 轨迹 −2.2% 时 BEV +4%，把总 loss 拉回。
5. **agent_box 是唯一持续改善的分支**（4.41 → 2.98，−32%），说明优化器本身工作正常，问题在泛化能力而非优化失效。
6. **两种口径再次背离**：ep28 `val/loss` 上升而 `val/paper_loss` 下降。若仍用旧指标选点会丢弃这一轮最好的 checkpoint——这是 v5 换口径的直接价值。

**净判断：** 该模型在「当前数据量 + 当前容量 + 当前正则」下已到达收敛上限。继续训练或继续调 LR 不会带来实质改善，必须改变数据或正则。

## 历史实验与冻结边界

> **状态：历史/冻结参照。** 不作为当前入口；配置状态见 [配置索引](#配置索引)，当前操作见 [快速启动](../../快速启动.md)。

最初batch2/FP32正式启动、5000步暂停、2→4 Worker以及后续bs8读取优化均为历史阶段，不能把旧PID、默认control、10538800步horizon与当前bs8/2634700步混用。原预检、源码审计、日志、JSON和Checkpoint继续保留；早期M0/M1/M2、42帧prepared和Pilot来源见 [v1_migration.md](../../analysis/diffusiondrive_b2d_architecture/v1_migration.md) 与 [训练准备证据](../../analysis/diffusiondrive_b2d_architecture/v1_validation/training_preparation/README.md)。

2026-10-19 的 Features 系统盘迁移曾保留78392份/249.83GiB原件。2026-10-23 按用户明确要求，对清单内每份系统盘原件和 PSSD 副本重新完成 SHA256/大小及冻结索引核验后，删除系统盘原件，PSSD 副本全部保留；系统盘可用空间由约35GiB增至约285GiB。源目录身份标记、配置、Manifest、SQLite、producer 和迁移注册均未改写，实际训练读取仍由 [storage_registration.json](../../outputs/diffusiondrive_v1/system_features_to_pssd_20261019/storage_registration.json) 和冻结Manifest解析到 PSSD。历史注册中的 `source_files_deleted=0` 是迁移当日记录，不代表现在仍有原件；本次删除结果见 [source_cleanup_result.json](../../outputs/diffusiondrive_v1/system_features_to_pssd_20261019/source_cleanup_result.json)，逐文件核验见同目录 `source_cleanup_verified.jsonl`。Targets保留现有位置。后续227864帧multi-producer属于独立数据版本；加入恢复标签应另建继续训练实验，保持固定验证集，不能称作当前同合同resume。gravity6/teleport本轮不实现。
