# Training Status

> **建立前提：** 需要一个稳定的当前训练状态入口，避免从历史报告/旧 Checkpoint 推断当前任务。
> **解决问题：** 记录正式实验、PID、数据/Split、初始化、验证与 Test 使用边界。
> **适用范围：** 当前 `status_r2_cos100_ema5ep_bevw3`（R2，从 R1 epoch 2 续训至 100 epoch，**已于 epoch 21 暂停**）。
> **不包含：** 不替代 `快速启动.md` 的操作命令，不追认历史实验指标。
> **当前状态：** **PAUSED**（2026-10-27 15:40，epoch 21 完整跑完含验证与保存后按用户要求停止）。R2 相对 R1 改三项：余弦跨度 25→100 epoch、EMA `decay` 0.999→0.99992（5 epoch 窗口）、`bev_semantic_weight` 14→42。
> **相关代码：** `tools/training/diffusiondrive_v1/status_experimental_train.py`（v6）、`route_split.py`、`tests/training/test_lr_scale_guard.py`、`/tmp/r2_supervisor.sh`、`/tmp/r2_stop_after_epoch21.py`。
> **最后更新：** 2026-10-27。

## 当前实验（R2，已暂停）

| 项目 | 值 |
|---|---|
| 实验目录 | `/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/status_r1_ema_bevlr5e1_lr3e4_20261026`（沿用 R1 目录，不新建） |
| 配置 | `tools/training/diffusiondrive_v1/experiments/status_r2_cos100_ema5ep_bevw3_20261027.json` |
| 初始化 | 从 R1 `checkpoints/last.ckpt`（epoch 2 / step 7,512）full-resume |
| 三项改动 | ① `scheduler_epochs=100`、`max_epochs=100`；② `ema_decay=0.99992`（12,520 步 ≈ 5 epoch 窗口）；③ `bev_semantic_weight=42`（官方 14 的 3 倍） |
| 守护 | `/tmp/r2_supervisor.sh`：`setsid nohup` 提权，非 0 退出自动从最新 `last.ckpt` 重启，最多 20 次，冷却 120 s，状态写入 `supervisor.log` |
| **状态** | **`state=paused`**，epoch 21 / 100 完整结束（step 55,088），按用户要求停止；全进程已退出，GPU 2.4 GB |
| **断点** | `checkpoints/last.ckpt`（epoch 21 / step 55,088，SHA 以启动时重算为准）；EMA `num_updates=52,581` |
| **恢复** | `setsid nohup /tmp/r2_supervisor.sh < /dev/null > /dev/null 2>&1 &`（supervisor 会自行取最新 `last.ckpt` 并钉 SHA） |
| **当前最佳** | `best-020-052584.ckpt`，`val/paper_loss=78.064`（ep21 为 78.632，退步 0.73%） |
| LR 曲线 | 100-epoch 余弦：ep25 仍有 2.64e-4、ep50 1.58e-4、ep99 1.08e-6（旧 25-epoch 在 ep25 已降到 1e-6） |
| 优化器 | AdamW，3 组：base 1.0×、image_encoder 0.5×、bev_semantic_head 0.5×；峰值 3e-4 / 1.5e-4 / 1.5e-4 |
| Monitor | `val/paper_loss`（EMA 权重上计算），`checkpoint_keep=10` |
| 已清空 | `ModelCheckpoint` 的 `best_k_models` 历史（旧口径 121.078 / 70.884 / 66.851 会永久压制新口径 ~83 的值） |

## 暂停时的分量状态（ep21）

| 指标 | ep20 | ep21 | 趋势 |
|---|---|---|---|
| `val/paper_loss` | **78.064** | 78.632（+0.73%） | 单次反弹 |
| `raw_traj` | **4.988** | 5.004（+0.32%） | 停止下降 |
| `raw_bev` | 0.3345 | 0.3401（+1.66%） | **连续两次上升**（ep17 最低 0.2960） |
| `val/ade_m` | **1.418** | 1.470（+3.63%） | 反弹 |
| `val/fde_m` | **3.249** | 3.420（+5.26%） | 反弹 |
| 航向误差 | **4.18°** | 4.38° | 反弹 |

**结论：ep20 是当前最优综合点，ep21 全面小幅退步。BEV 分量在 ep17 触底（raw 0.2960）后已连续两次回升；轨迹在 ep20 触底（raw 4.988）后停住。** 未触发配置里的「连续 3 个 epoch 上升」调查条件。

## R2 口径警告

`bev_semantic_weight` 14 → 42 使 `val/paper_bev_semantic_loss` 与 `val/paper_loss` 绝对值整体抬高，**不能与 R1 / cos50 直接比较**。跨版本对比必须先去权重：

```
raw_bev  = val/paper_bev_semantic_loss / bev_semantic_weight
raw_traj = val/paper_trajectory_loss  / trajectory_weight
raw_cls  = val/paper_agent_class_loss / agent_class_weight
```

参考点：cos50 最佳 `raw_bev = 5.68/14 = 0.406`；R1 ep2 `raw_bev = 6.28/14 = 0.449`。

## 上一实验（已暂停，cos50b）

| 项目 | 值 |
|---|---|
| 实验目录 | `/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/status_agentbox_cos50b_lr3e4_clip1000_20261026` |
| 状态 | `state=paused`（2026-10-26 20:41），epoch 32 / global_step **77,749 / 125,200**（62.1%） |
| 暂停断点 | `pause-step077749.ckpt`（729 MB）；Best 为 `best-030-077624.ckpt`（`val/paper_loss` 75.89） |
| 说明 | 作为 R1 的对照组保留，不删除；如需续训必须先确认不会与 R1 抢 GPU |

## 短程训练进展（armA → contA → cos50 → cos50b，历史）

四阶段共用同一数据划分与 loss 公式，每阶段只改一个变量。

| 阶段 | epoch | 改动 |
|---|---|---|
| `AGENTBOX-SHORT-A` | 0–4 | 修复 `lr_scale`（image_encoder 恢复 0.5×）、base 3e-4、clip 1000 |
| `AGENTBOX-CONT-A` | 5–24 | 续训至 25 epoch（余弦仍为 100） |
| `AGENTBOX-COS50` | 25–27 | `scheduler_epochs=50`，monitor 改 `val/paper_loss` |
| `cos50b` | 27–31 | 同上，断点续训 |

### 退火段指标（ep25–31）

| epoch | train/loss | `val/loss`（旧） | `val/paper_loss` | paper traj | paper BEV | ADE |
|---|---|---|---|---|---|---|
| 25 | 20.53 | 16.219 | 76.49 | 66.78 | 5.68 | 1.415 |
| 26 | 19.38 | 16.204 | 76.77 | 67.08 | 5.56 | 1.409 |
| 27 | 18.61 | 16.512 | 77.06 | 67.24 | 5.75 | 1.455 |
| 28 | 17.60 | 16.607 | **75.89** | **65.75** | 6.00 | 1.411 |
| 29 | 16.91 | 17.455 | 77.17 | 66.79 | 6.22 | 1.521 |
| 30 | 16.31 | 17.563 | 78.26 | 67.81 | 6.20 | 1.526 |
| 31 | **15.50** | 17.010 | **79.39** | **69.08** | 6.09 | 1.454 |

LR 在此区间从 1.65e-4 降到约 1.05e-4。

### 结论

1. **余弦退火不仅无效，反而加剧过拟合。** `val/paper_loss` 从 76.49 单调升到 **79.39（+3.8%）**，同期 train 从 20.53 降到 15.50（−24.5%）。ep28 的单次 −1.16 是噪声，随后连续两轮反弹。学习率变量已彻底排除。
2. **轨迹项同步恶化**：66.78 → 69.08（+3.4%）。低 LR 没有带来精细收敛，只让模型更深地拟合训练集。
3. **BEV 持续恶化** 5.68 → 6.09–6.22（约 +8%），与梯度失衡一致（`bev_head` 仅占 0.55% 梯度范数）。
4. **agent_box 是全程唯一持续改善的分支**（4.41 → 2.98，−32%），说明优化器本身工作正常。
5. **两种口径多次背离**：ep28 旧指标上升而 paper 下降；ep29–31 旧指标回落而 paper 继续恶化。任一口径单独使用都会误判。

### 当前瓶颈（详见 `docs/问题记录.md` P012）

| 瓶颈 | 证据 | 本轮处置 |
|---|---|---|
| 泛化不足（主因） | train −73.9% 而 val 上升；train/val gap 从 3.17 扩到 5.1 倍 | **R1 已开 EMA**（decay 0.999）；`weight_decay` 5e-4 / Dropout 0.2 待后续单变量验证 |
| BEV 分支失衡 | `bev_head` 0.55% 梯度，BEV loss +8% | **R1 已设 `lr_mult=0.5`** |
| 轨迹停滞与低速过预测 | 19 epoch 无突破；P008 oracle 1.09 vs selected 3.11 | 补零位移 Anchor、评分头校准（未在本轮） |

## R1 验收判据（训练完成后核对）

| 项 | 判据 |
|---|---|
| EMA 降低波动 | 最后 5 个 epoch 的 `val/paper_loss` 标准差低于 cos50 同区间 |
| BEV 守住 | `val/bev_loss` ≤ 6.00（cos50 末期为 6.09–6.22） |
| 无附带损伤 | paper traj ≤ 65.75 × 1.03；`val/agent_box` ≤ 3.30 |
| 总体 | 最佳 `val/paper_loss` < 75.89（cos50 最佳） |

## 训练边界

- 截至2026-10-27记录，当前R2从R1完整断点续训后已暂停；cos50b保留为历史对照。2026-10-28发布环境无法读取外盘实时状态，不能据本文执行恢复。
- Test 仍未加载；评估仅使用 Validation（17,191 行）。
- PSSD（`/dev/nvme0n1p1`）与 YMTC（`/dev/nvme1n1p3`）当前均已挂载；启动前仍须核对断点 SHA 与输出目录父级存在。
- 判断收敛只看 `train/loss_epoch` 与 `val/paper_loss` 的间距；`val/loss` 仅作历史连续性参考，不用于选点或设阈值。

## 已落地的修复与关键约束

前几轮暴露的五个缺陷都已修复并有回归覆盖，官方仓库保持逐字未改（仅 scheduler 的 `verbose` 兼容补丁）。

| 缺陷 | 根因 | 修复 | 验证 |
| --- | --- | --- | --- |
| `lr_scale` 丢失 | 官方 `get_coslr_optimizers` 只在新增组写 `lr`、从不写 `lr_scale`，`WarmupCosLR` 只在组 0 有该键时按组缩放 | 入口包装后写组 `lr_scale` 并重建同一官方 scheduler | 组 0（439 张量）1.0、组 1（108 个 `_backbone.image_encoder.*`）0.5 |
| 梯度裁剪误设 | 首次用 clip 1.0，实测总范数 608–645（trajectory_head 占 607–640），等于每步等比缩小约 640 倍 | 统一改 **clip 1000**，`grad_norms.jsonl` 增加 `clip_scale` | 全程 `clip_scale=1.0`，0 次裁剪 |
| 验证口径错误 | `forward_test` 不返回 loss 键，`transfuser_loss` 回退成单步 plain L1（约真实目标 1/4.5） | v4/v5 新增 `training_form_loss`：仅切轨迹头到训练分支、冻结 Dropout、固定采样种子 | 同批数据 B/A = 0.988，C/B = 0.160 |
| 调度器跨度固定 100 | 官方硬编码 `epochs=100`，短程实验只走完 33° 余弦（LR 仅降 13%） | v5 新增配置项 `scheduler_epochs`，并在 resume 后重新注入（`load_state_dict` 会覆盖） | ep25 LR 由 2.60e-4 跳到 1.65e-4 |
| 暂停无效 | Lightning 2.2.1 没有 `Trainer.should_stop`，赋值只在实例上挂惰性字段 | 改为把 `trainer.fit_loop.epoch_loop.max_steps` 封顶到当前 step | ep31 自我退出，无需 SIGTERM |

**必须保留的约束：**
- `paramwise_lr_scale` 必须为 `true`，否则 `image_encoder` 的 0.5× 会被静默抹平；
- `scheduler_epochs` 必须 ≥ `max_epochs`，否则拒绝启动；
- `--init-weights` 只加载 763 个权重张量；`--resume` 只接受本实验目录内完整断点，跨目录必须带 `--resume-sha256`；
- 非有限 loss 或梯度立即停止；输出盘未挂载时拒绝 `mkdir`，避免把断盘写成系统盘目录；
- 恢复目录以实际`resume_contract`、本实验配置和断点身份为准；R2允许本实验同目录完整恢复，跨目录必须显式SHA；原有历史断点保留。

回归：`tests/training` **146 项通过**，含 lr_scale 守卫、调度器跨度注入、暂停封顶与真实 Lightning 端到端停止。

## 硬边界

- 不以旧 Checkpoint、旧 optimizer/scheduler/global_step 或旧 1,024 行子集作为当前实验身份。
- 不在训练中读取 Test，不用 Test 调 LR/早停/Checkpoint。
- 旧 Checkpoint 对新 Test 有 Route 级污染，见 `checkpoint_dataset_contamination_audit.md`。
- 若进程退出、数据 SHA 变化或验证失败，状态必须改为 `BLOCKED/STOPPED`，不得用旁路重启。

## 相关证据

- `outputs/.../20261024_training_cleanup/diffusiondrive_v1_training_cleanup_audit.md`
- `outputs/.../20261024_training_cleanup/checkpoint_dataset_contamination_audit.md`
- `outputs/.../20261024_training_cleanup/training_preflight_v2.md`
- `outputs/.../20261024_training_cleanup/launch_identity.json`
- `outputs/.../20261024_training_cleanup/split_seed0_801010/dataset_split_manifest.json`
- `outputs/.../20261024_training_cleanup/split_seed0_801010/status_training_dataset_manifest.json`
- `outputs/.../20261025_eval_routepreview_v2_best5008_latest43650/full/evaluation_report.md`
- `outputs/.../20261025_eval_routepreview_v2_best5008_latest43650/train_probe/train_probe_report.md`
- `outputs/.../20261025_eval_routepreview_v2_best5008_latest43650/agent_box_degradation_diagnosis.md`
- `outputs/.../20261025_eval_routepreview_v2_best5008_latest43650/agent_box_loss_curve.png`
- `tools/training/diffusiondrive_v1/experiments/status_agentbox_shortA_lr3e4_lrscale_clip1000.json`
- `tools/training/diffusiondrive_v1/experiments/status_agentbox_shortB_lr6e4_lrscale_clip1000.json`
- `tools/training/diffusiondrive_v1/experiments/status_agentbox_shortC_lr3e4_flatlr_clip1000.json`
- `outputs/.../20261025_eval_routepreview_v2_best5008_latest43650/agent_box_diagnosis.json`
- `outputs/.../20261025_eval_routepreview_v2_best5008_latest43650/full/checkpoint_identity.json`
