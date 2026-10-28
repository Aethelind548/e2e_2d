# 第二轮发布范围审查

> **建立前提：** 用户授权Git RC收尾与第二轮精简。
> **解决问题：** 第二轮发布范围审查。
> **适用范围：** refactor/release-v1；基线6d7b709；2026-10-28。
> **不包含：** 训练恢复、GPU/真实数据/CARLA验收、修改冻结资产或推送合并。
> **当前状态：** 代码RC审查；外部集成另行验收。
> **相关代码：** tools/、agent/、tests/、tools/release/。
> **最后更新：** 2026-10-28。

第一方修改前Git干净，基线提交完整；官方V1子仓库仅有已知补丁和缓存，未重置。53项仅执行`git rm --cached`，本地原路径及字节不变；不将历史研究代码认定为可删除。独立归档`outputs/project_history/release_v1_phase3_archive_20261028.tar.gz`已逐文件SHA复核，原Git历史同时保留。

依赖审查复用已有函数引用清单，并补查AST调用、完整路径/模块名、短名称、JSON/Shell/CLI及文档。静态无引用只证明本次代码集合未检出，不证明未知外部命令不存在；历史CLI仍可在原本地工作区使用。下表给出每个排除项，完整源码docstring/调用方/精确SHA见本地outputs/release_refactor/phase3/cleanup_review.json。

| 路径 | 字节 | 用途与依据 | 当前有效引用 | 替代/保存与风险 |
|---|---:|---|---|---|
| `tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal.json` | 3368241 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_warmstart.json` | 3368002 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/check_labels.py` | 4340 | One bounded real R0/R2 label check; writes evidence and reusable smoke tensors.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/check_m1.py` | 3617 | Finite cross-map/rare-class inspection using the selected fit configuration.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/m5_abc_design.json` | 5326 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_agentbox_contA_lr3e4_lrscale_clip1000_20261025.json` | 4061 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_agentbox_cos50_lr3e4_clip1000_20261026.json` | 6048 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_agentbox_paperval_lr3e4_clip1000_20261026.json` | 4164 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_agentbox_shortA_lr3e4_lrscale_clip1000.json` | 3739 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_agentbox_shortB_lr6e4_lrscale_clip1000.json` | 3694 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_agentbox_shortC_lr3e4_flatlr_clip1000.json` | 3539 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/experiments/status_routepreview_v2_split801010_seed0_20261025_012809.json` | 3342 | 历史实验配置；与当前R1/R2分开保留；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/training/diffusiondrive_v1/status_training_preflight_v2.py` | 11969 | Historical preflight (no optimizer step, optional subset).；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/check_b2d_timestamps.py` | 5718 | Bounded, read-only clock/ego metadata inspection for explicit B2D routes.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/inspect_b2d_route.py` | 4311 | Read a small, explicit Bench2Drive route window and print a JSON summary.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/inspect_navsim_b2d_pair.py` | 10248 | Inspect one NAVSIM log/PCD and one short Bench2Drive annotation window.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/route_preview_alignment.py` | 7622 | Offline/online-proxy Route Preview alignment for V1.1.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/route_preview_inventory.py` | 4926 | Read-only dataset inventory for Route Preview V1.1 full generation.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/run_route_preview_v1.sh` | 2747 | 旧Route Preview启动Shell，依赖同批归档inventory；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/scratch_behavior_attribution.py` | 33836 | Read-only behavior-balanced checkpoint attribution for the Scratch V1 run.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/status_finetune_preflight.py` | 17766 | Read-only preflight for a proposed Status-only warm-start experiment.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/validate_b2d_nav_status.py` | 3609 | Read-only route-level checks for navigation and ego-state semantics.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/validate_b2d_semantics.py` | 7482 | Read-only, single-route checks for Bench2Drive pose, timing and LiDAR frames.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d/validate_b2d_trajectory_candidate.py` | 2633 | Read-only two-frame pose/sign check; delta=0 is Actor diagnostic, not rear-axle GT.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/audit_v1_camera_alignment.py` | 21363 | Read-only, five-real-frame camera audit. Candidates never enter training caches.；同批历史调用：tools/validation/b2d_adapter/render_sensor_report.py,tools/validation/b2d_adapter/validate_v1_camera_crop_fix.py,tools/validation/b2d_adapter/validate_v1_camera_views.py | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/audit_v1_semantics.py` | 25192 | Read-only, bounded M2 annotation/map audit; never regenerates training labels.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/baseline_loader_check.py` | 4958 | Bounded real Baseline reads with spawn workers, including relocated Features.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/certify_full_target_recovery.py` | 8162 | Certify bounded real recovery pilots before full profile apply is unlocked.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_feature_expansion.py` | 4655 | Real expansion acceptance without modifying old tensors, navigation or producer.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_feature_recovery.py` | 4066 | Inject recoverable failures in designated test cache artifacts, never raw data.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_gravity6_geometry.py` | 10426 | Read-only real geometry replay of the four isolated gravity6 prototype pairs.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_v1_camera_forward.py` | 3533 | Real prepared Dataset roundtrip and original initialized V1 loss, no optimizer.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_v1_incremental_target_retry.py` | 5891 | Probe recorded tilted-actor rejects only, then apply a bounded isolated batch.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_v1_m4_forward.py` | 2708 | One real M4 batch through unchanged V1 training forward/loss; no optimizer.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_v1_recovered_target_interfaces.py` | 7025 | World geometry, de-duplicated union and real paired reads of isolated recovery.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/check_v1_target_interfaces.py` | 8104 | Bounded real target roundtrips and strict pairing; sensor generation forbidden.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/cleanup_v1_mainline.py` | 12211 | One-time, explicit V1 mainline cleanup; plan/dry-run precede file deletion.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/render_adapter_overview.py` | 10786 | Render the implemented B2D -> DiffusionDrive V1 adapter, without reading data.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/render_full_target_recovery.py` | 7204 | A few real recovery comparisons; refused maps are never drawn as background.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/render_sensor_report.py` | 5829 | Export two same-frame real sensor illustrations using existing V1 renderers.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/render_v1_bev_examples.py` | 12147 | Render existing verified seven-class targets; never rebuild labels or train.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/run_baseline_preflight.py` | 5287 | Only a 12-step run and resume to 32 on a 16/8-example real Baseline subset.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/summarize_full_target_recovery.py` | 8357 | Authoritative union report over target-only indexes and verified 6DoF pairs.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/summarize_v1_pilot.py` | 11465 | Summarize recorded pilot evidence; no training or raw-data reads.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/trace_teleport_events.py` | 6678 | Evidence for each recorded Route/flag frame; never relaxes a rejection rule.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_camera_crop_fix.py` | 12465 | Five real camera pairs, versioned Dataset caches and one full V1 loss forward.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_camera_views.py` | 7660 | Verify the real Camera Builder and calibrated overlap without lost Pilot caches.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_feature_cache.py` | 4432 | Independent real two-root reading and selected fresh Adapter comparisons.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_m4.py` | 12165 | Bounded M4 checks: only the 48 historical frames and two actual map positions.；同批历史调用：tools/validation/b2d_adapter/validate_v1_m5.py | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_m4_contracts.py` | 3563 | Check cached M4/M2 datasets and original/enriched resume identities; no Trainer.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_m5.py` | 16308 | Six finite M5 sites, numerical caches/plots and unchanged M2/M4 contracts.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/validate_v1_training_preparation.py` | 8150 | Bounded real V1 regression and pilot cache/loader measurements after cleanup.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |
| `tools/validation/b2d_adapter/verify_system_feature_relocation.py` | 5939 | Read-only post-remount verification against frozen Feature index and migration journal.；正式训练/Agent/长期测试未发现导入；仅独立历史CLI或配置入口 | 无保留代码调用；历史文档引用可存在 | D；本地原路径+基线+校验归档；中风险；最终RC人工审查；无本地删除授权 |

旧Status subset预检的模块说明明确指向status_training_preflight_full.py；仅该项可宣称正式替代。其他历史工具不宣称完全等价替代，其样本/输出绑定和历史研究价值由原件、归档和旧提交保存。两个大型配置formal/warmstart不参与当前R1/R2或长期测试；scratch、formal_resume_fix、PSSD relocation、旧control配置等因合同/来源或潜在恢复引用继续保留。

## 保留项逐文件复审

| 路径 | 发布决定与依据 |
|---|---|
| `tests/agent/conftest.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tests/agent/test_agent_route_preview_contract.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_agent_status_integration.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_config_lifecycle.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_features_predictor.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_m2_m3.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_mock_integration.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_online_status_builder.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_pending_capabilities.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_sensors_navigation.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/agent/test_trajectory_control_runtime.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_candidate_quality_diagnostic.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_certification_level_audit.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_collection_source_recovery.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_experimental_semantics.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_full_route_preview_audit.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_full_status_overlay.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_navigation_semantics_check.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_online_global_plan.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_end_classifier.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_navigation.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_navigation_stable_centers.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_preview_contract.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_preview_resource_guard.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_preview_v11.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_source_certification.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_truncation_repair.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_truncation_root_cause.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_route_validation_inventory.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_session_continuous_audit.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/b2d_adapter/test_status_overlay_preflight.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/release/test_preflight.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_certified_status_subset.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_evaluate_checkpoint.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_lr_experiment.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_lr_scale_guard.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_route_split.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_staged_status_overlay_v3.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_finetune_entry.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_overlay_reader.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_route_eval_core.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_training_curves.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_training_manifest.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_training_monitor.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_status_turn_analysis.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tests/training/test_target_relocation.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/training/diffusiondrive_v1/__init__.py` | 保留Git；有效调用：tests/training/test_certified_status_subset.py, tests/training/test_route_split.py, tests/training/test_staged_status_overlay_v3.py, tests/training/test_status_finetune_entry.py 等 |
| `tools/training/diffusiondrive_v1/baseline.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal_resume_fix.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_warmstart_pssd_targets.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr6e4_official_route_split.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_original_7class_20261019.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_original_7class_bs8_bf16_20261020.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_original_7class_bs8_fresh.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/baseline_setup.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/bf16_policy.py` | 保留Git；有效调用：tests/training/test_lr_experiment.py, tools/training/diffusiondrive_v1/fit.py, tools/training/diffusiondrive_v1/status_experimental_train.py, tools/training/diffusiondrive_v1/status_training_preflight_full.py 等 |
| `tools/training/diffusiondrive_v1/cache_only_reader.py` | 保留Git；有效调用：tests/training/test_target_relocation.py, tools/training/diffusiondrive_v1/experimental_status_reader.py, tools/training/diffusiondrive_v1/recovery_paired.py, tools/validation/b2d/candidate_quality_diagnostic.py 等 |
| `tools/training/diffusiondrive_v1/camera_original_compatibility.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/camera_proportional_check.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/certified_overlay_reader_check.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/certified_overlay_v4.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_validation_inventory.py, tools/validation/b2d/route_validation_inventory.py |
| `tools/training/diffusiondrive_v1/certified_status_subset.py` | 保留Git；有效调用：tests/training/test_certified_status_subset.py |
| `tools/training/diffusiondrive_v1/evaluate_checkpoint.py` | 保留Git；有效调用：tests/training/test_evaluate_checkpoint.py, tests/training/test_lr_experiment.py, tests/training/test_status_route_eval_core.py, tools/training/diffusiondrive_v1/lr_experiment.py 等 |
| `tools/training/diffusiondrive_v1/experimental_status_reader.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/status_experimental_train.py, tools/training/diffusiondrive_v1/status_route_eval_core.py, tools/training/diffusiondrive_v1/status_training_preflight_full.py |
| `tools/training/diffusiondrive_v1/experiments/status_r1_ema_bevlr5e1_lr3e4_20261026.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/experiments/status_r2_cos100_ema5ep_bevw3_20261027.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/feature_flatten.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_migration.py, tools/validation/b2d_adapter/test_feature_shared_storage.py |
| `tools/training/diffusiondrive_v1/feature_jobs.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_migration.py, tools/training/diffusiondrive_v1/paired_dataset.py, tools/training/diffusiondrive_v1/recovery_paired.py, tools/training/diffusiondrive_v1/system_feature_relocation.py 等 |
| `tools/training/diffusiondrive_v1/feature_migration.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_flatten.py, tools/training/diffusiondrive_v1/feature_queue.py, tools/training/diffusiondrive_v1/feature_service.py, tools/validation/b2d_adapter/test_feature_migration.py 等 |
| `tools/training/diffusiondrive_v1/feature_queue.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_service.py, tools/validation/b2d_adapter/test_feature_queue.py |
| `tools/training/diffusiondrive_v1/feature_service.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_jobs.py, tools/training/diffusiondrive_v1/feature_migration.py, tools/training/diffusiondrive_v1/feature_queue.py, tools/training/diffusiondrive_v1/feature_storage.py 等 |
| `tools/training/diffusiondrive_v1/feature_shared_storage.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_flatten.py, tools/training/diffusiondrive_v1/feature_jobs.py, tools/training/diffusiondrive_v1/feature_migration.py, tools/training/diffusiondrive_v1/feature_queue.py 等 |
| `tools/training/diffusiondrive_v1/feature_storage.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/feature_jobs.py, tools/training/diffusiondrive_v1/feature_service.py, tools/validation/b2d_adapter/test_feature_system_storage.py |
| `tools/training/diffusiondrive_v1/features_cache.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/features_cache.py` | 保留Git；有效调用：tests/training/test_target_relocation.py, tools/training/diffusiondrive_v1/baseline_setup.py, tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/feature_jobs.py 等 |
| `tools/training/diffusiondrive_v1/features_jobs.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/features_manifests.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/features_remaining_20261015.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/features_shared_storage.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/features_system_storage.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/fit.py` | 保留Git；有效调用：tests/training/test_lr_experiment.py |
| `tools/training/diffusiondrive_v1/formal_training_proposal.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/full_recovery_rules.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/gravity6_recovery.py, tools/training/diffusiondrive_v1/target_retry_audit.py, tools/validation/b2d_adapter/test_full_target_recovery.py |
| `tools/training/diffusiondrive_v1/gravity6_recovery.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/recovery_paired.py |
| `tools/training/diffusiondrive_v1/lr_experiment.py` | 保留Git；有效调用：tests/training/test_lr_experiment.py, tools/training/diffusiondrive_v1/fit.py |
| `tools/training/diffusiondrive_v1/m1_subset.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/m2_subset.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/m4_original_compatibility.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/m4_subset.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/official_v1_audit.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/evaluate_checkpoint.py, tools/training/diffusiondrive_v1/fit.py, tools/training/diffusiondrive_v1/training_control.py, tools/validation/b2d/candidate_quality_diagnostic.py |
| `tools/training/diffusiondrive_v1/paired_check.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/paired_dataset.py` | 保留Git；有效调用：tests/training/test_target_relocation.py, tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/evaluate_checkpoint.py, tools/training/diffusiondrive_v1/fit.py 等 |
| `tools/training/diffusiondrive_v1/payload_sha_audit.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/pilot_original.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/pilot_proportional.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/preflight.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/prepare.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/prepared.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/features_cache.py, tools/training/diffusiondrive_v1/fit.py, tools/training/diffusiondrive_v1/prepare.py |
| `tools/training/diffusiondrive_v1/procfs_mount_reader.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/recovery_paired.py, tools/validation/b2d_adapter/validate_procfs_reader.py, tools/validation/b2d_adapter/validate_single_pass_reader.py |
| `tools/training/diffusiondrive_v1/recovery_paired.py` | 保留Git；有效调用：tests/training/test_status_overlay_reader.py, tests/training/test_target_relocation.py, tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/official_v1_audit.py 等 |
| `tools/training/diffusiondrive_v1/route_split.py` | 保留Git；有效调用：tests/training/test_route_split.py |
| `tools/training/diffusiondrive_v1/runtime.py` | 保留Git；有效调用：tests/agent/test_features_predictor.py, tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/evaluate_checkpoint.py, tools/training/diffusiondrive_v1/features_cache.py 等 |
| `tools/training/diffusiondrive_v1/sampling.py` | 保留Git；有效调用：tests/training/test_lr_experiment.py, tools/training/diffusiondrive_v1/fit.py |
| `tools/training/diffusiondrive_v1/single_pass_cache_reader.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/cache_only_reader.py, tools/validation/b2d_adapter/validate_single_pass_reader.py |
| `tools/training/diffusiondrive_v1/staged_status_overlay_v3.py` | 保留Git；有效调用：tests/training/test_staged_status_overlay_v3.py |
| `tools/training/diffusiondrive_v1/status_experimental_train.py` | 保留Git；有效调用：tests/training/test_lr_scale_guard.py |
| `tools/training/diffusiondrive_v1/status_finetune_entry.py` | 保留Git；有效调用：tests/training/test_status_finetune_entry.py |
| `tools/training/diffusiondrive_v1/status_overlay_contract.py` | 保留Git；有效调用：tests/training/test_status_finetune_entry.py, tests/training/test_status_overlay_reader.py, tools/training/diffusiondrive_v1/certified_overlay_reader_check.py, tools/training/diffusiondrive_v1/status_overlay_reader.py |
| `tools/training/diffusiondrive_v1/status_overlay_reader.py` | 保留Git；有效调用：tests/training/test_status_finetune_entry.py, tests/training/test_status_overlay_reader.py, tools/training/diffusiondrive_v1/status_finetune_entry.py |
| `tools/training/diffusiondrive_v1/status_route_eval.py` | 保留Git；有效调用：tests/training/test_status_route_eval_core.py |
| `tools/training/diffusiondrive_v1/status_route_eval_core.py` | 保留Git；有效调用：tests/training/test_status_route_eval_core.py, tools/training/diffusiondrive_v1/status_experimental_train.py, tools/training/diffusiondrive_v1/status_route_eval.py, tools/training/diffusiondrive_v1/status_route_eval_core.py |
| `tools/training/diffusiondrive_v1/status_training_curves.py` | 保留Git；有效调用：tests/training/test_status_training_curves.py |
| `tools/training/diffusiondrive_v1/status_training_manifest.py` | 保留Git；有效调用：tests/training/test_status_training_manifest.py |
| `tools/training/diffusiondrive_v1/status_training_monitor.py` | 保留Git；有效调用：tests/training/test_status_training_monitor.py |
| `tools/training/diffusiondrive_v1/status_training_preflight_full.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/status_turn_analysis.py` | 保留Git；有效调用：tests/training/test_status_turn_analysis.py |
| `tools/training/diffusiondrive_v1/system_feature_relocation.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/baseline_setup.py, tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/payload_sha_audit.py, tools/training/diffusiondrive_v1/procfs_mount_reader.py 等 |
| `tools/training/diffusiondrive_v1/target_cache.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/baseline_setup.py, tools/training/diffusiondrive_v1/fit.py, tools/training/diffusiondrive_v1/gravity6_recovery.py, tools/training/diffusiondrive_v1/paired_dataset.py 等 |
| `tools/training/diffusiondrive_v1/target_jobs.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/target_recovery.py, tools/training/diffusiondrive_v1/target_retry_audit.py, tools/training/diffusiondrive_v1/target_service.py, tools/validation/b2d/route_end_classifier.py 等 |
| `tools/training/diffusiondrive_v1/target_recovery.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/full_recovery_rules.py, tools/training/diffusiondrive_v1/target_retry_audit.py, tools/validation/b2d_adapter/audit_v1_target_rejections.py |
| `tools/training/diffusiondrive_v1/target_relocation.py` | 保留Git；有效调用：tests/training/test_target_relocation.py, tools/training/diffusiondrive_v1/paired_dataset.py, tools/training/diffusiondrive_v1/recovery_paired.py |
| `tools/training/diffusiondrive_v1/target_retry_audit.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/gravity6_recovery.py, tools/training/diffusiondrive_v1/recovery_paired.py, tools/validation/b2d_adapter/audit_v1_target_rejections.py, tools/validation/b2d_adapter/test_target_retry_audit.py |
| `tools/training/diffusiondrive_v1/target_service.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/target_storage.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/cache_only_reader.py, tools/training/diffusiondrive_v1/paired_dataset.py, tools/training/diffusiondrive_v1/recovery_paired.py, tools/training/diffusiondrive_v1/single_pass_cache_reader.py 等 |
| `tools/training/diffusiondrive_v1/targets_full.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/targets_recovery_irrelevant_v1.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/train.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/training_control.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/training_health.py` | 保留Git；有效调用：tests/training/test_lr_experiment.py, tools/training/diffusiondrive_v1/fit.py |
| `tools/training/diffusiondrive_v1/training_monitor.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/training/diffusiondrive_v1/training_progress.py` | 保留Git；有效调用：tests/training/test_lr_experiment.py, tools/training/diffusiondrive_v1/fit.py |
| `tools/training/diffusiondrive_v1/v1_target_contract.json` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d/__init__.py` | 保留Git；有效调用：tests/b2d_adapter/test_certification_level_audit.py, tests/b2d_adapter/test_navigation_semantics_check.py, tests/b2d_adapter/test_online_global_plan.py, tests/b2d_adapter/test_route_source_certification.py 等 |
| `tools/validation/b2d/adapter_semantics_prototype.py` | 保留Git；有效调用：tests/b2d_adapter/test_experimental_semantics.py |
| `tools/validation/b2d/candidate_quality_diagnostic.py` | 保留Git；有效调用：tests/b2d_adapter/test_candidate_quality_diagnostic.py |
| `tools/validation/b2d/carla_global_plan_capture.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d/certification_level_audit.py` | 保留Git；有效调用：tests/b2d_adapter/test_certification_level_audit.py, tools/validation/b2d/navigation_semantics_check.py, tools/validation/b2d/online_status_parity.py, tools/validation/b2d/session_continuous_audit.py |
| `tools/validation/b2d/collection_source_recovery.py` | 保留Git；有效调用：tests/b2d_adapter/test_collection_source_recovery.py, tools/b2d_adapter/terminal_policy.py |
| `tools/validation/b2d/full_route_preview_audit.py` | 保留Git；有效调用：tests/b2d_adapter/test_full_route_preview_audit.py |
| `tools/validation/b2d/full_status_overlay.py` | 保留Git；有效调用：tests/b2d_adapter/test_full_status_overlay.py |
| `tools/validation/b2d/leaderboard_session_capture.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d/measure_carla_rear_axle.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d/navigation_semantics_check.py` | 保留Git；有效调用：tests/b2d_adapter/test_navigation_semantics_check.py |
| `tools/validation/b2d/online_global_plan.py` | 保留Git；有效调用：tests/b2d_adapter/test_online_global_plan.py, tools/validation/b2d/carla_global_plan_capture.py, tools/validation/b2d/route_validation_inventory.py |
| `tools/validation/b2d/online_status_parity.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d/preview_trajectory_consistency.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d/route_end_classifier.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_end_classifier.py, tools/validation/b2d/full_route_preview_audit.py, tools/validation/b2d/route_truncation_repair.py, tools/validation/b2d/route_truncation_root_cause.py |
| `tools/validation/b2d/route_navigation_preview.py` | 保留Git；有效调用：tools/validation/b2d/full_route_preview_audit.py, tools/validation/b2d/route_preview_generate.py, tools/validation/b2d/route_truncation_root_cause.py, tools/validation/b2d/status_upstream_audit.py |
| `tools/validation/b2d/route_preview_generate.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_preview_resource_guard.py |
| `tools/validation/b2d/route_preview_v11.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_preview_v11.py, tools/validation/b2d/route_truncation_root_cause.py, tools/validation/b2d/status_upstream_audit.py |
| `tools/validation/b2d/route_source_certification.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_source_certification.py, tools/validation/b2d/certification_level_audit.py, tools/validation/b2d/navigation_semantics_check.py, tools/validation/b2d/online_status_parity.py 等 |
| `tools/validation/b2d/route_truncation_repair.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_truncation_repair.py |
| `tools/validation/b2d/route_truncation_root_cause.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_truncation_root_cause.py, tools/validation/b2d/full_route_preview_audit.py, tools/validation/b2d/route_truncation_repair.py |
| `tools/validation/b2d/route_validation_inventory.py` | 保留Git；有效调用：tests/b2d_adapter/test_route_validation_inventory.py |
| `tools/validation/b2d/session_continuous_audit.py` | 保留Git；有效调用：tests/b2d_adapter/test_session_continuous_audit.py |
| `tools/validation/b2d/session_plan_agent.py` | 保留Git；有效调用：tools/training/diffusiondrive_v1/certified_overlay_v4.py, tools/validation/b2d/route_validation_inventory.py |
| `tools/validation/b2d/status_overlay_preflight.py` | 保留Git；有效调用：tests/b2d_adapter/test_status_overlay_preflight.py |
| `tools/validation/b2d/status_upstream_audit.py` | 保留Git；有效调用：tests/b2d_adapter/test_experimental_semantics.py |
| `tools/validation/b2d_adapter/__init__.py` | 保留Git；有效调用：tools/validation/b2d_adapter/test_target_retry_audit.py |
| `tools/validation/b2d_adapter/audit_v1_target_rejections.py` | 保留Git；有效调用：tools/validation/b2d_adapter/test_target_retry_audit.py |
| `tools/validation/b2d_adapter/test_adapter.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_engineering_builders.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_feature_job_mount_errors.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_feature_migration.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_feature_queue.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_feature_shared_storage.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_feature_storage.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_feature_system_storage.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_full_target_recovery.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_level2.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_native_builders.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_real_routes.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_target_job_discovery.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_target_retry_audit.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_target_storage_codec.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_v1_builders.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_v1_m4.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/test_v1_static_layers.py` | 保留Git；长期回归原路径保留；未发现可证明等价且可整体代替的测试；opt-in/xfail边界保留 |
| `tools/validation/b2d_adapter/validate_b2d_adapter.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d_adapter/validate_bf16_training.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d_adapter/validate_cache_only_reader.py` | 保留Git；有效调用：tools/validation/b2d_adapter/validate_single_pass_reader.py |
| `tools/validation/b2d_adapter/validate_procfs_reader.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d_adapter/validate_recovery_collection.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |
| `tools/validation/b2d_adapter/validate_single_pass_reader.py` | 保留Git；配置/CLI/动态入口、来源合同或依赖用途未能证明可排除；保守保留，后续人工确认 |

## 研究文本

既有analysis与outputs研究材料此前已排除普通Git，本轮不扫描/移动其原件。docs/agent_m2_m3_interface_analysis.md、docs/agent_test_report.md和docs/问题记录.md仍有当前导航/未完成合同依据，保留Git；不为减少文件数移除。TRAINING_STATUS历史短实验引用明确标为本地/历史，不是远程可运行配置承诺。未移动到undo或冻结analysis区。
