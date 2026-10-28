# 发布文件范围与外部资产Manifest

> **建立前提：** 用户授权V1发布重构，先保护基线再分批修改。
> **解决问题：** 发布文件范围与外部资产Manifest。
> **适用范围：** refactor/release-v1；2026-10-28。
> **不包含：** 覆盖训练资产、修改官方网络/Loss、绕过冻结身份或推送main。
> **当前状态：** 受限发布审查完成；真实恢复/部署仍未验收。
> **相关代码：** tools/、agent/、tests/。
> **最后更新：** 2026-10-28。

## 发布边界

第一方源码、配置、既有长期测试、必要维护/部署说明和本次七份审查报告进入根Git。V1以.gitmodules及160000 gitlink固定上游提交；其已有scheduler补丁独立发布，当前本地官方目录保持原样。V2源码仅本地保留，固定提交及既有Python补丁/原MIT许可通过资源Manifest携带。没有克隆/下载或修改两个官方仓库。

.gitignore只控制发布纳入，不触发删除。研究analysis除release_refactor七报告外、outputs/data/datasets、模型/数组/SQLite、本地凭据/缓存排除；冻结运行配置保持Git原字节。outputs/assets和注册/索引虽不进普通Git，仍是必要独立资产。历史tools验证脚本因没有无用/替代证明继续保留；本轮不为减文件数牺牲恢复能力。

| 路径 | 工作区字节/类型 | 职责 |
| --- | --- | --- |
| .gitattributes | 186 | 仅上游补丁禁用文本转换与空白重排，原补丁字节保持 |
| .gitignore | 1133 | 源码/必要部署和维护文档 |
| .gitmodules | 130 | 源码/必要部署和维护文档 |
| AGENTS.md | 4200 | 源码/必要部署和维护文档 |
| README.md | 2332 | 源码/必要部署和维护文档 |
| TRAINING_STATUS.md | 12077 | 源码/必要部署和维护文档 |
| agent/README.md | 16963 | 源码/必要部署和维护文档 |
| agent/__init__.py | 59 | 源码/必要部署和维护文档 |
| agent/adapters/__init__.py | 73 | 源码/必要部署和维护文档 |
| agent/adapters/coordinate_transform.py | 1846 | 源码/必要部署和维护文档 |
| agent/adapters/online_feature_adapter.py | 5345 | 源码/必要部署和维护文档 |
| agent/bench2drive_agent.py | 10333 | 源码/必要部署和维护文档 |
| agent/common/__init__.py | 62 | 源码/必要部署和维护文档 |
| agent/common/errors.py | 1101 | 源码/必要部署和维护文档 |
| agent/common/types.py | 5066 | 源码/必要部署和维护文档 |
| agent/config/__init__.py | 148 | 源码/必要部署和维护文档 |
| agent/config/default.json | 1232 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| agent/config/settings.py | 7790 | 源码/必要部署和维护文档 |
| agent/control/__init__.py | 59 | 源码/必要部署和维护文档 |
| agent/control/lateral_controller.py | 997 | 源码/必要部署和维护文档 |
| agent/control/longitudinal_controller.py | 3650 | 源码/必要部署和维护文档 |
| agent/control/vehicle_controller.py | 2287 | 源码/必要部署和维护文档 |
| agent/environment_interface.py | 3355 | 源码/必要部署和维护文档 |
| agent/navigation/__init__.py | 77 | 源码/必要部署和维护文档 |
| agent/navigation/route_manager.py | 5221 | 源码/必要部署和维护文档 |
| agent/navigation/route_preview.py | 2171 | 源码/必要部署和维护文档 |
| agent/navigation/status_builder.py | 13046 | 源码/必要部署和维护文档 |
| agent/planning/__init__.py | 71 | 源码/必要部署和维护文档 |
| agent/planning/diffusiondrive_predictor.py | 1786 | 源码/必要部署和维护文档 |
| agent/planning/mock_predictor.py | 1879 | 源码/必要部署和维护文档 |
| agent/planning/model_loader.py | 1720 | 源码/必要部署和维护文档 |
| agent/planning/predictor.py | 697 | 源码/必要部署和维护文档 |
| agent/runtime/__init__.py | 62 | 源码/必要部署和维护文档 |
| agent/runtime/diagnostics.py | 1673 | 源码/必要部署和维护文档 |
| agent/runtime/safety_supervisor.py | 4243 | 源码/必要部署和维护文档 |
| agent/sensors/__init__.py | 49 | 源码/必要部署和维护文档 |
| agent/sensors/ego_state.py | 8669 | 源码/必要部署和维护文档 |
| agent/sensors/georeference.py | 1552 | 源码/必要部署和维护文档 |
| agent/sensors/observation_buffer.py | 1333 | 源码/必要部署和维护文档 |
| agent/sensors/sensor_manager.py | 8091 | 源码/必要部署和维护文档 |
| agent/trajectory/__init__.py | 63 | 源码/必要部署和维护文档 |
| agent/trajectory/trajectory_decoder.py | 1794 | 源码/必要部署和维护文档 |
| agent/trajectory/trajectory_processor.py | 3229 | 源码/必要部署和维护文档 |
| analysis/release_refactor/deprecated_interface_inventory.md | 1297742 | 本次审查交付；最终大小以Git树为准 |
| analysis/release_refactor/interface_migration_map.md | 4293 | 本次审查交付；最终大小以Git树为准 |
| analysis/release_refactor/refactor_plan.md | 3228 | 本次审查交付；最终大小以Git树为准 |
| analysis/release_refactor/regression_test_report.md | 3213 | 本次审查交付；最终大小以Git树为准 |
| analysis/release_refactor/release_file_manifest.md | 本次生成 | 本次审查交付；最终大小以Git树为准 |
| analysis/release_refactor/release_readiness.md | 本次生成 | 本次审查交付；最终大小以Git树为准 |
| analysis/release_refactor/removed_code_report.md | 4301 | 本次审查交付；最终大小以Git树为准 |
| docs/agent_m2_m3_interface_analysis.md | 4870 | 源码/必要部署和维护文档 |
| docs/agent_test_report.md | 17938 | 源码/必要部署和维护文档 |
| docs/问题记录.md | 109678 | 源码/必要部署和维护文档 |
| repositories/DiffusionDrive | gitlink | 固定上游提交9b52ed0e…；源码与现有补丁保持原位 |
| requirements.txt | 679 | 源码/必要部署和维护文档 |
| tests/agent/conftest.py | 8583 | 源码/必要部署和维护文档 |
| tests/agent/test_agent_route_preview_contract.py | 2079 | 长期回归，保持原路径 |
| tests/agent/test_agent_status_integration.py | 6035 | 长期回归，保持原路径 |
| tests/agent/test_config_lifecycle.py | 8232 | 长期回归，保持原路径 |
| tests/agent/test_features_predictor.py | 11803 | 长期回归，保持原路径 |
| tests/agent/test_m2_m3.py | 23214 | 长期回归，保持原路径 |
| tests/agent/test_mock_integration.py | 5611 | 长期回归，保持原路径 |
| tests/agent/test_online_status_builder.py | 6399 | 长期回归，保持原路径 |
| tests/agent/test_pending_capabilities.py | 3248 | 长期回归，保持原路径 |
| tests/agent/test_sensors_navigation.py | 6654 | 长期回归，保持原路径 |
| tests/agent/test_trajectory_control_runtime.py | 8646 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_candidate_quality_diagnostic.py | 4711 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_certification_level_audit.py | 4510 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_collection_source_recovery.py | 3975 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_experimental_semantics.py | 21850 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_full_route_preview_audit.py | 2763 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_full_status_overlay.py | 4497 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_navigation_semantics_check.py | 2061 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_online_global_plan.py | 2922 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_end_classifier.py | 4420 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_navigation.py | 3778 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_navigation_stable_centers.py | 2063 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_preview_contract.py | 4106 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_preview_resource_guard.py | 2408 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_preview_v11.py | 1239 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_source_certification.py | 2149 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_truncation_repair.py | 2513 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_truncation_root_cause.py | 3913 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_route_validation_inventory.py | 8241 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_session_continuous_audit.py | 2743 | 长期回归，保持原路径 |
| tests/b2d_adapter/test_status_overlay_preflight.py | 3507 | 长期回归，保持原路径 |
| tests/release/test_preflight.py | 3203 | 长期回归，保持原路径 |
| tests/training/test_certified_status_subset.py | 5954 | 长期回归，保持原路径 |
| tests/training/test_evaluate_checkpoint.py | 4734 | 长期回归，保持原路径 |
| tests/training/test_lr_experiment.py | 14350 | 长期回归，保持原路径 |
| tests/training/test_lr_scale_guard.py | 25579 | 长期回归，保持原路径 |
| tests/training/test_route_split.py | 4459 | 长期回归，保持原路径 |
| tests/training/test_staged_status_overlay_v3.py | 5095 | 长期回归，保持原路径 |
| tests/training/test_status_finetune_entry.py | 5019 | 长期回归，保持原路径 |
| tests/training/test_status_overlay_reader.py | 12292 | 长期回归，保持原路径 |
| tests/training/test_status_route_eval_core.py | 3214 | 长期回归，保持原路径 |
| tests/training/test_status_training_curves.py | 2618 | 长期回归，保持原路径 |
| tests/training/test_status_training_manifest.py | 4509 | 长期回归，保持原路径 |
| tests/training/test_status_training_monitor.py | 2756 | 长期回归，保持原路径 |
| tests/training/test_status_turn_analysis.py | 2580 | 长期回归，保持原路径 |
| tests/training/test_target_relocation.py | 7289 | 长期回归，保持原路径 |
| tools/README.md | 5663 | 源码/必要部署和维护文档 |
| tools/__init__.py | 75 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/README.md | 4605 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/__init__.py | 490 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/camera_adapter.py | 2250 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/config.py | 3670 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/coordinate_transform.py | 2683 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/errors.py | 265 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/experimental_causal_state.py | 5678 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/experimental_route_navigation.py | 28335 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/experimental_route_preview.py | 8026 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/experimental_route_progress.py | 19331 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/experimental_status_builder.py | 14588 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/frame_reader.py | 4462 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/lidar_adapter.py | 2107 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/lidar_decode.py | 3127 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/native_builders.py | 13240 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/native_config.py | 3977 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/native_dataset.py | 3579 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/native_navigation.py | 2723 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/navsim_bridge.py | 9928 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/object_adapter.py | 4784 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/route_index.py | 3688 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/route_preview_contract.py | 14183 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/static_layers/__init__.py | 263 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/static_layers/opendrive.py | 38731 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/temporal_sampler.py | 3132 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/terminal_policy.py | 8064 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/trajectory_target.py | 5530 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/v1_bev.py | 26581 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/v1_builders.py | 13233 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/v1_geometry.py | 10201 | 源码/必要部署和维护文档 |
| tools/b2d_adapter/v1_label_rules.py | 2265 | 源码/必要部署和维护文档 |
| tools/release/__init__.py | 80 | 长期部署预检/资源描述/上游补丁/许可证 |
| tools/release/licenses/HUST_Vision_Lab.LICENSE | 1072 | 长期部署预检/资源描述/上游补丁/许可证 |
| tools/release/patches/DiffusionDrive.patch | 726 | 长期部署预检/资源描述/上游补丁/许可证 |
| tools/release/patches/DiffusionDriveV2.patch | 1308 | 长期部署预检/资源描述/上游补丁/许可证 |
| tools/release/preflight.py | 6614 | 长期部署预检/资源描述/上游补丁/许可证 |
| tools/release/resources.json | 34836 | 长期部署预检/资源描述/上游补丁/许可证 |
| tools/training/README.md | 100926 | 源码/必要部署和维护文档 |
| tools/training/__init__.py | 44 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/__init__.py | 78 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/baseline.json | 1442 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal.json | 3368241 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_formal_resume_fix.json | 3368532 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json | 3368692 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_warmstart.json | 3368002 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_warmstart_pssd_targets.json | 3368469 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr6e4_official_route_split.json | 10383 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_original_7class_20261019.json | 3365966 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_original_7class_bs8_bf16_20261020.json | 3366230 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_original_7class_bs8_fresh.json | 3365965 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/baseline_setup.py | 9040 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/bf16_policy.py | 6400 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/cache_only_reader.py | 8056 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/camera_original_compatibility.json | 4025 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/camera_proportional_check.json | 5290 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/certified_overlay_reader_check.py | 3812 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/certified_overlay_v4.py | 13311 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/certified_status_subset.py | 9718 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/check_labels.py | 4340 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/check_m1.py | 3617 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/evaluate_checkpoint.py | 27910 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/experimental_status_reader.py | 5046 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/experiments/m5_abc_design.json | 5326 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_agentbox_contA_lr3e4_lrscale_clip1000_20261025.json | 4061 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_agentbox_cos50_lr3e4_clip1000_20261026.json | 6048 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_agentbox_paperval_lr3e4_clip1000_20261026.json | 4164 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_agentbox_shortA_lr3e4_lrscale_clip1000.json | 3739 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_agentbox_shortB_lr6e4_lrscale_clip1000.json | 3694 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_agentbox_shortC_lr3e4_flatlr_clip1000.json | 3539 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_r1_ema_bevlr5e1_lr3e4_20261026.json | 6716 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_r2_cos100_ema5ep_bevw3_20261027.json | 8584 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/experiments/status_routepreview_v2_split801010_seed0_20261025_012809.json | 3342 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/feature_flatten.py | 8134 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/feature_jobs.py | 14079 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/feature_migration.py | 27922 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/feature_queue.py | 6002 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/feature_service.py | 8637 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/feature_shared_storage.py | 4048 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/feature_storage.py | 9008 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/features_cache.json | 2734 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/features_cache.py | 28670 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/features_jobs.json | 440 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/features_manifests.json | 593 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/features_remaining_20261015.json | 79458 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/features_shared_storage.json | 1853 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/features_system_storage.json | 765 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/fit.py | 52572 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/formal_training_proposal.json | 3534 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/full_recovery_rules.py | 26694 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/gravity6_recovery.py | 14108 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/lr_experiment.py | 3960 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/m1_subset.json | 2365 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/m2_subset.json | 8756 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/m4_original_compatibility.json | 487 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/m4_subset.json | 4582 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/official_v1_audit.py | 8846 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/paired_check.json | 6047 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/paired_dataset.py | 12872 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/payload_sha_audit.py | 8451 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/pilot_original.json | 14701 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/pilot_proportional.json | 14816 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/preflight.py | 8029 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/prepare.py | 8886 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/prepared.py | 10678 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/procfs_mount_reader.py | 4779 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/recovery_paired.py | 19982 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/route_split.py | 16667 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/runtime.py | 5408 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/sampling.py | 2367 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/single_pass_cache_reader.py | 3879 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/staged_status_overlay_v3.py | 7983 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_experimental_train.py | 54406 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_finetune_entry.py | 6730 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_overlay_contract.py | 5616 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_overlay_reader.py | 10976 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_route_eval.py | 503 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_route_eval_core.py | 37094 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_training_curves.py | 5355 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_training_manifest.py | 10430 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_training_monitor.py | 6027 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_training_preflight_full.py | 16217 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_training_preflight_v2.py | 11969 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/status_turn_analysis.py | 5193 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/system_feature_relocation.py | 9603 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_cache.py | 20533 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_jobs.py | 22591 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_recovery.py | 11236 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_relocation.py | 6614 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_retry_audit.py | 38158 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_service.py | 8421 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/target_storage.py | 15687 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/targets_full.json | 4803 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/targets_recovery_irrelevant_v1.json | 4829 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/training/diffusiondrive_v1/train.py | 12020 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/training_control.py | 12115 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/training_health.py | 11692 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/training_monitor.py | 2241 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/training_progress.py | 7243 | 源码/必要部署和维护文档 |
| tools/training/diffusiondrive_v1/v1_target_contract.json | 2642 | 当前或冻结配置；原绝对路径/身份保留，不自动迁移 |
| tools/validation/__init__.py | 74 | 源码/必要部署和维护文档 |
| tools/validation/b2d/__init__.py | 80 | 源码/必要部署和维护文档 |
| tools/validation/b2d/adapter_semantics_prototype.py | 71373 | 源码/必要部署和维护文档 |
| tools/validation/b2d/candidate_quality_diagnostic.py | 44814 | 源码/必要部署和维护文档 |
| tools/validation/b2d/carla_global_plan_capture.py | 11917 | 源码/必要部署和维护文档 |
| tools/validation/b2d/certification_level_audit.py | 14021 | 源码/必要部署和维护文档 |
| tools/validation/b2d/check_b2d_timestamps.py | 5718 | 源码/必要部署和维护文档 |
| tools/validation/b2d/collection_source_recovery.py | 13430 | 源码/必要部署和维护文档 |
| tools/validation/b2d/full_route_preview_audit.py | 23758 | 源码/必要部署和维护文档 |
| tools/validation/b2d/full_status_overlay.py | 16773 | 源码/必要部署和维护文档 |
| tools/validation/b2d/inspect_b2d_route.py | 4311 | 源码/必要部署和维护文档 |
| tools/validation/b2d/inspect_navsim_b2d_pair.py | 10248 | 源码/必要部署和维护文档 |
| tools/validation/b2d/leaderboard_session_capture.py | 8696 | 源码/必要部署和维护文档 |
| tools/validation/b2d/measure_carla_rear_axle.py | 5226 | 源码/必要部署和维护文档 |
| tools/validation/b2d/navigation_semantics_check.py | 8111 | 源码/必要部署和维护文档 |
| tools/validation/b2d/online_global_plan.py | 7666 | 源码/必要部署和维护文档 |
| tools/validation/b2d/online_status_parity.py | 8599 | 源码/必要部署和维护文档 |
| tools/validation/b2d/preview_trajectory_consistency.py | 10913 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_end_classifier.py | 19476 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_navigation_preview.py | 26587 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_preview_alignment.py | 7622 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_preview_generate.py | 26525 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_preview_inventory.py | 4926 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_preview_v11.py | 27292 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_source_certification.py | 26525 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_truncation_repair.py | 13418 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_truncation_root_cause.py | 28294 | 源码/必要部署和维护文档 |
| tools/validation/b2d/route_validation_inventory.py | 40951 | 源码/必要部署和维护文档 |
| tools/validation/b2d/run_route_preview_v1.sh | 2747 | 源码/必要部署和维护文档 |
| tools/validation/b2d/scratch_behavior_attribution.py | 33836 | 源码/必要部署和维护文档 |
| tools/validation/b2d/session_continuous_audit.py | 10476 | 源码/必要部署和维护文档 |
| tools/validation/b2d/session_plan_agent.py | 12903 | 源码/必要部署和维护文档 |
| tools/validation/b2d/status_finetune_preflight.py | 17766 | 源码/必要部署和维护文档 |
| tools/validation/b2d/status_overlay_preflight.py | 15670 | 源码/必要部署和维护文档 |
| tools/validation/b2d/status_upstream_audit.py | 31979 | 源码/必要部署和维护文档 |
| tools/validation/b2d/validate_b2d_nav_status.py | 3609 | 源码/必要部署和维护文档 |
| tools/validation/b2d/validate_b2d_semantics.py | 7482 | 源码/必要部署和维护文档 |
| tools/validation/b2d/validate_b2d_trajectory_candidate.py | 2633 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/__init__.py | 69 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/audit_v1_camera_alignment.py | 21363 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/audit_v1_semantics.py | 25192 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/audit_v1_target_rejections.py | 21850 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/baseline_loader_check.py | 4958 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/certify_full_target_recovery.py | 8162 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_feature_expansion.py | 4655 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_feature_recovery.py | 4066 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_gravity6_geometry.py | 10426 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_v1_camera_forward.py | 3533 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_v1_incremental_target_retry.py | 5891 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_v1_m4_forward.py | 2708 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_v1_recovered_target_interfaces.py | 7025 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/check_v1_target_interfaces.py | 8104 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/cleanup_v1_mainline.py | 12211 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/render_adapter_overview.py | 10786 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/render_full_target_recovery.py | 7204 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/render_sensor_report.py | 5829 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/render_v1_bev_examples.py | 12147 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/run_baseline_preflight.py | 5287 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/summarize_full_target_recovery.py | 8357 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/summarize_v1_pilot.py | 11465 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/test_adapter.py | 14677 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_engineering_builders.py | 10710 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_feature_job_mount_errors.py | 2100 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_feature_migration.py | 7137 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_feature_queue.py | 4579 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_feature_shared_storage.py | 8927 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_feature_storage.py | 3449 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_feature_system_storage.py | 3076 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_full_target_recovery.py | 4237 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_level2.py | 7904 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_native_builders.py | 8075 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_real_routes.py | 5056 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_target_job_discovery.py | 4043 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_target_retry_audit.py | 5673 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_target_storage_codec.py | 5080 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_v1_builders.py | 16821 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_v1_m4.py | 5243 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/test_v1_static_layers.py | 8594 | 长期回归，保持原路径 |
| tools/validation/b2d_adapter/trace_teleport_events.py | 6678 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_b2d_adapter.py | 4963 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_bf16_training.py | 11341 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_cache_only_reader.py | 7151 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_procfs_reader.py | 5622 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_recovery_collection.py | 8584 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_single_pass_reader.py | 8581 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_camera_crop_fix.py | 12465 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_camera_views.py | 7660 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_feature_cache.py | 4432 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_m4.py | 12165 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_m4_contracts.py | 3563 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_m5.py | 16308 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/validate_v1_training_preparation.py | 8150 | 源码/必要部署和维护文档 |
| tools/validation/b2d_adapter/verify_system_feature_relocation.py | 5939 | 源码/必要部署和维护文档 |
| 快速启动.md | 6402 | 源码/必要部署和维护文档 |

## 外部运行资源（43项）

机器可读清单为tools/release/resources.json，记录位置参数、已有SHA、producer/根身份、用途及检查级别。下表是描述，不携带数值载荷；预检只stat，不认证payload SHA/owner/UUID。资源检查参数不会替旧训练配置改路径，目标服务器仍需独立且已验证的新路径合同。

| 资源ID | 模板路径 | 合同用途 | profiles |
| --- | --- | --- | --- |
| plan_anchor | {workspace}/outputs/diffusiondrive_v1/assets/official_kmeans_navsim_traj_20.npy | fixed official 20x8x2 anchor | baseline,r2 |
| image_backbone | {workspace}/outputs/diffusiondrive_v1/assets/resnet34_a1_0-46f8f793.pth | local ResNet34 weights; no download fallback | baseline,r2 |
| laz_decoder_python | {workspace}/outputs/diffusiondrive_v1/assets/laz_decoder_env/bin/python | separate LAZ Python; rebuild target environment instead of copying virtualenv | baseline,r2 |
| bench2drive_v004_v1 | {data_root} | explicit v1 data root; never recursively scan from preflight | baseline,r2 |
| map_Town01 | {map_root}/Town01.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town02 | {map_root}/Town02.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town03 | {map_root}/Town03.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town04 | {map_root}/Town04.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town05 | {map_root}/Town05.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town06 | {map_root}/Town06.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town07 | {map_root}/Town07.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town10HD | {map_root}/Town10HD.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town11 | {map_root}/Town11.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town12 | {map_root}/Town12.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town13 | {map_root}/Town13.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| map_Town15 | {map_root}/Town15.xodr | OpenDRIVE map source; original frozen config is not remapped by this manifest | baseline,r2 |
| r2_paired_manifest.json | {workspace}/outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/split_seed0_801010/paired_manifest.json | frozen data/split/index identity | r2 |
| r2_paired_index.sqlite | {workspace}/outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/split_seed0_801010/paired_index.sqlite | frozen data/split/index identity | r2 |
| r2_status_training_dataset_manifest.json | {workspace}/outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/split_seed0_801010/status_training_dataset_manifest.json | frozen data/split/index identity | r2 |
| r2_status_training_sample_index.jsonl | {workspace}/outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/split_seed0_801010/status_training_sample_index.jsonl | frozen data/split/index identity | r2 |
| r2_dataset_split_manifest.json | {workspace}/outputs/diffusiondrive_v1/route_preview_v11/20261024_training_cleanup/split_seed0_801010/dataset_split_manifest.json | frozen data/split/index identity | r2 |
| feature_sources_baseline | {workspace}/outputs/diffusiondrive_v1/features_service | frozen producer 79d47303b6c3549102fe7b8dfa7b7df0c9575de4addea7f6ec94e21e716a5ce9 | r2 |
| feature_sources_remaining_20261015 | {workspace}/outputs/diffusiondrive_v1/features_jobs/remaining_20261015 | frozen producer 79d47303b6c3549102fe7b8dfa7b7df0c9575de4addea7f6ec94e21e716a5ce9 | r2 |
| target_sources_baseline | {workspace}/outputs/diffusiondrive_v1/targets_full_v1 | frozen producer 7e341e1bb88667badbd9513d37c5adee917a56e040e2bbff634e792542800bdd | r2 |
| target_sources_compatible | {workspace}/outputs/diffusiondrive_v1/targets_full_recovery_20261017_r2/compatible | frozen producer 4c8d5956a0eccda778b48ab160d86c075785afa9d85ac9f0c232bdb6a6f0275c | r2 |
| target_sources_historical26 | {workspace}/outputs/diffusiondrive_v1/targets_recovery_irrelevant_v1 | frozen producer 8120da3e563e312c7a4d286a3f4467599ba0701915c54bda7e27442a4367b38e | r2 |
| target_sources_historical28 | {workspace}/outputs/diffusiondrive_v1/target_retry_audit_20261017 | frozen producer ca4b63c8a65d386b6cc044a818b28e4a3f6688433d36abd589012cb635adf400 | r2 |
| target_sources_projected | {workspace}/outputs/diffusiondrive_v1/targets_full_recovery_20261017_r2/projected | frozen producer 30a92b2dfc2373d88171444dadd1047beeeb748644163810c1a4aa9b7c66a983 | r2 |
| feature_storage_registration | {workspace}/outputs/diffusiondrive_v1/system_features_to_pssd_20261019/storage_registration.json | owner/producer/relocation registration; never rewrite in place | r2 |
| target_storage_registration | {workspace}/outputs/diffusiondrive_v1/targets_to_pssd_20261024/target_storage_registration_historical28_v2.json | owner/producer/relocation registration; never rewrite in place | r2 |
| r2_raw_resume_checkpoint | {experiment_root}/checkpoints/last.ckpt | actual R2 raw full checkpoint; no substitution of historical evaluation copies | r2 |
| r2_launch_identity | {experiment_root}/launch.json | actual run identity, not historical profile defaults | r2 |
| baseline_paired_manifest | {workspace}/analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/baseline_pssd_manifest.json | Baseline-only pairing; distinct from R2 split | baseline |
| feature_baseline_cache_a | {feature_baseline_cache_a} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| feature_baseline_cache_b | {feature_baseline_cache_b} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| feature_remaining_20261015_cache_a | {feature_remaining_20261015_cache_a} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| feature_remaining_20261015_cache_b | {feature_remaining_20261015_cache_b} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| feature_remaining_20261015_cache_system | {feature_remaining_20261015_cache_system} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| target_baseline_targets_data | {target_baseline_targets_data} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| target_compatible_full_recovery_compatible | {target_compatible_full_recovery_compatible} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| target_historical26_targets_recovery_irrelevant_v1 | {target_historical26_targets_recovery_irrelevant_v1} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| target_historical28_isolated_retry_audit | {target_historical28_isolated_retry_audit} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |
| target_projected_full_recovery_projected | {target_projected_full_recovery_projected} | payload root described by paired manifest; resolve effective registered location, not stale original location | r2 |

## 上游与许可证

| 来源 | 固定提交 | 代码许可 | 随包补丁 |
| --- | --- | --- | --- |
| repositories/DiffusionDrive | 9b52ed0ec06b073d82d6f392ab084c7b301c8681 | MIT | tools/release/patches/DiffusionDrive.patch |
| repositories/DiffusionDriveV2 | 1cd12a1e155c34dcc471261835444c8d5587580b | MIT | tools/release/patches/DiffusionDriveV2.patch |

原MIT全文随tools/release/licenses/HUST_Vision_Lab.LICENSE保留。两个补丁均从现有官方工作树Python改动获取，并用git apply --check --reverse只读确认已应用；未把V2跟踪pyc变化当源码补丁，也未清理它们。数据、地图、预训练权重、CARLA和其他依赖的许可独立于这两个代码许可证；第一方未擅自新增公开许可。

## 依赖与迁移限制

requirements.txt记录本次模型环境已观察的主要包版本，以及独立LAZ环境codec版本；原官方requirements/environment.yml仍按固定上游保留，不能与当前Torch2.7.1环境混为同一锁。尚未在干净Ubuntu/GPU主机安装验证；CUDA wheel来源、驱动、nuplan依赖解析和独立LAZ环境需要目标机验收。

R2实际raw恢复断点、launch身份和旧pin源码不可见；不能从仅有资源描述恢复完整训练。Agent真实推理/在线LiDAR未完成。代码发布包可审查，完整远程恢复发布仍有阻塞。
