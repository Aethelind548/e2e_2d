# Bench2Drive → 原版 DiffusionDrive V1 Adapter

> **建立前提：** 既有工程的接口、设计或有限验证记录；版本和样本以正文为准。
> **解决问题：** 说明Bench2Drive → 原版 DiffusionDrive V1 Adapter的结论、依据和适用边界。
> **适用范围：** 工具与数据转换。
> **不包含：** 重复维护根目录快速启动。
> **当前状态：** 当前维护。
> **相关代码：** tools/
> **最后更新：** 2026-10-22（新增道路级 Route Preview 隔离原型说明）。

当前训练入口为 `DiffusionDriveV1Dataset`、`DiffusionDriveV1FeatureBuilder` 与 `DiffusionDriveV1TargetBuilder`，定义在 `v1_builders.py`。训练使用原版 V1 模型仓库，不使用 NAVSIM SceneLoader，不依赖已清理的 BNDP 代码。

> **模式边界：** 当前正式训练使用 `paired_manifest + cache-only + verified-single-pass`；`prepare.py`/`PreparedDataset`/raw Dataset 是历史 Pilot 或数据准备路径。当前操作只在 [快速启动](../../快速启动.md) 维护；合同和配置状态见 [训练主文档](../training/README.md)。

## 功能与依赖

- `route_index.py` / `frame_reader.py`：只索引显式 Route，按需读取当前标注、三路前视 RGB 和 LAZ 点云。
- `experimental_route_navigation.py` / `experimental_route_preview.py`：隔离的道路级 Route 重建与 20 m 预瞄原型；显式区分 `ORIGINAL_ROUTE` / `RECONSTRUCTED_ROUTE` / `AMBIGUOUS` / `NOT_VERIFIED`，不接入正式 Feature Producer。
- `route_preview_contract.py`：离线与 Agent 共享的 `route_progress -> preview_point -> command4` 纯函数；UTurn 固定 `UNSUPPORTED_UTURN`，不进入 Status8。`agent/navigation/route_preview.py` 提供 opt-in 在线适配器。
- `coordinate_transform.py` / `temporal_sampler.py`：既有工程后轴参考、x 前/y 左/yaw 左转为正、10 Hz 源时钟与 0.5 s 监督采样，合同保留工程参数与来源限制。
- `camera_adapter.py` / `lidar_adapter.py` / `lidar_decode.py`：B2D 三相机使用独立的比例裁剪/拼接，保留官方 Resize/ToTensor 约定；LAZ 使用配置中的已有独立解释器解码。
- `native_navigation.py` / `object_adapter.py` / `native_builders.py`：当前近点命令、因果历史速度/加速度与共享特征。V1 覆盖旧的懒加载 V2 相机工厂；共享源码保留以维持历史 producer 身份。
- `trajectory_target.py` / `navsim_bridge.py` / `v1_builders.py`：后轴相对轨迹、官方车辆筛选与补齐、完整监督合同。
- `v1_bev.py`：12 Town 版本化 OpenDRIVE 地图、道路/步行道/中心线及 Actor 七类数值栅格，保留官方覆盖顺序与线宽。
- `v1_label_rules.py` / `v1_geometry.py`：原版与 M4 增强规则隔离；地图自交和不可靠几何显式拒绝。
- `static_layers/`：M5 世界静态几何/属性缓存及独立局部 Mask，不强制接入七类训练。

## 单样本训练合同

| 字典 | 张量 | shape / dtype |
|---|---|---|
| features | camera_feature | `(3,256,1024)` float32 |
| features | lidar_feature | `(1,256,256)` float32 |
| features | status_feature | `(8,)` float32 |
| targets | trajectory | `(8,3)` float32 |
| targets | agent_states / agent_labels | `(30,5)` float32 / `(30,)` bool |
| targets | bev_semantic_map | `(128,256)` int64；类别 0～6 |

七类为 Background、Road、Walkway、Centerline、Static、Vehicle、Pedestrian。缺少真实 BEV provider、原始帧/地图、导航或可靠几何会拒绝，不以背景或零张量替代。

原始 Adapter 按需模式仅使用 worker=0；这是历史 Pilot/诊断路径，不是当前正式训练入口。`tools/training/diffusiondrive_v1/prepare.py` 预生成完整特征/目标、接受 Manifest 与失败日志；`PreparedDataset` 验证源码/地图/源文件身份和张量 SHA。当前正式训练使用 paired cache-only Manifest，见 [训练主文档](../training/README.md)。

## 有限验证

```bash
python -m unittest tools.validation.b2d_adapter.test_v1_builders -v
python -m tools.validation.b2d_adapter.validate_v1_training_preparation
```

共享时钟/坐标/语义证据保留在 `analysis/data_interface_analysis/`；M2/M4/M5 与 Pilot 结果位于 `analysis/diffusiondrive_b2d_architecture/`。旧 V2 专用诊断入口与 BNDP 产物已出清；历史报告的旧产物链接只代表历史引用。

[功能总览图](../../analysis/diffusiondrive_b2d_architecture/v1_validation/adapter_overview/adapter_functional_overview.png) · [训练准备结果](../../analysis/diffusiondrive_b2d_architecture/v1_validation/training_preparation/README.md)
