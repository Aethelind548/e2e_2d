# DiffusionDrive V1 + Bench2Drive Agent V0.1

> **建立前提：** 既有工程的接口、设计或有限验证记录；版本和样本以正文为准。
> **解决问题：** 说明DiffusionDrive V1 + Bench2Drive Agent V0.1的结论、依据和适用边界。
> **适用范围：** Agent框架与CPU回归。
> **不包含：** 宣称已完成真实CARLA闭环。
> **当前状态：** 当前维护。
> **相关代码：** agent/、tests/agent/
> **最后更新：** 2026-10-21（结构治理：架构/接口与测试结果分离）。

本目录是独立的 Agent 基础框架。八个模块、统一数据结构、配置管理和主执行链已建立；M2/M3 已支持显式配置的当前帧时钟、GNSS 因果状态，以及**唯一**的稠密全局计划 Route Preview Status（Navigation4 20 m + A0 Motion4）。真实主链已在 CPU 合成观测上到达 M4，尚未接入完整在线特征、真实模型或车辆闭环。`CONFIGURED` 仅表示框架配置完成。

当前结果与回归方法见 [测试主文档](../docs/agent_test_report.md)、[统一问题记录](../docs/问题记录.md) 和 [M2/M3 输入依据](../docs/agent_m2_m3_interface_analysis.md)。完整 Mock 主链当前阻塞于 M4，不具备闭环驾驶能力。

## 文件与模块

初始框架文件位于 `agent/`；后续测试位于 `tests/agent/`，文档/证据位于 `docs/`。测试轮没有修改模型、生产 Adapter、训练程序或分析产物。

| 模块 | 文件（相对本目录） | 主要类与接口 |
| --- | --- | --- |
| M1 环境接口 | `bench2drive_agent.py`、`environment_interface.py` | `get_entry_point()`；`Bench2DriveAgent` 的 `setup/sensors/set_global_plan/run_step/destroy/to_vehicle_control`；`DiffusionDriveLeaderboardAgent` 绑定本机外部基类 |
| 公共契约 | `common/types.py`、`common/errors.py` | 统一 dataclass、坐标系、运行状态和分层异常 |
| 配置 | `config/settings.py`、`config/default.json` | `AgentConfig` 及传感器、特征、导航、规划器、PID、控制器、日志配置；`load_config()` |
| M2 传感器与状态 | `sensors/sensor_manager.py`、`sensors/ego_state.py`、`sensors/georeference.py`、`sensors/observation_buffer.py` | 当前帧校验；`CausalGNSSStateEstimator`；`CarlaGeoReference`；历史缓存 |
| M3 导航 | `navigation/route_manager.py`、`navigation/status_builder.py`、`navigation/route_preview.py` | `RouteManager.set_global_plan/update/reset/preview`；`OnlineStatusBuilder`（稠密路线、Navigation4、A0 Motion4、有效性）；`validate_command4` |
| M4 在线特征 | `adapters/online_feature_adapter.py`、`adapters/coordinate_transform.py` | `OnlineFeatureAdapter.build_features/build_lidar/build_status/validate_features/camera_from_verified_rgb`；世界与模型局部 XY 双向变换 |
| M5 预测器 | `planning/predictor.py`、`planning/diffusiondrive_predictor.py`、`planning/mock_predictor.py`、`planning/model_loader.py` | `TrajectoryPredictor.initialize/predict/reset/close`；真实与 Mock 实现；`V1ModelLoader.resolve_official_interfaces/build_model/load_checkpoint` |
| M6 轨迹处理 | `trajectory/trajectory_decoder.py`、`trajectory/trajectory_processor.py` | `TrajectoryDecoder.decode_poses`；`TrajectoryProcessor.validate/process/transform/resample/select_control_point/reset` |
| M7 控制 | `control/lateral_controller.py`、`control/longitudinal_controller.py`、`control/vehicle_controller.py` | 横向与纵向控制器的 `compute/update_parameters/reset`；`PIDController.step`；`clamp`；`VehicleController.compute/validate_command` |
| M8 运行管理 | `runtime/safety_supervisor.py`、`runtime/diagnostics.py` | `RuntimeSupervisor` 的生命周期、模块状态、异常钩子、`check_control/fallback_control`；`get_logger/runtime_snapshot` |

根目录及九个子包均含 `__init__.py`，当前包含 34 个 Python 文件、1 个 JSON 配置和本文档。

## 已核查的现有工程接口

1. `../repositories/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_features.py`：输入字段为 `camera_feature`、`lidar_feature`、`status_feature`；状态拼接顺序是 `[command4, vx, vy, ax, ay]`。
2. `../tools/b2d_adapter/v1_builders.py` 与 `native_builders.py`：当前工程单样本 float32 形状分别为 `(3,256,1024)`、`(1,256,256)`、`(8,)`；模型输入需要额外 batch 维。新增校验沿用此契约，不生成占位 Tensor。
3. `../repositories/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`：原版 **V1** 的网络类名就是 `V2TransfuserModel`；前向接口为 `forward(features, targets=None)`，预测轨迹字段是 `trajectory`。不引用 `DiffusionDriveV2`、RL 或 SEL。
4. 当前 V1 默认 `TransfuserConfig.trajectory_sampling` 与离线契约为 8 个未来点、间隔 0.5 s。Agent 不将该值硬编码为部署默认值；应从实际选择的训练配置、checkpoint 和时间约定绑定。模型输出是每点 `[x,y,heading]`，解码器要求调用方显式提供时间偏移。
5. 在线 M3 不再使用 `../tools/b2d_adapter/native_navigation.py` 的 near-bearing 策略；该函数只保留给离线冻结数据重建。在线唯一规则是 `navigation/status_builder.py`：稠密 `set_global_plan` 世界坐标 → `route_preview_contract.preview_command4`（20 m 弧长、2.0 m 侧向阈值、5.0 m 后轴投影）→ Navigation4；Motion4 用 A0（10 Hz 后轴位姿差分，当前后轴系 `[forward,left]`）。四维顺序 `[Left,Straight,Right,Unknown]`，Order 不映射原始 RoadOption；无效预瞄返回 `status8=None` 与显式原因，不静默回退 Unknown。
6. `../tools/b2d_adapter/object_adapter.py` 的 `candidate_causal_state` 依赖 reader、历史标注位姿及工程采样时钟。在线标量速度、原始 IMU 和 GNSS 不能直接冒充同一状态，因此不直接调用该离线 reader 流程。
7. 本仓库没有 Bench2Drive checkout，`repositories/NAVISIM_Agent_Info/` 为空。本机已有相邻工程 `/home/aethelind/work/environment-build_test/Bench2Drive/`，本轮只读核查其中 `leaderboard/leaderboard/autoagents/autonomous_agent.py`、`envs/sensor_interface.py` 和 `leaderboard_evaluator.py`，以及相邻 `Bench2DriveZoo/team_code/vad_b2d_agent.py`，没有修改它们。此处仅确认本机代码接口，不声称已认证它与 V0.0.4 采集版本完全一致。

本机评测器的实际顺序为 `get_entry_point()` → `(host,port,debug)` 构造 → `set_global_plan()` → `setup(config_path + '+' + save_name)` → `sensors()`。环境包装类保留外部基类的 `__call__`、传感器接口和指标接口，按最后一个 `+` 分离运行标签，并把原始完整路线保留给 M3；基类自身的下采样私有路线仍由基类管理。包装层也接受不带标签的普通配置路径；带 `+` 的原始文件名存在语义歧义，独立核心可直接接收该路径。更换评测器版本时需重新核查这些约定。

`bench2drive_agent.py` 导入不会导入 CARLA/Leaderboard；`get_entry_point()` 才导入包装层并把入口类绑定到入口模块供评测器查询。包装层构造按已有外部基类调用 `get_hero()`，需要运行环境已存在 CARLA world；本轮没有构造它。未来接入时工程根目录须在 `PYTHONPATH`，外部 CARLA/Leaderboard/scenario_runner 由现有环境提供。本文不提供本轮运行评测的命令。

## 统一数据与单位

公共定义全部位于 `common/types.py`。生产模块负责构造，消费者按只读约定使用；raw payload 不做昂贵深拷贝，外部生产者不得随后原地修改。

| 数据对象 | 生成模块 | 读取模块 | 关键字段与约定 |
| --- | --- | --- | --- |
| `EgoState` | M2 状态估计器 | M3/M4/M6/M7 | 工程后轴世界 `pose`、独立 `actor_pose`、GNSS 安装点 `navigation_world_xy_m`、测量速度、后轴局部二维因果速度/加速度和来源 |
| `SimulationTick` | M1 环境元数据 | M2 | 当前 CARLA 帧、GameTime 秒、固定 tick 秒；不以绝对帧号推算 GameTime |
| `SensorObservation` | M2 | M2/M4，M8 可检查 | 当前仿真时间与 `id -> SensorSample`；sample 保留原始 `frame_id`、payload、可选测量时间 |
| `GlobalRoute` | M1/M3 | M3 | 原始 GPS/世界路线 pair 序列，世界坐标系为 CARLA；RoadOption 不改写 |
| `NavigationState` | M3 | M4 | 当前时间、带坐标/参考点的局部目标、原始指令、可选 `command4`、生成策略版本 |
| `ModelFeatures` | M4 | M5 | 三个未加 batch 维的 Tensor、仿真时间、模型局部坐标系、可选世界原点 `origin` 和来源 |
| `TrajectoryPrediction` | M5，经 M6 解码器 | M6 | 点序列、预测起始时间、坐标系、来源、可选局部坐标原点 |
| `ReferenceTrajectory` | M6 | M7 | 与预测共享点结构及单位，保留时间、坐标、来源和原点 |
| `ControlCommand` | M7 | M8/M1 | 当前时间、CARLA 归一化 `steer[-1,1]`、`throttle[0,1]`、`brake[0,1]`、布尔标志 |
| `AgentRuntimeState` | M8 | M1/诊断接口 | 生命周期、成功步数、当前时间、活动模块、模块状态、最后错误类型与消息 |

`Timestamp.seconds` 是 GameTime 秒，帧号保持独立。启用当前帧策略后由显式 tick 锚点校验，再给样本写入测量时间；原始 `collect()` 工具仍保留未知测量时间。观测缓存要求时钟一致、时间严格递增，不插值或补历史帧。

`Pose2D` 记录米、弧度、坐标枚举和参考点名称；`TrajectoryPoint` 记录米、航向弧度及相对于预测起始时间的未来偏移秒。本轮显式未来轨迹要求偏移为正且严格递增。模型参考系为后轴局部 `x` 前、`y` 左、航向正向左；CARLA Actor 局部 `y` 右、航向正向右。未知坐标使用 `UNKNOWN`，禁止在特征/轨迹校验时通过。不得仅凭枚举名称假定 Actor 原点就是后轴；世界转换要求调用方提供已核实的后轴世界位姿。控制器仍需单独核实轮角到 CARLA steer 的比例与符号。

原始 Leaderboard payload 已从本机接口核查：相机为 uint8 BGRA `H×W×4`，LiDAR 为 float32 `N×4`，GNSS 为经纬高数组，IMU 为加速度/角速度/compass 七项，速度计为 `{'speed': ...}`。这些只是原始载荷布局，尚不代表完成颜色、外参、时钟、重力或后轴语义对齐。

## 调用链与生命周期

```text
M1 run_step(input_data, simulation_seconds)
  M8 begin_step / 模块状态跟踪
  M2 SensorManager.update -> (SensorObservation, EgoState)
  M3 RouteManager.update -> NavigationState
  M4 OnlineFeatureAdapter.build_features / validate_features -> ModelFeatures
  M5 TrajectoryPredictor.predict -> TrajectoryPrediction
  M6 TrajectoryProcessor.process -> ReferenceTrajectory
  M7 VehicleController.compute -> ControlCommand
  M8 RuntimeSupervisor.check_control
  M1 to_vehicle_control -> carla.VehicleControl
  M8 complete_step
```

生命周期包含 `CREATED -> CONFIGURED -> WARMING_UP`。20 Hz 下前四个有效观测抛 `WarmupRequiredError`，保留真实历史，不进入 FAILED；第五个观测具备 10 Hz 因果状态窗口后可通过 M2/M3，在当前 M4 TODO 进入 FAILED。未来完整成功步才进入 RUNNING。没有输出默认 warmup 控制；CARLA 运行器的启动响应仍待实现。默认配置不自动启用工程策略，缺少依据时仍拒绝。`setup()` 不初始化模型。

主流程记录具体失败模块、错误类型、消息及 traceback 后重新抛出。`PendingImplementationError` 同时属于项目 Agent 异常和 `NotImplementedError`。M8 还为传感器、推理、轨迹和控制故障保留独立处理钩子。未实现的失败响应接口会抛异常；没有把任意制动命令描述为已验证安全策略。日志使用具名 logger 和 `NullHandler`，由未来宿主配置输出 handler，不改动全局日志配置或写日志文件。

## 已实现与待补充部分

实际实现的确定性基础功能：

- JSON 配置加载、未知键检查、传感器 ID 检查、部分数值范围检查、配置目录相对资产路径解析；不读取资产。
- 原始观测封装、必需传感器检查、有界历史缓存和时间单调性检查；完整路线保存、`setup` 前路线传递与清理。
- 显式已核实 RGB 输入下直接复用离线相机函数；显式给定后轴世界原点时复用离线坐标数学。
- 真实输入 Tensor 的形状、dtype、有限值、传感器值域和 command4 单热校验。Tensor 校验按需导入 torch，本轮未执行。
- 显式给定坐标、时间和来源的单轨迹 pose 解码；空轨迹、长度、NaN/Inf、时间顺序检查；保留坐标/时间的参考轨迹生成。同坐标系变换可直接返回。
- 通用标量 PID、可选积分限幅、输出限幅、参数更新与状态复位；归一化控制值校验和 CARLA 对象转换接口。
- 显式配置、显式初始化后的恒速直线 Mock 轨迹生成代码。默认不选择、不初始化、不执行 Mock，真实预测失败时不回退到 Mock。
- 生命周期、模块状态记录、异常传播、诊断快照、幂等清理与完整结构性调用链。

已补充：显式固定步长当前帧校验、过期/未来帧拒绝、GNSS 安装点到 Actor/工程后轴的定位、10 Hz 因果差分和时延来源、路线进度/near 选择、导航编码、原子路线替换和可继续的真实历史 warmup。

仍未实现：部署传感器认证、异步/历史帧重采样、实际 GNSS 噪声滤波认证、在线 BGRA/RGB 与 LiDAR/状态 Feature 桥接、真实模型加载/推理、整轨迹变换/重采样、实际车控、CARLA warmup 控制响应及经验证的故障降级。

复用关系：

- 已编写直接调用：`tools.b2d_adapter.camera_adapter.proportional_camera_feature`、`coordinate_transform.ActorPose/world_to_ego/ego_to_world`、`v1_builders.official_v1_module`。这些入口本轮均未运行。
- 已接入在线 Status：`navigation/status_builder.py` 直接调用 `route_preview_contract.preview_command4` 与 `experimental_causal_state.causal_state(A0)`，构建 float32[8]；Camera/LiDAR 仍需在线桥接。
- 仅离线保留、在线不再调用：`native_navigation.near_route_command`、`B2DEgoStatusBuilder`（冻结数据重建契约）。不伪造完整 raw observation 去绕过 reader/provenance 检查。
- 原版 V1 构造会读取 anchor，并在 backbone 内使用 `timm.create_model(pretrained=True)`，因此不能在本轮顺手调用。未来需使用已核实本地资产和现有离线初始化策略，避免下载。
- checkpoint 应对照现有 `tools/training/diffusiondrive_v1/fit.py` 的 `state_dict`、`b2d_v1_fit` 契约与实际最终产物核查，不猜测去前缀或放宽权重加载。

## 配置来源与下一步

启用本轮 M2/M3 须显式配置 `sensors.alignment_policy="leaderboard-current-frame-v1"`、`ego_state.policy="b2d-causal-gnss-v1"`、`navigation.status_policy="b2d-route-preview-20m-v1"`（亦为默认值）。`ego_state` 还要求真实 GNSS/IMU/速度 ID、map `lat_ref_deg/lon_ref_deg`、`vehicle_type="vehicle.lincoln.mkz_2020"`、`geometry_policy="lincoln-10hz-rear-v1"`、显式接受 `planar_assumption=true`；GNSS/IMU 声明必须含有限的 x/y/z/roll/pitch/yaw，当前仅支持零旋转。地理参考会与 GPS/世界路线对核验；不自动求解或失败回退。

20 Hz 时 `sensors.buffer_size` 至少为 5；10 Hz 时至少为 3。状态固定读取真实的 t、t-0.1、t-0.2，非整数 tick 比或缺采样被拒绝。环境包装层每步读取当前 GameTime 与 world snapshot 的 frame/delta；CPU 驱动需先调用 `agent.set_simulation_tick(SimulationTick(frame_id, Timestamp(game_time), delta_seconds))`。这些元数据不能从传感器中猜测。warmup 异常由调用者显式处理；时钟重启使用新 Agent，当前车辆路线替换只重置导航进度，不清除物理历史。

显式 JSON 路径由外部 `setup()` 传入；独立核心传 `None` 时采用 dataclass 默认值。`config/default.json` 对应这些默认值，不自动从环境变量或训练配置目录取值。传感器物理参数、模型采样/资产路径、导航/状态策略、PID 增益和车辆参数缺省为未确定。`buffer_size=3` 仅为缓存容量，`planner.kind=diffusiondrive_v1` 为默认实现选择，日志默认 `diffusiondrive.agent/INFO`。

后续先固定部署传感器、时间/坐标及导航来源契约，再补在线特征与离线一致性；之后接入核实过的 V1 配置/checkpoint、轨迹到控制的策略和故障响应。任何运行验证应在后续明确授权的任务中开展。

## 测试与证据

Agent 的测试命令、通过/Blocked 统计、逐项用例和证据索引只在 [基础框架测试报告](../docs/agent_test_report.md) 维护；本文不重复这些结果。初始语法编译通过只代表当时文件可解析，不代表运行接口或闭环能力已验收。
