# Agent M2/M3 实现与验证报告

> **建立前提：** 既有工程的接口、设计或有限验证记录；版本和样本以正文为准。
> **解决问题：** 说明Agent M2/M3 实现与验证报告的结论、依据和适用边界。
> **适用范围：** Agent框架与CPU回归。
> **不包含：** 宣称已完成真实CARLA闭环。
> **当前状态：** 当前维护。
> **相关代码：** agent/、tests/agent/
> **最后更新：** 2026-10-21（结构治理：接收 Agent README 的历史语法记录；不追认历史验收）。

本轮完成了受限工程输入契约下的 M2/M3 实现，**真实 Agent 主链从 M2 阻塞推进到 M4**。没有替换传感器管理、状态估计、导航或在线 Adapter 来取得该结果；M4 仍显式拒绝未实现的 Feature 构建。

## 输入依据与实际功能

接口分析先于实现形成，见 [M2/M3 输入依据](agent_m2_m3_interface_analysis.md)。本机源码指纹记录在 [source_evidence.json](../outputs/diffusiondrive_v1/validation/agent/m2_m3/source_evidence.json)。没有推测缺失的 map 地理参考、IMU 安装旋转或 RoadOption 整数语义。

| 问题 | 本轮完成 | 保留边界 |
| --- | --- | --- |
| O01 | `SimulationTick` 显式记录当前帧、GameTime、delta；固定步长帧差/时间差校验；拒绝过期、未来、不一致/重复帧；样本写入已校验时间；失效时钟观测不入缓存 | 只支持本地 Leaderboard 的当前帧完整观测策略，不支持异步重采样 |
| O02 | `CarlaGeoReference`、GNSS 安装点→Actor→工程后轴转换；与 GPS/世界路线对核验地理参考；从真实 t/t-0.1/t-0.2 样本构造二维因果状态；记录速度/加速度时延；拒绝未知基准、缺历史、位置/航向跳变 | 明确选用 Lincoln 平面工程策略；合成对齐不是实际 GNSS 噪声、历史采集资产或非平面条件的认证 |
| O03 | 复现本地 Zoo RoutePlanner 进度/near 选择；单节点路线、游标不回退、原子路线替换和 generation | 终点 behind/过近、离线导航距离门限不满足时明确拒绝，不自动选择 far 或伪造目标 |
| O04 | 使用真实 near 世界坐标和当前 Actor/后轴位姿，按离线同一方位阈值生成 command4；原始 RoadOption 只保留，不参与编码 | 未解决原始整数映射，也不声称等价 NAVSIM 20 m 或认证 V0.0.4 采集端 |
| O10（部分） | 真实历史 warmup、WARMING_UP 生命周期、可继续下一 tick、实际控制间隔来源；路线切换只重置导航，保留物理历史；清理重置时钟和诊断对象 | warmup 不产生默认控制；真实 CARLA 调用者的启动响应、同一销毁实例重启和故障控制仍未实现 |

20 Hz 下前三个所需 10 Hz 状态样本来自 5 个实际观测，前 4 次 `run_step` 抛 `WarmupRequiredError`，记录 warmup 而非 FAILED；第 5 次通过 M2/M3。10 Hz 下需要 3 个观测。标量速度来自速度计，二维速度和加速度来自后轴位置差分；原始 IMU 加速度未用于替代模型状态。

## 最终执行结果

| 范围 | Passed | Failed / Error | Skipped | Blocked |
| --- | ---: | ---: | ---: | ---: |
| L1 模块测试 | 205 | 0 | 0 | 10 |
| L2 主入口/推进测试 | 8 | 0 | 0 | 8 |
| 已有 V1 / Adapter 回归 | 35 | 0 | 0 | 0 |
| **合计 266 项** | **248** | **0** | **0** | **18** |

最终耗时 4.68 s，原始 pytest 输出为 `248 passed, 18 xfailed, 14 warnings`。18 个严格 xfail 按 Blocked 单列，不计入通过；退出码 0 不表示完整 L2 控制链通过。14 条 warning 为既有 matplotlib/pyparsing 弃用提示，未屏蔽。

- 新增 `tests/agent/test_m2_m3.py` 的 **55 项全部通过**，其中 7 项属于主入口推进/包装层集成。
- 删除 M2 alignment、M2 ego state、M3 route target、M3 command 的 4 个旧 Pending 探针，替换为正向功能和异常测试；未配置策略的拒绝测试仍保留。
- 原有 8 个完整 Mock 场景重新运行，均在四次真实 warmup 后于 **M4.features** 阻塞。实际断言 `M2.sensors=completed`、`M3.navigation=completed`、`M4.features=failed`，predictor 调用数和控制输出数均为 0。
- 5 个直/左/右/减速/停车主链推进用例验证当前阶段目标已达成；这不等于模型已经预测或车辆已有控制响应。
- 30 次连续 **M2/M3** 有效更新通过（此前另有四次 warmup）；完整 Agent 的 30 个控制周期仍为 Blocked，完成数为 0，未用 M2/M3 成功替代该结果。
- 外部包装层使用 GameTime/world snapshot 元数据的测试通过，只替换外部模块，不提供 Actor 运动真值；真实 CARLA Smoke 为 **Not Run**。

证据：[最终日志](../outputs/diffusiondrive_v1/validation/agent/m2_m3/final.log)、[JUnit](../outputs/diffusiondrive_v1/validation/agent/m2_m3/final.xml)、[统计](../outputs/diffusiondrive_v1/validation/agent/m2_m3/summary.json)、[每项测试函数/源文件/状态](../outputs/diffusiondrive_v1/validation/agent/m2_m3/case_catalog.md)。

## 运行命令与环境

从工程根目录执行：

```bash
agent_python=/home/aethelind/miniconda3/envs/navsim_gpu/bin/python
agent_laz_site="$PWD/outputs/diffusiondrive_v1/assets/laz_decoder_env/lib/python3.9/site-packages"
env PYTHONPATH="$agent_laz_site:$PWD/repositories/DiffusionDrive:$PWD" \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/e2e-agent-test-mpl \
  "$agent_python" -m pytest -q -rx -p no:cacheprovider \
  tests/agent \
  tools/validation/b2d_adapter/test_v1_builders.py::V1ContractTests \
  tools/validation/b2d_adapter/test_adapter.py::AdapterTests \
  --junitxml=outputs/diffusiondrive_v1/validation/agent/m2_m3/final.xml \
  > outputs/diffusiondrive_v1/validation/agent/m2_m3/final.log 2>&1
```

沿用 Python 3.9.23、pytest 8.4.2、Torch 2.7.1+cu128、NumPy 1.23.4 的现有环境，屏蔽 GPU、限制 CPU 线程；LAZ 依赖只读引用现成安装，未下载/安装。种子仍为 20261016，GNSS 合成数据由独立 CARLA 正投影公式生成。环境基线见 [environment.json](../outputs/diffusiondrive_v1/validation/agent/environment.json)。

首轮命令为相同环境下 `pytest -q -p no:cacheprovider tests/agent --junitxml=outputs/diffusiondrive_v1/validation/agent/m2_m3/initial.xml`，日志重定向至 `initial.log`；第二轮同样仅选择 `tests/agent`，输出 `after_fixes.xml/log`；最后使用上述合并命令。

| 阶段 | 真实结果 | 原因/处理 |
| --- | --- | --- |
| 首轮 | 202 Passed、5 Failed、18 Blocked | 5 个离线公式对照夹具漏提供 `raw.ref`，在读取来源元数据时失败，未完成数值比较；生产状态代码未因此改成固定输出。见 [initial.log](../outputs/diffusiondrive_v1/validation/agent/m2_m3/initial.log) |
| 修复夹具并补连续/包装层测试 | 211 Passed、18 Blocked | 补齐 ref 后，直/左/右/减速/停车在线状态均与实际 `candidate_causal_state` 公式比较通过。见 [after_fixes.log](../outputs/diffusiondrive_v1/validation/agent/m2_m3/after_fixes.log) |
| 补地图参考/compass 边界并合并旧回归 | 248 Passed、18 Blocked | 新增 2 个异常检查；包含 35 个已有回归，全部无非预期失败 |

上一轮 193 Passed / 22 Blocked 的报告、计划和问题清单已保存在 先前快照（已由本文整合），旧日志/XML 也保留；不把本轮结果覆盖为上一轮曾经通过。

## 当前可验证位置

```text
显式当前 tick + 原始观测
  -> M2 当前帧/固定时钟校验
  -> 前四次：WarmupRequiredError，保留真实历史，不返回控制
  -> 第五次：后轴因果状态（含测量时间/参考点/时延）
  -> M3 路线推进 / near 世界目标 / command4
  -> M4 OnlineFeatureAdapter.build_features
  -> PendingImplementationError，Supervisor 记录 M4.features 失败
```

本轮没有构造网络、加载权重、训练、数据重生成、启动 CARLA 或完整评测；未修改原版 V1、生产 B2D Adapter、训练配置、处理任务、Feature 队列/缓存或保存路径。只改 Agent、测试和本轮文档/证据。下一轮 M4 的前置条件见 [未解决问题](问题记录.md)。

## 修改范围

- M2：新增 `agent/sensors/georeference.py`；实现 `ego_state.py` 的 `CausalGNSSStateEstimator` 和 `sensor_manager.py` 的对齐/时钟校验。
- M3：实现 `navigation/route_manager.py`、`command_encoder.py`。
- 公共/配置：`common/types.py`、`common/errors.py`、`config/settings.py`、`config/default.json`。
- 编排/诊断：`bench2drive_agent.py`、`environment_interface.py`、`runtime/safety_supervisor.py`、`runtime/diagnostics.py`。
- 测试：新增 `test_m2_m3.py`；更新 conftest、生命周期/导航用例的真实路线夹具、完整 Mock 用例的新阻塞位置、剩余 Pending 探针。
- 文档：更新 Agent README、测试方案、本文、问题清单；新增接口分析和 `m2_m3/` 证据目录。

## 长期回归范围与方法


本轮目标是让真实主入口在已有输入依据下通过 M2/M3 并到达真实 M4；不要求补齐 M4、模型或车控。先阅读 [接口分析](agent_m2_m3_interface_analysis.md)，采用显式固定步长/当前帧、明确地图参考和安装、Lincoln 平面工程状态策略与本地 Zoo near 路线策略。

## 实际模块与验证位置

| 模块 | 当前实现 | 本轮测试重点 |
| --- | --- | --- |
| M1 | 本地 Leaderboard 包装、SimulationTick 元数据入口、完整主链和清理 | 保留签名；warmup 可继续；GPS/世界路线参考校验；新路线不污染历史 |
| M2 | 当前帧/固定时钟校验、原始载荷校验、GNSS Mercator 定位、工程后轴因果状态、真实历史 warmup | 明确单位/参考点/时延；无插值/伪造；失效帧和缺依据拒绝 |
| M3 | 本地 RoutePlanner 进度、near 选择、方位 command4、路线 generation | 对照本地代码；对照离线编码；原始整数不参与；原子路线替换 |
| M4 | 既有 RGB 公共函数与 Tensor 契约校验；完整在线构建仍 TODO | 本轮主链必须实际进入该 TODO，不能以特征替身绕过 |
| M5–M8 | 既有 Mock/轨迹校验/PID/诊断；真实推理、实际车控与失败响应仍不完整 | 回归原有测试；不扩大实现范围 |

## 数据与隔离条件

- 现有 Python/pytest CPU 环境，GPU 屏蔽、线程限制为 1；合成随机输入种子 20261016，不启动训练/评测/Server。
- GPS 从独立的本机 CARLA `location_to_gps` 正投影公式生成；生产代码执行逆投影。fixture 中的 42°/2°只是明确的合成地图参考，不作为部署默认值。
- 合成输入包含声明尺寸的 BGRA、Nx4 LiDAR、IMU[7]、GNSS[3]、速度字典。IMU 的加速度刻意设置为与后轴差分不同的值，以验证它没有被误用。
- 时钟锚点独立提供：绝对 frame 从 1000 起、GameTime 从 10 s 起，明确证明没有按 `frame*dt` 猜测时钟；20 Hz 状态选择帧 1000/1002/1004。
- 只替换外部 CARLA 对象/外部世界 tick、Leaderboard 基类及允许的 predictor；M2/M3/M4 全部是真实代码。离线公式对照使用单元级只读 reader fixture，不替换 Agent 中间模块。
- 不修改生产 Adapter、训练配置、Feature 处理代码或任务；所有合成磁盘数据由 pytest 放在 `/tmp`。

## 新增功能测试映射

以下函数均位于 `tests/agent/test_m2_m3.py`；参数化后合计 55 项。每项完整参数 ID、具体源文件和实际结果见 [逐项索引](../outputs/diffusiondrive_v1/validation/agent/m2_m3/case_catalog.md)。

| 测试函数 | 源文件/接口 | 核查内容 |
| --- | --- | --- |
| `test_current_frame_anchor_and_real_history_warmup` | `agent/sensors/sensor_manager.py`、`ego_state.py` | 对齐、真实四次 warmup、后轴/Actor 参考、10 Hz 选帧、来源时延和 reset |
| `test_acceleration_and_lateral_velocity_are_position_based` | `ego_state.py::estimate` | 加速度/横向速度来自位置；标量速度不替代二维分量 |
| `test_state_matches_existing_candidate_causal_formula` | `ego_state.py`、`tools/b2d_adapter/object_adapter.py::candidate_causal_state` | 直、左、右、减速、停车五组独立合成位姿，与实际离线函数比较 |
| `test_bad_clock_or_expired_observation_does_not_enter_history` | `sensor_manager.py::validate_alignment/update` | 旧帧、未来帧、错误时间、帧跳、变量 delta、重复帧、缺 tick 拒绝 |
| `test_unsupported_sampling_and_insufficient_capacity_rejected`、`test_missing_required_past_sample_is_not_fabricated` | `ego_state.py`、`sensor_manager.py` | 非整数采样比、容量不足、缺 t-0.1 采样，不插值/补状态 |
| `test_missing_state_basis_rejected`、`test_nonzero_mount_rotation_rejected` | `ego_state.py::__init__`、`georeference.py` | 未知车型/参考、非平面声明、非零安装、缺地图参数拒绝 |
| `test_invalid_gnss_domain_and_motion_discontinuity_rejected`、`test_compass_outside_sensor_domain_is_rejected` | `georeference.py`、`ego_state.py` | Mercator 极点、瞬移、非法 compass 明确失败 |
| `test_route_near_target_and_commands_ignore_raw_roadoption`、`test_online_command_exactly_matches_offline_bearing` | `navigation/route_manager.py`、`command_encoder.py`、离线 `native_navigation.py` | 直/左/右 command4、9 组旋转坐标对照、任意原始整数不影响结果 |
| `test_progress_matches_local_zoo_planner_and_does_not_rewind` | `route_manager.py`、本机 `team_code/planner.py::RoutePlanner` | 使用本机算法作为独立对照，推进与返回 near 坐标相同且不回退 |
| `test_route_switch_resets_progress_and_bad_replacement_is_atomic`、`test_invalid_route_geometry_rejected`、`test_behind_or_out_of_range_near_is_not_replaced` | `route_manager.py`、`command_encoder.py` | 切换、generation、重复/非有限路线、behind/out-of-range 不伪造目标 |
| `test_real_chain_passes_m2_m3_and_reaches_real_m4` | `bench2drive_agent.py::run_step` → 真实 M2/M3/M4 | 五场景四次 warmup 后在 M4 明确停止，M2/M3 状态与时间/command 对应正确，无控制输出 |
| `test_agent_route_switch_during_warmup_preserves_physical_history` | `bench2drive_agent.py::set_global_plan/run_step` | 主入口切换路线，保留物理历史、清空旧导航、按新路线进入 M4 |
| `test_ten_hz_ticks_need_three_actual_observations` | `sensor_manager.py`、`ego_state.py` | 10 Hz 下三条真实样本可用，不把 20 Hz 五样本数量硬编码为状态算法 |
| `test_real_m2_m3_30_ready_updates_with_no_padding` | 真实 SensorManager + RouteManager | 四次 warmup 后连续 30 次 M2/M3 更新、滚动选帧；不宣称完整 Agent 30 控制周期 |
| `test_bad_state_or_navigation_basis_is_explicit` | `route_manager.py`、`command_encoder.py` | 缺 GNSS 世界位置、后轴几何不一致、未知导航策略拒绝 |
| `test_live_wrapper_uses_current_tick_without_actor_motion` | `environment_interface.py`、真实 M2/M3/M4 | 只替换外部 GameTime/world metadata，验证当前 tick 自动传递，不读取 Actor 运动真值 |
| `test_wrong_georeference_is_rejected_against_real_route_pairs` | `ego_state.py::validate_route_reference`、`bench2drive_agent.py::setup` | 地理参考数值合法却与路线不匹配时明确拒绝，不自动求解/回退 |

## 已有测试调整与回归

保留 `test_config_lifecycle.py`、`test_sensors_navigation.py`、`test_features_predictor.py`、`test_trajectory_control_runtime.py` 的既有断言。路线夹具改为实际 GPS/Transform 布局；未配置 alignment/state/navigation 的守卫测试明确标为未配置条件，不再声称 M2/M3 算法未实现。

`test_pending_capabilities.py::test_required_capability_blocked` 移除已完成的 4 个参数，保留 M4 三项、M6 三项、M7 两项、M8 失败响应和孤立首步控制共 10 个真实 TODO 探针。

`test_mock_integration.py::test_real_agent_mock_scenarios` 的 8 个完整 Mock 场景重新走真实 run_step：先断言四次 warmup，再断言 M2/M3 完成且真实 M4 阻塞。predictor 未调用，不将非法预测/推理异常等尚未到达的阶段伪报为覆盖成功；30 周期完整控制仍记 Blocked。

同时运行已有 `V1ContractTests` 10 项和 `AdapterTests` 25 项，核查现有公共配置/几何/传感器处理契约未被破坏。每个回归项的源文件/测试函数也在本轮索引中。

## 判定与边界

- 本阶段通过条件：受支持的显式输入下 M2/M3 正向与异常测试通过，真实主链到达 M4，报告新的具体阻塞位置。
- Blocked 用 `xfail(strict=True, raises=PendingImplementationError)`，其他异常是失败；完成旧能力后必须移除对应探针，不允许永久 xfail 掩盖已实现代码。
- warmup 使用专用异常和状态计数，不返回看似有效的零状态或车辆控制。CARLA 启动响应仍未验证。
- 仅在本机 oracle 源码不存在时条件 skip 并写出路径；本次最终 Skipped 为 0。
- 真实 CARLA Smoke 未运行，外部模块替身不构成真实环境通过证据。命令、统计和限制见 [测试报告](agent_test_report.md)。

## 初次框架测试的有效结论

初次结果为 193 Passed / 22 Blocked；后续 M2/M3 结果以本文前文为准。早期修复涉及配置类型与路径、传感器载荷、轨迹坐标/时间、PID、生命周期与故障传播；长期测试均保留。证据统一位于 `outputs/diffusiondrive_v1/validation/agent/`，新运行使用独立输出目录，不覆盖冻结结果。

初始搭建阶段另做过一次 `py_compile` 语法检查：`agent/` 下 33 个新增 Python 文件可解析，编译产物仅写入 `/tmp` 后删除。该记录只证明语法可解析，不代表运行接口、M2/M3 或闭环能力已验收。
