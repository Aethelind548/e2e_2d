# M2/M3 输入依据与实现边界

> **建立前提：** 既有工程的接口、设计或有限验证记录；版本和样本以正文为准。
> **解决问题：** 说明M2/M3 输入依据与实现边界的结论、依据和适用边界。
> **适用范围：** Agent框架与CPU回归。
> **不包含：** 宣称已完成真实CARLA闭环。
> **当前状态：** 当前维护。
> **相关代码：** agent/、tests/agent/
> **最后更新：** 2026-10-20（说明头/入口治理；不追认历史验收）。

本轮先核查本机代码，再按以下受限契约实现；不认证 V0.0.4 的历史采集资产，也不猜测缺失的地理参考或 RoadOption 映射。

1. **传感器与频率**：本机 `environment-build_test/Bench2DriveZoo/team_code/vad_b2d_agent.py::sensors` 声明六相机、IMU、GNSS、速度计，没有启用 LiDAR。前三相机为 1600×900；IMU `sensor_tick=0.05`、GNSS `0.01`、速度计声明 20 Hz。本机 Leaderboard evaluator 固定步长 1/20 s；`agent_wrapper.py` 实际将速度计频率设为仿真步频，LiDAR 若声明则硬设旋转 10 Hz（不等于每秒只回调十次）。wrapper 并未将普通传感器声明中的 sensor_tick 全部传入 blueprint；不能仅凭配置值宣称测量频率。本 Agent 尚无已认证部署传感器布局，新增配置只提供显式 opt-in 条件，不自动复制为默认部署配置。
2. **时钟**：`leaderboard/envs/sensor_interface.py::get_data(frame)` 丢弃不同帧的数据并返回 `(frame,payload)`；`AutonomousAgent.__call__` 使用 `GameTime.get_frame/get_time`。`scenario_runner/.../timer.py::on_carla_tick` 首次只累加一个 delta，之后按帧差累加，GameTime 起点与 CARLA elapsed 不同。采用外部提供的当前 `SimulationTick(frame,GameTime秒,delta秒)` 作为锚点，校验所有观测处于当前帧及跨步帧差/时间差；不由某个传感器推断当前帧，不使用绝对 `frame*dt`。只支持固定步长、当前帧完整观测；异步/旧帧/未来帧明确拒绝。
3. **定位与参考点**：本机 `team_code/planner.py::gps_to_location` 给出 CARLA Mercator 反投影，地球半径 6378137 m，必须显式提供 map 的 lat_ref/lon_ref，不能沿用示例默认 42/2 或失败时 0/0。`tools/data_collect.py` 以 `degrees(compass)-90` 构造 CARLA yaw，Zoo 中 `ego_theta=-compass+pi/2` 也对应模型左正航向。实现仅支持显式接受的平面、零旋转 IMU/GNSS 安装；先把 GNSS 安装点世界位置反算为 Actor，再用现有 `actor_to_reference` 转工程后轴。原始 IMU 加速度（含重力等影响）不直接进入模型。
4. **与离线因果状态对齐**：`tools/b2d_adapter/object_adapter.py::candidate_causal_state` 在当前及前两帧后轴位置上按 0.1 s 计算后向一阶/二阶差分，再旋转到当前后轴局部 x 前/y 左；速度时延 0.05 s、加速度时延 0.1 s。工程几何来自 `v1_native_config()`：Lincoln MKZ 2020、后轴 Actor offset (-1.388632877,0) m。本轮使用显式声明的同一工程策略，仅从真实历史中选择 t、t-0.1、t-0.2；20 Hz 时需至少五个观测。不足历史为可诊断 warmup 异常，不生成零速度/零加速度或默认控制；完整历史缺采样、非整数步比、未知车型/安装/地理参考均拒绝。GNSS 估计不是离线标注真值，数值对齐只在同源合成位姿上核查公式。
5. **路线与 command4**：本机 `team_code/planner.py::run_step` 在累计 50 m 范围内选择距离当前位置不超过 4 m 的最远节点推进 deque，保留至少两个点并返回第二点；当前位置是 GNSS 反投影的安装点位置。M3 在原始完整世界路线中复现此选择，保留原始 RoadOption 但不读取其整数值。`native_navigation.py::near_route_command` 在工程后轴坐标中检查 x>0、距离[0.5,60] m，bearing 超过 +10°为 Left、小于 -10°为 Right，否则 Straight，目标顺序 L/S/R/Unknown；不自动输出 Unknown，不改选 far 点。在线编码复用相同坐标数学和现有工程阈值，用对照离线函数的测试防止漂移；不修改正在被 Feature 任务使用的 Adapter 源码或指纹。
6. **生命周期与边界**：路线替换原子校验、重置导航进度/代号，保留同一车辆的物理历史；重启车辆/时钟则创建新 Agent。warmup 只积累真实观测并向调用者抛专门异常，Supervisor 不进入 FAILED；需要有效控制的 CARLA 运行器仍需后续定义启动响应，不能把此路径称为已能在 CARLA 行驶。M4 的完整特征构建保持 TODO；集成验收位置为 M2/M3 完成并由真实 M4 明确阻塞。

依据均来自本机已存在 checkout；本轮不连接 CARLA、不读取完整数据集、不修改 Feature 扩展任务。部署前仍须验证 GNSS 地理参考/噪声、真实安装和车辆资产身份。
