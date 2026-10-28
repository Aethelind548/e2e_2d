# 独立克隆验证

> **建立前提：** 用户授权Git RC收尾与第二轮精简。
> **解决问题：** 独立克隆验证。
> **适用范围：** refactor/release-v1；基线6d7b709；2026-10-28。
> **不包含：** 训练恢复、GPU/真实数据/CARLA验收、修改冻结资产或推送合并。
> **当前状态：** 代码RC审查；外部集成另行验收。
> **相关代码：** tools/、agent/、tests/、tools/release/。
> **最后更新：** 2026-10-28。

## 结果与范围

代码提交`67f95ea1eac8634f0b46d2d29838f15931644711`在独立目录/tmp/e2e_rc_candidate_20261028全新克隆验证：**702 passed、28 skipped、17 xfailed、0 failures/errors，747项，9.98秒**。2-worker DataLoader在允许IPC环境执行通过，没有deselect。GPU不可见、所有真实数据opt-in关闭、HF_HUB_OFFLINE=1。未从原工作区复制数据、权重、Anchor、outputs或历史配置之外的未跟踪源码。

复现命令在快速启动第4节；完整终端输出/JUnit在本地outputs/release_refactor/phase3/candidate_tests.log与candidate_tests.xml。本报告不会把JUnit的45个skipped全计作普通skip：其中28为skip、17为xfail。

## 克隆和补丁

- git clone --no-local --branch refactor/release-v1从根仓库的已提交Git对象建立新目录；根仓库暂无远程，未声称已从GitHub克隆第一方根仓库。
- git submodule update --init --checkout从**公开GitHub URL**实际下载V1，HEAD精确为9b52ed0ec06b073d82d6f392ab084c7b301c8681；不是复制原有dirty工作树。子模块未应用补丁时干净，随后git apply --check和应用DiffusionDrive.patch成功，唯一变更scheduler兼容补丁。gitlink没有改变。
- 网络在普通沙箱DNS受限；只读公开访问获准后git ls-remote与子模块下载均成功。另验证本地只读Git对象镜像初始化也可用；不混称联网测试。
- 历史V2在另一个临时干净克隆固定1cd12a1e155c34dcc471261835444c8d5587580b，原有DiffusionDriveV2.patch检查/应用成功，仅backbone的4增2删；不将V2接入当前V1。两份补丁SHA与MIT文件检查通过。

## 依赖与独立性

临时Python3.9 venv使用--system-site-packages复用已有模型软件依赖，laspy2.6.1/lazrs0.7.0经pip装入临时venv；未修改原环境，测试没有追加原outputs中的codec路径。**完整requirements在空Ubuntu环境的解析/安装、驱动/CUDA ABI仍UNVERIFIED**，不能把这次克隆测试当成干净操作系统重建验收。

227份第一方Python使用compile进行无pyc语法检查；31份JSON加载成功；14项第一方核心模块及4项官方V1模块导入成功，均验证__file__位于新克隆内部。核心formal entry帮助12项成功：release preflight、fit、status_experimental_train、status_route_eval、evaluate_checkpoint、prepare、features_cache、target_cache、feature_jobs、target_jobs、target_recovery、validate_b2d_adapter。根Git树无outputs/数据/模型/SQLite载荷。

默认依赖发现从冻结JSON读取显式资源位置；这些JSON没有重写。真实训练仍会因缺少资产/旧pin失败，而不是悄悄回退原工作区。本轮帮助、导入、合成与协议单测不打开那些路径。历史工具保留的绝对默认位置只属于显式历史/外部资源入口，不能拿其默认值当远程训练配置。

## 必须区分的验证

| 项目 | 结果 | 边界 |
|---|---|---|
| 语法/核心导入/配置/CLI | PASS | 227 Python、31 JSON、18模块、12帮助 |
| 训练数据、坐标/标签/shape合同 | PASS | 已有CPU合成/临时文件回归，无真实B2D载荷 |
| Checkpoint恢复协议 | PASS | 既有CPU状态/采样/边界/恢复单测，不是实际R2断点恢复 |
| Agent接口 | PASS（CPU协议） | fake宿主边界；17既有Pending xfail保留；真实闭环UNVERIFIED |
| 官方V1 CPU前后向 | PASS | 真实官方结构+原Loss；临时随机ResNet、合成Anchor/张量；梯度有限；0优化器步；不是官方预训练验收 |
| 可选官方Anchor与Leaderboard源码 | UNVERIFIED | 两项显式资产opt-in未设置；原SHA/数值/API/顺序断言保留，不能算PASS |
| 真实数据opt-in | UNVERIFIED | 其余26 skip；Level2旧实现缺失，启用仍不能通过 |
| GPU、R2 raw恢复、外部数据、CARLA | UNVERIFIED | 未执行，不改变冻结身份 |

基线6d7b709新克隆首次测试700 passed、26 skipped、1 failed、17 xfailed、1 deselected；失败证明官方Anchor隐式依赖，已改为显式资源检查，新增独立真实CPU模型测试，并将Leaderboard源码同样显式化。本次702通过不是把真实资产检查改称通过。

只读发布预检成功运行但结论按实际保留：Baseline BLOCKED（PASS38/BLOCKED17/UNVERIFIED1）；R2 BLOCKED（PASS41/BLOCKED42/UNVERIFIED5）；Agent UNVERIFIED（PASS5/UNVERIFIED2）。缺少外部资源及旧fit/lr源码pin并未修复或绕过。

代码测试提交之后仅维护交付报告/导航，代码树不变；最终报告提交将在同一独立克隆刷新并校验第一方/官方代码树完全一致，不把未提交原工作树当交付物。
