"""Render the implemented B2D -> DiffusionDrive V1 adapter, without reading data."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/adapter_overview'


def render(output):
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(26, 16))
    fig.patch.set_facecolor('#f5f7fb')
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 16)
    ax.axis('off')
    fig.subplots_adjust(left=.018, right=.982, top=.98, bottom=.025)
    ink, muted = '#17283f', '#60718a'
    blue, green, violet = '#3169b6', '#238565', '#7656ad'

    def text(x, y, value, size=15, color=ink, weight='normal', **kwargs):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                va='top', linespacing=1.6, **kwargs)

    def box(x, y, w, h, title, body='', code='', color=blue, fill='white', size=15):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.16',
                                  linewidth=1.2, edgecolor='#dce3ed', facecolor=fill, zorder=2))
        ax.add_patch(Rectangle((x+.16, y+h-.48), .05, .25, color=color, zorder=3))
        text(x+.32, y+h-.21, title, 18, color, 'bold', zorder=3)
        if body:
            text(x+.25, y+h-.79, body, size, zorder=3)
        if code:
            text(x+.25, y+.35, code, 11.5, muted, zorder=3)

    def arrow(points, color=blue, dashed=False):
        # Orthogonal polylines with a single final arrowhead.
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.7,
                    linestyle='--' if dashed else '-', zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                    mutation_scale=16, lw=1.7, color=color,
                                    linestyle='--' if dashed else '-', zorder=1))

    text(.4, 15.72, 'Bench2Drive → DiffusionDrive V1 · Adapter 功能总览', 29, weight='bold')
    text(.42, 15.01, '按当前实现组织：数据读取、时空转换、完整监督构建、缓存与训练接口', 17, muted)
    text(25.55, 15.08, '实现快照  2026-10-14', 12, muted, ha='right')

    for x, label in ((.45, '01  真实数据源'), (4.7, '02  读取与时空约定'),
                     (9.25, '03  特征与监督构建'), (19.15, '04  训练数据接口')):
        text(x, 14.28, label, 18, weight='bold')

    # Source cards, with future annotation explicitly limited to supervision.
    source_cards = [
        (11.9, 'RGB · 三路前视相机', '左前 / 正前 / 右前\nBench2Drive 原始图像'),
        (9.97, 'LiDAR · 点云', 'LAZ 点云读取与解码\n原始传感器坐标与高度'),
        (8.04, '逐帧标注 · 状态与 Actor', 'ego / Actor 包围盒、导航近点\n历史帧构造状态；未来帧仅用于监督'),
        (6.11, 'OpenDRIVE · 12 Town', '道路 / 车道 / 地图对象\n绑定 Town、CARLA 版本和地图 SHA'),
    ]
    for y, title, body in source_cards:
        box(.45, y, 3.82, 1.73, title, body, size=13.5)
        arrow([(4.28, y+.85), (4.52, y+.85), (4.52, 9.9), (4.68, 9.9)])

    box(4.7, 6.11, 4.12, 7.52, '共享 Adapter 基础层', color=blue)
    text(4.97, 12.79, '索引与按需读取', 17, weight='bold')
    text(4.97, 12.24, 'Route / Frame 绑定与完整性检查\n读取标注、图像和点云', 14)
    text(4.97, 11.38, 'route_index.py / frame_reader.py', 11.5, muted)
    ax.plot([4.98, 8.51], [10.91, 10.91], color='#e2e8f0', lw=1, zorder=3)
    text(4.97, 10.62, '时间与坐标转换', 17, weight='bold')
    text(4.97, 10.04, 'Actor 位姿 → 工程后轴参考\n世界坐标 → 自车局部坐标\nx 向前 · y 向左 · yaw 左转为正\n米 / 弧度；显式时钟与采样步长', 14)
    text(4.97, 8.61, 'coordinate_transform.py / temporal_sampler.py', 10.5, muted)
    ax.plot([4.98, 8.51], [8.17, 8.17], color='#e2e8f0', lw=1, zorder=3)
    text(4.97, 7.89, '导航与因果运动状态', 17, weight='bold')
    text(4.97, 7.33, '当前路线近点 → L / S / R\n历史后轴运动 → 速度 / 加速度\n工程策略携带来源与适用范围', 14)
    text(4.97, 6.43, 'native_navigation.py / native_builders.py', 10.5, muted)

    # One sample contains both dictionaries; the envelope feeds prepared caching.
    ax.add_patch(FancyBboxPatch((9.12, 5.97), 9.55, 7.92,
                               boxstyle='round,pad=0.02,rounding_size=0.16',
                               linewidth=1.1, edgecolor='#c8d9ed', facecolor='#edf3fa', zorder=0))
    # Two dictionaries: features and targets. Shapes are per sample, without B.
    text(9.26, 13.67, 'features · 推理输入', 15, blue, 'bold')
    text(13.98, 13.67, 'targets · 训练监督', 15, blue, 'bold')
    box(9.25, 10.7, 4.37, 2.46, '相机特征',
        '原版 V1 裁剪 / 拼接 / 缩放\ncamera_feature  [3, 256, 1024]\nfloat32',
        'v1_builders.py → 原版相机预处理', size=14)
    box(9.25, 8.25, 4.37, 2.23, 'LiDAR 特征',
        '已有坐标 / 高度转换 + 栅格计数\nlidar_feature  [1, 256, 256]\nfloat32',
        'lidar_decode.py / lidar_adapter.py', size=14)
    box(9.25, 6.11, 4.37, 1.92, '自车状态特征',
        '命令 4 + 速度 2 + 加速度 2\nstatus_feature  [8] · float32',
        'native_builders.py', size=14)

    box(13.99, 10.7, 4.52, 2.46, '未来轨迹监督',
        '8 个后轴相对位姿 · 每步 0.5 s\ntrajectory  [8, 3] · float32\n每点为 x / y / yaw',
        'trajectory_target.py', size=14)
    box(13.99, 8.25, 4.52, 2.23, '车辆检测监督',
        '原版范围筛选 / 最近 30 辆 / 补齐\nagent_states  [30, 5] · float32\nagent_labels  [30] · bool',
        'v1_builders.py → 原版车辆目标选择', size=14)
    box(13.99, 6.11, 4.52, 1.92, '七类 BEV 语义监督',
        'OpenDRIVE 几何 + 当前 Actor\nbev_semantic_map  [128, 256] · int64',
        'v1_bev.py → 原版七类栅格化', size=14)

    arrow([(8.85, 9.9), (9.04, 9.9), (9.04, 11.9), (9.23, 11.9)])
    arrow([(8.85, 9.9), (9.04, 9.9), (9.23, 9.35)])
    arrow([(8.85, 9.9), (9.04, 9.9), (9.04, 7.1), (9.23, 7.1)])
    # Build target branch separately, avoiding a suggestion that labels derive from features.
    arrow([(8.84, 13.36), (13.81, 13.36), (13.81, 11.9), (13.97, 11.9)])
    arrow([(13.81, 11.9), (13.81, 9.36), (13.97, 9.36)])
    arrow([(13.81, 9.36), (13.81, 7.1), (13.97, 7.1)])

    box(19.15, 10.32, 6.37, 3.31, '预生成缓存 → Dataset / Batch',
        '完整 features + targets → .pt\n接受 Manifest / 失败记录 / 内容 SHA\nPreparedDataset 验证合同并复读\n训练 worker 无需解析原始传感器与地图',
        'prepare.py / prepared.py', size=15)
    arrow([(18.69, 12), (19.13, 12)])
    text(18.48, 5.74, '每帧样本 = (features, targets)', 12, muted, ha='right')
    box(19.15, 6.11, 6.37, 3.49, '原版 DiffusionDrive V1',
        '输入：相机 / LiDAR / 状态特征\n监督：轨迹 / 车辆检测 / 七类 BEV\n复用原版网络、BEV Head 与损失函数\n已完成有限步训练与断点恢复验证',
        'repositories/DiffusionDrive · fit.py', size=15)
    arrow([(22.32, 10.3), (22.32, 9.62)])

    # Independent semantic additions: solid = implemented; dashed = future integration.
    box(.45, 1.9, 8.37, 3.21, 'M4 · 可选七类语义增强',
        'GuardRail：有可靠轮廓与高度依据时 → Static / 类别 4\n倾斜 Actor：真实 3D 包围盒 → XY 占用投影\n跨层 Actor：默认诊断；可选道路支撑过滤\nv1_original_7class  /  v1_semantic_enriched：规则与缓存隔离',
        'v1_label_rules.py / v1_geometry.py / v1_bev.py', color=green, fill='#edf7f1', size=15)
    arrow([(8.84, 3.56), (13.64, 3.56), (13.64, 6.94), (13.97, 6.94)], color=green)
    text(9.1, 3.98, '可选增强 → 同一七类合同', 12, green)

    box(9.25, .69, 9.26, 2.51, 'M5 · 独立地图几何与语义层（已实现）',
        '车道边界 / RoadMark 区间属性 / Crosswalk Mask\n按地图版本保存世界几何 → 按后轴位姿查询局部多层 Mask\n独立 uint8 数值层；不覆盖原版七类标签',
        'static_layers/opendrive.py · 地图缓存与局部查询', color=violet, fill='#f3eefb', size=14)
    arrow([(2.36, 6.08), (2.36, 5.5), (9.03, 5.5), (9.03, 1.92), (9.23, 1.92)], color=violet)
    text(9.26, 5.18, '地图 + 自车后轴位姿 / 世界高度', 12, violet)
    arrow([(18.53, 1.92), (19.13, 1.92)], color=violet, dashed=True)
    box(19.15, .69, 6.37, 2.51, '后续实验接口 · 尚未接入训练',
        'B：独立 Boundary 监督组合\nC：附加边界 / 标线 / Crosswalk 预测头\n当前为实验设计；CARLA Agent 后续开发',
        'experiments/m5_abc_design.json', color=muted, fill='#f0f2f6', size=14)

    # The legend states semantic IDs, preserving the official seven-class contract.
    labels = ['0 背景', '1 道路', '2 步行道', '3 中心线', '4 静态物', '5 车辆', '6 行人']
    colors = ['#eeeeee', '#666666', '#89cce0', '#ffdf00', '#eb5757', '#5268f5', '#40c97b']
    text(19.2, 5.73, 'BEV：前方 32 m × 横向 64 m · 0.25 m / 像素', 12, muted)
    for i, (label, color) in enumerate(zip(labels, colors)):
        x, y = 19.2+(i % 4)*1.56, 5.09-(i // 4)*.46
        ax.add_patch(Rectangle((x, y), .17, .17, facecolor=color, edgecolor='#a0aabb', lw=.5))
        text(x+.24, y+.2, label, 11.5)
    text(19.2, 3.92, '全链路检查：来源 / 单位 / 几何 / dtype / shape / SHA\n缺失或不可靠几何显式拒绝，保留定位信息', 12, muted)
    text(.47, 1.33, '实线：已实现数据流\n虚线：后续训练接入', 13, muted)
    text(.47, .53, '说明：尺寸不含 batch 维；七类 BEV 为监督标签，非预测结果。工程坐标、时钟与导航策略不表示与 NAVSIM 数据采集完全等效。', 11.5, muted)

    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'adapter_functional_overview.{suffix}', dpi=180,
                    facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    render(parser.parse_args().output_dir)
