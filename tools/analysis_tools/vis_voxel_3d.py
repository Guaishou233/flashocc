#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
可视化 pkl 文件中 target_1_16 的 3D 体素图
使用 voxel_origin 作为原点绘制体素网格
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_palette():
    """获取 ScanNet 类别调色板"""
    return np.array([
        (174, 199, 232),  # 0: wall
        (152, 223, 138),  # 1: floor
        (31, 119, 180),   # 2: cabinet
        (255, 187, 120),  # 3: bed
        (188, 189, 34),   # 4: chair
        (140, 86, 75),    # 5: sofa
        (255, 152, 150),  # 6: table
        (214, 39, 40),    # 7: door
        (197, 176, 213),  # 8: window
        (148, 103, 189),  # 9: bookshelf
        (196, 156, 148),  # 10: picture
        (23, 190, 207),   # 11: counter
        (247, 182, 210),  # 12: desk
        (219, 219, 141),  # 13: curtain
        (255, 127, 14),   # 14: refrigerator
        (158, 218, 229),  # 15: showercurtrain
        (44, 160, 44),    # 16: toilet
        (112, 128, 144),  # 17: sink
        (227, 119, 194),  # 18: bathtub
        (82, 84, 163),    # 19: otherfurniture
        (128, 128, 128),  # 20: unknown (255映射)
    ], dtype=np.uint8)


def get_class_names():
    """获取类别名称"""
    return (
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
        'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'unknown'
    )


def create_voxel_cube(center, size):
    """
    创建一个体素立方体的顶点
    
    Args:
        center: (x, y, z) 体素中心坐标
        size: 体素大小（标量或 (dx, dy, dz)）
    
    Returns:
        vertices: (8, 3) 立方体的8个顶点
    """
    if isinstance(size, (int, float)):
        size = np.array([size, size, size])
    else:
        size = np.array(size)
    
    half_size = size / 2.0
    x, y, z = center
    
    vertices = np.array([
        [x - half_size[0], y - half_size[1], z - half_size[2]],  # 0
        [x + half_size[0], y - half_size[1], z - half_size[2]],  # 1
        [x + half_size[0], y + half_size[1], z - half_size[2]],  # 2
        [x - half_size[0], y + half_size[1], z - half_size[2]],  # 3
        [x - half_size[0], y - half_size[1], z + half_size[2]],  # 4
        [x + half_size[0], y - half_size[1], z + half_size[2]],  # 5
        [x + half_size[0], y + half_size[1], z + half_size[2]],  # 6
        [x - half_size[0], y + half_size[1], z + half_size[2]],  # 7
    ])
    
    return vertices


def get_cube_faces(vertices):
    """
    获取立方体的6个面
    
    Args:
        vertices: (8, 3) 立方体的8个顶点
    
    Returns:
        faces: list of (4, 3) 每个面的4个顶点
    """
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
    ]
    return faces


def visualize_voxel_3d(pkl_path, output_path=None, voxel_size=None, 
                       show_empty=False, alpha=0.6, figsize=(12, 10)):
    """
    可视化 target_1_16 的 3D 体素图
    
    Args:
        pkl_path: pkl 文件路径
        output_path: 输出图像路径（如果为 None 则显示）
        voxel_size: 体素大小，如果为 None 则根据 grid_config 推算
        show_empty: 是否显示空体素（标签为0）
        alpha: 体素透明度
        figsize: 图像大小
    """
    # 加载数据
    print(f"加载数据: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # 获取数据
    voxel_origin = np.array(pkl_data['voxel_origin'])  # (3,)
    target_1_16 = pkl_data['target_1_16']  # (15, 15, 9)
    
    print(f"voxel_origin: {voxel_origin}")
    print(f"target_1_16 形状: {target_1_16.shape}")
    print(f"target_1_16 数据类型: {target_1_16.dtype}")
    print(f"target_1_16 值范围: [{target_1_16.min()}, {target_1_16.max()}]")
    print(f"唯一标签: {np.unique(target_1_16)}")
    
    # 确定体素大小
    # 根据配置文件，target_1_16 的体素大小约为 1.6m (x, y), 0.6m (z)
    if voxel_size is None:
        # 根据 grid_config 推算：target_1_4 是 0.4m，target_1_16 是 4 倍
        voxel_size = np.array([1.6, 1.6, 0.6])  # (x, y, z) 单位：米
    else:
        if isinstance(voxel_size, (int, float)):
            voxel_size = np.array([voxel_size, voxel_size, voxel_size])
        else:
            voxel_size = np.array(voxel_size)
    
    print(f"体素大小: {voxel_size} 米")
    
    # 获取调色板
    palette = get_palette()
    class_names = get_class_names()
    
    # 创建 3D 图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取非空体素的位置和标签
    Dx, Dy, Dz = target_1_16.shape
    voxel_count = 0
    
    for i in range(Dx):
        for j in range(Dy):
            for k in range(Dz):
                label = target_1_16[i, j, k]
                
                # 跳过空体素（如果 show_empty=False）
                if not show_empty and label == 0:
                    continue
                
                # 计算体素在世界坐标系中的中心位置
                # 体素索引 (i, j, k) 对应的世界坐标
                voxel_center = voxel_origin + np.array([
                    (i + 0.5) * voxel_size[0],
                    (j + 0.5) * voxel_size[1],
                    (k + 0.5) * voxel_size[2]
                ])
                
                # 创建体素立方体
                vertices = create_voxel_cube(voxel_center, voxel_size)
                faces = get_cube_faces(vertices)
                
                # 获取颜色
                if label == 255:
                    # 255 映射到 20 (unknown)
                    color_idx = 20
                elif 0 <= label < len(palette):
                    color_idx = int(label)
                else:
                    # 无效标签，使用灰色
                    color_idx = 20
                
                color = palette[color_idx] / 255.0  # 归一化到 [0, 1]
                
                # 绘制体素
                voxel = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                                       edgecolor='black', linewidths=0.1)
                ax.add_collection3d(voxel)
                voxel_count += 1
    
    print(f"绘制了 {voxel_count} 个体素")
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'3D Voxel Visualization (target_1_16)\nOrigin: {voxel_origin}', fontsize=14)
    
    # 计算边界框
    all_centers = []
    for i in range(Dx):
        for j in range(Dy):
            for k in range(Dz):
                if show_empty or target_1_16[i, j, k] != 0:
                    center = voxel_origin + np.array([
                        (i + 0.5) * voxel_size[0],
                        (j + 0.5) * voxel_size[1],
                        (k + 0.5) * voxel_size[2]
                    ])
                    all_centers.append(center)
    
    if all_centers:
        all_centers = np.array(all_centers)
        x_min, x_max = all_centers[:, 0].min(), all_centers[:, 0].max()
        y_min, y_max = all_centers[:, 1].min(), all_centers[:, 1].max()
        z_min, z_max = all_centers[:, 2].min(), all_centers[:, 2].max()
        
        # 添加一些边距
        margin = 0.5
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        ax.set_zlim([z_min - margin, z_max + margin])
    
    # 添加图例
    unique_labels = np.unique(target_1_16)
    unique_labels = unique_labels[unique_labels != 0]  # 排除空体素
    
    if len(unique_labels) > 0:
        legend_elements = []
        for label in unique_labels[:10]:  # 最多显示10个类别
            if label == 255:
                color_idx = 20
                name = 'unknown'
            elif 0 <= label < len(class_names):
                color_idx = int(label)
                name = class_names[color_idx]
            else:
                continue
            
            color = palette[color_idx] / 255.0
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color, label=f'{label}: {name}'))
        
        if len(unique_labels) > 10:
            legend_elements.append(Patch(facecolor='gray', label=f'... and {len(unique_labels) - 10} more'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {output_path}")
    else:
        plt.show()


def list_available_samples(data_dir):
    """列出可用的样本"""
    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return []
    
    samples = []
    for scene_name in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene_name)
        if os.path.isdir(scene_path):
            pkl_files = [f for f in os.listdir(scene_path) if f.endswith('.pkl')]
            pkl_files.sort()
            for pkl_file in pkl_files:
                pkl_path = os.path.join(scene_path, pkl_file)
                samples.append({
                    'scene': scene_name,
                    'frame': pkl_file.replace('.pkl', ''),
                    'path': pkl_path
                })
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='可视化 pkl 文件中 target_1_16 的 3D 体素图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化指定文件
  python vis_voxel_3d.py --pkl_path /path/to/scene0000_00/00000.pkl
  
  # 列出可用样本
  python vis_voxel_3d.py --list_samples --data_dir /path/to/gathered_data
  
  # 可视化并保存
  python vis_voxel_3d.py --pkl_path /path/to/scene0000_00/00000.pkl --output_path output.png
  
  # 显示空体素
  python vis_voxel_3d.py --pkl_path /path/to/scene0000_00/00000.pkl --show_empty
        """
    )
    
    parser.add_argument('--pkl_path', type=str, default=None,
                       help='pkl 文件路径')
    parser.add_argument('--data_dir', type=str, 
                       default='/data/tangqiansong/raw_data/scannet_occ_mini/gathered_data',
                       help='数据目录（用于列出样本）')
    parser.add_argument('--output_path', type=str, default=None,
                       help='输出图像路径（如果为 None 则显示）')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=None,
                       metavar=('X', 'Y', 'Z'),
                       help='体素大小 (x, y, z)，单位：米。如果为 None 则使用默认值 [1.6, 1.6, 0.6]')
    parser.add_argument('--show_empty', action='store_true',
                       help='是否显示空体素（标签为0）')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='体素透明度 (0-1)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 10],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='图像大小（宽度，高度）')
    parser.add_argument('--list_samples', action='store_true',
                       help='列出可用的样本')
    
    args = parser.parse_args()
    
    # 列出样本
    if args.list_samples:
        samples = list_available_samples(args.data_dir)
        print(f"\n找到 {len(samples)} 个样本:\n")
        print(f"{'序号':<6} {'场景':<20} {'帧':<10} {'路径'}")
        print("-" * 100)
        for idx, sample in enumerate(samples):
            print(f"{idx:<6} {sample['scene']:<20} {sample['frame']:<10} {sample['path']}")
        return
    
    # 检查 pkl_path
    if args.pkl_path is None:
        print("错误: 请指定 --pkl_path 或使用 --list_samples 查看可用样本")
        parser.print_help()
        return
    
    if not os.path.exists(args.pkl_path):
        print(f"错误: 文件不存在: {args.pkl_path}")
        return
    
    # 可视化
    visualize_voxel_3d(
        pkl_path=args.pkl_path,
        output_path=args.output_path,
        voxel_size=args.voxel_size,
        show_empty=args.show_empty,
        alpha=args.alpha,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()

