"""
分析 ScanNet OCC 数据集的深度图像，统计 x, y, z 范围，
为 grid_config 设置提供建议。

用法:
    python analyze_depth_for_grid_config.py \
        --data_root /data/tangqiansong/raw_data/scannet_occ \
        --sample_ratio 0.01  # 采样1%的数据进行分析
"""
import argparse
import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


def load_pkl_file(pkl_path):
    """加载 pkl 文件"""
    print(f"加载 pkl 文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def depth_image_to_points(depth_array, intrinsic, cam_pose=None, max_depth=50.0):
    """
    将深度图像转换为3D点云
    
    Args:
        depth_array: (H, W) 深度数组，单位为米（或毫米，需要根据实际情况转换）
        intrinsic: (3, 3) 相机内参矩阵
        cam_pose: (4, 4) 相机到世界坐标的变换矩阵（可选）
        max_depth: 最大深度阈值（米）
    
    Returns:
        points: (N, 3) 点云坐标，单位米
    """
    height, width = depth_array.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 假设深度图像单位可能是毫米，转换为米
    # 如果已经是米，则不需要转换
    depth_m = depth_array.astype(np.float32)
    
    # 检查深度值的范围来判断单位
    # 如果最大值超过100，可能是毫米单位，转换为米
    if depth_m.max() > 100:  # 如果最大值超过100，可能是毫米单位
        depth_m = depth_m / 1000.0
    
    # 过滤无效深度
    valid_mask = (depth_m > 0) & (depth_m < max_depth)
    
    if np.sum(valid_mask) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_m[valid_mask]
    
    # 像素坐标转相机坐标系
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid
    
    # 相机坐标系转世界坐标系（如果提供了 cam_pose）
    if cam_pose is not None:
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
        points_cam_homogeneous = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1))], axis=1)  # (N, 4)
        points_world_homogeneous = (cam_pose @ points_cam_homogeneous.T).T  # (N, 4)
        points = points_world_homogeneous[:, :3]  # (N, 3)
    else:
        points = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    return points.astype(np.float32)


def analyze_depth_images(data_root, ann_file, sample_ratio=0.01, max_samples=None):
    """
    分析深度图像，统计 x, y, z 范围
    
    Args:
        data_root: 数据根目录
        ann_file: 标注文件路径（相对于 data_root 或绝对路径）
        sample_ratio: 采样比例
        max_samples: 最大样本数（如果指定，优先使用这个）
    
    Returns:
        stats: 统计结果字典
    """
    # 加载标注文件
    if os.path.isabs(ann_file):
        pkl_path = ann_file
    else:
        pkl_path = os.path.join(data_root, ann_file)
    
    data_dict = load_pkl_file(pkl_path)
    data_list = data_dict['data_list']
    
    print(f"总样本数: {len(data_list)}")
    
    # 确定采样数量
    if max_samples is not None:
        sample_count = min(max_samples, len(data_list))
    else:
        sample_count = max(1, int(len(data_list) * sample_ratio))
    
    # 随机采样
    if sample_count < len(data_list):
        indices = np.random.choice(len(data_list), sample_count, replace=False)
        sampled_data = [data_list[i] for i in indices]
        print(f"采样 {sample_count} 个样本进行分析")
    else:
        sampled_data = data_list
        print(f"使用全部 {len(data_list)} 个样本")
    
    # 收集所有点云坐标
    all_points = []
    depth_stats = {'min': [], 'max': [], 'mean': [], 'median': []}
    processed_count = 0
    failed_count = 0
    
    for item in tqdm(sampled_data, desc="处理深度图像"):
        try:
            # 获取文件路径
            depth_path = item['depth_path']
            pkl_path_item = item['pkl_path']
            
            # 加载深度图像
            if os.path.isabs(depth_path):
                depth_file = depth_path
            else:
                depth_file = os.path.join(data_root, depth_path)
            
            if not os.path.exists(depth_file):
                failed_count += 1
                continue
            
            # 加载深度图像（ScanNet 深度图通常是16位PNG，单位毫米）
            depth_img = Image.open(depth_file)
            depth_array = np.array(depth_img).astype(np.float32)
            
            # 统计深度值（排除无效值）
            valid_depths = depth_array[(depth_array > 0) & (depth_array < 65535)]
            if len(valid_depths) > 0:
                depth_stats['min'].append(valid_depths.min())
                depth_stats['max'].append(valid_depths.max())
                depth_stats['mean'].append(valid_depths.mean())
                depth_stats['median'].append(np.median(valid_depths))
            else:
                failed_count += 1
                continue
            
            # 加载 pkl 获取相机参数
            if os.path.isabs(pkl_path_item):
                pkl_file = pkl_path_item
            else:
                pkl_file = os.path.join(data_root, pkl_path_item)
            
            if not os.path.exists(pkl_file):
                failed_count += 1
                continue
            
            with open(pkl_file, 'rb') as f:
                pkl_data = pickle.load(f)
            
            intrinsic = pkl_data['intrinsic'][:3, :3]
            cam_pose = pkl_data.get('cam_pose', None)
            
            # 转换深度图像为点云
            points = depth_image_to_points(
                depth_array, intrinsic, cam_pose, max_depth=50.0)
            
            if points.shape[0] > 0:
                all_points.append(points)
                processed_count += 1
            
        except Exception as e:
            print(f"\n处理失败: {e}")
            failed_count += 1
            continue
    
    print(f"\n成功处理: {processed_count} 个样本")
    print(f"失败: {failed_count} 个样本")
    
    if len(all_points) == 0:
        print("错误: 没有成功处理任何样本！")
        return None
    
    # 合并所有点云
    all_points = np.concatenate(all_points, axis=0)
    print(f"\n总点数: {all_points.shape[0]:,}")
    
    # 统计 x, y, z 范围
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    x_mean, y_mean, z_mean = all_points[:, 0].mean(), all_points[:, 1].mean(), all_points[:, 2].mean()
    x_std, y_std, z_std = all_points[:, 0].std(), all_points[:, 1].std(), all_points[:, 2].std()
    
    # 深度统计
    depth_min = np.min(depth_stats['min']) if depth_stats['min'] else 0
    depth_max = np.max(depth_stats['max']) if depth_stats['max'] else 0
    depth_mean = np.mean(depth_stats['mean']) if depth_stats['mean'] else 0
    depth_median = np.median(depth_stats['median']) if depth_stats['median'] else 0
    
    stats = {
        'points_count': all_points.shape[0],
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'z_range': (z_min, z_max),
        'x_stats': {'min': x_min, 'max': x_max, 'mean': x_mean, 'std': x_std},
        'y_stats': {'min': y_min, 'max': y_max, 'mean': y_mean, 'std': y_std},
        'z_stats': {'min': z_min, 'max': z_max, 'mean': z_mean, 'std': z_std},
        'depth_stats': {
            'min': depth_min,
            'max': depth_max,
            'mean': depth_mean,
            'median': depth_median
        }
    }
    
    return stats


def recommend_grid_config(stats, target_resolution=(200, 200, 16), voxel_size_xy=None, 
                          match_target_size=False):
    """
    根据统计结果推荐 grid_config
    
    Args:
        stats: 统计结果
        target_resolution: 目标分辨率 (Dx, Dy, Dz)
        voxel_size_xy: 期望的 xy 平面体素大小（米），如果 None 则自动计算
        match_target_size: 如果为 True，精确匹配目标网格大小（60, 60, 36）
    
    Returns:
        recommended_config: 推荐的 grid_config 字典
    """
    x_min, x_max = stats['x_range']
    y_min, y_max = stats['y_range']
    z_min, z_max = stats['z_range']
    
    Dx, Dy, Dz = target_resolution
    
    if match_target_size:
        # 精确匹配目标网格大小模式
        # 扩展范围以包含边界（添加10%的margin）
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_min_rec = x_min - x_range * margin
        x_max_rec = x_max + x_range * margin
        y_min_rec = y_min - y_range * margin
        y_max_rec = y_max + y_range * margin
        z_min_rec = z_min - z_range * margin
        z_max_rec = z_max + z_range * margin
        
        # 根据目标网格大小和范围反推间隔
        # 首先确定一个合理的间隔（优先使用用户指定的 voxel_size_xy）
        if voxel_size_xy is None:
            # 根据范围和目标大小计算一个合理的间隔
            voxel_size_x_candidate = (x_max_rec - x_min_rec) / Dx
            voxel_size_y_candidate = (y_max_rec - y_min_rec) / Dy
            # 选择较小的，确保范围能覆盖数据
            voxel_size_xy = min(voxel_size_x_candidate, voxel_size_y_candidate)
            # 四舍五入到合理的小数位
            voxel_size_xy = round(voxel_size_xy * 10) / 10  # 保留一位小数
        
        # 根据目标网格大小和间隔反推范围
        # (max - min) / interval = target_size
        # 所以: max - min = target_size * interval
        x_total_range = Dx * voxel_size_xy
        y_total_range = Dy * voxel_size_xy
        
        # 计算 z 的间隔（基于扩展后的范围）
        z_interval = (z_max_rec - z_min_rec) / Dz
        z_total_range = Dz * z_interval
        
        # 以中心点为中心，对称扩展
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        x_min_rec = x_center - x_total_range / 2
        x_max_rec = x_center + x_total_range / 2
        y_min_rec = y_center - y_total_range / 2
        y_max_rec = y_center + y_total_range / 2
        z_min_rec = z_center - z_total_range / 2
        z_max_rec = z_center + z_total_range / 2
        
        # 对齐到间隔的倍数（确保精确匹配）
        x_min_rec = np.floor(x_min_rec / voxel_size_xy) * voxel_size_xy
        x_max_rec = x_min_rec + Dx * voxel_size_xy
        y_min_rec = np.floor(y_min_rec / voxel_size_xy) * voxel_size_xy
        y_max_rec = y_min_rec + Dy * voxel_size_xy
        z_min_rec = np.floor(z_min_rec / z_interval) * z_interval
        z_max_rec = z_min_rec + Dz * z_interval
        
        voxel_size_z = z_interval
    else:
        # 原始方法：根据目标分辨率和范围计算体素大小
        if voxel_size_xy is None:
            voxel_size_x = (x_max - x_min) / Dx
            voxel_size_y = (y_max - y_min) / Dy
            voxel_size_xy = max(voxel_size_x, voxel_size_y)
        else:
            voxel_size_xy = voxel_size_xy
        
        voxel_size_z = (z_max - z_min) / Dz
        
        # 扩展范围以包含边界（添加10%的margin）
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_min_rec = x_min - x_range * margin
        x_max_rec = x_max + x_range * margin
        y_min_rec = y_min - y_range * margin
        y_max_rec = y_max + y_range * margin
        z_min_rec = z_min - z_range * margin
        z_max_rec = z_max + z_range * margin
        
        # 调整范围使其能被体素大小整除（向上取整）
        x_min_rec = np.floor(x_min_rec / voxel_size_xy) * voxel_size_xy
        x_max_rec = np.ceil(x_max_rec / voxel_size_xy) * voxel_size_xy
        y_min_rec = np.floor(y_min_rec / voxel_size_xy) * voxel_size_xy
        y_max_rec = np.ceil(y_max_rec / voxel_size_xy) * voxel_size_xy
        z_min_rec = np.floor(z_min_rec / voxel_size_z) * voxel_size_z
        z_max_rec = np.ceil(z_max_rec / voxel_size_z) * voxel_size_z
        voxel_size_z = (z_max_rec - z_min_rec) / Dz
    
    # 深度范围建议（从深度统计）
    # 判断深度单位：如果最大值大于1000，可能是毫米单位
    depth_min_raw = stats['depth_stats']['min']
    depth_max_raw = stats['depth_stats']['max']
    
    if depth_max_raw > 1000:
        # 毫米单位，转换为米
        depth_min = depth_min_raw / 1000.0
        depth_max = depth_max_raw / 1000.0
    else:
        # 已经是米单位
        depth_min = depth_min_raw
        depth_max = depth_max_raw
    
    # 限制最大深度为50米
    depth_max = min(depth_max, 50.0)
    depth_interval = 0.5  # 建议的深度间隔
    
    # 确保深度范围合理
    if depth_min < 0.5:
        depth_min = 0.5
    if depth_max < depth_min + depth_interval:
        depth_max = depth_min + depth_interval * 10
    
    recommended_config = {
        'x': [float(x_min_rec), float(x_max_rec), float(voxel_size_xy)],
        'y': [float(y_min_rec), float(y_max_rec), float(voxel_size_xy)],
        'z': [float(z_min_rec), float(z_max_rec), float(voxel_size_z)],
        'depth': [float(depth_min), float(depth_max), float(depth_interval)]
    }
    
    return recommended_config


def print_analysis_results(stats, recommended_config):
    """打印分析结果"""
    print("\n" + "="*80)
    print("深度图像和点云统计分析结果")
    print("="*80)
    
    print("\n【点云统计】")
    print(f"  总点数: {stats['points_count']:,}")
    print(f"\n  X 轴范围: [{stats['x_range'][0]:.2f}, {stats['x_range'][1]:.2f}] 米")
    print(f"    平均值: {stats['x_stats']['mean']:.2f} 米, 标准差: {stats['x_stats']['std']:.2f} 米")
    print(f"\n  Y 轴范围: [{stats['y_range'][0]:.2f}, {stats['y_range'][1]:.2f}] 米")
    print(f"    平均值: {stats['y_stats']['mean']:.2f} 米, 标准差: {stats['y_stats']['std']:.2f} 米")
    print(f"\n  Z 轴范围: [{stats['z_range'][0]:.2f}, {stats['z_range'][1]:.2f}] 米")
    print(f"    平均值: {stats['z_stats']['mean']:.2f} 米, 标准差: {stats['z_stats']['std']:.2f} 米")
    
    print("\n【深度图像统计】")
    depth_stats = stats['depth_stats']
    print(f"  深度范围: [{depth_stats['min']:.1f}, {depth_stats['max']:.1f}] 单位")
    print(f"  平均深度: {depth_stats['mean']:.1f} 单位")
    print(f"  中位深度: {depth_stats['median']:.1f} 单位")
    if depth_stats['max'] > 100:
        print(f"  注意: 深度值较大，可能是毫米单位（转换为米后约为 {depth_stats['max']/1000:.1f}m）")
    
    print("\n【推荐的 grid_config】")
    print("="*80)
    print("grid_config = {")
    print(f"    'x': {recommended_config['x']},")
    print(f"    'y': {recommended_config['y']},")
    print(f"    'z': {recommended_config['z']},")
    print(f"    'depth': {recommended_config['depth']},")
    print("}")
    
    # 计算实际网格大小
    x_size = int((recommended_config['x'][1] - recommended_config['x'][0]) / recommended_config['x'][2])
    y_size = int((recommended_config['y'][1] - recommended_config['y'][0]) / recommended_config['y'][2])
    z_size = int((recommended_config['z'][1] - recommended_config['z'][0]) / recommended_config['z'][2])
    depth_size = int((recommended_config['depth'][1] - recommended_config['depth'][0]) / recommended_config['depth'][2])
    
    print(f"\n实际网格大小: ({x_size}, {y_size}, {z_size})")
    print(f"深度维度数: {depth_size}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="分析 ScanNet OCC 数据集的深度图像，统计 x, y, z 范围")
    parser.add_argument(
        '--data_root',
        type=str,
        default='/data/tangqiansong/raw_data/scannet_occ',
        help='数据集根目录'
    )
    parser.add_argument(
        '--train_ann',
        type=str,
        default='scannet_occ_infos_train.pkl',
        help='训练集标注文件（相对于 data_root 或绝对路径）'
    )
    parser.add_argument(
        '--test_ann',
        type=str,
        default='scannet_occ_infos_test.pkl',
        help='测试集标注文件（相对于 data_root 或绝对路径）'
    )
    parser.add_argument(
        '--sample_ratio',
        type=float,
        default=0.01,
        help='采样比例（0.0-1.0），用于加速分析'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大样本数（如果指定，优先使用这个）'
    )
    parser.add_argument(
        '--voxel_size_xy',
        type=float,
        default=0.4,
        help='期望的 xy 平面体素大小（米）'
    )
    parser.add_argument(
        '--target_resolution',
        type=int,
        nargs=3,
        default=[60, 60, 36],
        help='目标分辨率 (Dx, Dy, Dz)，默认 [60, 60, 36] 匹配 GT 大小'
    )
    parser.add_argument(
        '--use_train_only',
        action='store_true',
        help='仅使用训练集进行分析'
    )
    parser.add_argument(
        '--match_target_size',
        action='store_true',
        default=True,
        help='精确匹配目标网格大小（默认启用）'
    )
    
    args = parser.parse_args()
    
    # 分析训练集
    print("\n" + "="*80)
    print("开始分析训练集...")
    print("="*80)
    train_stats = analyze_depth_images(
        args.data_root, args.train_ann, 
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples
    )
    
    if train_stats is None:
        print("训练集分析失败！")
        return
    
    # 分析测试集（如果指定）
    test_stats = None
    if not args.use_train_only:
        print("\n" + "="*80)
        print("开始分析测试集...")
        print("="*80)
        test_stats = analyze_depth_images(
            args.data_root, args.test_ann,
            sample_ratio=args.sample_ratio,
            max_samples=args.max_samples
        )
    
    # 合并统计结果
    if test_stats is not None:
        print("\n合并训练集和测试集的统计结果...")
        # 合并点云统计（简单合并，实际应该合并点云）
        combined_stats = {
            'points_count': train_stats['points_count'] + test_stats['points_count'],
            'x_range': (
                min(train_stats['x_range'][0], test_stats['x_range'][0]),
                max(train_stats['x_range'][1], test_stats['x_range'][1])
            ),
            'y_range': (
                min(train_stats['y_range'][0], test_stats['y_range'][0]),
                max(train_stats['y_range'][1], test_stats['y_range'][1])
            ),
            'z_range': (
                min(train_stats['z_range'][0], test_stats['z_range'][0]),
                max(train_stats['z_range'][1], test_stats['z_range'][1])
            ),
            'x_stats': train_stats['x_stats'],  # 使用训练集的统计
            'y_stats': train_stats['y_stats'],
            'z_stats': train_stats['z_stats'],
            'depth_stats': {
                'min': min(train_stats['depth_stats']['min'], test_stats['depth_stats']['min']),
                'max': max(train_stats['depth_stats']['max'], test_stats['depth_stats']['max']),
                'mean': (train_stats['depth_stats']['mean'] + test_stats['depth_stats']['mean']) / 2,
                'median': (train_stats['depth_stats']['median'] + test_stats['depth_stats']['median']) / 2,
            }
        }
        final_stats = combined_stats
    else:
        final_stats = train_stats
    
    # 推荐配置
    recommended_config = recommend_grid_config(
        final_stats,
        target_resolution=tuple(args.target_resolution),
        voxel_size_xy=args.voxel_size_xy,
        match_target_size=args.match_target_size
    )
    
    # 打印结果
    print_analysis_results(final_stats, recommended_config)
    
    # 保存结果到文件
    output_file = os.path.join(args.data_root, 'grid_config_analysis.json')
    import json
    results = {
        'stats': {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v) 
                 for k, v in final_stats.items() if k != 'points_count'},
        'recommended_config': recommended_config,
        'target_resolution': args.target_resolution
    }
    # 转换 numpy 类型为 Python 原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()

