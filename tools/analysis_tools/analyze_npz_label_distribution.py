import argparse
import os
import sys
from collections import Counter
from glob import glob
import numpy as np
import pickle
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有 tqdm，创建一个简单的替代品
    def tqdm(iterable, *args, **kwargs):
        return iterable


def analyze_label_distribution(data_dir, verbose=False):
    """
    分析指定目录下所有 npz 文件中的标签 ID 分布
    
    Args:
        data_dir: npz 文件所在的目录路径
        verbose: 是否显示详细输出
    
    Returns:
        dict: 包含统计结果的字典
            - pred_counts: Counter 对象，记录 pred 中每个标签 ID 的出现次数
            - gt_counts: Counter 对象，记录 gt 中每个标签 ID 的出现次数
            - files_processed: 处理的文件列表
            - total_voxels: 总voxel数
    """
    # 查找目录下所有 .npz 文件
    npz_files = glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        print(f"警告: 在目录 {data_dir} 中未找到 .npz 文件")
        return None
    
    print(f"找到 {len(npz_files)} 个 npz 文件")
    
    pred_counts = Counter()
    gt_counts = Counter()
    files_processed = []
    total_voxels = 0
    files_skipped = 0
    
    # 使用 tqdm 显示进度条
    progress_bar = tqdm(npz_files, desc="处理 npz 文件", disable=verbose or not HAS_TQDM)
    
    for npz_file in progress_bar:
        if verbose:
            print(f"处理文件: {os.path.basename(npz_file)}")
        else:
            # 更新进度条描述
            progress_bar.set_postfix(file=os.path.basename(npz_file))
        
        try:
            data = np.load(npz_file)
            
            # 检查是否有 pred 和 gt 键
            if 'pred' not in data and 'gt' not in data:
                if verbose:
                    print(f"  跳过: 文件不包含 'pred' 或 'gt' 键")
                    print(f"  可用键: {list(data.keys())}")
                files_skipped += 1
                continue
            
            file_voxels = 0
            
            # 统计 pred 中的标签 ID
            if 'pred' in data:
                pred = data['pred']
                pred_flat = pred.flatten()
                pred_file_counts = Counter(pred_flat)
                pred_counts.update(pred_file_counts)
                file_voxels = len(pred_flat)
                if verbose:
                    print(f"  pred: shape={pred.shape}, 唯一标签数={len(pred_file_counts)}")
            
            # 统计 gt 中的标签 ID
            if 'gt' in data:
                gt = data['gt']
                gt_flat = gt.flatten()
                gt_file_counts = Counter(gt_flat)
                gt_counts.update(gt_file_counts)
                if file_voxels == 0:
                    file_voxels = len(gt_flat)
                if verbose:
                    print(f"  gt: shape={gt.shape}, 唯一标签数={len(gt_file_counts)}")
            
            total_voxels += file_voxels
            files_processed.append(os.path.basename(npz_file))
        
        except Exception as e:
            if verbose:
                print(f"  错误: 处理文件 {npz_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
            files_skipped += 1
            continue
    
    if not files_processed:
        print("警告: 未成功处理任何文件")
        return None
    
    if not verbose:
        print(f"\n处理完成: 成功处理 {len(files_processed)} 个文件，跳过 {files_skipped} 个文件")
    
    result = {
        "pred_counts": pred_counts,
        "gt_counts": gt_counts,
        "files_processed": files_processed,
        "total_voxels": total_voxels,
    }
    
    return result


def analyze_pkl_label_distribution(data_dir, field_name='gt_occ_1_4', verbose=False):
    """
    分析指定目录下所有 pkl 文件中的标签 ID 分布（递归搜索所有子目录）
    
    Args:
        data_dir: pkl 文件所在的目录路径
        field_name: 要统计的字段名，默认为 'gt_occ_1_4'
        verbose: 是否显示详细输出
    
    Returns:
        dict: 包含统计结果的字典
            - label_counts: Counter 对象，记录每个标签 ID 的出现次数
            - files_processed: 处理的文件列表
            - total_voxels: 总voxel数
    """
    # 递归查找目录下所有 .pkl 文件
    pkl_files = glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)
    
    if not pkl_files:
        print(f"警告: 在目录 {data_dir} 中未找到 .pkl 文件")
        return None
    
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    
    label_counts = Counter()
    files_processed = []
    total_voxels = 0
    files_skipped = 0
    field_fallback_count = 0
    
    # 使用 tqdm 显示进度条
    progress_bar = tqdm(pkl_files, desc="处理 pkl 文件", disable=verbose or not HAS_TQDM)
    
    for pkl_file in progress_bar:
        rel_path = os.path.relpath(pkl_file, data_dir)
        if verbose:
            print(f"处理文件: {rel_path}")
        else:
            # 更新进度条描述
            progress_bar.set_postfix(file=os.path.basename(rel_path))
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 检查是否是字典类型
            if not isinstance(data, dict):
                if verbose:
                    print(f"  跳过: 文件不是字典类型，而是 {type(data)}")
                files_skipped += 1
                continue
            
            # 尝试查找目标字段（先尝试用户指定的字段名，再尝试 target_1_4）
            target_field = None
            if field_name in data:
                target_field = data[field_name]
            elif 'target_1_4' in data:
                target_field = data['target_1_4']
                field_fallback_count += 1
                if verbose:
                    print(f"  注意: 未找到 '{field_name}'，使用 'target_1_4' 代替")
            else:
                if verbose:
                    print(f"  跳过: 文件不包含 '{field_name}' 或 'target_1_4' 键")
                    print(f"  可用键: {list(data.keys())}")
                files_skipped += 1
                continue
            
            # 转换为 numpy 数组（如果不是的话）
            if not isinstance(target_field, np.ndarray):
                target_field = np.array(target_field)
            
            # 统计标签 ID
            target_flat = target_field.flatten()
            # 转换为整数类型（如果是浮点数的话）
            if target_field.dtype == np.float64 or target_field.dtype == np.float32:
                target_flat = target_flat.astype(np.int64)
            
            file_counts = Counter(target_flat)
            label_counts.update(file_counts)
            file_voxels = len(target_flat)
            total_voxels += file_voxels
            
            if verbose:
                print(f"  shape={target_field.shape}, 唯一标签数={len(file_counts)}, voxel数={file_voxels}")
            files_processed.append(rel_path)
        
        except Exception as e:
            if verbose:
                print(f"  错误: 处理文件 {pkl_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
            files_skipped += 1
            continue
    
    if not files_processed:
        print("警告: 未成功处理任何文件")
        return None
    
    if not verbose:
        print(f"\n处理完成: 成功处理 {len(files_processed)} 个文件，跳过 {files_skipped} 个文件")
        if field_fallback_count > 0:
            print(f"注意: 有 {field_fallback_count} 个文件使用了 'target_1_4' 代替 '{field_name}'")
    
    result = {
        "label_counts": label_counts,
        "files_processed": files_processed,
        "total_voxels": total_voxels,
    }
    
    return result


def print_statistics(result):
    """打印统计结果（用于 npz 文件）"""
    if result is None:
        return
    
    print("\n" + "=" * 60)
    print("标签 ID 分布统计结果")
    print("=" * 60)
    print(f"处理的文件数: {len(result['files_processed'])}")
    print(f"总voxel数: {result['total_voxels']}")
    
    print("\n处理的文件:")
    for fname in result['files_processed']:
        print(f"  - {fname}")
    
    # 打印 pred 统计
    if result['pred_counts']:
        print("\n" + "-" * 60)
        print("PRED 标签 ID 分布 (按出现次数降序):")
        print("-" * 60)
        sorted_pred = sorted(result['pred_counts'].items(), key=lambda x: x[1], reverse=True)
        total_pred = sum(result['pred_counts'].values())
        
        print(f"总标签数: {total_pred}")
        print(f"唯一标签数: {len(sorted_pred)}")
        print("\n标签 ID 统计:")
        for i, (label_id, count) in enumerate(sorted_pred, 1):
            percentage = (count / total_pred) * 100
            print(f"  {i:3d}. 标签 ID: {label_id:4d}  出现次数: {count:10d}  ({percentage:6.2f}%)")
    
    # 打印 gt 统计
    if result['gt_counts']:
        print("\n" + "-" * 60)
        print("GT 标签 ID 分布 (按出现次数降序):")
        print("-" * 60)
        sorted_gt = sorted(result['gt_counts'].items(), key=lambda x: x[1], reverse=True)
        total_gt = sum(result['gt_counts'].values())
        
        print(f"总标签数: {total_gt}")
        print(f"唯一标签数: {len(sorted_gt)}")
        print("\n标签 ID 统计:")
        for i, (label_id, count) in enumerate(sorted_gt, 1):
            percentage = (count / total_gt) * 100
            print(f"  {i:3d}. 标签 ID: {label_id:4d}  出现次数: {count:10d}  ({percentage:6.2f}%)")
    
    # 比较 pred 和 gt 的差异
    if result['pred_counts'] and result['gt_counts']:
        print("\n" + "-" * 60)
        print("PRED vs GT 标签 ID 差异分析:")
        print("-" * 60)
        
        pred_labels = set(result['pred_counts'].keys())
        gt_labels = set(result['gt_counts'].keys())
        
        only_pred = pred_labels - gt_labels
        only_gt = gt_labels - pred_labels
        common = pred_labels & gt_labels
        
        print(f"仅在 PRED 中出现的标签 ID: {sorted(only_pred) if only_pred else '无'}")
        print(f"仅在 GT 中出现的标签 ID: {sorted(only_gt) if only_gt else '无'}")
        print(f"共同标签 ID 数量: {len(common)}")
        
        if common:
            print("\n共同标签 ID 的数量对比:")
            print(f"{'标签 ID':<10} {'PRED 数量':<15} {'GT 数量':<15} {'差异':<15}")
            print("-" * 60)
            for label_id in sorted(common):
                pred_count = result['pred_counts'][label_id]
                gt_count = result['gt_counts'][label_id]
                diff = pred_count - gt_count
                print(f"{label_id:<10} {pred_count:<15} {gt_count:<15} {diff:<15}")


def print_pkl_statistics(result, field_name='target_1_4'):
    """打印 pkl 文件的统计结果"""
    if result is None:
        return
    
    print("\n" + "=" * 60)
    print(f"标签 ID 分布统计结果 ({field_name})")
    print("=" * 60)
    print(f"处理的文件数: {len(result['files_processed'])}")
    print(f"总voxel数: {result['total_voxels']}")
    
    # 打印标签统计
    if result['label_counts']:
        print("\n" + "-" * 60)
        print(f"{field_name.upper()} 标签 ID 分布 (按出现次数降序):")
        print("-" * 60)
        sorted_labels = sorted(result['label_counts'].items(), key=lambda x: x[1], reverse=True)
        total_labels = sum(result['label_counts'].values())
        
        print(f"总标签数: {total_labels}")
        print(f"唯一标签数: {len(sorted_labels)}")
        print("\n标签 ID 统计:")
        for i, (label_id, count) in enumerate(sorted_labels, 1):
            percentage = (count / total_labels) * 100
            print(f"  {i:3d}. 标签 ID: {label_id:4d}  出现次数: {count:10d}  ({percentage:6.2f}%)")
    
    # 显示前10个处理的文件（如果文件太多）
    print("\n处理的文件 (前10个):")
    for fname in result['files_processed'][:10]:
        print(f"  - {fname}")
    if len(result['files_processed']) > 10:
        print(f"  ... 还有 {len(result['files_processed']) - 10} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="分析 npz 或 pkl 文件中的标签 ID 分布"
    )
    parser.add_argument(
        "--data-dir",
        default="/data/tangqiansong/raw_data/scannet_occ/gathered_data",
        required=False,
        help="包含 npz 或 pkl 文件的目录路径",
    )
    parser.add_argument(
        "--file-type",
        choices=['npz', 'pkl', 'auto'],
        default='auto',
        help="文件类型: 'npz', 'pkl', 或 'auto' (自动检测)",
    )
    parser.add_argument(
        "--field-name",
        default='target_1_4',
        help="pkl 文件中要统计的字段名（默认为 'gt_occ_1_4'）",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="显示详细输出（每个文件的处理信息）",
    )
    args = parser.parse_args()
    
    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    
    if not os.path.isdir(data_dir):
        print(f"错误: 目录不存在: {data_dir}")
        sys.exit(1)
    
    # 自动检测文件类型
    if args.file_type == 'auto':
        npz_files = glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        pkl_files = glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)
        
        if pkl_files and not npz_files:
            file_type = 'pkl'
        elif npz_files and not pkl_files:
            file_type = 'npz'
        elif pkl_files and npz_files:
            # 如果两种都有，优先使用 pkl
            file_type = 'pkl'
            print(f"检测到两种文件类型，使用 pkl 文件")
        else:
            print(f"错误: 在目录 {data_dir} 中未找到 .npz 或 .pkl 文件")
            sys.exit(1)
    else:
        file_type = args.file_type
    
    # 根据文件类型选择处理函数
    if file_type == 'pkl':
        result = analyze_pkl_label_distribution(data_dir, args.field_name, args.verbose)
        print_pkl_statistics(result, args.field_name)
    else:
        result = analyze_label_distribution(data_dir, args.verbose)
        print_statistics(result)


if __name__ == "__main__":
    main()

