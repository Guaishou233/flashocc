import argparse
import os
import pickle
import sys
from collections import Counter
from glob import glob
import numpy as np


def describe_value(name, value, indent: int = 0, key_limit: int = 50):
    prefix = " " * indent
    vtype = type(value).__name__
    extra = []
    # numpy/torch 等对象可能有 shape/dtype
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            extra.append(f"shape={tuple(shape)}")
        except Exception:
            pass
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        extra.append(f"dtype={dtype}")

    if isinstance(value, dict):
        keys = list(value.keys())
        kshow = keys[:key_limit]
        more = "" if len(keys) <= key_limit else f" (+{len(keys)-key_limit} more)"
        print(f"{prefix}- {name}: dict[{len(keys)}] {more}")
        if kshow:
            print(f"{prefix}  keys: {kshow}")
    elif isinstance(value, (list, tuple)):
        length = len(value)
        elem_type = type(value[0]).__name__ if length > 0 else "?"
        print(f"{prefix}- {name}: {vtype}[{length}] elem={elem_type}")
        if length > 0 and isinstance(value[0], dict):
            keys = list(value[0].keys())
            kshow = keys[:key_limit]
            more = "" if len(keys) <= key_limit else f" (+{len(keys)-key_limit} more)"
            print(f"{prefix}  first_elem_keys: {kshow}{more}")
    else:
        extras = f" ({', '.join(extra)})" if extra else ""
        try:
            length_info = f", len={len(value)}"  # type: ignore[arg-type]
        except Exception:
            length_info = ""
        print(f"{prefix}- {name}: {vtype}{length_info}{extras}")


def collect_numbers_from_target_1_4(target_1_4):
    """
    从 target_1_4 中收集所有数字（扁平化处理）
    
    Args:
        target_1_4: 可以是 list, numpy array, 或其他可迭代的嵌套结构
    
    Returns:
        list: 所有数字的扁平列表
    """
    numbers = []
    
    def flatten(item):
        """递归扁平化嵌套结构"""
        if isinstance(item, (int, float, np.integer, np.floating)):
            numbers.append(float(item))
        elif isinstance(item, np.ndarray):
            numbers.extend(item.flatten().tolist())
        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                flatten(sub_item)
        else:
            # 尝试转换为数字
            try:
                numbers.append(float(item))
            except (ValueError, TypeError):
                pass
    
    flatten(target_1_4)
    return numbers


def count_target_1_4_numbers(data_dir):
    """
    统计指定目录下所有 pkl 文件中 data_list 里的 target_1_4 出现的数字
    
    Args:
        data_dir: pkl 文件所在的目录路径
    
    Returns:
        dict: 包含统计结果的字典
            - value_counts: Counter 对象，记录每个数字的出现次数
            - total_count: 总数字个数
            - unique_count: 唯一数字的个数
            - files_processed: 处理的文件列表
            - items_processed: 处理的 data_list 项目总数
    """
    # 查找目录下所有 .pkl 文件
    pkl_files = glob(os.path.join(data_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"警告: 在目录 {data_dir} 中未找到 .pkl 文件")
        return None
    
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    
    all_numbers = []
    files_processed = []
    items_processed = 0
    
    for pkl_file in pkl_files:
        print(f"处理文件: {os.path.basename(pkl_file)}")
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict) or "data_list" not in data:
                print(f"  跳过: 文件结构不符合预期（缺少 data_list）")
                continue
            
            data_list = data["data_list"]
            file_numbers_count = 0
            
            for item in data_list:
                if not isinstance(item, dict) or "target_1_4" not in item:
                    continue
                
                target_1_4 = item["target_1_4"]
                numbers = collect_numbers_from_target_1_4(target_1_4)
                all_numbers.extend(numbers)
                file_numbers_count += len(numbers)
                items_processed += 1
            
            if file_numbers_count > 0:
                files_processed.append(os.path.basename(pkl_file))
                print(f"  提取了 {file_numbers_count} 个数字，处理了 {len([x for x in data_list if 'target_1_4' in x])} 个数据项")
        
        except Exception as e:
            print(f"  错误: 处理文件 {pkl_file} 时出错: {e}")
            continue
    
    if not all_numbers:
        print("警告: 未找到任何 target_1_4 数据")
        return None
    
    # 统计数字出现次数
    value_counts = Counter(all_numbers)
    
    result = {
        "value_counts": value_counts,
        "total_count": len(all_numbers),
        "unique_count": len(value_counts),
        "files_processed": files_processed,
        "items_processed": items_processed,
    }
    
    return result


def print_statistics(result):
    """打印统计结果"""
    if result is None:
        return
    
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"处理的文件数: {len(result['files_processed'])}")
    print(f"处理的数据项数: {result['items_processed']}")
    print(f"总数字个数: {result['total_count']}")
    print(f"唯一数字个数: {result['unique_count']}")
    print("\n处理的文件:")
    for fname in result['files_processed']:
        print(f"  - {fname}")
    
    print("\n数字出现次数统计 (按出现次数降序):")
    # 按出现次数排序
    sorted_counts = sorted(result['value_counts'].items(), key=lambda x: x[1], reverse=True)
    
    # 打印前50个最常见的数字
    print_limit = min(50, len(sorted_counts))
    for i, (value, count) in enumerate(sorted_counts[:print_limit], 1):
        percentage = (count / result['total_count']) * 100
        print(f"  {i:3d}. 值: {value:10.2f}  出现次数: {count:10d}  ({percentage:6.2f}%)")
    
    if len(sorted_counts) > print_limit:
        print(f"\n  ... 还有 {len(sorted_counts) - print_limit} 个唯一值未显示")


def main():
    parser = argparse.ArgumentParser(
        description="查看 .pkl 顶层字段与结构概览，或统计 target_1_4 中的数字"
    )
    parser.add_argument(
        "--pkl",
        default=None,
        required=False,
        help="要查看的 .pkl 文件绝对路径或相对路径",
    )
    parser.add_argument(
        "--count-target",
        action="store_true",
        help="统计 target_1_4 中的数字（需要指定 --data-dir）",
    )
    parser.add_argument(
        "--data-dir",
        default="/data/tangqiansong/raw_data/scannet_occ_mini",
        required=False,
        help="包含 pkl 文件的目录路径（用于 --count-target 模式）",
    )
    args = parser.parse_args()

    # 如果指定了统计模式
    if args.count_target:
        data_dir = os.path.expanduser(args.data_dir)
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        
        if not os.path.isdir(data_dir):
            print(f"错误: 目录不存在: {data_dir}")
            sys.exit(1)
        
        result = count_target_1_4_numbers(data_dir)
        print_statistics(result)
        return

    # 默认模式：查看 pkl 文件结构
    if args.pkl is None:
        args.pkl = "/data/tangqiansong/raw_data/scannet_occ_mini/scannet_occ_infos_test.pkl"

    pkl_path = os.path.expanduser(args.pkl)
    if not os.path.isabs(pkl_path):
        pkl_path = os.path.abspath(pkl_path)

    if not os.path.exists(pkl_path):
        print(f"文件不存在: {pkl_path}")
        sys.exit(1)

    print(f"加载: {pkl_path}")
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"读取/反序列化失败: {e}")
        sys.exit(1)

    print("顶层类型:", type(data).__name__)

    if isinstance(data, dict):
        print(f"顶层字段数: {len(data)}")
        for k, v in data.items():
            describe_value(str(k), v, indent=2)
    elif isinstance(data, (list, tuple)):
        print(f"顶层为序列: {type(data).__name__}, 长度: {len(data)}")
        if len(data) > 0:
            first = data[0]
            print("首元素类型:", type(first).__name__)
            if isinstance(first, dict):
                print("首元素字典字段预览:")
                for k, v in first.items():
                    describe_value(str(k), v, indent=2)
    else:
        describe_value("value", data, indent=0)
    

if __name__ == "__main__":
    main()