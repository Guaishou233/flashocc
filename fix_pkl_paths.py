#!/usr/bin/env python3
"""
修复pkl文件中的图像路径问题
将错误的路径替换为正确的路径
"""

import pickle
import os
import shutil
import gc

def fix_pkl_paths():
    # 原始路径前缀（需要替换的）
    old_prefix = "/home/hongxiao.yu/projects/mmdetection3d/data/scannet/posed_images"
    # 新的正确路径前缀
    new_prefix = "/data/tangqiansong/raw_data/scannet_occ/posed_images"
    
    # 处理训练集pkl文件
    train_pkl_path = "/data/tangqiansong/raw_data/scannet_occ/scannet_occ_infos_train.pkl"
    test_pkl_path = "/data/tangqiansong/raw_data/scannet_occ/scannet_occ_infos_test.pkl"
    
    for pkl_path in [train_pkl_path, test_pkl_path]:
        if not os.path.exists(pkl_path):
            print(f"文件不存在: {pkl_path}")
            continue
            
        print(f"正在处理: {pkl_path}")
        
        # 备份原文件
        backup_path = pkl_path + ".backup"
        if not os.path.exists(backup_path):
            print(f"创建备份文件: {backup_path}")
            shutil.copy2(pkl_path, backup_path)
            print(f"备份完成")
        
        # 读取pkl文件
        print("正在读取pkl文件...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"原始样本数量: {len(data)}")
        
        # 修复路径
        fixed_count = 0
        for i, sample in enumerate(data):
            if 'img' in sample and sample['img'].startswith(old_prefix):
                # 替换图像路径
                old_img_path = sample['img']
                new_img_path = old_img_path.replace(old_prefix, new_prefix)
                sample['img'] = new_img_path
                fixed_count += 1
                
                if i < 5:  # 打印前5个修复的路径作为示例
                    print(f"  样本 {i}: {old_img_path} -> {new_img_path}")
            
            # 每处理1000个样本打印一次进度
            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i + 1}/{len(data)} 个样本，修复了 {fixed_count} 个路径")
        
        print(f"总共修复了 {fixed_count} 个图像路径")
        
        # 保存修复后的pkl文件
        print("正在保存修复后的文件...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"已保存修复后的文件: {pkl_path}")
        
        # 清理内存
        del data
        gc.collect()
        print()

def verify_fix():
    """验证修复结果"""
    print("验证修复结果...")
    
    train_pkl_path = "/data/tangqiansong/raw_data/scannet_occ/scannet_occ_infos_train.pkl"
    
    with open(train_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查前几个样本的路径
    for i in range(min(5, len(data))):
        img_path = data[i]['img']
        print(f"样本 {i}: {img_path}")
        
        # 检查文件是否存在
        if os.path.exists(img_path):
            print(f"  ✓ 文件存在")
        else:
            print(f"  ✗ 文件不存在")
    
    del data
    gc.collect()

if __name__ == "__main__":
    print("开始修复pkl文件中的图像路径...")
    print("=" * 50)
    
    fix_pkl_paths()
    
    print("=" * 50)
    print("路径修复完成！")
    
    # 验证修复结果
    verify_fix()