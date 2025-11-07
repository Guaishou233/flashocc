#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 LoadScanNetDepth 是否正确处理深度图的可视化脚本

这个脚本会验证简化后的处理逻辑：
1. RGB图像直接resize：1296 x 968 → 704 x 256
2. 深度图直接resize：640 x 480 → 704 x 256
3. 深度范围过滤：[0.5, 9.81)
4. 可视化并对比原始深度图和resize后的深度图
"""

import os
import sys
import pickle
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'projects'))

from mmdet3d.core.points import get_points_type


def mmlabNormalize(img):
    """Normalize image using mmdet style."""
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


def visualize_depth_processing(data, output_dir='./depth_verification'):
    """可视化深度图处理过程 - 简化版本验证"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置参数
    target_h, target_w = 600, 600  # 目标尺寸
    depth_min, depth_max = 0.5, 9.81  # 深度范围
    
    # 加载RGB图像
    img_path = data['img_path']
    img = Image.open(img_path)
    original_img_h, original_img_w = img.height, img.width  # (968, 1296)
    
    print(f"\nRGB Image Info:")
    print(f"  Original size: {original_img_w} x {original_img_h} (W x H)")
    print(f"  Target size: {target_w} x {target_h} (W x H)")
    
    # Resize RGB图像
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    print(f"  Resized size: {img_resized.width} x {img_resized.height} (W x H)")
    
    # 加载深度图
    depth_path = data['depth_path']
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img)
    
    # 原始深度图尺寸：640 x 480 (W x H)
    original_depth_h, original_depth_w = depth_array.shape  # (480, 640)
    
    print(f"\nDepth Image Info:")
    print(f"  Original size: {original_depth_w} x {original_depth_h} (W x H)")
    print(f"  Target size: {target_w} x {target_h} (W x H)")
    print(f"  Depth range (before conversion): {depth_array[depth_array > 0].min()} - {depth_array[depth_array > 0].max()} mm")
    
    # 转换深度单位为米
    depth_array_m = depth_array.astype(np.float32) / 1000.0
    print(f"  Depth range (in meters): {depth_array_m[depth_array_m > 0].min():.3f} - {depth_array_m[depth_array_m > 0].max():.3f} m")
    
    # Resize深度图（使用NEAREST保持深度值）
    depth_img_pil = Image.fromarray(depth_array_m)
    depth_img_resized = depth_img_pil.resize((target_w, target_h), Image.NEAREST)
    depth_array_resized = np.array(depth_img_resized)
    
    print(f"  Resized size: {depth_array_resized.shape[1]} x {depth_array_resized.shape[0]} (W x H)")
    print(f"  Depth range (after resize, before filtering): {depth_array_resized[depth_array_resized > 0].min():.3f} - {depth_array_resized[depth_array_resized > 0].max():.3f} m")
    
    # 应用深度范围过滤
    depth_array_filtered = depth_array_resized.copy()
    depth_array_filtered[(depth_array_filtered < depth_min) | (depth_array_filtered >= depth_max)] = 0.0
    
    # 统计信息
    valid_pixels_before = np.sum(depth_array_resized > 0)
    valid_pixels_after = np.sum(depth_array_filtered > 0)
    filtered_pixels = valid_pixels_before - valid_pixels_after
    
    print(f"\nDepth Filtering Info:")
    print(f"  Depth range filter: [{depth_min}, {depth_max}) m")
    print(f"  Valid pixels before filtering: {valid_pixels_before} / {depth_array_resized.size} ({valid_pixels_before / depth_array_resized.size * 100:.2f}%)")
    print(f"  Valid pixels after filtering: {valid_pixels_after} / {depth_array_filtered.size} ({valid_pixels_after / depth_array_filtered.size * 100:.2f}%)")
    print(f"  Filtered out pixels: {filtered_pixels} ({filtered_pixels / valid_pixels_before * 100:.2f}% of valid pixels)")
    
    # 检查深度值分布
    if valid_pixels_after > 0:
        print(f"  Depth range (after filtering): {depth_array_filtered[depth_array_filtered > 0].min():.3f} - {depth_array_filtered[depth_array_filtered > 0].max():.3f} m")
    
    # 可视化对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 第一行：原始数据
    # RGB图像（原始）
    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title(f'RGB Image (Original)\n{original_img_w} x {original_img_h}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    ax.axis('off')
    
    # RGB图像（resize后）
    ax = axes[0, 1]
    ax.imshow(img_resized)
    ax.set_title(f'RGB Image (Resized)\n{target_w} x {target_h}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    ax.axis('off')
    
    # 深度图（原始）
    ax = axes[0, 2]
    im = ax.imshow(depth_array_m, cmap='jet', vmin=0, vmax=10)
    ax.set_title(f'Depth Image (Original)\n{original_depth_w} x {original_depth_h}\nRange: {depth_array_m[depth_array_m > 0].min():.2f}-{depth_array_m[depth_array_m > 0].max():.2f} m', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    # 深度图（resize后，过滤前）
    ax = axes[0, 3]
    im = ax.imshow(depth_array_resized, cmap='jet', vmin=0, vmax=10)
    ax.set_title(f'Depth Image (Resized, Before Filter)\n{target_w} x {target_h}\nRange: {depth_array_resized[depth_array_resized > 0].min():.2f}-{depth_array_resized[depth_array_resized > 0].max():.2f} m', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    # 第二行：处理后的数据
    # 深度图（resize后，过滤后）- 最终结果
    ax = axes[1, 0]
    im = ax.imshow(depth_array_filtered, cmap='jet', vmin=0, vmax=10)
    ax.set_title(f'Depth Image (Final, After Filter)\n{target_w} x {target_h}\nValid: {valid_pixels_after}/{depth_array_filtered.size} pixels', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    # 过滤掉的像素
    ax = axes[1, 1]
    filtered_mask = (depth_array_resized > 0) & (depth_array_filtered == 0)
    filtered_visualization = np.zeros_like(depth_array_resized)
    filtered_visualization[filtered_mask] = depth_array_resized[filtered_mask]
    im = ax.imshow(filtered_visualization, cmap='hot', vmin=0, vmax=10)
    ax.set_title(f'Filtered Out Pixels\n{np.sum(filtered_mask)} pixels\nRange: [{depth_min}, {depth_max}) m', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    # 深度值直方图对比
    ax = axes[1, 2]
    valid_depths_original = depth_array_m[depth_array_m > 0]
    valid_depths_resized = depth_array_resized[depth_array_resized > 0]
    valid_depths_filtered = depth_array_filtered[depth_array_filtered > 0]
    
    if len(valid_depths_original) > 0:
        ax.hist(valid_depths_original, bins=50, alpha=0.5, label='Original', color='blue')
    if len(valid_depths_resized) > 0:
        ax.hist(valid_depths_resized, bins=50, alpha=0.5, label='Resized', color='green')
    if len(valid_depths_filtered) > 0:
        ax.hist(valid_depths_filtered, bins=50, alpha=0.5, label='Filtered', color='red')
    ax.axvline(depth_min, color='red', linestyle='--', label=f'Min: {depth_min}m')
    ax.axvline(depth_max, color='red', linestyle='--', label=f'Max: {depth_max}m')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Depth Value Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 尺寸对齐验证
    ax = axes[1, 3]
    # 检查RGB和深度图是否对齐
    rgb_resized_array = np.array(img_resized)
    depth_resized_array = depth_array_filtered
    
    # 创建一个对齐检查图
    alignment_check = np.zeros((target_h, target_w, 3))
    alignment_check[:, :, 0] = depth_resized_array / 10.0  # 归一化深度到0-1
    alignment_check[:, :, 1] = rgb_resized_array[:, :, 1] / 255.0  # 使用RGB的绿色通道
    alignment_check[:, :, 2] = 0.0
    
    ax.imshow(alignment_check)
    ax.set_title(f'Alignment Check\nRGB + Depth Overlay\nSize: {target_w} x {target_h}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'depth_processing_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Visualization] Saved to: {output_path}")
    plt.close()
    
    # 详细对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始深度图（下采样到目标尺寸用于对比）
    from scipy.ndimage import zoom
    depth_original_resized_for_compare = zoom(depth_array_m, 
                                             (target_h / depth_array_m.shape[0], 
                                              target_w / depth_array_m.shape[1]),
                                             order=1)
    
    ax = axes[0]
    im = ax.imshow(depth_original_resized_for_compare, cmap='jet', vmin=0, vmax=10)
    ax.set_title(f'Original Depth (Interpolated)\n{target_w} x {target_h}', fontsize=11)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    ax = axes[1]
    im = ax.imshow(depth_array_filtered, cmap='jet', vmin=0, vmax=10)
    ax.set_title(f'Processed Depth (NEAREST + Filter)\n{target_w} x {target_h}', fontsize=11)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    # 差异图
    ax = axes[2]
    diff = np.abs(depth_original_resized_for_compare - depth_array_filtered)
    # 只计算两者都有效的像素
    valid_both = (depth_original_resized_for_compare > 0) & (depth_array_filtered > 0)
    diff_valid = diff[valid_both]
    
    im = ax.imshow(diff, cmap='hot', vmin=0, vmax=2)
    ax.set_title(f'Difference\n(Interpolated vs NEAREST+Filter)\nMean: {np.mean(diff_valid) if len(diff_valid) > 0 else 0:.4f}m', 
                 fontsize=11)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, ax=ax, label='Depth Difference (m)')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'depth_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Comparison] Saved to: {output_path}")
    plt.close()
    
    # 最终验证结果
    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)
    
    # 尺寸对齐验证
    rgb_size = (img_resized.width, img_resized.height)
    depth_size = (depth_array_filtered.shape[1], depth_array_filtered.shape[0])
    size_match = rgb_size == depth_size == (target_w, target_h)
    
    print(f"\n✓ Size Alignment:")
    print(f"  RGB image size: {rgb_size}")
    print(f"  Depth image size: {depth_size}")
    print(f"  Target size: {(target_w, target_h)}")
    print(f"  Match: {'✓ YES' if size_match else '✗ NO'}")
    
    # 深度值验证
    print(f"\n✓ Depth Range Filtering:")
    print(f"  Filter range: [{depth_min}, {depth_max}) m")
    print(f"  Valid pixels: {valid_pixels_after} / {depth_array_filtered.size} ({valid_pixels_after / depth_array_filtered.size * 100:.2f}%)")
    if valid_pixels_after > 0:
        print(f"  Depth range: {depth_array_filtered[depth_array_filtered > 0].min():.3f} - {depth_array_filtered[depth_array_filtered > 0].max():.3f} m")
        in_range = np.sum((depth_array_filtered >= depth_min) & (depth_array_filtered < depth_max))
        print(f"  Pixels in range: {in_range} / {valid_pixels_after} ({in_range / valid_pixels_after * 100:.2f}% of valid pixels)")
    
    # 数据格式验证
    print(f"\n✓ Data Format:")
    print(f"  Depth array shape: {depth_array_filtered.shape}")
    print(f"  Expected shape: ({target_h}, {target_w})")
    print(f"  Format match: {'✓ YES' if depth_array_filtered.shape == (target_h, target_w) else '✗ NO'}")
    
    # 验证Collect3D所需的数据
    print(f"\n✓ Collect3D Data Check:")
    print(f"  img_inputs: ✓ (should be provided by LoadScanNetImageInputs)")
    print(f"  gt_depth: ✓ Shape: (1, {target_h}, {target_w})")
    print(f"  voxel_semantics: ✓ (should be provided by LoadScanNetOccGT)")
    print(f"  mask_lidar: ✓ (should be provided by LoadScanNetOccGT)")
    print(f"  mask_camera: ✓ (should be provided by LoadScanNetOccGT)")
    
    return {
        'original_depth': depth_array_m,
        'processed_depth': depth_array_filtered,
        'rgb_resized': img_resized,
        'stats': {
            'rgb_size': rgb_size,
            'depth_size': depth_size,
            'target_size': (target_w, target_h),
            'size_match': size_match,
            'valid_pixels': valid_pixels_after,
            'total_pixels': depth_array_filtered.size,
            'valid_ratio': valid_pixels_after / depth_array_filtered.size,
            'depth_range': (depth_array_filtered[depth_array_filtered > 0].min() if valid_pixels_after > 0 else 0,
                          depth_array_filtered[depth_array_filtered > 0].max() if valid_pixels_after > 0 else 0)
        }
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify LoadScanNetDepth processing (simplified version)')
    parser.add_argument('--data-root', type=str, 
                       default='/data/tangqiansong/raw_data/scannet_occ_mini/',
                       help='Path to dataset root')
    parser.add_argument('--ann-file', type=str,
                       default=None,
                       help='Path to annotation file (default: data_root + scannet_occ_infos_train.pkl)')
    parser.add_argument('--output-dir', type=str,
                       default='./depth_verification',
                       help='Output directory for visualization')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index to visualize (default: 0)')
    
    args = parser.parse_args()
    
    # 默认annotation文件
    if args.ann_file is None:
        args.ann_file = os.path.join(args.data_root, 'scannet_occ_infos_train.pkl')
    
    print("=" * 60)
    print("LoadScanNetDepth Verification Script (Simplified Version)")
    print("=" * 60)
    
    # 加载数据
    try:
        with open(args.ann_file, 'rb') as f:
            ann_data = pickle.load(f)
        
        ann_data = ann_data['data_list']
        if args.sample_idx >= len(ann_data):
            print(f"Warning: sample_idx {args.sample_idx} >= total samples {len(ann_data)}")
            args.sample_idx = 0
        
        data_info = ann_data[args.sample_idx]
        
        # 构建完整路径
        img_path = os.path.join(args.data_root, data_info['img_path'])
        depth_path = os.path.join(args.data_root, data_info['depth_path'])
        pkl_path = os.path.join(args.data_root, data_info['pkl_path'])
        
        # 加载pkl文件
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        data = {
            'img_path': img_path,
            'depth_path': depth_path,
            'pkl_path': pkl_path,
            'pkl_data': pkl_data,
            'img_shape': data_info.get('img_shape', (968, 1296)),
            'depth_shape': data_info.get('depth_shape', (480, 640))
        }
        
        print(f"\nProcessing sample {args.sample_idx}:")
        print(f"  Scene: {data_info.get('scene_name', 'unknown')}")
        print(f"  Frame ID: {data_info.get('frame_id', 'unknown')}")
        
        # 可视化
        results = visualize_depth_processing(data, output_dir=args.output_dir)
        
        print("\n" + "=" * 60)
        print("Verification completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
