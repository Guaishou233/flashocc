# 深度图处理验证脚本使用说明

## 功能说明

`verify_depth_processing.py` 脚本用于验证 `LoadScanNetDepth` 是否正确处理深度图。

该脚本会：
1. 从深度图生成点云
2. 从点云重新生成深度图（两种方法：直接投影和带变换投影）
3. 可视化原始深度图、点云、以及重新生成的深度图
4. 对比原始深度图和重建深度图，计算差异统计

## 使用方法

### 基本用法

```bash
cd /data/tangqiansong/FlashOCC
python verify_depth_processing.py
```

### 指定参数

```bash
python verify_depth_processing.py \
    --data-root /data/tangqiansong/raw_data/scannet_occ_mini/ \
    --ann-file /data/tangqiansong/raw_data/scannet_occ_mini/scannet_occ_infos_train.pkl \
    --output-dir ./depth_verification \
    --sample-idx 0
```

### 参数说明

- `--data-root`: 数据集根目录路径（默认：`/data/tangqiansong/raw_data/scannet_occ_mini/`）
- `--ann-file`: 标注文件路径（默认：`data_root + scannet_occ_infos_train.pkl`）
- `--output-dir`: 输出可视化图像的目录（默认：`./depth_verification`）
- `--sample-idx`: 要可视化的样本索引（默认：0）

## 输出结果

脚本会在输出目录生成以下文件：

1. **depth_processing_verification.png**: 主要验证可视化图
   - 原始深度图（原始尺寸）
   - 原始深度图（下采样到256x704）
   - 点云可视化（X-Z视图）
   - 重建深度图（直接投影）
   - 重建深度图（带变换，模拟LoadScanNetDepth）
   - 差异图

2. **depth_comparison.png**: 详细对比图
   - 两行对比：直接投影 vs 带变换投影
   - 每行包含：原始、重建、差异

## 验证要点

脚本会输出以下统计信息：

1. **点云统计**：
   - 总点数
   - X、Y、Z坐标范围

2. **直接投影方法**：
   - 有效像素数及百分比
   - 平均深度差异
   - 最大深度差异

3. **带变换投影方法（LoadScanNetDepth）**：
   - 有效像素数及百分比
   - 平均深度差异
   - 最大深度差异

## 预期结果

如果 `LoadScanNetDepth` 处理正确：
- 重建的深度图应该与原始深度图（下采样后）相似
- 深度差异应该很小（通常 < 0.1m）
- 有效像素覆盖率应该较高（> 80%）

## 注意事项

1. 确保已安装所需依赖：
   - numpy
   - torch
   - PIL (Pillow)
   - matplotlib
   - scipy

2. 脚本需要访问数据集文件和标注文件

3. 如果遇到导入错误，确保在正确的目录下运行脚本

