"""
可视化深度图生成的体素标注与GT体素标注的对比

用法:
    python vis_depth_voxel_comparison.py \
        --pkl_file /data/tangqiansong/raw_data/scannet_occ_mini/gathered_data/scene0000_00/00001.pkl \
        --out_dir /data/tangqiansong/FlashOCC/work_dirs/voxel_comparison
"""
import argparse
import os
import pickle
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_palette():
    """获取20类调色板"""
    return [
        (174, 199, 232),
        (152, 223, 138),
        (31, 119, 180),
        (255, 187, 120),
        (188, 189, 34),
        (140, 86, 75),
        (255, 152, 150),
        (214, 39, 40),
        (197, 176, 213),
        (148, 103, 189),
        (196, 156, 148),
        (23, 190, 207),
        (247, 182, 210),
        (219, 219, 141),
        (255, 127, 14),
        (158, 218, 229),
        (44, 160, 44),
        (112, 128, 144),
        (227, 119, 194),
        (82, 84, 163),
    ]


def get_class_names():
    """获取类别名称"""
    return (
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
        'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
    )


def depth_image_to_points(depth_array, intrinsic, cam_pose=None, max_depth=50.0):
    """
    将深度图像转换为3D点云
    
    Args:
        depth_array: (H, W) 深度数组，单位为毫米
        intrinsic: (3, 3) 相机内参矩阵
        cam_pose: (4, 4) 相机到世界坐标的变换矩阵（可选）
        max_depth: 最大深度阈值（米）
    
    Returns:
        points: (N, 3) 点云坐标，单位米
    """
    height, width = depth_array.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 深度图像单位通常是毫米，转换为米
    depth_m = depth_array.astype(np.float32) / 1000.0
    
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


def points_to_voxel_grid(points, voxel_origin, voxel_size, grid_shape):
    """
    将点云转换为体素网格
    
    Args:
        points: (N, 3) 点云坐标，单位米
        voxel_origin: (3,) 体素原点，单位米
        voxel_size: (3,) 或 float 体素大小，单位米
        grid_shape: (Dx, Dy, Dz) 网格形状
    
    Returns:
        voxel_grid: (Dx, Dy, Dz) 体素网格，1表示占用，0表示空
    """
    Dx, Dy, Dz = grid_shape
    
    # 处理体素大小：如果是标量，转换为向量
    if isinstance(voxel_size, (int, float)):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.array(voxel_size)
    
    # 将点云坐标转换为体素索引
    # 体素索引 = (点坐标 - 体素原点) / 体素大小
    voxel_indices = ((points - voxel_origin) / voxel_size).astype(np.int32)
    
    # 过滤超出网格范围的点
    valid_mask = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < Dx) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < Dy) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < Dz)
    )
    
    if np.sum(valid_mask) == 0:
        return np.zeros(grid_shape, dtype=np.uint8)
    
    voxel_indices = voxel_indices[valid_mask]
    
    # 创建体素网格
    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1
    
    return voxel_grid


def project_voxel_to_image(voxel_grid, voxel_origin, voxel_size, cam_pose, intrinsic, img_shape):
    """
    将体素网格投影到图像平面
    
    Args:
        voxel_grid: (Dx, Dy, Dz) 体素网格
        voxel_origin: (3,) 体素原点
        voxel_size: (3,) 或 float 体素大小
        cam_pose: (4, 4) 相机位姿（世界到相机）
        intrinsic: (3, 3) 相机内参
        img_shape: (H, W) 图像形状
    
    Returns:
        proj_image: (H, W, 3) 投影图像
    """
    H, W = img_shape
    proj_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 处理体素大小：如果是标量，转换为向量
    if isinstance(voxel_size, (int, float)):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.array(voxel_size)
    
    # 获取占用的体素索引
    occupied_indices = np.argwhere(voxel_grid > 0)
    
    if len(occupied_indices) == 0:
        return proj_image
    
    # 将体素索引转换为世界坐标（体素中心）
    voxel_coords = occupied_indices.astype(np.float32) + 0.5  # 体素中心
    world_coords = voxel_origin[None, :] + voxel_coords * voxel_size  # (N, 3)
    
    # 世界坐标转相机坐标
    world_coords_homogeneous = np.concatenate(
        [world_coords, np.ones((world_coords.shape[0], 1))], axis=1)  # (N, 4)
    
    # 尝试两种变换：cam_pose 可能是 cam2world 或 world2cam
    def project_with_ext(Ext):
        """使用给定的外参进行投影"""
        R = Ext[:3, :3]
        t = Ext[:3, 3:4]
        Pw = world_coords.T  # (3, N)
        Pc = (R @ Pw + t)  # (3, N)
        z = Pc[2, :]
        valid = z > 1e-4
        Pc = Pc[:, valid]
        if Pc.shape[1] == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
        return Pc.T, np.where(valid)[0]
    
    # 尝试两种变换
    E1 = np.linalg.inv(cam_pose)  # 假设 cam_pose 是 cam2world
    E2 = cam_pose  # 假设 cam_pose 是 world2cam
    
    cam_coords1, idx1 = project_with_ext(E1)
    cam_coords2, idx2 = project_with_ext(E2)
    
    # 选择投影到图像内更多点的那种
    if cam_coords2.shape[0] > cam_coords1.shape[0]:
        cam_coords = cam_coords2
        valid_idx = idx2
    else:
        cam_coords = cam_coords1
        valid_idx = idx1
    
    if cam_coords.shape[0] == 0:
        return proj_image
    
    # 投影到图像平面
    pixel_coords = (intrinsic @ cam_coords.T).T  # (N, 3)
    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:3]  # (N, 2)
    
    # 过滤在图像范围内的点
    in_image_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
    )
    
    if np.sum(in_image_mask) == 0:
        return proj_image
    
    pixel_coords = pixel_coords[in_image_mask].astype(np.int32)
    
    # 绘制投影点
    for u, v in pixel_coords:
        cv2.circle(proj_image, (u, v), 2, (0, 255, 0), -1)
    
    return proj_image


def visualize_voxel_3d(voxel_grid, cam_pose=None, title="Voxel Grid"):
    """
    可视化3D体素网格
    
    Args:
        voxel_grid: (Dx, Dy, Dz) 体素网格
        cam_pose: (4, 4) 相机位姿，用于设置视角
        title: 图像标题
    
    Returns:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取占用的体素
    occupied = voxel_grid > 0
    occupied_ds = occupied[::2, ::2, ::2]  # 降采样以加速渲染
    
    # 创建颜色
    colors = np.ones((*occupied_ds.shape, 4), dtype=np.float32)
    colors[:, :, :, 0] = 0.5  # 红色
    colors[:, :, :, 1] = 0.5  # 绿色
    colors[:, :, :, 2] = 1.0  # 蓝色
    colors[:, :, :, 3] = 0.8  # 透明度
    
    # 绘制体素
    ax.voxels(occupied_ds.transpose(1, 0, 2),
              facecolors=colors.transpose(1, 0, 2, 3),
              edgecolor='k', linewidth=0.1)
    
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置视角
    if cam_pose is not None:
        R = cam_pose[:3, :3]
        # 相机主视向方向
        look_vec = R @ np.array([0, 0, -1])
        yaw = np.degrees(np.arctan2(look_vec[1], look_vec[0]))
        pitch = np.degrees(np.arctan2(look_vec[2], np.linalg.norm(look_vec[:2])))
        ax.view_init(elev=pitch, azim=yaw)
    else:
        ax.view_init(elev=20, azim=45)
    
    ax.set_box_aspect((occupied_ds.shape[1], occupied_ds.shape[0], occupied_ds.shape[2]))
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='可视化深度图生成的体素标注与GT体素标注的对比')
    parser.add_argument('--pkl_file', type=str, required=True,
                        help='pkl文件路径')
    parser.add_argument('--out_dir', type=str, default='/data/tangqiansong/FlashOCC/work_dirs/voxel_comparison',
                        help='输出目录')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[0.4, 0.4, 0.15],
                        help='体素大小（米），格式: x y z，默认 [0.4, 0.4, 0.15]')
    parser.add_argument('--max_depth', type=float, default=50.0,
                        help='最大深度（米）')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载pkl文件
    print(f"加载pkl文件: {args.pkl_file}")
    with open(args.pkl_file, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # 获取数据
    target_1_4 = np.array(pkl_data['target_1_4'], dtype=np.int32)  # (60, 60, 36)
    depth_gt_path = pkl_data['depth_gt']
    img_path = pkl_data['img']
    cam_pose = np.array(pkl_data['cam_pose'], dtype=np.float64)  # (4, 4)
    intrinsic = np.array(pkl_data['intrinsic'], dtype=np.float64)[:3, :3]  # (3, 3)
    voxel_origin = np.array(pkl_data['voxel_origin'], dtype=np.float64).reshape(3)  # (3,)
    
    print(f"target_1_4 shape: {target_1_4.shape}")
    print(f"voxel_origin: {voxel_origin}")
    
    # 加载深度图
    print(f"加载深度图: {depth_gt_path}")
    if not os.path.exists(depth_gt_path):
        print(f"错误: 深度图文件不存在: {depth_gt_path}")
        return
    
    depth_img = Image.open(depth_gt_path)
    depth_array = np.array(depth_img).astype(np.float32)
    print(f"深度图形状: {depth_array.shape}")
    
    # 加载原始图像
    print(f"加载原始图像: {img_path}")
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在: {img_path}")
        return
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"图像形状: {img.shape}")
    
    # 从深度图生成点云
    print("从深度图生成点云...")
    points = depth_image_to_points(depth_array, intrinsic, cam_pose, max_depth=args.max_depth)
    print(f"生成点云数量: {points.shape[0]}")
    
    # 从点云生成体素网格
    print("从点云生成体素网格...")
    grid_shape = target_1_4.shape  # (60, 60, 36)
    voxel_size = np.array(args.voxel_size) if isinstance(args.voxel_size, list) else args.voxel_size
    voxel_from_depth = points_to_voxel_grid(points, voxel_origin, voxel_size, grid_shape)
    print(f"体素网格形状: {voxel_from_depth.shape}")
    print(f"占用体素数量: {np.sum(voxel_from_depth > 0)}")
    
    # 将GT体素网格转换为占用网格（非0和255的为占用）
    voxel_gt = (target_1_4 != 0) & (target_1_4 != 255)
    voxel_gt = voxel_gt.astype(np.uint8)
    print(f"GT占用体素数量: {np.sum(voxel_gt > 0)}")
    
    # 生成三个对比图
    
    # 1. 原始图片和生成体素图的对比
    print("生成对比图1: 原始图片和生成体素图...")
    voxel_size_for_proj = voxel_size[0] if isinstance(voxel_size, np.ndarray) else voxel_size
    proj_depth = project_voxel_to_image(voxel_from_depth, voxel_origin, voxel_size_for_proj, 
                                       cam_pose, intrinsic, img.shape[:2])
    img_with_depth_voxel = cv2.addWeighted(img, 0.7, proj_depth, 0.3, 0)
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))
    axes1[0].imshow(img)
    axes1[0].set_title('Original Image')
    axes1[0].axis('off')
    axes1[1].imshow(img_with_depth_voxel)
    axes1[1].set_title('Original Image + Depth-based Voxel Projection')
    axes1[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'image_depth_voxel_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. 标注数据和原始图片的对比
    print("生成对比图2: 标注数据和原始图片...")
    proj_gt = project_voxel_to_image(voxel_gt, voxel_origin, voxel_size_for_proj,
                                     cam_pose, intrinsic, img.shape[:2])
    img_with_gt_voxel = cv2.addWeighted(img, 0.7, proj_gt, 0.3, 0)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
    axes2[0].imshow(img)
    axes2[0].set_title('Original Image')
    axes2[0].axis('off')
    axes2[1].imshow(img_with_gt_voxel)
    axes2[1].set_title('Original Image + GT Voxel Projection')
    axes2[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'image_gt_voxel_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. 标注数据和生成体素图的对比
    print("生成对比图3: 标注数据和生成体素图...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左侧：GT体素投影
    axes3[0].imshow(img_with_gt_voxel)
    axes3[0].set_title('GT Voxel Projection')
    axes3[0].axis('off')
    
    # 右侧：深度图生成的体素投影
    axes3[1].imshow(img_with_depth_voxel)
    axes3[1].set_title('Depth-based Voxel Projection')
    axes3[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'voxel_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 额外：3D可视化对比（使用相同的相机视角）
    print("生成3D可视化对比...")
    fig_gt_3d = visualize_voxel_3d(voxel_gt, cam_pose, "GT Voxel Grid (Camera View)")
    plt.savefig(os.path.join(args.out_dir, 'voxel_gt.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_gt_3d)
    
    fig_depth_3d = visualize_voxel_3d(voxel_from_depth, cam_pose, "Depth-based Voxel Grid (Camera View)")
    plt.savefig(os.path.join(args.out_dir, 'voxel_from_depth.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_depth_3d)
    
    print(f"所有可视化结果已保存到: {args.out_dir}")


if __name__ == '__main__':
    main()

