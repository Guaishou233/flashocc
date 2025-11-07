import os
import argparse
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 从数据集获取调色板和类别名称
import sys
# 添加项目路径以导入数据集类
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)
from projects.mmdet3d_plugin.datasets.scannet_occ_dataset import ScanNetOccDataset


def get_palette():
    """
    从 ScanNetOccDataset 获取调色板
    
    Returns:
        list: RGB颜色列表，每个元素为 (R, G, B) 元组
    """
    return ScanNetOccDataset.METAINFO['palette']


def get_class_names():
    """
    从 ScanNetOccDataset 获取类别名称
    
    Returns:
        tuple: 类别名称元组
    """
    return ScanNetOccDataset.METAINFO['classes']


def build_legend_image(palette, class_names: tuple, present_classes: list, 
                       width: int, title: str = 'Legend') -> np.ndarray:
    """
    构建图例图像
    
    Args:
        palette: 调色板（列表或numpy数组），RGB格式，每个元素为 (R, G, B)
        class_names: 类别名称元组
        present_classes: 出现的类别ID列表
        width: 图像宽度
        title: 图例标题
    
    Returns:
        RGB 图像数组 (H, W, 3)
    """
    # 确保palette是numpy数组
    if isinstance(palette, list):
        palette = np.array(palette, dtype=np.uint8)
    
    # 计算图例高度
    item_height = 30
    title_height = 30
    padding = 10
    num_items = len(present_classes)
    height = title_height + num_items * item_height + padding * 2
    
    # 创建白色背景
    legend_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 转换为 BGR 进行绘制
    legend_img_bgr = cv2.cvtColor(legend_img, cv2.COLOR_RGB2BGR)
    
    # 绘制标题
    cv2.putText(legend_img_bgr, title, (padding, title_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 绘制每个类别
    y_offset = title_height + padding
    color_box_size = 20
    text_x_offset = color_box_size + 10
    
    for i, class_id in enumerate(present_classes):
        y = y_offset + i * item_height
        
        # 获取颜色（确保类别ID在有效范围内）
        if class_id < len(palette):
            if isinstance(palette[class_id], np.ndarray):
                color_rgb = palette[class_id].tolist()
            else:
                color_rgb = list(palette[class_id])
            color_bgr = tuple(reversed(color_rgb))  # RGB -> BGR
        else:
            color_bgr = (128, 128, 128)  # 灰色作为默认颜色
        
        # 绘制颜色框
        cv2.rectangle(legend_img_bgr, 
                      (padding, y - color_box_size // 2),
                      (padding + color_box_size, y + color_box_size // 2),
                      color_bgr, -1)
        cv2.rectangle(legend_img_bgr,
                      (padding, y - color_box_size // 2),
                      (padding + color_box_size, y + color_box_size // 2),
                      (0, 0, 0), 1)  # 黑色边框
        
        # 绘制类别名称
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f'Class {class_id}'
        
        label = f'{class_id}: {class_name}'
        cv2.putText(legend_img_bgr, label, (padding + text_x_offset, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # 转回 RGB
    legend_img_rgb = cv2.cvtColor(legend_img_bgr, cv2.COLOR_BGR2RGB)
    return legend_img_rgb


def vis_vox(occ: np.ndarray, cam_pose: np.ndarray = None, vox_stride: int = 2) -> np.ndarray:
    """
    使用 vox 模式可视化体素网格
    
    Args:
        occ: 体素标签 (Dx, Dy, Dz)
        cam_pose: 相机位姿矩阵 (4x4)，用于设置视角
        vox_stride: 降采样步长
    
    Returns:
        RGB 图像数组 (H, W, 3)
    """
    # 准备调色板
    base_palette = np.array(get_palette(), dtype=np.uint8)
    if base_palette.shape[0] < 256:
        extra = 256 - base_palette.shape[0]
        extra_colors = np.array([
            ((i * 37) % 255, (i * 59) % 255, (i * 83) % 255) for i in range(extra)
        ], dtype=np.uint8)
        palette = np.concatenate([base_palette, extra_colors], axis=0)
    else:
        palette = base_palette

    # 占用判断：隐藏标签 0（背景）、19（otherfurniture）和 20（255映射的未知类别）
    occupied = (occ != 0) & (occ != 19) & (occ != 20) & (occ != 255)

    # 降采样以加速渲染
    s = max(1, int(vox_stride))
    occ_ds = occ[::s, ::s, ::s]
    occupied_ds = occupied[::s, ::s, ::s]

    # 准备颜色 (Dx,Dy,Dz,4)：逐体素用调色板索引
    occ_ds_clipped = np.clip(occ_ds, 0, 255)
    colors_rgb = palette[occ_ds_clipped]  # (Dx,Dy,Dz,3)
    # 将背景/未占用设为透明
    alpha = (occupied_ds.astype(np.uint8) * 255)  # (Dx,Dy,Dz)
    colors_rgba = np.concatenate([colors_rgb, alpha[..., None]], axis=-1)  # (Dx,Dy,Dz,4)
    colors_rgba = (colors_rgba / 255.0).astype(np.float32)

    # 使用 matplotlib 绘制体素
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(occupied_ds.transpose(1, 0, 2),  # 调整到 (Y,X,Z) 提升直观性
              facecolors=colors_rgba.transpose(1, 0, 2, 3),
              edgecolor='k', linewidth=0.1)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    
    # 设置视角
    if cam_pose is not None:
        R = cam_pose[:3, :3]
        # 相机主视向方向，改为-z轴
        look_vec = R @ np.array([0, 0, -1])  # 修正为-z轴（朝外）
        # yaw, pitch
        yaw = np.degrees(np.arctan2(look_vec[1], look_vec[0]))
        pitch = np.degrees(np.arctan2(look_vec[2], np.linalg.norm(look_vec[:2])))
        # matplotlib: elev(上仰), azim(新yaw)
        ax.view_init(elev=pitch, azim=yaw)
    else:
        ax.view_init(elev=20, azim=45)
    
    ax.set_box_aspect((occupied_ds.shape[1], occupied_ds.shape[0], occupied_ds.shape[2]))
    plt.tight_layout()
    
    # 保存到临时文件并读取
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        plt.savefig(tmp_path, dpi=200)
        plt.close(fig)
        img_bgr = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        else:
            return np.zeros((600, 800, 3), dtype=np.uint8)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def find_cam_pose_from_data(scene_name: str, frame_id: str, info_pkl: str = None) -> np.ndarray:
    """
    从数据集中查找对应的相机位姿
    
    Args:
        scene_name: 场景名称，如 'scene0002_00'
        frame_id: 帧ID，如 '00000'
        info_pkl: info pkl 文件路径（可选）
    
    Returns:
        相机位姿矩阵 (4x4)，如果找不到则返回 None
    """
    if info_pkl is None:
        # 尝试常见的路径
        possible_paths = [
            '/data/tangqiansong/raw_data/scannet_occ_mini/scannet_occ_infos_test.pkl',
            '/data/tangqiansong/raw_data/scannet_occ_mini/scannet_occ_infos_val.pkl',
            '/data/tangqiansong/raw_data/scannet_occ_mini/scannet_occ_infos_train.pkl',
        ]
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        info_pkl = found_path
    
    if info_pkl is None or not os.path.exists(info_pkl):
        return None
    
    try:
        with open(info_pkl, 'rb') as f:
            data = pickle.load(f)
        
        # 兼容两种结构
        if isinstance(data, dict) and 'data_list' in data:
            data_list = data['data_list']
        elif isinstance(data, list):
            data_list = data
        else:
            return None
        
        # 查找匹配的样本
        for sample in data_list:
            sample_scene = sample.get('scene_name', '')
            sample_frame = sample.get('frame_id', '')
            if str(sample_scene) == str(scene_name) and str(sample_frame) == str(frame_id):
                # 尝试从 sample 获取 cam_pose
                if 'cam_pose' in sample:
                    return np.array(sample['cam_pose'])
                # 或者从 pkl_path 加载
                if 'pkl_path' in sample:
                    with open(sample['pkl_path'], 'rb') as f:
                        pkl_data = pickle.load(f)
                    if 'cam_pose' in pkl_data:
                        return np.array(pkl_data['cam_pose'])
    except Exception as e:
        print(f'Warning: Failed to load cam_pose: {e}')
    
    return None


def visualize_pred_gt_comparison(
    result_dir: str,
    out_dir: str,
    scene_name: str = None,
    info_pkl: str = None,
    vox_stride: int = 2,
    start_idx: int = 0,
    num_frames: int = None
):
    """
    可视化预测结果和GT的对比
    
    Args:
        result_dir: 结果目录路径，如 '/data/.../scene0002_00/'
        out_dir: 输出目录
        scene_name: 场景名称，如果为 None 则从 result_dir 推断
        info_pkl: info pkl 文件路径，用于获取相机位姿
        vox_stride: 体素降采样步长
        start_idx: 起始帧索引
        num_frames: 要可视化的帧数量，None 表示全部
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 推断场景名称
    if scene_name is None:
        scene_name = os.path.basename(os.path.normpath(result_dir))
    
    # 获取所有子目录（帧目录）
    frame_dirs = sorted([d for d in os.listdir(result_dir) 
                        if os.path.isdir(os.path.join(result_dir, d))])
    
    if num_frames is not None:
        frame_dirs = frame_dirs[start_idx:start_idx + num_frames]
    else:
        frame_dirs = frame_dirs[start_idx:]
    
    print(f'Processing {len(frame_dirs)} frames from scene {scene_name}')
    
    # 准备调色板
    base_palette = np.array(get_palette(), dtype=np.uint8)
    if base_palette.shape[0] < 256:
        extra = 256 - base_palette.shape[0]
        extra_colors = np.array([
            ((i * 37) % 255, (i * 59) % 255, (i * 83) % 255) for i in range(extra)
        ], dtype=np.uint8)
        palette = np.concatenate([base_palette, extra_colors], axis=0)
    else:
        palette = base_palette
    
    class_names = get_class_names()
    
    for frame_id in frame_dirs:
        frame_path = os.path.join(result_dir, frame_id)
        pred_gt_path = os.path.join(frame_path, 'pred_gt.npz')
        
        if not os.path.exists(pred_gt_path):
            print(f'Warning: {pred_gt_path} not found, skipping')
            continue
        
        # 加载预测和GT
        data = np.load(pred_gt_path)
        pred = data['pred'].astype(np.int32)
        gt = data['gt'].astype(np.int32)
        
        print(f'Processing frame {frame_id}: pred shape {pred.shape}, gt shape {gt.shape}')
        
        # 获取相机位姿
        cam_pose = find_cam_pose_from_data(scene_name, frame_id, info_pkl)
        
        # 可视化 GT 和预测
        gt_img = vis_vox(gt, cam_pose, vox_stride)
        pred_img = vis_vox(pred, cam_pose, vox_stride)
        
        # 合并图像（左右并排）
        h = max(gt_img.shape[0], pred_img.shape[0])
        w = gt_img.shape[1] + pred_img.shape[1]
        
        # 如果高度不一致，调整到相同高度
        if gt_img.shape[0] != h:
            gt_img = cv2.resize(gt_img, (gt_img.shape[1], h))
        if pred_img.shape[0] != h:
            pred_img = cv2.resize(pred_img, (pred_img.shape[1], h))
        
        combined_img = np.concatenate([gt_img, pred_img], axis=1)
        
        # 添加标题
        title_h = 40
        title_img = np.ones((title_h, w, 3), dtype=np.uint8) * 255
        # 先转换为 BGR 进行绘制
        title_img_bgr = cv2.cvtColor(title_img, cv2.COLOR_RGB2BGR)
        cv2.putText(title_img_bgr, f'GT (Left)', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(title_img_bgr, f'Prediction (Right)', (gt_img.shape[1] + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        # 转回 RGB 以与 combined_img 拼接
        title_img_rgb = cv2.cvtColor(title_img_bgr, cv2.COLOR_BGR2RGB)
        combined_img = np.concatenate([title_img_rgb, combined_img], axis=0)
        
        # 添加图例（隐藏标签 0、19、20）
        present_gt = sorted(set(int(x) for x in np.unique(gt) if x not in (0, 19, 20)))
        present_pred = sorted(set(int(x) for x in np.unique(pred) if x not in (0, 19, 20)))
        present_all = sorted(set(present_gt + present_pred))
        legend = build_legend_image(palette, class_names, present_all, w, title='Legend')
        combined_img = np.concatenate([combined_img, legend], axis=0)
        
        # 保存
        out_path = os.path.join(out_dir, f'{scene_name}_{frame_id}_comparison.png')
        cv2.imwrite(out_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize prediction vs GT comparison for scene results')
    parser.add_argument('--result_dir', type=str, default='/data/tangqiansong/FlashOCC/work_dirs/predict_result/gt_pred/scene0002_00',
                        help='结果目录路径，如 /data/.../predict_result/scene0002_00/')
    parser.add_argument('--out_dir', type=str, default='/data/tangqiansong/FlashOCC/work_dirs/predict_result/gt_pred',
                        help='输出目录，默认为 result_dir + "_vis"')
    parser.add_argument('--info_pkl', type=str, default="/data/tangqiansong/raw_data/scannet_occ_mini/scannet_occ_infos_test.pkl",
                        help='info pkl 文件路径，用于获取相机位姿（可选）')
    parser.add_argument('--vox_stride', type=int, default=2,
                        help='体素降采样步长，值越大渲染越快')
    parser.add_argument('--start', type=int, default=0,
                        help='起始帧索引')
    parser.add_argument('--num', type=int, default=10,
                        help='要可视化的帧数量，None 表示全部')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = args.result_dir.rstrip('/') + '_vis'
    
    visualize_pred_gt_comparison(
        result_dir=args.result_dir,
        out_dir=args.out_dir,
        scene_name=None,
        info_pkl=args.info_pkl,
        vox_stride=args.vox_stride,
        start_idx=args.start,
        num_frames=args.num
    )
    
    print(f'Done. Results saved to {args.out_dir}')


if __name__ == '__main__':
    main()

