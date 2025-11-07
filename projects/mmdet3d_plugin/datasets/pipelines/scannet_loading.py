# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class LoadScanNetImageInputs(object):
    """Load ScanNet image inputs including RGB images, camera poses and intrinsics."""
    
    def __init__(self, data_config, is_train=False, sequential=False, height_offset=-0.5):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.height_offset = height_offset  # 高度偏移，默认-0.5米

    def choose_cams(self):
        """Choose cameras for processing."""
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def get_sensor_transforms(self, cam_pose, intrinsic):
        """Get sensor transformation matrices.
        
        For monocular camera OCC task:
        - Camera coordinate = Ego coordinate = World coordinate
        - Only height offset is applied (default -0.5m)
        """
        # 单目相机场景：相机坐标系 = 自车坐标系 = 世界坐标系
        # camera to ego (identity for single camera)
        sensor2ego = torch.eye(4)
        
        # ego to global: 单位矩阵 + 高度偏移
        # 世界坐标系就是相机本身，只需要在z轴上添加高度偏移
        ego2global = torch.eye(4)
        ego2global[2, 3] = self.height_offset  # z轴偏移（高度偏移）
        
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """Get image inputs for ScanNet - simplified version with only resize."""
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        # Load data from pkl file
        pkl_path = results['pkl_path']
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)

        # Get target size
        target_h, target_w = self.data_config['input_size']  # (256, 704)

        for cam_name in cam_names:
            # Load image
            img_path = results['img_path']
            img = Image.open(img_path)
            original_h, original_w = img.height, img.width  # (968, 1296)

            # Simple resize - no crop, flip, or rotate
            img_resized = img.resize((target_w, target_h), Image.BILINEAR)

            # Calculate resize scale factors
            scale_w = target_w / original_w
            scale_h = target_h / original_h

            # Get camera intrinsic (3x3 part)
            intrin = torch.from_numpy(pkl_data['intrinsic'][:3, :3]).float()
            
            # Update intrinsic to match resized image
            # Scale focal lengths and principal point
            intrin_resized = intrin.clone()
            intrin_resized[0, 0] *= scale_w  # fx
            intrin_resized[1, 1] *= scale_h  # fy
            intrin_resized[0, 2] *= scale_w  # cx
            intrin_resized[1, 2] *= scale_h  # cy
            
            # Get camera pose and convert to sensor transforms
            cam_pose = pkl_data['cam_pose']
            sensor2ego, ego2global = self.get_sensor_transforms(cam_pose, pkl_data['intrinsic'])

            # For simple resize, post_rot and post_tran are identity
            # post_rot represents the transformation matrix (identity for simple resize)
            post_rot = torch.eye(3)
            post_tran = torch.zeros(3)

            canvas.append(np.array(img_resized))
            imgs.append(self.normalize_img(img_resized))

            intrins.append(intrin_resized)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        imgs = torch.stack(imgs)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas

        # Add BDA (Bird's Eye View Data Augmentation) matrix
        # For ScanNet, we use identity matrix as default BDA
        bda = torch.eye(3)
        
        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda

    def __call__(self, results):
        # Check if this is test-time augmentation
        if 'flip' in results and 'scale' in results:
            # Test-time augmentation case
            flip = results['flip']
            scale = results.get('scale', None)
            # Convert scale from tuple (img_scale) to float (scale_factor)
            if isinstance(scale, tuple):
                # For MultiScaleFlipAug3D, scale is (img_width, img_height)
                # We need to convert this to a scale factor
                scale = 0.0  # Use default scale factor for now
            results['img_inputs'] = self.get_inputs(results, flip=flip, scale=scale)
        else:
            # Normal case
            results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadScanNetOccGT(object):
    """Load ScanNet occupancy ground truth."""
    
    def __init__(self, grid_config):
        self.grid_config = grid_config

    def __call__(self, results):
        # Load data from pkl file
        pkl_path = results['pkl_path']
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)

        # Get occupancy labels
        # target_1_4 is the high resolution occupancy (60x60x36)
        # target_1_16 is the low resolution occupancy (15x15x9)
        target_1_4 = pkl_data['target_1_4']  # (60, 60, 36)
        target_1_16 = pkl_data['target_1_16']  # (15, 15, 9)
        
        # Convert to torch tensors
        voxel_semantics = torch.from_numpy(target_1_4).long()  # (60, 60, 36)
        
        # 将255映射到0（empty类别），0-11为语义类别
        voxel_semantics = voxel_semantics.clone()
        voxel_semantics[voxel_semantics == 255] = 0
        
        # Create mask_lidar and mask_camera (assuming all voxels are valid)
        mask_lidar = torch.ones_like(voxel_semantics, dtype=torch.bool)
        mask_camera = torch.ones_like(voxel_semantics, dtype=torch.bool)
        

        # Resize voxel data to match model input dimensions (200x200x16)
        # Original: (60, 60, 36) -> Target: (200, 200, 16)
        import torch.nn.functional as F
        
        # Add batch and channel dimensions for interpolation
        voxel_semantics = voxel_semantics.unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 60, 36)
        mask_lidar = mask_lidar.unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 60, 36)
        mask_camera = mask_camera.unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 60, 36)
        
        # # Resize to target dimensions
        # target_size = (200, 200, 16)
        # voxel_semantics = F.interpolate(voxel_semantics, size=target_size, mode='nearest')
        # mask_lidar = F.interpolate(mask_lidar.float(), size=target_size, mode='nearest').bool()
        # mask_camera = F.interpolate(mask_camera.float(), size=target_size, mode='nearest').bool()
        
        # # Remove batch and channel dimensions
        # voxel_semantics = voxel_semantics.squeeze(0).squeeze(0)  # (200, 200, 16)
        # mask_lidar = mask_lidar.squeeze(0).squeeze(0)  # (200, 200, 16)
        # mask_camera = mask_camera.squeeze(0).squeeze(0)  # (200, 200, 16)

        results['voxel_semantics'] = voxel_semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results


@PIPELINES.register_module()
class LoadScanNetDepth(object):
    """Load ScanNet depth image - simplified version with direct resize."""
    
    def __init__(self, grid_config, downsample=1, data_config=None):
        self.downsample = downsample
        self.grid_config = grid_config
        self.data_config = data_config

    def __call__(self, results):
        # Get target size from data_config or img_inputs
        # img_inputs should already be processed by LoadScanNetImageInputs
        if 'img_inputs' in results:
            imgs = results['img_inputs'][0]  # (B, C, H, W)
            target_h, target_w = imgs.shape[2], imgs.shape[3]  # (256, 704)
        elif self.data_config is not None:
            target_h, target_w = self.data_config['input_size']  # (256, 704)
        else:
            # Fallback: use default size
            target_h, target_w = 256, 704

        # Load depth image
        depth_path = results['depth_path']
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)
        
        # Original depth image size: 480 x 640 (H x W)
        original_h, original_w = depth_array.shape  # (480, 640)
        
        # Convert depth from millimeters to meters
        depth_array_m = depth_array.astype(np.float32) / 1000.0
        
        # Resize depth image to match RGB image size
        # Use NEAREST to preserve depth values (avoid interpolation artifacts)
        # Note: PIL Image.resize expects (width, height) order
        depth_img_pil = Image.fromarray(depth_array_m)
        depth_img_resized = depth_img_pil.resize((target_w, target_h), Image.NEAREST)
        depth_array_resized = np.array(depth_img_resized)
        
        # Apply depth range filtering
        depth_min = self.grid_config['depth'][0]  # 0.5
        depth_max = self.grid_config['depth'][1]  # 9.81
        depth_array_resized[(depth_array_resized < depth_min) | (depth_array_resized >= depth_max)] = 0.0
        
        # Convert to torch tensor
        depth_map = torch.from_numpy(depth_array_resized).float()
        
        # Format: (N, H, W) where N is number of cameras
        # For single camera, shape should be (1, H, W)
        depth_map = depth_map.unsqueeze(0)  # (1, 256, 704)
        
        results['gt_depth'] = depth_map
        
        return results


@PIPELINES.register_module()
class ScanNetPointToMultiViewDepth(object):
    """Convert ScanNet points to multi-view depth maps."""
    
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """Convert points to depth map."""
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:6]
        
        # Load camera pose from pkl
        pkl_path = results['pkl_path']
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            # Get camera pose
            cam_pose = pkl_data['cam_pose']
            
            # Transform points to camera coordinate
            points_cam = points_lidar.tensor[:, :3].matmul(
                torch.from_numpy(cam_pose[:3, :3].T).float()) + \
                torch.from_numpy(cam_pose[:3, 3]).float().unsqueeze(0)
            
            # Project to image plane
            points_img = points_cam.matmul(intrins[cid].T)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
            
            # Apply image augmentation
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            
            # Convert to depth map
            depth_map = self.points2depthmap(points_img,
                                           imgs.shape[2],  # H
                                           imgs.shape[3]   # W
                                           )
            depth_map_list.append(depth_map)
        
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        
        return results
