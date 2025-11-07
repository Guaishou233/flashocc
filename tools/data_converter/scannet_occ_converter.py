# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
import numpy as np
from os import path as osp
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

class ScanNetOccData:
    """ScanNet OCC dataset.
    
    This class is used to create the ScanNet OCC dataset info file.
    
    Args:
        root_path (str): Path of dataset root.
        split (str): Dataset split used.
        workers (int): Number of threads to be used.
    """
    
    def __init__(self, root_path, split='train', workers=4):
        self.root_path = root_path
        self.split = split
        self.workers = workers
        
        # 获取所有场景
        self.scenes = self._get_scenes()
        
        # 根据split划分数据
        self.data_infos = self._create_data_infos()
        
    def _get_scenes(self):
        """获取所有场景列表"""
        gathered_data_path = osp.join(self.root_path, 'gathered_data')
        scenes = []
        
        if osp.exists(gathered_data_path):
            for scene_name in os.listdir(gathered_data_path):
                scene_path = osp.join(gathered_data_path, scene_name)
                if osp.isdir(scene_path):
                    scenes.append(scene_name)
        
        scenes.sort()
        return scenes
    
    def _create_data_infos(self):
        """创建数据信息列表"""
        data_infos = []
        
        # 8:2的训练/测试划分
        total_scenes = len(self.scenes)
        train_split_idx = int(total_scenes * 0.8)
        
        if self.split == 'train':
            target_scenes = self.scenes[:train_split_idx]  # 前80%的场景用于训练
        elif self.split == 'test':
            target_scenes = self.scenes[train_split_idx:]  # 后20%的场景用于测试
        else:
            target_scenes = self.scenes
        
        # 添加场景处理进度条
        scene_progress = tqdm(target_scenes, desc=f'Processing {self.split} scenes', unit='scene')
        for scene_name in scene_progress:
            scene_path = osp.join(self.root_path, 'gathered_data', scene_name)
            if not osp.exists(scene_path):
                continue
                
            # 获取该场景下的所有pkl文件
            pkl_files = [f for f in os.listdir(scene_path) if f.endswith('.pkl')]
            pkl_files.sort()
            
            # 更新进度条描述
            scene_progress.set_postfix({
                'scene': scene_name,
                'frames': len(pkl_files),
                'total_samples': len(data_infos)
            })
            
            for pkl_file in pkl_files:
                pkl_path = osp.join(scene_path, pkl_file)
                frame_id = pkl_file.split('.')[0]
                
                # 构建数据信息
                data_info = {
                    'scene_name': scene_name,
                    'frame_id': frame_id,
                    'pkl_path': pkl_path,
                    'img_path': osp.join(self.root_path, 'posed_images', scene_name, f'{frame_id}.jpg'),
                    'depth_path': osp.join(self.root_path, 'posed_images', scene_name, f'{frame_id}.png'),
                }
                
                data_infos.append(data_info)
        
        return data_infos
    
    def _load_single_data(self, data_info):
        """加载单个数据文件的信息"""
        try:
            # 加载pkl文件获取数据信息
            with open(data_info['pkl_path'], 'rb') as f:
                pkl_data = pickle.load(f)
            
            # 构建mmdetection3d格式的数据信息
            mmdet3d_info = {
                'scene_name': data_info['scene_name'],
                'frame_id': data_info['frame_id'],
                'img_path': data_info['img_path'],
                'depth_path': data_info['depth_path'],
                'pkl_path': data_info['pkl_path'],
                'cam_pose': pkl_data['cam_pose'].tolist(),
                'intrinsic': pkl_data['intrinsic'].tolist(),
                'voxel_origin': pkl_data['voxel_origin'],
                'target_1_4': pkl_data['target_1_4'].tolist(),
                'target_1_16': pkl_data['target_1_16'].tolist(),
                'img_shape': (968, 1296, 3),  # 实际图像尺寸
                'depth_shape': (480, 640),   # 实际深度图尺寸
            }
            
            return mmdet3d_info
            
        except Exception as e:
            print(f'Error loading {data_info["pkl_path"]}: {str(e)}')
            return None
    
    def get_data_infos(self):
        """获取所有数据信息"""
        if self.workers <= 1:
            data_infos = []
            # 添加单线程数据加载进度条
            data_progress = tqdm(self.data_infos, desc=f'Loading {self.split} data', unit='sample')
            for data_info in data_progress:
                result = self._load_single_data(data_info)
                if result is not None:
                    data_infos.append(result)
                # 更新进度条信息
                data_progress.set_postfix({
                    'loaded': len(data_infos),
                    'failed': len(self.data_infos) - len(data_infos) - (len(self.data_infos) - data_progress.n)
                })
        else:
            data_infos = []
            # 添加多线程数据加载进度条
            data_progress = tqdm(total=len(self.data_infos), desc=f'Loading {self.split} data (multi-thread)', unit='sample')
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # 使用submit和as_completed来支持进度条更新
                from concurrent.futures import as_completed
                future_to_data = {executor.submit(self._load_single_data, data_info): data_info for data_info in self.data_infos}
                
                for future in as_completed(future_to_data):
                    result = future.result()
                    if result is not None:
                        data_infos.append(result)
                    data_progress.update(1)
                    data_progress.set_postfix({
                        'loaded': len(data_infos),
                        'failed': data_progress.n - len(data_infos)
                    })
        
        return data_infos


def create_scannet_occ_info_file(data_path, pkl_prefix='scannet_occ', save_path=None, workers=4):
    """Create ScanNet OCC information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'scannet_occ'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path), f'Data path {data_path} does not exist'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path), f'Save path {save_path} does not exist'

    # 创建训练、测试数据集信息（8:2划分）
    splits = ['train', 'test']
    
    # 添加整体进度条
    overall_progress = tqdm(splits, desc='Creating dataset info files', unit='split')
    
    for split in overall_progress:
        overall_progress.set_description(f'Creating {split} dataset info')
        print(f'Creating {split} dataset info...')
        
        # 创建数据集实例
        dataset = ScanNetOccData(root_path=data_path, split=split, workers=workers)
        
        # 获取数据信息
        data_infos = dataset.get_data_infos()
        
        # 保存信息文件
        info_filename = os.path.join(save_path, f'{pkl_prefix}_infos_{split}.pkl')
        print(f'Saving {len(data_infos)} {split} samples to {info_filename}')
        
        # 将数据包装成字典格式
        data_dict = {
            'metainfo': {
                        # ScanNet官方20个语义类别（按照ScanNet官方标签ID顺序）
                        'classes': ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                                'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture'),
                        # ScanNet官方标签ID（与classes对应）
                        'class_ids': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
                        # 为每个类别设置对应的颜色（RGB格式）
                        'palette': [
                            (174, 199, 232),  # 1: wall - 浅蓝色
                            (152, 223, 138),  # 2: floor - 浅绿色
                            (31, 119, 180),   # 3: cabinet - 蓝色
                            (255, 187, 120),  # 4: bed - 橙色
                            (188, 189, 34),   # 5: chair - 黄绿色
                            (140, 86, 75),    # 6: sofa - 棕色
                            (255, 152, 150),  # 7: table - 粉红色
                            (214, 39, 40),    # 8: door - 红色
                            (197, 176, 213),  # 9: window - 紫色
                            (148, 103, 189),  # 10: bookshelf - 深紫色
                            (196, 156, 148),  # 11: picture - 灰褐色
                            (23, 190, 207),   # 12: counter - 青色
                            (247, 182, 210),  # 13: desk - 浅粉色
                            (219, 219, 141),  # 14: curtain - 浅黄色
                            (255, 127, 14),   # 15: refrigerator - 深橙色
                            (158, 218, 229),  # 16: showercurtrain - 浅蓝色
                            (44, 160, 44),    # 17: toilet - 绿色
                            (112, 128, 144),  # 18: sink - 灰色
                            (227, 119, 194),  # 19: bathtub - 粉色
                            (82, 84, 163)     # 20: otherfurniture - 深蓝色
                        ],
                        # 有效的类别ID（ScanNet官方标签ID）
                        'seg_valid_class_ids': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
                        # 所有类别ID（0-19）
                        'seg_all_class_ids': tuple(range(19)),
                        # 忽略索引（255表示未知/忽略的体素）
                        'ignore_index': 255
                    },
            'data_list': data_infos
        }
        
        # 添加保存进度条
        with tqdm(total=1, desc=f'Saving {split} info file', unit='file', leave=False) as save_progress:
            with open(info_filename, 'wb') as f:
                pickle.dump(data_dict, f)
            save_progress.update(1)
        
        print(f'Successfully created {info_filename} with {len(data_infos)} samples')
        overall_progress.set_postfix({
            'completed': f'{split} ({len(data_infos)} samples)'
        })
    
    print('Successfully created ScanNet OCC dataset info files')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ScanNet OCC dataset info files')
    parser.add_argument('--data_path',default='/data/tangqiansong/raw_data/scannet_occ/', type=str, required=True,
                        help='Path to the ScanNet OCC dataset root directory')
    parser.add_argument('--save_path', type=str, default='/data/tangqiansong/raw_data/scannet_occ',
                        help='Path to save the info files (default: same as data_path)')
    parser.add_argument('--pkl_prefix', type=str, default='scannet_occ',
                        help='Prefix for the pkl files (default: scannet_occ)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers for parallel processing (default: 4)')
    
    args = parser.parse_args()
    
    # 执行数据集转换
    create_scannet_occ_info_file(
        data_path=args.data_path,
        pkl_prefix=args.pkl_prefix,
        save_path=args.save_path,
        workers=args.workers
    )
