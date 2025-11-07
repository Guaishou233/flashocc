# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import pickle

from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset


@DATASETS.register_module()
class ScanNetOccDataset(Custom3DDataset):
    r"""ScanNet OCC Dataset for Occupancy Prediction Task.

    This class serves as the API for experiments on the ScanNet OCC Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for data. Defaults to
            dict(img='posed_images', depth='posed_images').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=True, use_lidar=False).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to False.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        # ScanNet 21个类别：0-19为语义类别，20为255映射的未知类别（与配置文件保持一致）
        'classes': ("empty","ceiling","floor","wall","window","chair","bed","sofa","table","tvs","furn","objs",),
        # 类别ID（与classes对应，0-19为语义类别，20为255映射的未知类别）
        'class_ids': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        # 为每个类别设置对应的颜色（RGB格式）
        'palette': [
            (174, 199, 232),  # 0: empty - 浅蓝色
            (152, 223, 138),  # 1: ceiling - 浅绿色
            (31, 119, 180),   # 2: floor - 蓝色
            (255, 187, 120),  # 3: wall - 橙色
            (188, 189, 34),   # 4: window - 黄绿色
            (140, 86, 75),    # 5: chair - 棕色
            (255, 152, 150),  # 6: bed - 粉红色
            (214, 39, 40),    # 7: sofa - 红色
            (197, 176, 213),  # 8: table - 紫色
            (148, 103, 189),  # 9: tvs - 深紫色
            (196, 156, 148),  # 10: furn - 灰褐色
            (23, 190, 207),   # 11: objs - 青色
        ],
        # 有效的类别ID（0-20，包括unknown类别）
        'seg_valid_class_ids': (0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        # 所有类别ID（0-20）
        'seg_all_class_ids': tuple(range(12)),
        # 忽略索引（255表示未知/忽略的体素，在训练时映射到类别20）
        'ignore_index': None
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     img='posed_images',
                     depth='posed_images'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = False,
                 test_mode: bool = False,
                 classes: Optional[Union[tuple, list]] = None,
                 **kwargs) -> None:

        # 处理 metainfo 参数
        if metainfo is not None:
            # 如果提供了 metainfo，用它更新 METAINFO
            self.METAINFO.update(metainfo)
        
        # 保存classes参数（如果提供的话）
        if classes is not None:
            self.classes = classes
        else:
            # 如果没有提供classes，使用METAINFO中的默认值
            self.classes = self.METAINFO['classes']
        
        # 创建标签映射：0-11保持不变，255映射到0（empty类别）
        # 标签映射数组大小为256（0-255）
        seg_label_mapping = np.arange(256, dtype=np.int64)
        # 对于超出有效范围的标签（>=12且!=255），映射到0
        seg_max_cat_id = max(self.METAINFO['seg_all_class_ids'])  # 11
        seg_label_mapping[(seg_label_mapping > seg_max_cat_id) & (seg_label_mapping != 255)] = 0
        # 将255映射到0（empty类别）
        seg_label_mapping[255] = 0
        self.seg_label_mapping = seg_label_mapping

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            classes=classes,
            **kwargs)

        # 设置标签映射信息
        self.METAINFO['seg_label_mapping'] = self.seg_label_mapping

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str or BufferedReader): Path of the annotation file or file handle.

        Returns:
            list[dict]: List of annotations.
        """
        # 处理父类传递的BufferedReader对象
        if hasattr(ann_file, 'read'):
            # ann_file是一个文件句柄（BufferedReader）
            data_dict = pickle.load(ann_file)
        else:
            # ann_file是一个文件路径字符串
            ann_file_path = osp.join(self.data_root, ann_file)
            with open(ann_file_path, 'rb') as f:
                data_dict = pickle.load(f)
        
        # 返回data_list
        return data_dict['data_list']

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Args:
            idx (int): Index of the data.

        Returns:
            dict: Data info of the corresponding index.
        """
        if idx >= len(self.data_infos):
            raise IndexError(f'Index {idx} out of range for data_infos of length {len(self.data_infos)}')
            
        data_info = self.data_infos[idx]
        
        # 构建标准的数据字典
        input_dict = dict(
            sample_idx=data_info.get('scene_name', f'scene_{idx}'),
            frame_id=data_info.get('frame_id', idx),
            img_path=data_info['img_path'],
            depth_path=data_info['depth_path'],
            pkl_path=data_info['pkl_path'],
            img_shape=data_info['img_shape'],
            depth_shape=data_info['depth_shape'],
        )
        
        return input_dict

    def get_ann_info(self, idx: int) -> dict:
        """Get annotation info by index.

        Args:
            idx (int): Index of the annotation data to get.

        Returns:
            dict: Annotation info.
        """
        data_info = self.data_infos[idx]
        
        # 加载pkl文件中的完整数据
        pkl_path = data_info['pkl_path']
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        ann_info = dict()
        
        # 语义占用标签作为GT - 直接使用ScanNet原始语义标签
        ann_info['gt_occ_1_4'] = pkl_data['target_1_4']  # 高分辨率语义标签（ScanNet原始标签ID）
        ann_info['gt_occ_1_16'] = pkl_data['target_1_16']  # 低分辨率语义标签（ScanNet原始标签ID）
        
        # 体素原点
        ann_info['voxel_origin'] = pkl_data['voxel_origin']
        
        # 相机参数
        ann_info['cam_pose'] = pkl_data['cam_pose']
        ann_info['intrinsic'] = pkl_data['intrinsic']
        
        # 标签映射信息
        ann_info['seg_label_mapping'] = self.seg_label_mapping
        
        return ann_info


    def _filter_data(self) -> List[dict]:
        """Filter data according to filter_cfg.

        Returns:
            List[dict]: Filtered data list.
        """
        if not self.filter_empty_gt:
            return self.data_list
        
        # 对于占用预测任务，通常不过滤数据
        # 因为每个场景都有有效的占用标签
        return self.data_list

    def __getitem__(self, idx: int) -> dict:
        """Get item from infos according to the given index.

        Args:
            idx (int): Index of the data.

        Returns:
            dict: Data dict of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, idx: int) -> Union[dict, None]:
        """Prepare training data.

        Args:
            idx (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(idx)
        if input_dict is None:
            return None
        
        # 获取标注信息
        ann_info = self.get_ann_info(idx)
        input_dict['ann_info'] = ann_info
        
        # 应用数据变换管道
        if self.pipeline is not None:
            input_dict = self.pipeline(input_dict)
        
        return input_dict

    def prepare_test_data(self, idx: int) -> dict:
        """Prepare testing data.

        Args:
            idx (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(idx)
        
        # 应用数据变换管道
        if self.pipeline is not None:
            input_dict = self.pipeline(input_dict)
        
        return input_dict

    def evaluate(self,
                 occ_results: List[dict],
                 metric: Union[str, List[str]] = 'mIoU',
                 logger=None,
                 show_dir: Optional[str] = None,
                 **eval_kwargs) -> dict:
        """Evaluate occupancy on ScanNet with ScanNet-specific preprocessing.

        参考 NuScenes 的评估流程：逐样本载入 GT，做与训练一致的体素缩放，
        然后累计 mIoU 指标；同时支持将结果可视化保存。

        Args:
            occ_results (List[dict]): 测试得到的占用结果；元素可为 dict(含 'pred_occ') 或直接为体素预测。
            metric (Union[str, List[str]]): 指标名，默认 'mIoU'。
            logger (logging.Logger | str, optional): 日志。
            show_dir (str, optional): 可视化保存目录。
            **eval_kwargs: 其他透传参数。

        Returns:
            dict: 评估结果字典。
        """
        # 统一 metric 为字符串
        if isinstance(metric, list):
            metric_name = metric[0] if len(metric) > 0 else 'mIoU'
        else:
            metric_name = metric

        # 仅实现 mIoU（占用评估主用）；如需 ray-iou 可后续扩展
        if metric_name not in ['mIoU', 'IoU']:
            raise ValueError(f'Unsupported metric for ScanNetOccDataset: {metric_name}')

        # 延迟导入，避免不必要的依赖开销
        from ..core.evaluation.occ_metrics import Metric_mIoU
        import torch
        import os
        import numpy as np
        import mmcv

        # 使用METAINFO中定义的类别数（21个类别：0-19语义类别 + 20未知类别）
        num_classes = len(self.METAINFO.get('classes', []))
        eval_class_names = list(self.METAINFO.get('classes', []))

        occ_eval_metrics = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=True)
        # 覆盖类名用于打印
        occ_eval_metrics.class_names = eval_class_names

        # 遍历每个样本：载入 GT，直接使用原始标签形状，累计评估
        for index, occ_pred in enumerate(occ_results):
            data_info = self.data_infos[index]

            # 1) 读取 pkl 获取 GT 源数据，直接使用原始形状
            pkl_path = data_info['pkl_path']
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
            gt_semantics = pkl_data['target_1_4']  # 使用原始形状，ScanNet 原始语义ID

            # 2) 处理GT标签：将255映射到0（与训练时一致），其他标签保持不变
            gt_semantics_np = np.asarray(gt_semantics, dtype=np.int64)
            # 将255映射到0（empty类别），其他0-11保持不变
            # 对于超出范围的标签（>=12且!=255），映射到0
            gt_semantics_np = gt_semantics_np.copy()
            gt_semantics_np[gt_semantics_np == 255] = 0  # 将255映射到类别0
            # 将无效标签（<0或>=12）映射到0
            invalid_mask = (gt_semantics_np < 0) | (gt_semantics_np >= num_classes)
            gt_semantics_np[invalid_mask] = 0
            
            # 构造与标签形状相同的 mask（全部有效）
            mask_lidar_np = np.ones_like(gt_semantics_np, dtype=bool)
            mask_camera_np = np.ones_like(gt_semantics_np, dtype=bool)

            # 3) 取预测体素（支持 dict 或直接数组），并转换为 (Dx,Dy,Dz) 的整型类别
            if isinstance(occ_pred, dict) and 'pred_occ' in occ_pred:
                pred_occ = occ_pred['pred_occ']
            else:
                pred_occ = occ_pred

            if hasattr(pred_occ, 'detach'):
                pred_occ = pred_occ.detach()
            if hasattr(pred_occ, 'cpu'):
                pred_occ = pred_occ.cpu()
            pred_occ_t = torch.as_tensor(pred_occ)

            # 自适应处理预测形状：
            #  - 若形如 (C,Dx,Dy,Dz) 或 (Dx,Dy,Dz,C)，取 argmax 得到类别索引
            #  - 若本就是 (Dx,Dy,Dz) 则直接使用
            # 注意：模型输出是21个类别（0-20），需要根据实际通道数判断
            if pred_occ_t.dim() == 4:
                # (C,Dx,Dy,Dz) or (Dx,Dy,Dz,C)
                # 检查通道数：模型输出21个类别
                if pred_occ_t.shape[0] == num_classes:
                    pred_occ_t = pred_occ_t.argmax(dim=0)
                elif pred_occ_t.shape[-1] == num_classes:
                    pred_occ_t = pred_occ_t.argmax(dim=-1)
                else:
                    # 无法判断通道维，默认按最后一维 argmax
                    pred_occ_t = pred_occ_t.argmax(dim=-1)
            elif pred_occ_t.dim() != 3:
                # 非期望形状，尝试 squeeze 到 3D
                pred_occ_t = pred_occ_t.squeeze()
                if pred_occ_t.dim() == 4:
                    if pred_occ_t.shape[0] == num_classes:
                        pred_occ_t = pred_occ_t.argmax(dim=0)
                    elif pred_occ_t.shape[-1] == num_classes:
                        pred_occ_t = pred_occ_t.argmax(dim=-1)
                
            pred_occ_t = pred_occ_t.to(torch.long)
            # 将类别索引裁剪到有效范围（0-20）
            # 超出范围的设置为255以便评估时忽略
            pred_occ_t = torch.clamp(pred_occ_t, 0, num_classes - 1)

            # 4) 累计 mIoU（occ_metrics 期望 GT 为 (Dx,Dy,Dz)，Pred 为同维类别索引）
            # 预测体素转为 numpy（安全）：
            pred_np = pred_occ_t.numpy() if hasattr(pred_occ_t, 'numpy') else np.asarray(pred_occ_t)
            # GT 和 mask 已经是 numpy 数组，直接使用

            occ_eval_metrics.add_batch(
                pred_np,
                gt_semantics_np,
                mask_lidar_np,
                mask_camera_np
            )

            # 5) 可视化保存（可选）
            if show_dir is not None:
                mmcv.mkdir_or_exist(show_dir)
                scene_name = data_info.get('scene_name', f'scene_{index}')
                frame_id = data_info.get('frame_id', index)
                out_dir = osp.join(show_dir, str(scene_name), str(frame_id))
                mmcv.mkdir_or_exist(out_dir)
                save_path = osp.join(out_dir, 'pred_gt.npz')
                np.savez_compressed(
                    save_path,
                    pred=pred_np,
                    gt=gt_semantics_np
                )

        eval_results = occ_eval_metrics.count_miou()
        return eval_results

    def vis_occ(self, semantics):
        """简单的 BEV 可视化（借鉴 NuScenes 版本）。

        Args:
            semantics (np.ndarray): (Dx,Dy,Dz) 的体素语义类别。

        Returns:
            np.ndarray: (H,W,3) 的可视化图。
        """
        import numpy as np
        import torch
        import cv2

        Dx, Dy, Dz = semantics.shape
        # 选择每个 (x,y) 列上的一个代表层（这里选最大 z 索引）
        d = np.arange(Dz).reshape(1, 1, Dz).astype(np.float32)
        d = np.repeat(d, Dx, axis=0)
        d = np.repeat(d, Dy, axis=1)
        selected = np.argmax(d, axis=2)  # 形状 (Dx,Dy)

        semantics_torch = torch.from_numpy(semantics)
        selected_torch = torch.from_numpy(selected)
        occ_bev_torch = torch.gather(semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy().astype(np.int32).flatten()

        # 简单调色板：使用 METAINFO 中 palette；不足则循环
        palette = self.METAINFO.get('palette', [])
        if not palette:
            palette = [(i * 37 % 255, i * 59 % 255, i * 83 % 255) for i in range(256)]
        palette_np = np.array(palette + palette[: max(0, 256 - len(palette))], dtype=np.uint8)
        occ_bev_vis = palette_np[occ_bev][:, :3]
        occ_bev_vis = occ_bev_vis.reshape(Dx, Dy, 3)[::-1, ::-1, :]
        occ_bev_vis = cv2.resize(occ_bev_vis, (400, 400))
        return occ_bev_vis

    def _evaluate_miou(self, results: List[dict]) -> float:
        """Evaluate mean IoU.

        Args:
            results (List[dict]): Testing results.

        Returns:
            float: Mean IoU.
        """
        # 计算各类别IoU
        iou_results = self._evaluate_iou(results)
        # 计算平均IoU
        valid_ious = [iou for iou in iou_results.values() if iou > 0]
        if len(valid_ious) > 0:
            return sum(valid_ious) / len(valid_ious)
        return 0.0

    def _evaluate_iou(self, results: List[dict]) -> dict:
        """Evaluate IoU for each class.

        Args:
            results (List[dict]): Testing results.

        Returns:
            dict: IoU for each class.
        """
        # 实现各类别IoU计算逻辑
        class_names = self.METAINFO['classes']
        iou_results = {}
        
        # 这里需要根据具体的预测结果格式来实现
        # 目前返回占位符值
        for class_name in class_names:
            iou_results[class_name] = 0.0
            
        return iou_results

    def _evaluate_accuracy(self, results: List[dict]) -> float:
        """Evaluate accuracy.

        Args:
            results (List[dict]): Testing results.

        Returns:
            float: Accuracy.
        """
        # 实现准确率计算逻辑
        # 这里需要根据具体的预测结果格式来实现
        return 0.0

    def _evaluate_map(self, results: List[dict]) -> float:
        """Evaluate mean Average Precision (mAP).

        Args:
            results (List[dict]): Testing results.

        Returns:
            float: Mean Average Precision.
        """
        # 对于占用预测任务，mAP 通过计算每个类别的平均精度来实现
        class_names = self.METAINFO['classes']
        num_classes = len(class_names)
        
        # 收集所有预测和真实标签
        all_predictions = []
        all_targets = []
        
        for result in results:
            # 假设结果格式为 {'pred_occ': tensor, 'gt_occ': tensor}
            if 'pred_occ' in result and 'gt_occ' in result:
                pred_occ = result['pred_occ']
                gt_occ = result['gt_occ']
                
                # 如果是tensor，转换为numpy
                if hasattr(pred_occ, 'cpu'):
                    pred_occ = pred_occ.cpu().numpy()
                if hasattr(gt_occ, 'cpu'):
                    gt_occ = gt_occ.cpu().numpy()
                
                all_predictions.append(pred_occ.flatten())
                all_targets.append(gt_occ.flatten())
        
        if not all_predictions:
            return 0.0
        
        # 合并所有预测和标签
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # 计算每个类别的AP
        aps = []
        for class_id in range(num_classes):
            # 创建二分类标签：当前类别 vs 其他类别
            binary_targets = (all_targets == class_id).astype(int)
            
            if np.sum(binary_targets) == 0:
                # 如果该类别在真实标签中不存在，跳过
                continue
            
            # 对于占用预测，我们使用预测概率作为置信度
            # 这里简化处理，使用预测类别作为置信度
            binary_predictions = (all_predictions == class_id).astype(int)
            
            # 计算该类别的AP
            ap = self._compute_ap(binary_predictions, binary_targets)
            aps.append(ap)
        
        # 计算平均AP
        if aps:
            return np.mean(aps)
        else:
            return 0.0

    def _compute_ap(self, predictions, targets, num_thresholds=11):
        """Compute Average Precision for binary classification.

        Args:
            predictions (np.ndarray): Binary predictions.
            targets (np.ndarray): Binary targets.
            num_thresholds (int): Number of thresholds to evaluate.

        Returns:
            float: Average Precision.
        """
        # 对于二分类，我们使用不同的阈值来计算AP
        thresholds = np.linspace(0, 1, num_thresholds)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            # 应用阈值
            pred_binary = (predictions >= threshold).astype(int)
            
            # 计算TP, FP, FN
            tp = np.sum((pred_binary == 1) & (targets == 1))
            fp = np.sum((pred_binary == 1) & (targets == 0))
            fn = np.sum((pred_binary == 0) & (targets == 1))
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算AP (使用梯形积分)
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        
        return ap
