custom_imports = dict(imports=['tools.mmcv_custom_hooks.swanlab_logger_hook_final'], allow_failed_imports=False)


_base_ = ['../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# For ScanNet we use 21-class detection (0-19为语义类别，20为255映射的未知类别)
class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                   'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'unknown'
]

# ScanNet data configuration
data_config = {
    'cams': ['CAM_FRONT'],  # Single camera for ScanNet
    'Ncams': 1,
    'input_size': (600, 600),
    'src_size': (968, 1296),  # ScanNet image size

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-9.2, 14.8, 0.4],  # (14.8 - (-9.2)) / 0.4 = 24.0 / 0.4 = 60
    'y': [-8.4, 15.6, 0.4],  # (15.6 - (-8.4)) / 0.4 = 24.0 / 0.4 = 60
    'z': [-0.58, 4.58, 5.16],  # interval = 5.16，确保 Dz = 1（与 collapse_z=True 兼容）
    'depth': [0.5, 9.81, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 60




model = dict(
    type='BEVDepthOCC',     # single-frame
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,  # 禁用 checkpoint 以避免与 DDP + find_unused_parameters 的兼容性问题
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 2 + numC_Trans * 4,  # 120 + 240 = 360，匹配input_feature_index=(0, 1)
        out_channels=128,
        input_feature_index=(0, 1),  # 使用第0层和第1层，而不是(0, 2)，避免60不能被8整除的问题
        scale_factor=2),  # 上采样2倍而不是4倍：15 * 2 = 30，与第0层的30匹配
    occ_head=dict(
        type='BEVOCCHead2D_V2',
        in_dim=128,
        out_dim=128,
        Dz=36,
        use_mask=False,
        num_classes=20,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CustomFocalLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
    )
)

# Data
dataset_type = 'ScanNetOccDataset'
data_root = '/data/tangqiansong/raw_data/scannet_occ_mini/'
file_client_args = dict(backend='disk') 


train_pipeline = [
    dict(
        type='LoadScanNetImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(type='LoadScanNetOccGT', grid_config=grid_config),
    dict(type='LoadScanNetDepth', grid_config=grid_config, data_config=data_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='LoadScanNetImageInputs', data_config=data_config, sequential=False),
    dict(type='LoadScanNetOccGT', grid_config=grid_config),
    dict(type='LoadScanNetDepth', grid_config=grid_config, data_config=data_config),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

# Test-time augmentation pipeline
test_pipeline_tta = [
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=[(256, 704)],
        pts_scale_ratio=1.0,
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='LoadScanNetImageInputs', data_config=data_config, sequential=False),
            dict(type='LoadScanNetOccGT', grid_config=grid_config),
            dict(type='LoadScanNetDepth', grid_config=grid_config),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                        'mask_lidar', 'mask_camera'])
        ]
    )
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    filter_empty_gt=False,
)

test_data_config = dict(
    pipeline=test_pipeline_tta,  # Use TTA pipeline
    ann_file=data_root + 'scannet_occ_infos_test.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'scannet_occ_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='Depth'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=10)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SwanLabLoggerHook',
        project='flashocc-scannet',
        run_name='scannet_occ_mini',
        interval=50,
        enable_progress_bar=True
    ),
]

# load_from = "/data/tangqiansong/FlashOCC/ckpts/flashocc-r50-256x704.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=1, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# 添加验证工作流，确保验证钩子被正确调用
workflow = [('train', 1), ('val', 1)]

# 设置 find_unused_parameters=True 以处理未使用的参数（由于修改了 FPN 配置，第2层参数未使用）
# 注意：禁用了 checkpoint (with_cp=False) 以避免兼容性问题
find_unused_parameters = True





# with det pretrain; use_mask=True;
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 6.74
# ===> barrier - IoU = 37.65
# ===> bicycle - IoU = 10.26
# ===> bus - IoU = 39.55
# ===> car - IoU = 44.36
# ===> construction_vehicle - IoU = 14.88
# ===> motorcycle - IoU = 13.4
# ===> pedestrian - IoU = 15.79
# ===> traffic_cone - IoU = 15.38
# ===> trailer - IoU = 27.44
# ===> truck - IoU = 31.73
# ===> driveable_surface - IoU = 78.82
# ===> other_flat - IoU = 37.98
# ===> sidewalk - IoU = 48.7
# ===> terrain - IoU = 52.5
# ===> manmade - IoU = 37.89
# ===> vegetation - IoU = 32.24
# ===> mIoU of 6019 samples: 32.08

# with det pretrain; use_mask=False; class_balance=True
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 4.49
# ===> barrier - IoU = 29.59
# ===> bicycle - IoU = 7.38
# ===> bus - IoU = 30.32
# ===> car - IoU = 32.22
# ===> construction_vehicle - IoU = 13.04
# ===> motorcycle - IoU = 11.91
# ===> pedestrian - IoU = 8.61
# ===> traffic_cone - IoU = 8.11
# ===> trailer - IoU = 7.66
# ===> truck - IoU = 20.84
# ===> driveable_surface - IoU = 48.59
# ===> other_flat - IoU = 26.62
# ===> sidewalk - IoU = 26.08
# ===> terrain - IoU = 20.86
# ===> manmade - IoU = 7.62
# ===> vegetation - IoU = 7.14
# ===> mIoU of 6019 samples: 18.3