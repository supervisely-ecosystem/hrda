# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'SuperviselyDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[768,768],
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=[1024,1024], pad_val=0, seg_pad_val=255),
            dict(type='Pad', size_divisor=16),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val=dict(
        type='SuperviselyDataset',
        data_root="data/cracks_diverse_set_seg/ds-splitted-part-1",
        img_dir='img',
        ann_dir='seg2',
        img_suffix='.jpg', seg_map_suffix='.jpg.png',
        pipeline=test_pipeline),
    test=dict(
        type='SuperviselyDataset',
        # data_root=data_root+"/validation_test",
        data_root="data/cracks_diverse_set_seg/ds-splitted-part-1",
        img_dir='img',
        ann_dir='seg2',
        img_suffix='.jpg', seg_map_suffix='.jpg.png',
        pipeline=test_pipeline))
