batch_size = 2
num_workers = 1
crop_pseudo_margins = [10, 10, 10, 10]
img_scale = (640, 640)
crop_size = (512, 512)
augs_config = ""


# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

source_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="SlyImgAugs"),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
target_train_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type='LoadAnnotations'),
    # dict(type="SlyImgAugs"),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=16),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_workers,
    train=dict(
        type="UDADataset",
        source=dict(
            type="SuperviselyDataset",
            pipeline=source_train_pipeline,
        ),
        # valid_mask_size = [512, 512] (default)
        target=dict(
            type="SuperviselyDataset",
            test_mode=True,
            pipeline=target_train_pipeline,
        ),
    ),
    val=dict(
        type="SuperviselyDataset",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="SuperviselyDataset",
        pipeline=test_pipeline,
    ),
)
