batch_size = 2
num_workers = 2
crop_pseudo_margins = [10, 10, 10, 10]
img_scale = (640, 640)
crop_size = (512, 512)
data_root = "data/cracks_seg"
augs_config = ""


# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

source_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="SlyImgAugs"),
    dict(type="Resize", img_scale=img_scale),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
target_train_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type='LoadAnnotations'),
    # dict(type="SlyImgAugs"),
    dict(type="Resize", img_scale=img_scale),
    dict(type="RandomCrop", crop_size=crop_size),
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
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomCrop", crop_size=crop_size),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
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
            data_root=data_root + "/cracks_synth",
            img_dir="img",
            ann_dir="seg2",
            img_suffix=".jpg",
            seg_map_suffix=".jpg.png",
            pipeline=source_train_pipeline,
        ),
        # valid_mask_size = [512, 512] (default)
        target=dict(
            type="SuperviselyDataset",
            data_root=data_root + "/training",
            img_dir="img",
            ann_dir="seg2",
            img_suffix=".jpg",
            seg_map_suffix=".jpg.png",
            test_mode=True,
            pipeline=target_train_pipeline,
            crop_pseudo_margins=crop_pseudo_margins,
        ),
    ),
    val=dict(
        type="SuperviselyDataset",
        data_root=data_root + "/validation",
        img_dir="img",
        ann_dir="seg2",
        img_suffix=".jpg",
        seg_map_suffix=".jpg.png",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="SuperviselyDataset",
        data_root=data_root + "/validation",
        img_dir="img",
        ann_dir="seg2",
        img_suffix=".jpg",
        seg_map_suffix=".jpg.png",
        pipeline=test_pipeline,
    ),
)
