from mmseg.datasets.sly_dataset import SuperviselyDataset

classes = ["bg", "cracks"]
palette = [[0,0,0], [0,255,0]]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(768, 768)),
    dict(type='RandomCrop', crop_size=crop_size),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    # dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
ds = SuperviselyDataset(pipeline=pipeline, data_root="data/cracks_seg/cracks_synth", test_mode=False,
                        img_dir='img', ann_dir='seg2',
                        img_suffix='.jpg', seg_map_suffix='.jpg.png')
print(ds[0]['img'].data.shape)
