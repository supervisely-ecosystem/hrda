# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Default HRDA Configuration for GTA->Cityscapes
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/uda_cracks_test.py',
    # DAFormer Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 1
# HRDA Configuration
model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1,
        num_classes=2),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=[256, 256],
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[256, 256],
        crop_size=[512, 512]))
data = dict(
    train=dict(
        # Rare Class Sampling
        # rare_class_sampling=dict(
        #     min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        target=dict(crop_pseudo_margins=[10, 10, 10, 10]),
    ))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=1e-04,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
# gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=25000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=2500, max_keep_ckpts=-1, save_optimizer=False)
evaluation = dict(interval=500, metric='mIoU')
# Meta Information for Result Analysis
name = 'hrda_v2'
exp = 'basic'
name_dataset = 'v7_style'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

# For the other configurations used in the paper, please refer to experiment.py
