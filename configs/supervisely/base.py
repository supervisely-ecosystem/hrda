# Base HRDA Configuration
_base_ = [
    "../_base_/default_runtime.py",
    # DAFormer Network Architecture
    "../_base_/models/daformer_sepaspp_mitb5.py",
    # custom dataset
    "uda_dataset.py",
    # DAFormer Self-Training
    "../_base_/uda/dacs.py",
]


# Runtime
total_iters = 25000
val_interval = 500
checkpoint_interval = 2500
max_keep_ckpts = -1
save_optimizer = False

n_gpus = 1
runner = dict(type="IterBasedRunner", max_iters=total_iters)
evaluation = dict(interval=val_interval, metric="mIoU")
checkpoint_config = dict(
    by_epoch=False,
    interval=checkpoint_interval,
    max_keep_ckpts=max_keep_ckpts,
    save_optimizer=save_optimizer,
)

custom_hooks = [dict(type="SuperviselyHook")]

# HRDA Configuration
input_size = [512, 512]
hr_crop_size = [input_size[0] // 2, input_size[1] // 2]
stride = [input_size[0] // 2, input_size[1] // 2]
num_classes = 2
hr_loss_weight = 0.1

model = dict(
    type="HRDAEncoderDecoder",
    decode_head=dict(
        type="HRDAHead",
        # Use the DAFormer decoder for each scale.
        single_scale_head="DAFormerHead",
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=hr_loss_weight,
        num_classes=num_classes,
    ),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=hr_crop_size,
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(mode="slide", batched_slide=True, stride=stride, crop_size=input_size),
)


# Optimizer Hyperparameters
optim_type = "AdamW"
base_lr = 1e-4
weight_decay = 0.0001

optimizer_config = None
optimizer = dict(
    type=optim_type,
    lr=base_lr,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0), pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0)
        )
    ),
)


# LR scheduler
warmup_iters = 1000
policy = "poly"

lr_config = dict(
    policy=policy,
    warmup="linear",
    warmup_iters=warmup_iters,
    warmup_ratio=1e-3,
    power=1.0,
    min_lr=1e-7,
    by_epoch=False,
)
