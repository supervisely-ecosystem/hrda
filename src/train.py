import warnings
from src import globals as g
from src.globals import state
from src import utils, sly_dataset, sly_utils
import mmcv
from tools import train as train_cli


def train():
    state.update()
    if state.is_custom_model_selected:
        sly_utils.download_custom_model_weights(state.remote_weights_path, g.iter_progress)
    sly_dataset.download_datasets(g.PROJECT_ID, state.selected_dataset_ids)
    sly_dataset.prepare_datasets(state.classes)
    # TODO: Can we don't import training_ui?
    g.iter_progress(message="Training initialization...", total=1)
    cfg = mmcv.Config.fromfile("configs/supervisely/base.py")
    update_config(cfg)
    cfg.dump("config.py")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        train_cli.main(["config.py"])


def update_config(cfg):
    general_params = state.general_params
    checkpoint_params = state.checkpoint_params
    optimizer_params = state.optimizer_params
    input_size = (
        utils.round_to_divisor(general_params.shorter_input_size, 16),
        utils.round_to_divisor(general_params.longer_input_size, 16),
    )

    # General
    cfg.work_dir = g.WORK_DIR
    cfg.log_config.interval = g.LOG_INTERVAL

    # Runtime
    cfg.runner.max_iters = general_params.total_iters
    cfg.evaluation.interval = general_params.val_interval
    cfg.data.samples_per_gpu = general_params.batch_size_train
    cfg.data.workers_per_gpu = utils.get_num_workers(general_params.batch_size_train)

    # Checkpoints
    checkpoint_cfg = cfg.checkpoint_config
    checkpoint_cfg.interval = checkpoint_params.checkpoint_interval
    checkpoint_cfg.max_keep_ckpts = checkpoint_params.max_keep_ckpts
    checkpoint_cfg.save_optimizer = checkpoint_params.save_optimizer

    # Datasets
    cfg.data.train.source.dataset_name = state.source_dataset
    cfg.data.train.target.dataset_name = state.target_dataset
    cfg.data.val.dataset_name = state.val_dataset
    cfg.data.test.dataset_name = state.val_dataset

    # Pipelines
    source_pipeline = cfg.data.train.source.pipeline
    target_pipeline = cfg.data.train.target.pipeline
    val_pipeline = cfg.data.val.pipeline
    test_pipeline = cfg.data.test.pipeline
    source_pipeline[3].img_scale = input_size
    source_pipeline[4].crop_size = input_size
    source_pipeline[7].size = input_size
    target_pipeline[1].img_scale = input_size
    target_pipeline[2].crop_size = input_size
    target_pipeline[5].size = input_size
    val_pipeline[1].img_scale = input_size
    test_pipeline[1].img_scale = input_size

    # Model
    # hr_crop_size will be input_size//2
    hr_crop_size = [input_size[0] // 2, input_size[1] // 2]
    stride = [input_size[0] // 2, input_size[1] // 2]
    num_classes = len(state.classes) + 1
    hr_loss_weight = 0.1
    model = cfg.model
    model.decode_head.num_classes = num_classes
    model.decode_head.hr_loss_weight = hr_loss_weight
    model.hr_crop_size = hr_crop_size
    model.test_cfg.crop_size = input_size
    model.test_cfg.stride = stride
    model.pretrained = g.PRETRAINED_PATH

    cfg.load_from = state.local_weights_path

    # Optimizer
    opt = cfg.optimizer
    opt.type = optimizer_params.optimizer_type
    opt.lr = optimizer_params.base_lr
    opt.weight_decay = optimizer_params.weight_decay

    # LR Scheduler
    lr_scheduler = cfg.lr_config
    lr_scheduler.warmup_iters = optimizer_params.warmup_iters
    lr_scheduler.power = optimizer_params.scheduler_power
