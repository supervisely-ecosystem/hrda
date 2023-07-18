import mmcv
from tools import train as train_cli
from src import sly_dataset, state
from src import globals as g
from src.globals import state
import supervisely as sly


def train():
    sly_dataset.download_datasets(g.PROJECT_ID)
    sly_dataset.prepare_datasets()
    cfg = mmcv.Config.fromfile("configs/supervisely/base.py")
    update_config(cfg)
    cfg.dump("config.py")
    train_cli.main(["config.py"])


def update_config(cfg):
    cfg.data.train.source.dataset_name = state.source_dataset
    cfg.data.train.target.dataset_name = state.target_dataset
    cfg.data.val.dataset_name = state.val_dataset
    cfg.data.test.dataset_name = state.val_dataset
    cfg.model.decode_head.num_classes = len(state.classes) + 1
    cfg.log_config.interval = 1


sly.fs.remove_dir("app_data")
train()
