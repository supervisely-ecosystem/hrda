import gc
import os
import supervisely as sly
from supervisely.app.widgets import (
    Card,
    Button,
    Container,
    Progress,
    Empty,
    FolderThumbnail,
    DoneLabel,
    GridGallery,
)
import torch

import src.globals as g
from src import train
from src.monitoring import Monitoring, StageMonitoring
from src import sly_utils

start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

iter_progress = Progress("Iterations", hide_on_finish=False)
g.iter_progress = iter_progress
iter_progress.hide()

success_msg = DoneLabel("Training completed. Training artifacts were uploaded to Team Files.")
success_msg.hide()

folder_thumb = FolderThumbnail()
folder_thumb.hide()

btn_container = Container(
    [start_train_btn, stop_train_btn, Empty()],
    "horizontal",
    overflow="wrap",
    fractions=[1, 1, 10],
    gap=1,
)

# Charts
train_stage = StageMonitoring("train", "Train")
train_stage.create_metric("Loss", ["loss"])
train_stage.create_metric("LR", ["lr"], decimals_in_float=6)
val_stage = StageMonitoring("val", "Validation")
val_stage.create_metric("mIoU", ["mIoU"])
val_stage.create_metric("Per-class IoU")
monitoring = Monitoring()
monitoring.add_stage(train_stage)
monitoring.add_stage(val_stage)

# gp: GridPlot = monitoring._stages["val"]["raw"]
# gp._widgets["Classwise mAP"].hide()


# Prediction preview
prediction_preview = GridGallery(2, enable_zoom=True, sync_views=True)
prediction_preview.hide()

container = Container(
    [
        success_msg,
        folder_thumb,
        btn_container,
        iter_progress,
        monitoring.compile_monitoring_container(hide=True),
        prediction_preview,
    ]
)

card = Card(
    "7️⃣ Training progress",
    "Task progress, detailed logs, metrics charts, and other visualizations",
    content=container,
    lock_message="Select a model to unlock.",
)


def show_train_widgets():
    monitoring.container.show()
    stop_train_btn.enable()
    iter_progress.show()
    prediction_preview.show()


def reset_buttons():
    stop_train_btn.disable()
    start_train_btn.enable()
    iter_progress.hide()


@start_train_btn.click
def start_train():
    g.state.stop_training = False
    stop_train_btn.enable()
    iter_progress.show()
    iter_progress(message="Preparing the model and data...", total=1)
    monitoring.clean_up()

    clear_working_dirs()

    try:
        train.train()
    except StopIteration as exc:
        sly.logger.info("The training was stopped.")
    finally:
        reset_buttons()
        gc.collect()
        torch.cuda.empty_cache()

        # upload artifacts
        sly.fs.silent_remove(f"{g.WORK_DIR}/latest.pth")
        sly_utils.save_augs_config(g.state.augs_config_path, g.WORK_DIR)
        sly_utils.save_open_app_lnk(g.WORK_DIR)
        sly_utils.upload_artifacts(
            g.WORK_DIR, g.state.general_params.experiment_name, progress_widget=g.iter_progress
        )


@stop_train_btn.click
def stop_train():
    g.state.stop_training = True
    stop_train_btn.disable()


def update_prediction_preview(img_path: str, ann_pred: sly.Annotation, ann_gt: sly.Annotation):
    # copy to static dir
    fname = os.path.basename(img_path)
    dst_path = g.STATIC_DIR + "/" + fname
    static_path = "static/" + fname
    sly.fs.copy_file(img_path, dst_path)
    prediction_preview.clean_up()
    prediction_preview.append(static_path, annotation=ann_gt, title=f"Ground Truth ({fname})")
    prediction_preview.append(static_path, annotation=ann_pred, title=f"Prediction ({fname})")

    # TODO: debug
    if sly.is_development():
        img = sly.image.read(dst_path)
        ann_pred.draw_pretty(img, thickness=0)
        sly.image.write("debug_ann_pred.jpg", img)


def clear_working_dirs():
    sly.fs.remove_dir(g.app_dir)
    # sly.fs.remove_dir(g.PROJECT_DIR)
    # sly.fs.remove_dir(g.PROJECT_SEG_DIR)
    # sly.fs.remove_dir(g.app_dir + "/work_dir")
