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

start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

iter_progress = Progress("Iterations", hide_on_finish=False)
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


@start_train_btn.click
def start_train():
    g.state.stop_training = False
    stop_train_btn.enable()
    iter_progress.show()
    iter_progress(message="Preparing the model and data...", total=1)
    monitoring.clean_up()

    if sly.is_development():
        sly.fs.remove_dir("app_data")

    try:
        train.train()
    except StopIteration as exc:
        sly.logger.info("The training was stopped.")

    gc.collect()
    torch.cuda.empty_cache()


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
