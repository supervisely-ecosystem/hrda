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
    Field,
    ImagePairSequence
)
import torch

import src.globals as g
from src import train
from src.monitoring import Monitoring, StageMonitoring
from src import sly_utils
from src.ui.utils import HContainer

start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()
finish_btn = Button("Save & finish", "danger")
finish_btn.hide()

iter_progress = Progress("", hide_on_finish=False)
g.iter_progress = iter_progress
iter_progress.hide()

success_msg = DoneLabel(
    "The training completed. All checkpoints and logs were uploaded to Team Files."
)
success_msg.hide()

folder_thumb = FolderThumbnail()
folder_thumb.hide()

btn_container = HContainer(
    [start_train_btn, stop_train_btn, finish_btn],
    overflow="wrap",
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
prediction_preview = ImagePairSequence(opacity=0.6, enable_zoom=True, slider_title="")
prediction_preview.hide()
prediction_preview_f = Field(
    prediction_preview,
    "Prediction visualizations",
    "After each validation we will draw prediction for the first image in val dataset.",
)
prediction_preview_f.hide()

container = Container(
    [
        success_msg,
        folder_thumb,
        btn_container,
        iter_progress,
        monitoring.compile_monitoring_container(hide=True),
        prediction_preview_f,
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
    prediction_preview_f.show()


def reset_buttons():
    stop_train_btn.disable()
    start_train_btn.enable()
    iter_progress.hide()


def show_done_widgets(file_info):
    folder_thumb.set(info=file_info)
    folder_thumb.show()
    success_msg.show()
    iter_progress.hide()


def upload_and_finish():
    # prepare work_dir for uploading
    sly.fs.remove_dir(f"{g.WORK_DIR}/class_mix_debug")
    sly.fs.silent_remove(f"{g.WORK_DIR}/latest.pth")
    sly_utils.save_augs_config(g.state.augs_config_path, g.WORK_DIR)
    sly_utils.save_open_app_lnk(g.WORK_DIR)

    # upload work_dir
    out_path = sly_utils.upload_artifacts(
        g.WORK_DIR, g.state.experiment_name, progress_widget=g.iter_progress
    )
    config_file_info = g.api.file.get_info_by_path(g.TEAM_ID, f"/{out_path}/config.py")

    show_done_widgets(config_file_info)

    if sly.is_production():
        # set link to artifacts in workspace tasks
        g.api.task.set_output_directory(g.api.task_id, config_file_info.id, out_path)
        g.app.stop()


def update_prediction_preview(img_path: str, ann_pred: sly.Annotation, ann_gt: sly.Annotation):
    # copy to static dir
    fname = os.path.basename(img_path)
    dst_path = g.STATIC_DIR + "/" + fname
    static_path = "static/" + fname
    sly.fs.copy_file(img_path, dst_path)
    prediction_preview.append_left(static_path, ann=ann_gt, title=f"Ground Truth ({fname})")
    prediction_preview.append_right(static_path, ann=ann_pred, title=f"Prediction ({fname})")

    if os.environ.get("LOG_LEVEL", "DEBUG") == "DEBUG":
        img = sly.image.read(dst_path)
        ann_pred.draw_pretty(img, thickness=0)
        sly.image.write(f"{g.WORK_DIR}/debug_ann_pred.jpg", img)

    if prediction_preview.is_hidden():
        prediction_preview.show()


def clear_working_dirs():
    sly.fs.remove_dir(g.app_dir)


@start_train_btn.click
def start_train():
    g.state.stop_training = False
    stop_train_btn.enable()
    iter_progress.show()
    monitoring.clean_up()
    prediction_preview.clean_up()
    finish_btn.hide()

    # currently we can't clear the app_dir, due to conflicts in AugmentationsWithTabs
    # clear_working_dirs()

    no_errors = True
    try:
        train.train()
    except StopIteration as exc:
        sly.logger.info("The training was stopped.")
    except Exception as exc:
        no_errors = False
        raise exc
    finally:
        reset_buttons()
        # free cuda memory
        gc.collect()
        torch.cuda.empty_cache()

    if no_errors:
        upload_and_finish()
    else:
        finish_btn.show()


@stop_train_btn.click
def stop_train():
    g.state.stop_training = True
    stop_train_btn.disable()


@finish_btn.click
def on_finish():
    upload_and_finish()
