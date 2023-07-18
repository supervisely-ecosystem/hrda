import supervisely as sly
from supervisely.app.widgets import (
    Card,
    Button,
    Container,
    Progress,
    Empty,
    FolderThumbnail,
    DoneLabel,
)

import src.globals as g
from src import train
from src.monitoring import Monitoring, StageMonitoring

start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

epoch_progress = Progress("Epochs")
epoch_progress.hide()

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
train_stage.create_metric("Learning Rate", ["lr"], decimals_in_float=6)
val_stage = StageMonitoring("val", "Validation")
val_stage.create_metric("Metrics", "mIoU")
# val_stage.create_metric("Classwise mAP")
monitoring = Monitoring()
monitoring.add_stage(train_stage, True)
monitoring.add_stage(val_stage, True)


# gp: GridPlot = monitoring._stages["val"]["raw"]
# gp._widgets["Classwise mAP"].hide()


container = Container(
    [
        success_msg,
        folder_thumb,
        btn_container,
        epoch_progress,
        iter_progress,
        # monitoring.compile_monitoring_container(True),
    ]
)

card = Card(
    "7️⃣ Training progress",
    "Task progress, detailed logs, metrics charts, and other visualizations",
    content=container,
    lock_message="Select a model to unlock.",
)


@start_train_btn.click
def start_train():
    g.state.stop_training = False
    # monitoring.container.show()
    stop_train_btn.enable()
    # epoch_progress.show()
    iter_progress.show()
    try:
        train.train()
    except StopIteration as exc:
        sly.logger.info("The training is stoped.")


@stop_train_btn.click
def stop_train():
    g.state.stop_training = True
    stop_train_btn.disable()
