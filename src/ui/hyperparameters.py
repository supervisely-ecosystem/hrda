from supervisely.app.widgets import (
    Container,
    InputNumber,
    Field,
    Switch,
    Tabs,
    Card,
    SelectString,
    Input,
)
from src.ui.utils import HContainer
from src.input_container import InputForm
from src import sly_utils


class GeneralParams(InputForm):
    def __init__(self) -> None:
        self.experiment_name = Field(
            Input(f"hrda_{sly_utils.get_project_name()}"),
            "Experiment name",
            "This name will be used as a dir name in Team Files where model checkpoints will be uploaded after the training.",
        )
        self.total_iters = Field(
            InputNumber(25000, min=1),
            "Training iterations",
            "The number of total training iterations",
        )
        self.val_interval = Field(
            InputNumber(1000, min=1), "Validation interval", "Evaluate the model every N iterations"
        )

        self.longer_input_size = Field(InputNumber(640, 1, step=16), "Longer edge")
        self.shorter_input_size = Field(InputNumber(640, 1, step=16), "Shorter edge")

        input_size_container = HContainer([self.longer_input_size, self.shorter_input_size])
        input_size_container = Field(
            input_size_container,
            title="Input size",
            description="Images will be cropped and pad to this size",
        )

        self.batch_size_train = Field(InputNumber(2, 1), "Train batch size")

        final_container = Container(
            [
                self.experiment_name,
                self.total_iters,
                self.val_interval,
                input_size_container,
                self.batch_size_train,
            ]
        )
        self.compile(final_container)


class CheckpointParams(InputForm):
    def __init__(self) -> None:
        self.checkpoint_interval = Field(
            InputNumber(2500, min=1), "Checkpoint interval", "Save checkpoint every N iterations"
        )
        self.max_keep_ckpts = Field(
            InputNumber(3, min=-1),
            "Max keep checkpoints",
            'The maximum number of checkpoints to keep. Earlier checkpoints will be removed if the number of checkpoints will exceed this value. ("-1" will disable it)',
        )
        self.save_optimizer = Field(
            Switch(),
            "Save optimizer",
            "Whether to save the optimizer state along with model weights.",
        )

        final_container = Container(
            [self.checkpoint_interval, self.max_keep_ckpts, self.save_optimizer]
        )
        self.compile(final_container)


class OptimizerParams(InputForm):
    def __init__(self) -> None:
        self.optimizer_type = Field(
            SelectString(["AdamW", "SGD"]), "Optimizer", "The type of the optimizer."
        )
        self.base_lr = Field(InputNumber(1e-4, min=0), "Initial LR", "The initial learning rate after warmup.")
        self.weight_decay = Field(
            InputNumber(1e-4),
            "Weight decay",
            "Weight decay helps regularize the model to prevent overfitting.",
        )

        self.warmup_iters = Field(
            InputNumber(200, min=0), "Warmup", "The number of warmup iterations."
        )
        self.scheduler_power = Field(
            InputNumber(1.0, min=0),
            "Scheduler's power",
            "The power parameter of the Polynomial scheduler. This scheduler will gradually decrease the learning rate.",
        )

        final_container = Container(
            [
                self.optimizer_type,
                self.base_lr,
                self.weight_decay,
                self.warmup_iters,
                self.scheduler_power,
            ]
        )
        self.compile(final_container)


general_params = GeneralParams()
checkpoint_params = CheckpointParams()
optimizer_params = OptimizerParams()


content = Tabs(
    labels=[
        "General",
        "Checkpoints",
        "Optimizer and LR",
    ],
    contents=[
        general_params.get_content(),
        checkpoint_params.get_content(),
        optimizer_params.get_content(),
    ],
)

card = Card(
    title="6️⃣ Training hyperparameters",
    description="Configure the training process.",
    lock_message="Select a model to unlock.",
    content=content,
)
