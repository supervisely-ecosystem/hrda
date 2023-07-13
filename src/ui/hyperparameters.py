from supervisely.app.widgets import (
    Container,
    InputNumber,
    BindedInputNumber,
    Select,
    Field,
    Text,
    Empty,
    Switch,
    Tabs,
    Card,
    SelectString
)
from src.ui.utils import HContainer
from src.input_container import InputContainer


class GeneralParams(InputContainer):
    def __init__(self) -> None:
        self.num_epochs = Field(InputNumber(15, min=1), "Number of epochs")

        self.longer_input_size = Field(InputNumber(1000, 1), "Longer edge")
        self.shorter_input_size = Field(InputNumber(600, 1), "Shorter edge")

        input_size_container = HContainer([self.longer_input_size, self.shorter_input_size])
        input_size_container = Field(
            input_size_container,
            title="Input size",
            description="Images will be scaled approximately to the specified sizes while keeping the aspect ratio "
            "(internally, the sizes are passed as 'scale' parameter of the 'Resize' transform in mmcv).",
        )
        
        self.batch_size_train = Field(InputNumber(2, 1), "Train batch size")

        final_container = Container([self.num_epochs, input_size_container, self.batch_size_train])
        self.compile(final_container)

general_params = GeneralParams()


content = Tabs(
    labels=[
        "General",
        # "Checkpoints",
        # "Optimizer (Advanced)",
        # "Learning rate scheduler (Advanced)",
    ],
    contents=[
        general_params.get_content(),
    ],
)

card = Card(
    title="6️⃣ Training hyperparameters",
    description="Configure the training process.",
    lock_message="Select a model to unlock.",
    content=content,
)
