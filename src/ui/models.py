from typing import List
import supervisely as sly
from supervisely.app.widgets import (
    RadioTabs,
    RadioTable,
    SelectString,
    Card,
    Container,
    Button,
    Text,
    Field,
    TeamFilesSelector,
    Switch,
)

from src.globals import TEAM_ID


class ModelItem:
    def __init__(self, name, architecture_name, num_params):
        self.name = name
        self.architecture_name = architecture_name
        self.num_params = num_params


def get_architecture_list():
    architectures = ["SegFormer"]
    return architectures


def get_model_list(architecture_name):
    models = [
        ModelItem("SegFormer MiT-b5", architecture_name, 82.0),
    ]
    return models


def _get_table_data(models: List[ModelItem]):
    columns = ["Model variant", "Params (M)"]
    rows = [[m.name, m.num_params] for m in models]
    subtitles = [None] * len(columns)
    return columns, rows, subtitles


def is_pretrained_model_selected():
    custom_path = get_selected_custom_path()
    if radio_tabs.get_active_tab() == "Pretrained models":
        if custom_path:
            raise Exception(
                "Active tab is Pretrained models, but the path to the custom weights is selected. This is ambiguous."
            )
        return True
    else:
        if custom_path:
            return False
        else:
            raise Exception(
                "Active tab is Custom weights, but the path to the custom weights isn't selected."
            )


def get_selected_architecture_name() -> str:
    return "SegFormer"


def get_selected_pretrained_model() -> ModelItem:
    architecture_name = get_selected_architecture_name()
    models = get_model_list(architecture_name)
    idx = model_table.get_selected_row_index()
    return models[idx]


def get_selected_custom_path() -> str:
    paths = input_file.get_selected_paths()
    return paths[0] if len(paths) > 0 else ""


model_table = RadioTable([""], [[""]])
text = Text(status="info")

load_from = Switch(True)
load_from_field = Field(
    load_from,
    "Download pre-trained model",
    "Whether to download pre-trained weights and finetune the model or train it from scratch.",
)

input_file = TeamFilesSelector(TEAM_ID, selection_file_type="file")
path_field = Field(
    title="Path to weights file",
    description="Copy path in Team Files",
    content=input_file,
)

radio_tabs = RadioTabs(
    titles=["Pretrained models", "Custom weights"],
    contents=[
        Container(widgets=[model_table, text, load_from_field]),
        path_field,
    ],
)

card = Card(
    title=f"2️⃣ Model select",
    description="Choose model architecture and how its weights should be initialized",
    content=Container([radio_tabs]),
    lock_message="Select a task to unlock.",
)


def update_models(arch_name):
    models = get_model_list(arch_name)
    columns, rows, subtitles = _get_table_data(models)
    model_table.set_data(columns, rows, subtitles)
    model_table.select_row(0)
    update_selected_model(model_table.get_selected_row())


def update_selected_model(selected_row):
    idx = model_table.get_selected_row_index()
    text.text = f"Selected model: {selected_row[0]}"


def reset_widgets():
    update_models()


reset_widgets()
