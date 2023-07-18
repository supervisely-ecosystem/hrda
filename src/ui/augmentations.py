from pathlib import Path
import os
from supervisely.app.widgets import AugmentationsWithTabs, Card, Container, Switch
import src.globals as g


def name_from_path(aug_path):
    name = os.path.basename(aug_path).split(".json")[0].capitalize()
    name = " + ".join(name.split("_"))
    return name


template_dir = "aug_templates"
template_paths = list(map(str, Path(template_dir).glob("*.json")))
template_paths = sorted(template_paths, key=lambda x: x.replace(".", "_"))[::-1]

templates = [{"label": name_from_path(path), "value": path} for path in template_paths]


swithcer = Switch(True)
augments = AugmentationsWithTabs(g, task_type=g.TASK_NAME, templates=templates)


container = Container([swithcer, augments])

card = Card(
    title="5️⃣ Training augmentations",
    description="Choose one of the prepared templates or provide a custom pipeline",
    content=container,
    lock_message="Select a model to unlock.",
)


def get_selected_config_path():
    # path to aug pipline (.json file)
    if swithcer.is_switched():
        return augments._current_augs._template_path
    else:
        return None


def update_task(task_name):
    augments._augs1._task_type = task_name
    augments._augs2._task_type = task_name


def reset_widgets():
    if swithcer.is_switched():
        augments.show()
    else:
        augments.hide()


@swithcer.value_changed
def on_switch(is_switched: bool):
    reset_widgets()


reset_widgets()
