import src.globals as g
import supervisely as sly
from supervisely.app.widgets import Container

import src.ui.input_project as input_project
import src.ui.models as models
import src.ui.classes as classes
import src.ui.train_val_split as train_val_split

# import src.ui.graphics as graphics
import src.ui.hyperparameters as hyperparameters
import src.ui.training as training
import src.ui.augmentations as augmentations
import src.ui.handlers

# register modules (don't remove):
from src import sly_dataset, sly_imgaugs


widgets = [
    input_project.card,
    models.card,
    classes.card,
    train_val_split.card,
    augmentations.card,
    hyperparameters.card,
    training.card,
]
layout = Container(widgets=widgets)
app = sly.Application(layout=layout)

g.app = app
