import supervisely as sly
from supervisely.app.widgets import Card, SelectString, Container, Field
import src.globals as g
from src.ui.utils import HContainer


dataset_infos = g.api.dataset.get_list(g.PROJECT_ID)
dataset_names = [ds.name for ds in dataset_infos]
# dataset_items = [Select.Item(ds, ds.name) for ds in dataset_infos]

select_train_labeled = SelectString(dataset_names)
select_train_labeled_f = Field(
    select_train_labeled,
    "Source dataset (labeled)",
    "Labeled dataset in source domain (e.g. synthesized images).",
)
select_train_unlabeled = SelectString(dataset_names)
select_train_unlabeled_f = Field(
    select_train_unlabeled,
    "Target dataset (unlabeled)",
    "Unlabaled dataset in target domain to which the adaptation will be performed (e.g. real images).",
)
select_val_labeled = SelectString(dataset_names)
select_val_labeled_f = Field(
    select_val_labeled, "Val dataset (labeled)", "Labeled dataset in target domain to validate on."
)


card = Card(
    title="4️⃣ Select datasets",
    description="Define datasets for train and validation",
    content=Container([select_train_labeled_f, select_train_unlabeled_f, select_val_labeled_f]),
)
