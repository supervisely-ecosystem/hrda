import supervisely as sly
from supervisely.app.widgets import TrainValSplits, Card, Select, Container, Field
import src.globals as g

# splits = TrainValSplits(project_id=g.PROJECT_ID)

dataset_infos = g.api.dataset.get_list(g.PROJECT_ID)
dataset_names = [ds.name for ds in dataset_infos]
dataset_items = [Select.Item(ds, ds.name) for ds in dataset_infos]

select_train_labeled = Select(dataset_items)
select_train_labeled_f = Field(select_train_labeled, "Source-domain train dataset (labeled)")
select_train_unlabeled = Select(dataset_items)
select_train_unlabeled_f = Field(select_train_unlabeled, "Target-domain train dataset (unlabeled)")
select_val_labeled = Select(dataset_items)
select_val_labeled_f = Field(select_val_labeled, "Target-domain val dataset (labeled)")


card = Card(
    title="4️⃣ Select dataset",
    description="Define datasets for train and validation",
    content=Container([
        select_train_labeled_f,
        select_train_unlabeled_f,
        select_val_labeled_f
    ], "horizontal", overflow="wrap"),
)


# def dump_train_val_splits(project_dir):
#     # dump splits into json files
#     splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
#     train_split, val_split = splits.get_splits()
#     app_dir = g.app_dir
#     sly.json.dump_json_file(train_split, f"{app_dir}/train_split.json")
#     sly.json.dump_json_file(val_split, f"{app_dir}/val_split.json")
