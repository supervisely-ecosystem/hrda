import cv2
import supervisely as sly
from dotenv import load_dotenv
import os
import numpy as np
import random

from src import rename_ext


load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id = 22965

data_dir = "data/cracks"
data_dir_seg = data_dir+"_seg"

assert not (sly.fs.dir_exists(data_dir) or sly.fs.dir_exists(data_dir_seg)), "data_dir exists"

sly.Project.download(api, project_id, data_dir)

# check .png
paths_png = rename_ext.find_png(data_dir)
if paths_png:
    print(f"There are {len(paths_png)} .png files!")

# rename ext
paths = rename_ext.find_candidates(data_dir)
print("Will be renamed:", len(paths))
rename_ext.rename(paths)

# to_segmentation_task
sly.Project.to_segmentation_task(data_dir, data_dir_seg)

# human masks to machine masks
project = sly.Project(data_dir_seg, sly.OpenMode.READ)
palette = [cls.color for cls in project.meta.obj_classes if cls.name != "__bg__"]
datasets = project.datasets

print(f"{project.total_items=}")
print(f"{palette=}")

for dataset in datasets:
    dataset : sly.Dataset
    os.makedirs(f"{dataset.directory}/seg2", exist_ok=False)
    for item in dataset.get_items_names():
        ann_path = dataset.get_seg_path(item)
        mask = cv2.cvtColor(cv2.imread(ann_path), cv2.COLOR_BGR2RGB)
        result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
        for color_idx, color in enumerate(palette, 1):
            colormap = np.where(np.all(mask == color, axis=-1))
            result[colormap] = color_idx
        cv2.imwrite(f"{dataset.directory}/seg2/{item}.png", result)