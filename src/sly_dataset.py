import os
import cv2
import mmcv
from shutil import rmtree
from mmseg.datasets import CustomDataset, DATASETS
import numpy as np
import supervisely as sly
from src import globals as g


def collect_children_ds_ids(ds_tree, dataset_ids):
    for dsinfo, children in ds_tree.items():
        if dsinfo.id in dataset_ids:
            for child in children:
                dataset_ids.append(child.id)
                if isinstance(child, dict):
                    collect_children_ds_ids(child, dataset_ids)


def extract_nested_datasets(directory):
    for path in os.scandir(directory):
        if not path.is_dir():
            continue
        nested_path = os.path.join(path, "datasets")
        if os.path.exists(nested_path):  #  -> has nested datasets
            for item in os.scandir(nested_path):  # each dataset
                for folder in os.scandir(item):  # ann/img folders
                    for subitem in os.scandir(folder):  # files
                        os.rename(subitem.path, os.path.join(path.path, folder.name, subitem.name))
                    os.rmdir(folder.path)  # remove empty ann/img folders
                if item.name == "datasets":  # -> has extra nested dataset, repeat
                    extract_nested_datasets(item.path)
            rmtree(nested_path)  # remove dataset folder and its contents


def download_datasets(project_id, dataset_ids=None):
    project_dir = g.PROJECT_DIR
    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)
    dataset_ids = list(set(dataset_ids))

    selected_ds_cnt = len(dataset_ids)
    ds_tree = g.api.dataset.get_tree(project_id)
    nested_datasets = False
    collect_children_ds_ids(ds_tree, dataset_ids)
    if selected_ds_cnt != len(dataset_ids):
        sly.logger.info("Found nested datasets. Downloading all nested datasets.")
        nested_datasets = True
    # TODO: hardcoded progress_cb is bad here
    progress_cb = g.iter_progress(message="Downloading datasets...", total=g.state.n_images).update
    sly.Project.download(g.api, project_id, project_dir, dataset_ids, progress_cb=progress_cb)
    if nested_datasets:
        try:
            extract_nested_datasets(project_dir)
        except Exception as e:
            raise RuntimeError("Failed to extract nested datasets") from e
    return project_dir


def prepare_datasets(selected_classes: list):
    # TODO: hardcoded progress_cb is bad here
    progress_cb = g.iter_progress(
        message="Converting annotations...", total=g.state.n_images
    ).update
    if sly.fs.dir_exists(g.PROJECT_SEG_DIR):
        sly.fs.remove_dir(g.PROJECT_SEG_DIR)
    sly.Project.to_segmentation_task(
        g.PROJECT_DIR, g.PROJECT_SEG_DIR, target_classes=selected_classes.copy(), progress_cb=progress_cb
    )
    project = sly.Project(g.PROJECT_SEG_DIR, sly.OpenMode.READ)
    convert_project_masks(project, ann_dir=g.ANN_DIR)


def get_classes_and_palette(project_meta: sly.ProjectMeta):
    class_names = [cls.name for cls in project_meta.obj_classes if cls.name != "__bg__"]
    palette = [cls.color for cls in project_meta.obj_classes if cls.name != "__bg__"]
    class_names.insert(0, "__bg__")
    palette.insert(0, [0, 0, 0])
    return class_names, palette


def convert_project_masks(project_fs: sly.Project, ann_dir="seg2"):
    # convert human masks to machine masks

    class_names, palette = get_classes_and_palette(project_fs.meta)
    datasets = project_fs.datasets

    for dataset in datasets:
        dataset: sly.Dataset
        os.makedirs(f"{dataset.directory}/{ann_dir}", exist_ok=False)
        for item in dataset.get_items_names():
            ann_path = dataset.get_seg_path(item)
            mask = cv2.cvtColor(cv2.imread(ann_path), cv2.COLOR_BGR2RGB)
            result = _convert_mask_values(mask, palette)
            cv2.imwrite(f"{dataset.directory}/{ann_dir}/{item}.png", result)


def _convert_mask_values(mask: np.ndarray, palette: list):
    result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    for label_idx, color in enumerate(palette):
        if label_idx == 0:
            # skip background
            continue
        colormap = np.where(np.all(mask == color, axis=-1))
        result[colormap] = label_idx
    return result


@DATASETS.register_module()
class SuperviselyDataset(CustomDataset):
    img_suffix = (".jpg", ".jpeg", ".png")

    def __init__(self, pipeline, dataset_name, test_mode=False):
        super().__init__(
            pipeline,
            img_suffix=self.img_suffix,
            img_dir=g.IMG_DIR,
            ann_dir=g.ANN_DIR,
            seg_map_suffix=".png",
            data_root=g.PROJECT_SEG_DIR + "/" + dataset_name,
            test_mode=test_mode,
        )

        self.last_eval_results = None
        self.last_model_outputs = None
        self.project = sly.Project(g.PROJECT_SEG_DIR, sly.OpenMode.READ)
        self.CLASSES, self.PALETTE = get_classes_and_palette(self.project.meta)
        self.pseudo_margins = None
        self.valid_mask_size = [512, 512]  # TODO: pseudo_margins is not used yet

        if self.pseudo_margins is not None:
            assert pipeline[-1]["type"] == "Collect"
            pipeline[-1]["keys"].append("valid_pseudo_mask")

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        if self.pseudo_margins is not None:
            results["valid_pseudo_mask"] = np.ones(self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results["valid_pseudo_mask"][: self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results["valid_pseudo_mask"][-self.pseudo_margins[1] :, :] = 0
            if self.pseudo_margins[2] > 0:
                results["valid_pseudo_mask"][:, : self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results["valid_pseudo_mask"][:, -self.pseudo_margins[3] :] = 0
            results["seg_fields"].append("valid_pseudo_mask")

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = []
        for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
            img_info = dict(filename=img)
            if ann_dir is not None:
                seg_map = img + seg_map_suffix
                img_info["ann"] = dict(seg_map=seg_map)
            img_infos.append(img_info)

        sly.logger.info(f"Loaded {len(img_infos)} images from {img_dir}")
        return img_infos

    def evaluate(self, results, metric="mIoU", logger=None, efficient_test=False, **kwargs):
        eval_results = super().evaluate(results, metric, logger, efficient_test, **kwargs)
        self.last_eval_results = eval_results
        self.last_model_outputs = results
        return eval_results
