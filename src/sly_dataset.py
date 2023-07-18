import os
import cv2
import mmcv
from mmseg.datasets import CustomDataset, DATASETS
import numpy as np
import supervisely as sly
from src import globals as g


def download_datasets(project_id, dataset_ids=None, progress_cb=None):
    project_dir = g.PROJECT_DIR

    sly.Project.download(g.api, project_id, project_dir, dataset_ids, progress_cb=progress_cb)

    return project_dir


def prepare_datasets():
    sly.Project.to_segmentation_task(g.PROJECT_DIR, g.PROJECT_SEG_DIR)
    project = sly.Project(g.PROJECT_SEG_DIR, sly.OpenMode.READ)
    convert_project_masks(project, ann_dir=g.ANN_DIR)


def get_classes_and_palette(project_meta: sly.ProjectMeta):
    class_names = [cls.name for cls in project_meta.obj_classes if cls.name != "__bg__"]
    palette = [cls.color for cls in project_meta.obj_classes if cls.name != "__bg__"]
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
    for color_idx, color in enumerate(palette, 1):
        colormap = np.where(np.all(mask == color, axis=-1))
        result[colormap] = color_idx
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

        self.project = sly.Project(g.PROJECT_SEG_DIR, sly.OpenMode.READ)
        self.CLASSES, self.PALETTE = get_classes_and_palette(self.project.meta)
        self.pseudo_margins = None
        self.valid_mask_size = [512, 512]

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
