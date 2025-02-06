import os
import cv2
import mmcv

from shutil import rmtree, move
from mmseg.datasets import CustomDataset, DATASETS
import numpy as np
import supervisely as sly
from src import globals as g
import time

def collect_children_ds_ids(ds_tree, dataset_ids):
    for dsinfo, children in ds_tree.items():
        if dsinfo.id in dataset_ids and children:
            dataset_ids.extend([info.id for info in children])
            collect_children_ds_ids(children, dataset_ids)


def download_datasets(project_id, dataset_ids=None):
    project_dir = g.PROJECT_DIR
    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)
    dataset_ids = list(set(dataset_ids))

    selected_ds_cnt = len(dataset_ids)
    ds_tree = g.api.dataset.get_tree(project_id)
    collect_children_ds_ids(ds_tree, dataset_ids)
    nested_datasets = selected_ds_cnt != len(dataset_ids)
    if nested_datasets:
        sly.logger.info("Found nested datasets. Downloading all nested datasets.")
    # TODO: hardcoded progress_cb is bad here
    progress_cb = g.iter_progress(message="Downloading datasets...", total=g.state.n_images).update
    sly.Project.download(g.api, project_id, project_dir, dataset_ids, progress_cb=progress_cb)
    return project_dir


def prepare_datasets(selected_classes: list):
    # TODO: hardcoded progress_cb is bad here
    progress_cb = g.iter_progress(
        message="Converting annotations...", total=g.state.n_images
    ).update
    sly.Project.to_segmentation_task(
        g.PROJECT_DIR,
        inplace=True,
        target_classes=selected_classes.copy(),
        progress_cb=progress_cb,
    )
    project = sly.Project(g.PROJECT_DIR, sly.OpenMode.READ)
    progress_cb = g.iter_progress(message="Converting masks...", total=g.state.n_images).update
    convert_project_masks(project, ann_dir=g.ANN_DIR, progress_cb=progress_cb)


def get_classes_and_palette(project_meta: sly.ProjectMeta):
    class_names = [cls.name for cls in project_meta.obj_classes if cls.name != "__bg__"]
    palette = [cls.color for cls in project_meta.obj_classes if cls.name != "__bg__"]
    class_names.insert(0, "__bg__")
    palette.insert(0, [0, 0, 0])
    return class_names, palette


def convert_project_masks(project_fs: sly.Project, ann_dir="seg2", progress_cb=None):
    class_names, palette = get_classes_and_palette(project_fs.meta)

    # * Create LUT for mask conversion
    palette_arr = np.array(palette, dtype=np.uint32)
    lut = np.zeros(1 << 24, dtype=np.uint8)
    palette_hashes = palette_arr[:, 0] + (palette_arr[:, 1] << 8) + (palette_arr[:, 2] << 16)
    for label_idx, hash_val in enumerate(palette_hashes):
        lut[hash_val] = label_idx

    unsupported_exts = []
    for ds in project_fs.datasets:
        t = time.time()
        ds: sly.Dataset
        res_ds_dir = os.path.join(project_fs.parent_dir, project_fs.name, ds.name.split("/")[0])
        os.makedirs(res_ds_dir, exist_ok=True)
        os.makedirs(os.path.join(res_ds_dir, ann_dir), exist_ok=True)
        os.makedirs(os.path.join(res_ds_dir, "seg"), exist_ok=True)
        os.makedirs(os.path.join(res_ds_dir, "img"), exist_ok=True)
        os.makedirs(os.path.join(res_ds_dir, "ann"), exist_ok=True)
        existed_files = set(sly.fs.list_dir_recursively(os.path.join(res_ds_dir, ann_dir)))
        for item in ds.get_items_names():
            name = sly.generate_free_name(existed_files, f"{ds.short_name}_{item}", True, True)

            # * Move/rename image
            img_source = ds.get_img_path(item)
            ext = sly.fs.get_file_ext(img_source)
            img_dest_dir = os.path.join(res_ds_dir, "img")
            if ext.lower() not in [".png", ".jpeg", ".jpg"]:
                # * Convert image to jpg if it has unsupported extension
                try:
                    img = cv2.imread(img_source)
                    name = name + ".jpg"
                    cv2.imwrite(os.path.join(img_dest_dir, name), img)
                    unsupported_exts.append(ext.lower())
                except:
                    sly.logger.warn(f"Failed to convert image: {img_source}")
                    continue
            else:
                move(img_source, os.path.join(img_dest_dir, name))
            sly.fs.silent_remove(img_source)

            # * Generate machine mask
            mask_name = name + ".png"
            seg_ann_path = ds.get_seg_path(item)
            mask_bgr = cv2.imread(seg_ann_path)
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

            pixel_hashes = (
                mask_rgb[:, :, 0].astype(np.uint32)
                + (mask_rgb[:, :, 1].astype(np.uint32) << 8)
                + (mask_rgb[:, :, 2].astype(np.uint32) << 16)
            )
            result = lut[pixel_hashes]
            ann_out_path = os.path.join(res_ds_dir, ann_dir, mask_name)
            cv2.imwrite(ann_out_path, result)

            # * Move/rename annotations
            source_ann_path = ds.get_ann_path(item)
            move(source_ann_path, os.path.join(res_ds_dir, "ann", name + ".json"))
            sly.fs.silent_remove(source_ann_path)
            move(seg_ann_path, os.path.join(res_ds_dir, "seg", mask_name))
            sly.fs.silent_remove(seg_ann_path)

            if progress_cb is not None:
                progress_cb(1)
        sly.logger.info(f"Conversion of dataset {ds.name} took {time.time() - t:.2f} seconds")

    if len(unsupported_exts) > 0:
        sly.logger.info(
            "Converted {} images with unsupported extensions: ".format(len(unsupported_exts))
            + ", ".join(f'"{ext}"' for ext in set(unsupported_exts)),
        )


# def _convert_mask_values(mask: np.ndarray, palette: list):
#     result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
#     if mask.max() == 0:
#         return result
#     for label_idx, color in enumerate(palette):
#         if label_idx == 0:
#             # skip background
#             continue
#         colormap = np.where(np.all(mask == np.array(color, dtype=mask.dtype), axis=-1))
#         result[colormap] = label_idx
#     return result


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
            data_root=g.PROJECT_DIR + "/" + dataset_name,
            test_mode=test_mode,
        )

        self.last_eval_results = None
        self.last_model_outputs = None
        project_meta_json = sly.json.load_json_file(os.path.join(g.PROJECT_DIR, "meta.json"))
        project_meta = sly.ProjectMeta.from_json(project_meta_json)
        self.CLASSES, self.PALETTE = get_classes_and_palette(project_meta)
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
