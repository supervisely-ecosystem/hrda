import os
from pathlib import Path
import shutil
import numpy as np
from requests_toolbelt import MultipartEncoderMonitor
from supervisely.app.widgets import Progress

from tqdm import tqdm
import src.globals as g
import supervisely as sly


def mask2annotation(x: np.ndarray, classes, palette, skip_bg=True):
    assert len(classes) == len(palette)

    if x.ndim == 3:
        if x.shape[2] == 3:
            assert np.all(x[..., 0] == x[..., 1])
            assert np.all(x[..., 0] == x[..., 2])
            x = x[..., 0]
        elif x.shape[2] == 1:
            x = x.squeeze()

    labels = []
    for cls_idx in range(int(skip_bg), len(classes)):
        mask = x == cls_idx
        if mask.any():
            b = sly.Bitmap(mask)
            obj_cls = sly.ObjClass(classes[cls_idx], sly.Bitmap, palette[cls_idx])
            l = sly.Label(b, obj_cls)
            labels.append(l)
    ann = sly.Annotation(x.shape[:2], labels)
    return ann


def download_custom_config(remote_weights_path: str):
    # # download config_xxx.py
    # save_dir = remote_weights_path.split("checkpoints")
    # files = g.api.file.listdir(g.TEAM_ID, save_dir)
    # # find config by name in save_dir
    # remote_config_path = [f for f in files if f.endswith(".py")]
    # assert len(remote_config_path) > 0, f"Can't find config in {save_dir}."

    # download config.py
    remote_dir = os.path.dirname(remote_weights_path)
    remote_config_path = remote_dir + "/config.py"
    config_name = remote_config_path.split("/")[-1]
    config_path = g.app_dir + f"/{config_name}"
    g.api.file.download(g.TEAM_ID, remote_config_path, config_path)
    return config_path


def download_custom_model_weights(remote_weights_path: str):
    # download .pth
    file_name = os.path.basename(remote_weights_path)
    weights_path = g.app_dir + f"/{file_name}"
    g.api.file.download(g.TEAM_ID, remote_weights_path, weights_path)
    return weights_path


def download_custom_model(remote_weights_path: str):
    config_path = download_custom_config(remote_weights_path)
    weights_path = download_custom_model_weights(remote_weights_path)
    return weights_path, config_path


def upload_artifacts(work_dir: str, experiment_name: str = None, progress_widget: Progress = None):
    task_id = g.api.task_id or ""
    paths = [path for path in os.listdir(work_dir) if path.endswith(".py")]
    assert len(paths) > 0, "Can't find config file saved during training."
    assert len(paths) == 1, "Found more then 1 .py file"
    cfg_path = f"{work_dir}/{paths[0]}"
    shutil.move(cfg_path, f"{work_dir}/config.py")

    # rm symlink
    sly.fs.silent_remove(f"{work_dir}/last_checkpoint")

    if not experiment_name:
        experiment_name = f"{g.config_name.split('.py')[0]}"
    sly.logger.debug("Uploading checkpoints to Team Files...")

    if progress_widget:
        progress_widget.show()
        work_dir_p = Path(work_dir)
        nbytes = sum(f.stat().st_size for f in work_dir_p.glob("**/*") if f.is_file())
        pbar = progress_widget(
            message="Uploading to Team Files...",
            total=int(nbytes / 1024 / 1024),
            unit="MB",
        )

        def cb(monitor: MultipartEncoderMonitor):
            pbar.update(int(monitor.bytes_read / 1024 / 1024 - pbar.n))

    else:
        cb = None

    out_path = g.api.file.upload_directory(
        g.TEAM_ID,
        work_dir,
        f"/mmdetection-3/{task_id}_{experiment_name}",
        progress_size_cb=cb,
    )
    return out_path


def download_project(progress_widget):
    project_dir = f"{g.app_dir}/sly_project"

    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)

    n = get_images_count()
    with progress_widget(message="Downloading project...", total=n) as pbar:
        sly.Project.download(g.api, g.PROJECT_ID, project_dir, progress_cb=pbar.update)

    return project_dir


def get_images_count():
    return g.IMAGES_COUNT


def save_augs_config(augs_config_path: str, work_dir: str):
    sly.fs.copy_file(augs_config_path, work_dir + "/augmentations.json")


def save_open_app_lnk(work_dir: str):
    with open(work_dir + "/open_app.lnk", "w") as f:
        f.write(f"{g.api.server_address}/apps/sessions/{g.api.task_id}")


def get_images_count(dataset_ids):
    count_fn = lambda ds_id: len(g.api.image.get_list(ds_id))
    return sum(map(count_fn, dataset_ids))
