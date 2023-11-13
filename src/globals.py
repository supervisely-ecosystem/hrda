import os
import supervisely as sly
from dotenv import load_dotenv
from src.state import State

# from src.train_parameters import TrainParameters

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()
app_dir = sly.app.get_synced_data_dir()
app: sly.Application = None
iter_progress: sly.app.widgets.Progress = None

TASK_NAME = "segmentation"
PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()

# TODO: only when dataset is selected in the modal window
_project_info = api.project.get_info_by_id(PROJECT_ID)
IMAGES_COUNT = _project_info.items_count
PROJECT_NAME = _project_info.name

LOG_INTERVAL = 1 if sly.is_development() else 50
PRETRAINED_PATH = "/root/mit_b5.pth"
PROJECT_DIR = app_dir + "/sly_project"
PROJECT_SEG_DIR = app_dir + "/sly_project_seg"
IMG_DIR = "img"
ANN_DIR = "seg2"
WORK_DIR = app_dir + "/work_dir"
TEAMFILES_UPLOAD_DIR = "HRDA"
STATIC_DIR = app_dir + "/static"
os.makedirs(STATIC_DIR, exist_ok=True)

# params
MAX_CLASSES_FOR_PERCLASS_METRICS = 10


# for Augmentations widget:
data_dir = app_dir
team = api.team.get_info_by_id(TEAM_ID)
project_meta: sly.ProjectMeta = sly.ProjectMeta.from_json(api.project.get_meta(PROJECT_ID))
# project_fs: sly.Project = None

state = State()
