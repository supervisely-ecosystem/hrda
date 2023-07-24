import os
from dotenv import load_dotenv
from src.serve.hrda import HRDA
import supervisely as sly
from supervisely.nn import PredictionSegmentation
import mmcv
from src.sly_utils import mask2annotation

load_dotenv(os.path.expanduser("~/supervisely.env"))

device = "cuda:0"
config_path = "app_data/work_dir/config.py"
weights_path = "app_data/work_dir/iter_70.pth"
img_path = "app_data/sly_project_seg/ds1/img/IMG_1836.jpeg"

model = HRDA()
model.load_model(config_path, weights_path, device)

pred = model.predict(img_path)


def to_annotation(pred: PredictionSegmentation):
    b = sly.Bitmap(pred)
    obj_class = sly.ObjClass(pred.class_name, sly.Bitmap)
    l = sly.Label(b, obj_class)
    ann = sly.Annotation(pred.mask.shape, [l])
    return ann


ann = mask2annotation(pred[0].mask, model.class_names, model.model.PALETTE)
# ann = to_annotation(pred)

img = sly.image.read(img_path)
ann.draw_pretty(img)
sly.image.write("test.jpg", img)
