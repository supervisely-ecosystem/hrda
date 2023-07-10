import supervisely as sly
from dotenv import load_dotenv
import os
import mmcv

load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
workspace_id = sly.env.workspace_id()

gt_project_id = 22449

preds_file = "outputs.pkl"
filenames_pkl = "output_filenames.pkl"
dst_project_name = "validation HRDA 768x"

outputs = mmcv.load(preds_file)
filenames = mmcv.load(filenames_pkl)

api.project.clone_advanced(gt_project_id, workspace_id, dst_project_name)

project_id = api.project.get_info_by_name(workspace_id, dst_project_name).id
meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
obj_cls = meta.obj_classes.get("cracks")
assert obj_cls is not None

datasets = api.dataset.get_list(project_id)
assert len(datasets) == 1
dataset = datasets[0]
image_infos = api.image.get_list(dataset.id)
sly_filenames = [x.name for x in image_infos]

assert set(sly_filenames) == set(filenames)
filename2id = {f.name:f.id for f in image_infos}


img_ids = []
anns = []
for pred, filename in zip(outputs, filenames):
    pred = pred.astype(bool)
    if pred.sum() == 0:
        continue
    b = sly.Bitmap(pred)
    label = sly.Label(b, obj_cls)
    ann = sly.Annotation(pred.shape, [label])
    anns.append(ann)
    img_ids.append(filename2id[filename])

api.annotation.upload_anns(img_ids, anns)
