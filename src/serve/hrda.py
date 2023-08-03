import os
from typing import Any, Dict, List, Literal
from typing_extensions import Literal
import torch
import numpy as np

from supervisely.nn.inference import SemanticSegmentation
from supervisely.nn.inference.gui import InferenceGUI
from supervisely.nn.prediction_dto import PredictionSegmentation
import supervisely as sly

import mmcv
from mmcv.parallel import collate, scatter
from mmseg.apis.inference import inference_segmentor, init_segmentor, LoadImage
from mmseg.datasets.pipelines import Compose


class HRDA(SemanticSegmentation):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        config_path, weights_path = self.download_model_files()
        self.load_model(config_path, weights_path, device)
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def load_model(self, config_path, weights_path, device="cuda:0"):
        self.model = init_segmentor(
            config_path,
            weights_path,
            device=device,
            revise_checkpoint=[(r"^module\.", ""), ("model.", "")],
        )
        cfg = self.model.cfg
        classes, palette = self.model.CLASSES, self.model.PALETTE
        test_pipeline_cfg = [LoadImage()] + cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(test_pipeline_cfg)
        self.class_names = classes
        obj_classes = [
            sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(classes, palette)
        ]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self.device = device

    def get_classes(self) -> List[str]:
        return self.class_names

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any] = None
    ) -> List[PredictionSegmentation]:
        # prepare image
        data = self.prepare_image(image_path)
        
        # we will unpad and rescale later
        img_meta = data["img_metas"][0][0]
        ori_shape = img_meta["ori_shape"]
        if img_meta.get("pad_shape"):
            data["rescale"] = False

        # predict
        with torch.no_grad():
            result = self.model(return_loss=False, **data)

        # unpad and resize manually
        if img_meta.get("pad_shape"):
            h, w, c = img_meta["img_shape"]
            dtype = result[0].dtype
            result = result[0][:h, :w].astype(np.uint8)
            result = mmcv.imresize(result, ori_shape[:2][::-1], interpolation="nearest")
            result = result.astype(dtype)

        return [sly.nn.PredictionSegmentation(result)]

    def prepare_image(self, image_path: str) -> dict:
        data = dict(img=image_path)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data["img_metas"] = [i.data[0] for i in data["img_metas"]]
        return data

    def download_model_files(self):
        if self.gui is not None:
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                # selected_model = self.gui.get_checkpoint_info()
                # weights_path, config_path = self.download_pretrained_files(
                #     selected_model, model_dir
                # )
                pass
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights_path = self.download(custom_weights_link)
                remote_config_path = os.path.dirname(custom_weights_link)+"/config.py"
                config_path = self.download(remote_config_path)
            return config_path, weights_path

    def initialize_gui(self) -> None:
        self._gui = InferenceGUI(
            [],
            self.api,
            support_pretrained_models=False,
            support_custom_models=True,
        )