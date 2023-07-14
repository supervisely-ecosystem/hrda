import supervisely as sly
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SlyImgAugs(object):
    def __init__(self, config_path):
        self.config_path = config_path
        if self.config_path is not None:
            config = sly.json.load_json_file(self.config_path)
            self.augs = sly.imgaug_utils.build_pipeline(
                config["pipeline"], random_order=config["random_order"]
            )

            sly.logger.debug(
                "SlyImgAugs loaded: ",
                extra=dict(config_path=self.config_path, pipeline=config["pipeline"]),
            )
        else:
            sly.logger.debug("SlyImgAugs: config_path is None.")

    def _apply_augs(self, results):
        if self.config_path is None:
            return
        img = results["img"]
        mask = results["gt_semantic_seg"]
        res_img, res_mask = sly.imgaug_utils.apply_to_image_and_mask(self.augs, img, mask)
        results["img"] = res_img
        results["gt_semantic_seg"] = res_mask
        results["img_shape"] = res_img.shape

    def __call__(self, results):
        self._apply_augs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(config_path={self.config_path})"
        return repr_str
