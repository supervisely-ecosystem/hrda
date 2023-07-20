import numpy as np
import supervisely as sly
from mmcv.runner.hooks import HOOKS, Hook, EvalHook, CheckpointHook
from mmcv.runner import Runner

from src.ui import training as train_ui
import src.globals as g
from src.sly_dataset import SuperviselyDataset
from src.sly_utils import get_ann_from_np_mask


@HOOKS.register_module()
class SuperviselyHook(Hook):
    def __init__(self, chart_update_interval: int = 1, **kwargs):
        self.chart_update_interval = chart_update_interval
        self.iter_progress = None

    def before_run(self, runner: Runner) -> None:
        train_ui.show_train_widgets()
        self.iter_progress = train_ui.iter_progress(message="Iterations", total=runner.max_iters)

    def after_train_iter(self, runner: Runner) -> None:
        loss: float = runner.outputs["log_vars"]["decode.loss_seg"]
        i = runner.iter

        # Check nans
        if not np.isfinite(loss):
            sly.logger.warn(f"Loss become infinite or NaN! (loss={loss})")
            raise StopIteration()

        # Update progress bars
        self.iter_progress.update(1)

        # Update training charts (loss, lr, grad, etc.)
        if self.every_n_iters(runner, self.chart_update_interval):
            lr = runner.current_lr()[0]
            train_ui.monitoring.add_scalar("train", "Loss", "loss", i, loss)
            train_ui.monitoring.add_scalar("train", "LR", "lr", i, lr)

        # Validation charts,
        # Prediction previews
        eval_hook = self._get_eval_hook(runner)
        assert eval_hook is not None, "can't find the EvalHook"
        if eval_hook._should_evaluate(runner):
            dataset = eval_hook.dataloader.dataset
            assert isinstance(
                dataset, SuperviselyDataset
            ), f"Evaluation dataset {dataset} is not a SuperviselyDataset instance."

            # Update validation charts
            if dataset.last_eval_results is not None:
                results: dict = dataset.last_eval_results
                m_iou, per_class_iou = self.extract_metrics(results)
                train_ui.monitoring.add_scalar("val", "mIoU", "mIoU", i, m_iou)
                for class_name, value in per_class_iou.items():
                    train_ui.monitoring.add_scalar("val", "Per-class IoU", class_name, i, value)
            else:
                sly.logger.warn("dataset.last_eval_results is None")

            # Update prediction preview
            # we will draw the first image in val dataset
            if dataset.last_model_outputs is not None:
                outputs = dataset.last_model_outputs
                img_info = dataset.img_infos[0]
                img_path = f"{dataset.img_dir}/{img_info['filename']}"
                ann_path = f"{dataset.ann_dir}/{img_info['ann']['seg_map']}"
                gt = sly.image.read(ann_path)
                pred = outputs[0]
                ann_pred = get_ann_from_np_mask(pred, dataset.CLASSES, dataset.PALETTE)
                ann_gt = get_ann_from_np_mask(gt, dataset.CLASSES, dataset.PALETTE)
                train_ui.update_prediction_preview(img_path, ann_pred, ann_gt)

            else:
                sly.logger.warn("dataset.last_model_outputs is None")

        # Stop training
        if g.state.stop_training:
            checkpoint_hook: CheckpointHook = self._get_checkpoint_hook(runner)
            checkpoint_hook._save_checkpoint(runner)
            raise StopIteration()

    def extract_metrics(self, results: dict):
        per_class = {}
        for k, v in results.items():
            if k.startswith("IoU."):
                k_new = k[4:]
                per_class[k_new] = v
        return results["mIoU"], per_class

    def _get_eval_hook(self, runner: Runner):
        # by default it should be at index 2
        if isinstance(runner.hooks[2], EvalHook):
            return runner.hooks[2]
        else:
            for hook in runner.hooks:
                if isinstance(hook, EvalHook):
                    return hook

    def _get_checkpoint_hook(self, runner: Runner):
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                return hook
