from src import sly_utils


class State:
    def __init__(self) -> None:
        # TODO: remove hardcoded:
        self.model = None
        self.classes = ["lemon", "kiwi"]
        self.augs_config_path = "aug_templates/medium.json"
        self.source_dataset = "ds1"
        self.target_dataset = "ds2"
        self.val_dataset = "ds2"
        self.selected_dataset_ids = None
        self.batch_size = 2

        self.is_custom_model_selected = None
        self.remote_weights_path = None
        self.local_weights_path = None

        self.general_params = None
        self.checkpoint_params = None
        self.optimizer_params = None

        self.stop_training = False

        self._previous_classes = None

    def update(self):
        from src.ui import augmentations, classes, models, train_val_split
        from src.ui.hyperparameters import general_params, checkpoint_params, optimizer_params

        self._previous_classes = self.classes
        self.classes = classes.classes.get_selected_classes()
        if "__bg__" in self.classes:
            self.classes.remove("__bg__")
        assert len(self.classes) > 0, f"Please, select at least 1 class for training."

        self.model = models.get_selected_pretrained_model()
        self.is_custom_model_selected = not models.is_pretrained_model_selected()
        if self.is_custom_model_selected:
            self.remote_weights_path = models.get_selected_custom_path()
            self.local_weights_path = sly_utils.get_local_weights_path(self.remote_weights_path)
        else:
            self.remote_weights_path = None
            self.local_weights_path = None

        self.augs_config_path = augmentations.get_selected_config_path()

        self.source_dataset = train_val_split.select_train_labeled.get_value()
        self.target_dataset = train_val_split.select_train_unlabeled.get_value()
        self.val_dataset = train_val_split.select_val_labeled.get_value()

        datasets = {info.name: info.id for info in train_val_split.dataset_infos}
        sleceted_ds_names = [self.source_dataset, self.target_dataset, self.val_dataset]
        self.selected_dataset_ids = [datasets[name] for name in sleceted_ds_names]
        self.n_images = sly_utils.get_images_count(set(self.selected_dataset_ids))

        self.general_params = general_params
        self.checkpoint_params = checkpoint_params
        self.optimizer_params = optimizer_params

        self.experiment_name = self.general_params.experiment_name
