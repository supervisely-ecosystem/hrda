IGNORE__BG__CLASS = True


class State:
    def __init__(self) -> None:
        self.model = None
        self.classes = ["lemon", "kiwi"]
        self.augs_config_path = "aug_templates/medium.json"
        self.source_dataset = "ds1"
        self.target_dataset = "ds2"
        self.val_dataset = "ds2"
        self.batch_size = 2

        self.general_params = None
        self.checkpoint_params = None
        self.optimizer_params = None

        self.stop_training = False

    def update(self):
        from src.ui import augmentations, classes, models, train_val_split
        from src.ui.hyperparameters import general_params, checkpoint_params, optimizer_params

        self.classes = classes.classes.get_selected_classes()
        if IGNORE__BG__CLASS and "__bg__" in self.classes:
            self.classes.remove("__bg__")
        self.model = models.get_selected_pretrained_model()
        self.augs_config_path = augmentations.get_selected_config_path()
        self.source_dataset = train_val_split.select_train_labeled.get_value()
        self.target_dataset = train_val_split.select_train_unlabeled.get_value()
        self.val_dataset = train_val_split.select_val_labeled.get_value()

        self.general_params = general_params
        self.checkpoint_params = checkpoint_params
        self.optimizer_params = optimizer_params
