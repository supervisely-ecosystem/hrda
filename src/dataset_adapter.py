import supervisely as sly


class DatasetAdapter:
    def __init__(self) -> None:
        pass

    def convert_annotations(self):
        # sly2X
        pass

    def filter_by_classes(self, selected_classes):
        pass

    def filter_without_gt(self):
        pass

    def process(self):
        # naive copy
        self.convert_annotations()
        self.filter_by_classes()
        # return paths to train/val/etc in X
