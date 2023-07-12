from src.ui import models, classes, train_val_split, augmentations

@models.select_btn.click
def slb():
    classes.card.unlock()