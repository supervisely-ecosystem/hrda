import shutil
from pathlib import Path
from itertools import chain


def find_candidates(data_dir, patterns_to_rename=None):
    patterns_to_rename = patterns_to_rename or ["*.jpeg", "*.jpeg.json"]
    paths = map(str, chain(*[Path(data_dir).rglob(pattern) for pattern in patterns_to_rename]))
    paths = list(paths)
    return paths

def rename(paths):
    for path in paths:
        new_name = path.replace(".jpeg", ".jpg")
        shutil.move(path, new_name)

def find_png(data_dir):
    # check for .png
    paths = list(map(str, Path(data_dir).rglob("*.png")))
    return paths
