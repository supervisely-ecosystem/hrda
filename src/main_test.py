import os

import numpy as np
import src.globals as g
import supervisely as sly
from supervisely.app.widgets import Container, Image, LabeledImage, GridGallery, Button


def get_image():
    p = f"{g.STATIC_DIR}/IMG_0748.jpeg"
    h, w, c = sly.image.read(p).shape

    x = np.linspace(0, 1, h * w)
    y = np.sin(x * np.random.rand() * 1000) > 0.5
    y = y.reshape(h, w)
    l = sly.Label(sly.Bitmap(y), sly.ObjClass("test", sly.Bitmap, [255, 0, 0]))
    ann = sly.Annotation((h, w), [l])
    return f"static/IMG_0748.jpeg", ann


image_preview = GridGallery(2, enable_zoom=True, sync_views=True)
image_preview.append(*get_image())
image_preview.append(*get_image())
btn = Button("Click")


widgets = [image_preview, btn]
layout = Container(widgets=widgets)
app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)

g.app = app


@btn.click
def on_click():
    pass
