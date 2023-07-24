from supervisely.app.widgets import ClassesTable, Card, Container, Button, Switch, Field
from supervisely.app.content import StateJson
import supervisely as sly

from src.globals import PROJECT_ID


allowed_shapes = [sly.Bitmap, sly.Polygon]
classes = ClassesTable(project_id=PROJECT_ID, allowed_types=allowed_shapes)
classes.select_all()

filter_images_without_gt_input = Switch(True)
filter_images_without_gt_field = Field(
    filter_images_without_gt_input,
    title="Filter images without annotations",
    description="After selecting classes, some images may not have any annotations. Whether to remove them?",
)

card = Card(
    title="3️⃣ Training classes",
    description=(
        "Select classes that will be used for training. "
        "Supported shapes are Bitmap, Polygon, Rectangle."
    ),
    content=Container([classes, filter_images_without_gt_field]),
)
