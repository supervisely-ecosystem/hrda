from collections import OrderedDict
from typing import Dict, Optional, List, Tuple, Union
from collections import OrderedDict
from supervisely.app.widgets import GridPlot, Container, Field, Empty
from supervisely.app.content import StateJson, DataJson

NumT = Union[int, float]


class StageMonitoring(object):
    def __init__(self, stage_id: str, title: str, description: str = None) -> None:
        self._name = stage_id
        self._metrics = OrderedDict()
        self._title = title
        self._description = description

    def create_metric(self, metric: str, series: Optional[List[str]] = None, **kwargs):
        if metric in self._metrics:
            raise ArithmeticError("Metric already exists.")

        if series is None:
            srs = []
        else:
            srs = [
                {
                    "name": ser,
                    "data": [],
                }
                for ser in series
            ]

        self._metrics[metric] = {
            "title": metric,
            "series": srs,
        }
        self._metrics[metric].update(kwargs)

    def create_series(self, metric: str, series: Union[List[str], str]):
        if isinstance(series, str):
            series = [series]
        new_series = [{"name": ser, "data": []} for ser in series]
        self._metrics[metric]["series"].extend(new_series)

    def compile_grid_field(
        self,
        make_right_indent=True,
    ) -> Tuple[Union[Field, Container], GridPlot]:
        if make_right_indent is True:
            self.create_metric("right_indent_empty_plot", ["right_indent_empty_plot"])

        data = list(self._metrics.values())
        grid = GridPlot(data, columns=len(data))

        if make_right_indent is True:
            grid._widgets["right_indent_empty_plot"].hide()

        field = Field(grid, self._title, self._description)
        return field, grid

    @property
    def name(self):
        return self._name


class Monitoring(object):
    def __init__(self) -> None:
        self._stages = {}
        self.container = None

    def add_stage(self, stage: StageMonitoring, make_right_indent=True):
        field, grid = stage.compile_grid_field(make_right_indent)
        self._stages[stage.name] = {}
        self._stages[stage.name]["compiled"] = field
        self._stages[stage.name]["raw"] = grid

    def add_scalar(
        self,
        stage_id: str,
        metric_name: str,
        serise_name: str,
        x: NumT,
        y: NumT,
    ):
        self._stages[stage_id]["raw"].add_scalar(f"{metric_name}/{serise_name}", y, x)

    def add_scalars(
        self,
        stage_id: str,
        metric_name: str,
        new_values: Dict[str, NumT],
        x: NumT,
    ):
        self._stages[stage_id]["raw"].add_scalars(
            metric_name,
            new_values,
            x,
        )

    def compile_monitoring_container(self, hide: bool = False) -> Container:
        if self.container is None:
            self.container = Container([stage["compiled"] for stage in self._stages.values()])
        if hide:
            self.container.hide()
        return self.container

    def clean_up(self):
        for stage in self._stages.values():
            for line_plot in stage["raw"]._widgets.values():
                line_plot.clean_up()
