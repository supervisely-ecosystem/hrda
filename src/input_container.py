from typing import Callable, Dict, Any, List, Optional
from supervisely.app.widgets import (
    Button,
    Widget,
    Container,
    Switch,
    Card,
    InputNumber,
    Field,
    SelectString,
)


def unwrap_field(widget: Widget):
    if isinstance(widget, Field):
        return widget._content
    else:
        return widget


def get_value(widget: Widget):
    if isinstance(widget, Switch):
        return widget.is_switched()
    else:
        assert hasattr(widget, "value"), f'The widget {widget} hasn\'t the "value" attr'
        return widget.value


def set_value(widget: Widget, value):
    if isinstance(widget, Switch):
        if value:
            widget.on()
        else:
            widget.off()
    else:
        assert hasattr(widget, "value"), f'The widget {widget} hasn\'t the "value" attr'
        widget.value = value


class InputContainer:
    def __init__(self) -> None:
        self._container = None

    def get_widget(self, name):
        # as this class returns a value of the widget (not widget itself),
        # this method is useful when you need to get the widget itself
        attr = super().__getattribute__(name)
        return attr

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if (
            isinstance(attr, Widget)
            and not isinstance(attr, Container)
            and hasattr(self, "_container")
            and self._container is not None
        ):
            widget = unwrap_field(attr)
            return get_value(widget)
        return attr

    def __setattr__(self, name, value):
        try:
            attr = super().__getattribute__(name)
            if (
                isinstance(attr, Widget)
                and not isinstance(attr, Container)
                and hasattr(self, "_container")
                and self._container is not None
            ):
                widget = unwrap_field(attr)
                set_value(widget, value)
                return
        except AttributeError:
            pass
        super().__setattr__(name, value)

    def compile(self, container):
        self._container = container

    def get_content(self):
        return self._container
