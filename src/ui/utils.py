import torch
from collections import OrderedDict
from typing import Callable, Dict, Any, List, Optional
from supervisely.app import DataJson
from supervisely.app.widgets import Button, Widget, Container, Switch, Card, InputNumber, Empty


select_params = {"icon": None, "plain": False, "text": "Select"}
reselect_params = {"icon": "zmdi zmdi-refresh", "plain": True, "text": "Reselect"}


def update_custom_params(
    btn: Button,
    params_dct: Dict[str, Any],
) -> None:
    btn_state = btn.get_json_data()
    for key in params_dct.keys():
        if key not in btn_state:
            raise AttributeError(f"Parameter {key} doesn't exists.")
        else:
            DataJson()[btn.widget_id][key] = params_dct[key]
    DataJson().send_changes()


def update_custom_button_params(
    btn: Button,
    params_dct: Dict[str, Any],
) -> None:
    params = params_dct.copy()
    if "icon" in params and params["icon"] is not None:
        new_icon = f'<i class="{params["icon"]}" style="margin-right: {btn._icon_gap}px"></i>'
        params["icon"] = new_icon
    update_custom_params(btn, params)


def get_switch_value(switch: Switch):
    return switch.is_switched()


def set_switch_value(switch: Switch, value: bool):
    if value:
        switch.on()
    else:
        switch.off()


def disable_enable(widgets: List[Widget], disable: bool = True):
    for w in widgets:
        if disable:
            w.disable()
        else:
            w.enable()


def unlock_lock(cards: List[Card], unlock: bool = True):
    for w in cards:
        if unlock:
            w.unlock()
        else:
            w.lock()


def button_selected(
    select_btn: Button,
    disable_widgets: List[Widget],
    lock_cards: List[Card],
    lock_without_click: bool = False,
):
    if lock_without_click:
        disable_enable(disable_widgets, disable=False)
        unlock_lock(lock_cards, unlock=False)
        update_custom_button_params(select_btn, select_params)
        select_btn._click_handled = True
        return

    disable_enable(disable_widgets, select_btn._click_handled)
    unlock_lock(lock_cards, select_btn._click_handled)

    if select_btn._click_handled:
        update_custom_button_params(select_btn, reselect_params)
        select_btn._click_handled = False
    else:
        update_custom_button_params(select_btn, select_params)
        select_btn._click_handled = True


def get_devices():
    cuda_names = [
        f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(torch.cuda.device_count())
    ]
    cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    device_names = cuda_names + ["cpu"]
    torch_devices = cuda_devices + ["cpu"]
    return device_names, torch_devices


def create_linked_getter(
    widget1: InputNumber,
    widget2: InputNumber,
    switcher: Switch,
    get_first: bool = True,
) -> Callable[[Widget], Any]:
    """Return getter for widgets depends on switcher value.

    :param widget1: first input
    :type widget1: InputNumber
    :param widget2: second input
    :type widget2: InputNumber
    :param switcher: switcher widget
    :type switcher: Switch
    :param get_first: if True return getter for first widget, defaults to True
    :type get_first: bool, optional
    :return: getter function
    :rtype: Callable[[InputNumber], Any]
    """

    def getter(any_widget: InputNumber):
        widget1_val = widget1.value
        widget2_val = widget2.value

        if switcher.is_switched():
            widget1_val = None
        else:
            widget2_val = None

        if get_first:
            return widget1_val
        return widget2_val

    return getter


def HContainer(widgets: list, tight=True, gap=10, fractions=None, overflow="scroll"):
    if tight:
        fractions = [1]*len(widgets) + [10-len(widgets)]
        widgets.append(Empty())
    return Container(widgets, "horizontal", gap=gap, fractions=fractions, overflow=overflow)

def VContainer(widgets: list, gap=10, fractions=None, overflow="scroll"):
    return Container(widgets, "vertical", gap=gap, fractions=fractions, overflow=overflow)
