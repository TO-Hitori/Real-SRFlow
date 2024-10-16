# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 09 
"""
from typing import Mapping, Any
import importlib


def get_obj_from_str(string: str, reload: bool = False) -> object:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


import torch
if __name__ == "__main__":
    nn2 = getattr(torch.nn, "Conv2d")
    cnn = torch.nn.Conv2d(3, 3, 3)
    cnn2 = nn2(3, 3, 3)
    mode = get_obj_from_str("torch.nn.Conv2d")
    cnn3 = mode(3, 3, 3)

    print(type(cnn))
    print(type(cnn2))
    print(type(cnn3))

    print(cnn)
    print(cnn2)
    print(cnn3)

    cnn_dict = {
        'in_channels': 32,
        'out_channels': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
    }
    cnn4 = mode(**cnn_dict)
    print(cnn4)