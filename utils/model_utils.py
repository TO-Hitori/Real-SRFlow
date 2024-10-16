# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 13 
"""
import torch
from thop import profile
from torch import nn
from safetensors.torch import save_file, load_file

def print_info(x: torch.Tensor):
    print("=-"*20)
    print("type:", type(x))
    print("shape: ", x.shape)
    # print("sum: ", x.sum())
    # print("abs_sum: ", x.abs().sum())
    # print("abs_mean: ", x.abs().mean())
    print("range: ", x.max(), x.min(), x.max() - x.min())

def disabled_train(self: nn.Module=None) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def frozen_module(module: nn.Module) -> None:
    module.eval()
    # module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False

def model_info(model, inputs=None, device=None):
    if inputs is None:
        x = torch.randn(1, 6, 256, 256).to(device)
        t = torch.randn(1).to(device)

        inputs = (x, t)
    flops, params = profile(model, inputs=inputs)
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"FLOPs: {flops / 1e12}TFLOPs, Params: {params / 1e6}M")

