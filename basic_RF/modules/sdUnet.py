# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 03
Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py

"""
import math

import torch
from diffusers import UNet2DConditionModel
from torch import nn
from typing import Optional


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}

class Unet2DModel_sd(nn.Module):
    """
    in_channels:  输入样本的通道数
    out_channels：输出样本的通道数
    block_out_channels: 每一层的输出通道数
    time_embedding_dim: 时间编码的维度，不设置则为 block_out_channels[0] x 4
    act_fn: 激活函数的种类，在 ACTIVATION_FUNCTIONS 字典中
    down_block_types: 下采样块的组成类型
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=[64, 128, 256, 256],
        time_embedding_dim=None,
        act_fn="silu",
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),


    ):
        super(Unet2DModel_sd, self).__init__()
        # input
        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=block_out_channels[0], kernel_size=3, padding=1
        )
        # time
        timestep_input_dim, time_embed_dim = self._set_time_proj(time_embedding_dim, block_out_channels)
        self.time_proj = Timesteps(block_out_channels[0])
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        # block
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])


    def forward(self, sample, timestep):
        pass

    def _set_time_proj(self, time_embedding_dim, block_out_channels):
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        timestep_input_dim = block_out_channels[0]
        return timestep_input_dim, time_embed_dim



'''
time
'''
class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def get_timestep_embedding(
            self,
            timesteps: torch.Tensor,
            max_period: int = 10000,
    ):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

        Args
            timesteps (torch.Tensor):
                a 1-D Tensor of N indices, one per batch element. These may be fractional.
            self.num_channels (int):
                the dimension of the output.
            max_period (int):
                Controls the maximum frequency of the embeddings
        Returns
            torch.Tensor: an [N x dim] Tensor of positional embeddings.
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        half_dim = self.num_channels // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / half_dim

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # concat sine and cosine embeddings
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

        # zero pad
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, timesteps):
        t_emb = self.get_timestep_embedding(timesteps)
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        timestep_input_dim: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(timestep_input_dim, time_embed_dim, sample_proj_bias)

        self.act = ACTIVATION_FUNCTIONS[act_fn]

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        return sample