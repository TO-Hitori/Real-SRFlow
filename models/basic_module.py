# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 08 
"""
import math
from typing import Any, Tuple

from diffusers import UNet2DConditionModel
import torch
from torch import nn
import torch.nn.functional as F

def exist(x: Any) -> bool:
    return x is not None

'''
position embedding block
'''
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, timestep_input_dim: int, time_embedding_dim: int, time_scale: float = 1000, max_period: int = 10000):
        """
        :param timestep_input_dim: = base_channels
        :param time_embedding_dim: = base_channels x 4
        :param time_scale:
        :param max_period:
        """
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.timestep_input_dim = min(timestep_input_dim, 1024)
        self.time_embedding_dim = time_embedding_dim

        self.max_period = max_period
        self.time_scale = time_scale

        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

    def sinusoidal_positional_encoding(self, time_step: torch.Tensor) -> torch.Tensor:
        device = time_step.device

        half_dim = self.timestep_input_dim // 2
        freqs = torch.exp(-torch.arange(half_dim) / (half_dim - 1) * math.log(self.max_period)).to(device)
        arc = self.time_scale * time_step[:, None] * freqs[None, :]
        embeddings = torch.cat((arc.sin(), arc.cos()), dim=-1)
        return embeddings

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        time_embedding = self.sinusoidal_positional_encoding(time_step)
        time_embedding_proj = self.time_mlp(time_embedding)
        return time_embedding_proj

'''
basic cnn block
'''
class UpSample(nn.Module):
    def __init__(self, dim: int, use_conv: bool = True):
        super(UpSample, self).__init__()
        self.dim = dim
        self.use_conv = use_conv
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1) if self.use_conv else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            h = self.conv(h)
        return h


class DownSample(nn.Module):
    def __init__(self, dim: int, use_conv: bool = True):
        super(DownSample, self).__init__()
        self.dim = dim
        self.use_conv = use_conv
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1) if self.use_conv else nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int = None, num_groups: int = 8):
        super(ResnetBlock, self).__init__()
        self.block1 = Block(in_channels, out_channels, num_groups)
        self.block2 = Block(out_channels, out_channels, num_groups)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        ) if exist(time_embedding_dim) else None

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor = None) -> torch.Tensor:
        h = self.block1(x)

        if exist(self.mlp) and exist(time_embedding):
            time_embedding = self.mlp(time_embedding)
            h = time_embedding[:, :, None, None] + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetDownsampleBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_embedding_dim: int,
            add_downsample: bool = True
    ):
        super(ResnetDownsampleBlock2D, self).__init__()

        self.resnet1 = ResnetBlock(in_channels, out_channels, time_embedding_dim)
        self.resnet2 = ResnetBlock(out_channels, out_channels, time_embedding_dim)
        self.down_sample = DownSample(out_channels) if add_downsample else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.resnet1(x, t)
        h2 = self.resnet2(h1, t)
        output = self.down_sample(h2)
        return output, h2


class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels: int, time_embedding_dim: int):
        super(UNetMidBlock2D, self).__init__()
        self.resnet1 = ResnetBlock(in_channels, in_channels, time_embedding_dim)
        self.resnet2 = ResnetBlock(in_channels, in_channels, time_embedding_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h1 = self.resnet1(x, t)
        h2 = self.resnet2(h1, t)
        return h2


class ResnetUpsampleBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_embedding_dim: int,
            add_upsample: bool = True
    ):
        super(ResnetUpsampleBlock2D, self).__init__()
        self.resnet1 = ResnetBlock(in_channels, out_channels, time_embedding_dim)
        self.resnet2 = ResnetBlock(out_channels, out_channels, time_embedding_dim)
        self.up_sample = UpSample(out_channels) if add_upsample else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h1 = self.resnet1(x, t)
        h2 = self.resnet2(h1, t)
        output = self.up_sample(h2)
        return output


"""
U-net for print info
"""
class Unet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            block_out_channels=(64, 128, 256, 512)
    ):
        super(Unet, self).__init__()
        base_channels = block_out_channels[0]
        mid_channels = block_out_channels[-1]
        print(base_channels, mid_channels)
        # time dim
        timestep_input_dim = base_channels
        time_embedding_dim = base_channels * 4
        # unet dim
        all_channels = (base_channels,) + block_out_channels
        in_out = list(zip(all_channels[:-1], all_channels[1:]))

        # time
        self.time_embedding_layer = SinusoidalPositionEmbeddings(
            timestep_input_dim=timestep_input_dim,
            time_embedding_dim=time_embedding_dim
        )

        # input
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        # down
        num_resolutions = len(in_out)
        for idx, (input_channels, output_channels) in enumerate(in_out):
            add_downsample = idx < (num_resolutions - 1)
            self.down_blocks.append(
                ResnetDownsampleBlock2D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    time_embedding_dim=time_embedding_dim,
                    add_downsample=add_downsample
                )
            )
        # mid
        self.mid_block = UNetMidBlock2D(mid_channels, time_embedding_dim)

        # up
        for idx, (input_channels, output_channels) in enumerate(reversed(in_out)):
            add_upsample = idx < (num_resolutions - 1)
            self.up_blocks.append(
                ResnetUpsampleBlock2D(
                    in_channels=output_channels * 2,
                    out_channels=input_channels,
                    time_embedding_dim=time_embedding_dim,
                    add_upsample=add_upsample
                )
            )

        # out
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x, t):
        # input & time
        x = self.conv_in(x)
        t = self.time_embedding_layer(t)

        hidden_states = []
        # downsample
        for i, down_block in enumerate(self.down_blocks):
            x, hidden = down_block(x, t)
            hidden_states.append(hidden)

        # mid
        x = self.mid_block(x, t)

        # upsample
        for i, up_block in enumerate(self.up_blocks):
            x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_block(x, t)

        output = self.conv_out(x)
        return output

    def forward_print(self, x, t):
        print("input x", x.shape)
        print("input t", t.shape)
        # input & time
        x = self.conv_in(x)
        t = self.time_embedding_layer(t)
        print("conv in x", x.shape)
        print("time embedding", t.shape)


        hidden_states = []
        # downsample
        print("--downsample")
        for i, down_block in enumerate(self.down_blocks):
            x, hidden = down_block(x, t)
            hidden_states.append(hidden)
            print(f"down{i}-hidden:{hidden.shape}")
            print(f"down{i}-x:{x.shape}")

        # mid
        print("--mid:", x.shape)
        x = self.mid_block(x, t)


        print("hidden state")
        for h in hidden_states:
            print(h.shape)

        # upsample
        print("--upsample")
        for i, up_block in enumerate(self.up_blocks):
            x = torch.cat([x, hidden_states.pop()], dim=1)
            print(f"up{i}-input:{x.shape}")
            x = up_block(x, t)
            print(f"up{i}-out:{x.shape}")

        output = self.conv_out(x)
        print(f"output: {output.shape}")
        return output





from math import pi
from torch.utils.tensorboard import SummaryWriter
from diffusers import UNet2DConditionModel
if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.__version__)

    sw = SummaryWriter('./logs/model/unet')

    model = Unet()

    x = torch.randn(1, 3, 256, 256)
    t = torch.randn(1)

    sw.add_graph(model, [x, t])
    sw.close()

    output = model(x, t)
    print(model)
    print(output.shape)



