# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 08 
"""
from typing import Tuple

import torch
from torch import nn
from models import SinusoidalPositionEmbeddings, ResnetDownsampleBlock2D, ResnetUpsampleBlock2D, UNetMidBlock2D

class Unet(nn.Module):
    def __init__(
            self,
            in_channels: int = 6,
            out_channels: int = 3,
            block_out_channels: Tuple[int, ...] = (64, 128, 128, 256)
    ):
        super(Unet, self).__init__()
        base_channels = block_out_channels[0]
        mid_channels = block_out_channels[-1]
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        :param x: shape = [b, c=6, h ,w] dim=4
        :param t: shape = [b] dim=1
        :return:  shape = [b, c=3, h, w] dim=4
        '''
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

        output_velocity = self.conv_out(x)
        return output_velocity


from utils import model_info
from thop import profile
if __name__ == "__main__":
    device = torch.device("cuda")
    model = Unet(
        in_channels=6,
        out_channels=3,
        block_out_channels=(64, 128, 128, 256)
    ).to(device)
    x = torch.randn(1, 6, 512, 512).to(device)
    t = torch.randn(1).to(device)
    model_info(model, (x, t))
    print(type(model))

    # model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    # output = model(x, t)
    # print(model)
    # print(output.shape)

