# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 13 
"""
import os.path
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from tqdm import tqdm


class RecrifiedFlow(nn.Module):
    def __init__(self, model: nn.Module):
        super(RecrifiedFlow, self).__init__()
        self.model = model
        self.loss_func = F.mse_loss

    def sample_x_t(self, x_0: Tensor, x_1:Tensor, t: Tensor) -> Tensor:
        '''
        获取图像x1在t时刻的噪声图像xt

        :param x_1: 高分辨率图像 shape = [b, c, h, w], range = [-1, 1]
               x_0: 噪声图像    shape = [b, c, h, w]
        :param t: 时间 shape = [b]
                       range = [0, 1]
        :return: x_t: 时刻的图像 和 x_0 加入的噪声
        '''
        # [b] -> [b, 1, 1, 1]
        t = t[:, None, None, None]
        x_t = t * x_1 + (1 - t) * x_0
        return x_t

    def sample_x_t_demo(self, x_1: Tensor, sample_step: int = 5) -> List[Tensor]:
        device = x_1.device

        t = torch.linspace(0, 1, sample_step + 1).to(device)
        x_0 = torch.randn_like(x_1, device=device)

        xt_list = []
        for ti in t:
            ti = ti.view(1)
            xt = self.sample_x_t(x_0, x_1, ti)
            xt_list.append(xt)
        return xt_list

    def eular_sample(self, cond: Tensor, sample_step: int = 5) -> List[Tensor]:
        """
        :param cond: shape = [b, c, h, w]
        :param sample_step:
        :return:
        """
        self.model.eval()
        device = cond.device
        batch_size = cond.shape[0]

        # 计算步长
        dt = 1.0 / sample_step
        dt_shape = (batch_size,) + (1,) * len(cond.shape[1:])
        dt = torch.tensor([dt] * batch_size, device=device).view(*dt_shape)
        # 获取初始噪声 x_t, t=0
        x_t = torch.randn_like(cond, device=device)
        # 保存每个状态
        x_t_list = [x_t]
        for i in tqdm(range(sample_step)):
            # 计算当前时间
            t = i / sample_step
            t = torch.tensor([t] * batch_size, device=device)
            # 计算速度
            with torch.no_grad():
                v_pred = self.model(torch.cat([x_t, cond], dim=1), t)
            x_t = x_t + v_pred * dt
            x_t_list.append(x_t)
        self.model.train()
        x_t_list = [xt * 0.5 + 0.5 for xt in x_t_list]
        x_t_list = [xt.clamp(0, 1) for xt in x_t_list]
        return x_t_list

    def forward(self, x_1, cond):
        '''
        :param x_1: 高分辨率图像 shape = [b, c, h, w], b >= 2
        :param cond: 退化图像   shape = [b, c, h, w]
        :return: loss
        '''
        device = x_1.device
        batch_size = x_1.shape[0]
        # 生成随机时间序列
        t_half = torch.randn(batch_size // 2).to(device)
        t_half = torch.sigmoid(t_half)
        t = torch.cat([t_half, 1 - t_half], dim=0)
        # 获取随机噪声x_0 和 加噪图片 x_t
        x_0 = torch.randn_like(x_1, device=device)
        x_t = self.sample_x_t(x_0, x_1, t)
        # 预测速度
        v_pred = self.model(torch.cat([x_t, cond], dim=1), t)
        # 计算 损失函数
        loss = self.loss_func((x_1 - x_0), v_pred)
        return loss



import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_image2tensor, save_tensor, print_info
from torchvision.utils import make_grid

from models import Unet
if __name__ == "__main__":
    device = torch.device("cuda")
    model = Unet()
    rf = RecrifiedFlow(model=model)

    # batch_size = 10
    #
    # dt = 1.0 / batch_size
    # print(dt)
    # dt_shape = (batch_size,) + (1,) * 3
    # print(dt_shape)
    # dt = torch.tensor([dt] * batch_size).view(*dt_shape)
    # print(dt, dt.shape)

    """
    eular_sample
    """
    image_path = r"./datasets/val_image_up/up0030.jpg"
    save_path = "./rf_sample/eular_sample/"

    image = load_image2tensor(image_path)
    x_t_list = rf.eular_sample(image, 5)

    xt_tensor = torch.cat(x_t_list, dim=0)


    save_tensor(xt, image_save_path)


    for i, xt in enumerate(x_t_list):
        print_info(xt)
        image_save_path = os.path.join(save_path, f"sample{i}.png")
        save_tensor(xt, image_save_path)


    '''
    sample xt demo
    '''
    # sample_step = 10
    # image_path = r"C:\Users\WJQpe\Downloads\pixiv\116270273_p0.jpg"
    # image_name = ""
    # save_path = './rf_sample/get_xt/'
    #
    #
    # image = load_image2tensor(image_path)
    #
    # t = torch.linspace(0, 1, sample_step+1)
    # xt_list = rf.sample_x_t_demo(image, sample_step=sample_step)
    # for i, xt in enumerate(xt_list):
    #     image_save_path = os.path.join(save_path, image_name + f"x{t[i].item(): .3f}.png")
    #     save_tensor(xt, image_save_path)
    #     print(type(xt))
    #     print_info(xt)



