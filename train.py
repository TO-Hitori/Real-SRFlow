# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 13 
"""
import os.path
from argparse import ArgumentParser
import pytorch_lightning as pl
import torchvision
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import instantiate_from_config
from utils import model_info, save_tensor
from recrified_flows import RecrifiedFlow


def main(config) -> None:
    pl.seed_everything(config.seed)
    device = torch.device(config.device)
    print("训练设备：", device)

    # 数据准备
    dataset_train = instantiate_from_config(config.train_dataset)
    print("训练图像总数：", dataset_train.__len__())
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        num_workers=config.dataloader_train.num_workers,
        shuffle=config.dataloader_train.shuffle,
        persistent_workers=config.dataloader_train.persistent_workers
    )
    print("每个epoch的步数：", len(dataloader_train))

    dataset_val = instantiate_from_config(config.val_dataset)
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    print("验证图像总数：", len(dataloader_val))

    # 构建模型
    model = instantiate_from_config(config.model)
    if config.model.resume is not None:
        model.load_state_dict(torch.load(config.model.resume, weights_only=True))
        print("--载入权重：" + config.model.resume)
    RF = RecrifiedFlow(model).to(device)
    model_info(model, device=device)

    # 构建优化器 学习率策略
    optimizer = AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    print(optimizer)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.optim.max_lr,
        steps_per_epoch=len(dataloader_train),
        epochs=config.epochs
    )
    print(scheduler)

    # 实验数据记录
    experiments_root = os.path.join("./Experiment", config.experiments_name)
    log_root = os.path.join(experiments_root, "logs")
    weight_root = os.path.join(experiments_root, "weights")
    os.makedirs(experiments_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(weight_root, exist_ok=True)

    print("实验数据保存位置：" + experiments_root)
    print("tensorboard位置：" + log_root)
    print("权重位置：" + log_root)
    # 本次实验log位置
    num_logs = len(os.listdir(log_root))
    log_now = os.path.join(log_root, f"log{num_logs}")
    os.makedirs(log_now, exist_ok=True)
    sw = SummaryWriter(log_now)
    OmegaConf.save(config, os.path.join(log_now, "config.yaml"))


    global_step = 0
    for epoch in range(config.epochs):
        loop = tqdm(dataloader_train)
        for i, sample in enumerate(loop):
            image_hr = sample["image_hr"].to(device)
            image_deg = sample["image_deg"].to(device)

            loss = RF(image_hr, image_deg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step < 100:
                sw.add_scalar("loss", loss.item(), global_step=global_step)
                sw.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=global_step)
            if global_step % config.log_step_freq and global_step >= 100:
                sw.add_scalar("loss", loss.item(), global_step=global_step)
                sw.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=global_step)
            if global_step == 1 or global_step % 200 == 0:
                for val_sample in dataloader_val:
                    image_name = val_sample["image_name"][0]
                    image = val_sample["up_image"].to(device)

                    x_t_list = RF.eular_sample(image)
                    xt_tensor = torch.cat(x_t_list, dim=0)
                    img_grid = torchvision.utils.make_grid(
                        xt_tensor, normalize=False, nrow=6
                    )
                    sw.add_image(image_name + "eural", img_grid, global_step=global_step)

                    if global_step == 1:
                        img = val_sample["image"]
                        img_grid_ori = torchvision.utils.make_grid(
                            img, normalize=True
                        )
                        sw.add_image(image_name, img_grid_ori, global_step=1)


        if epoch < 2 or epoch % config.weight_save_per_epoch == 0:
            weight_name = f"logs_{num_logs}_epoch_{epoch}.pth"
            torch.save(
                model.state_dict(),
                os.path.join(weight_root, weight_name)
            )
            print("保存权重：" + os.path.join(weight_root, weight_name))


if __name__ == "__main__":
    # 获取配置文件
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/basic_train.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(config))

    main(config)
