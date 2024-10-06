# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 06 
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def print_info(img):
    print("=-"*20)
    print(type(img))
    print(img.shape)
    print(img.dtype)
    print(img.min(), img.max())
    print("=-" * 20)

def save_numpy(img, save_path="./result/save_numpy.png"):
    img = Image.fromarray(img)
    img.save(save_path)


class LaionHRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        crop_size: int = 512,

    ):
        super(LaionHRDataset, self).__init__()
        self.data_root = data_root
        self.image_list = os.listdir(data_root)
        self.crop_size = crop_size

        self.set_module()

    def set_module(self):
        self.image_crop_module = A.RandomCrop(self.crop_size, self.crop_size)
        self.image_aug_module = A.Compose([
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.2),
            A.RandomRotate90(p=0.8)
        ])
        self.image_degra_module = A.Compose([
            A.AdvancedBlur(
                blur_limit=(3, 21),
                sigma_x_limit=(0.3, 3.0),
                sigma_y_limit=(0.3, 3.0),
                p=1.0
            )
        ])

    def __getitem__(self, item):
        image_path = os.path.join(self.data_root, self.image_list[item])
        image = np.array(Image.open(image_path).convert('RGB'))
        image_hr = self.random_crop(image)
        image_hr = self.image_aug(image_hr)

        image_deg = self.image_degra(image)


        return image_deg

    def random_crop(self, img):
        return self.image_crop_module(image=img)["image"]

    def image_aug(self, img):
        return self.image_aug_module(image=img)["image"]

    def image_degra(self, img):
        return self.image_degra_module(image=img)["image"]

    def __len__(self) -> int:
        return len(self.image_list)


from torch.utils.data import DataLoader

if __name__ == "__main__":
    data_root = "C:/MyDataset/BSR/laion-high-resolution/00000/"
    batch_size = 16

    laion_ds = LaionHRDataset(data_root)

    sample = laion_ds.__getitem__(25)
    print_info(sample)
    save_numpy(sample, "./result/deg1.png")

    data_loader = DataLoader(
        dataset=laion_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )



