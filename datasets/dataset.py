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

        self.first_degra_module = A.Compose([
            A.AdvancedBlur(
                blur_limit=(7, 21),
                sigma_x_limit=(0.2, 3.0),
                sigma_y_limit=(0.2, 3.0),
                beta_limit=(0.5, 4),
                p=1.0
            ),
            A.OneOf([
                A.Resize(self.crop_size // 2, self.crop_size // 2, interpolation=cv2.INTER_LINEAR_EXACT, p=0.25),
                A.Resize(self.crop_size // 2, self.crop_size // 2, interpolation=cv2.INTER_CUBIC, p=0.25),
                A.Resize(self.crop_size // 2, self.crop_size // 2, interpolation=cv2.INTER_AREA, p=0.25),
                A.Resize(self.crop_size // 2, self.crop_size // 2, interpolation=cv2.INTER_LANCZOS4, p=0.25),
            ], p=1.0),
            A.OneOf([
                A.GaussNoise(var_limit=(1, 30), p=0.5),
                A.ISONoise(intensity=(0.05, 2), p=0.5),
            ], p=1.0),


        ])

    def __getitem__(self, item):
        image_path = os.path.join(self.data_root, self.image_list[item])
        image = np.array(Image.open(image_path).convert('RGB'))

        image_hr = self.random_crop(image)
        image_hr = self.image_aug(image_hr)

        image_deg = self.first_degra_module(image=image_hr)["image"]

        return image_hr, image_deg

    def set_module(self):
        '''
        INTER_NEAREST (0): 最近邻插值。速度最快，但可能会产生块状效果。
        INTER_LINEAR (1): 双线性插值。速度和质量之间的平衡，常用于放大图像。
        INTER_CUBIC (2): 双三次插值。基于4x4像素邻域，生成比双线性插值更平滑的图像，但速度较慢。
        INTER_AREA (3): 基于像素区域关系的重采样。适用于图像缩小，能产生无摩尔纹的结果。
        INTER_LANCZOS4 (4): Lanczos插值，基于8x8像素邻域。提供最高质量，但速度最慢。
        INTER_LINEAR_EXACT (5): 类似于双线性插值，但计算更精确。
        INTER_NEAREST_EXACT (6): 类似于最近邻插值，但计算更精确。
        '''

        self.test_module = A.Downscale(0.5, 0.5, interpolation=cv2.INTER_LANCZOS4, always_apply=True, p=1.0)

        self.image_crop_module = A.RandomCrop(self.crop_size, self.crop_size, p=1.0)
        self.image_aug_module = A.Compose([
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.2),
            A.RandomRotate90(p=0.8)
        ])


        self.second_degra_module = A.Compose([
            A.AdvancedBlur(
                blur_limit=(7, 21),
                sigma_x_limit=(0.2, 1.5),
                sigma_y_limit=(0.2, 1.5),
                beta_limit=(0.5, 4),
                p=0.8
            ),
        ])

    def random_crop(self, img):
        return self.image_crop_module(image=img)["image"]

    def image_aug(self, img):
        return self.image_aug_module(image=img)["image"]

    def image_degra(self, img):
        return self.first_degra_module(image=img)["image"]
    def image_test(self, img):
        return self.test_module(image=img)["image"]

    def __len__(self) -> int:
        return len(self.image_list)


from torch.utils.data import DataLoader

if __name__ == "__main__":

    data_root = "D:/Dateset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/"
    batch_size = 16

    laion_ds = LaionHRDataset(data_root, crop_size=1024)

    image, image1  = laion_ds.__getitem__(30)
    print_info(image)
    print_info(image1)
    save_numpy(image, "./result/crop.png")
    save_numpy(image1, "./result/deg1.png")

    data_loader = DataLoader(
        dataset=laion_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )




