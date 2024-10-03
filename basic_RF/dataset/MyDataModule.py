# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 02 
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch as th
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AnimeFace(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data_list = os.listdir(self.dataset_path)
        self.transform = A.Compose(
            [
                A.CenterCrop(height=64, width=64),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2()
            ]
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]
        image_path = os.path.join(self.dataset_path, image_name)
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.transform(image=image)['image']

        return image


from torch.utils.data import DataLoader
from torchvision.utils import save_image
if __name__ == "__main__":
    path = 'C:/MyDataset/animeface/'
    dataset = AnimeFace(path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False
    )

    for i, sample_batch in enumerate(dataloader):
        save_image(sample_batch, 'AnimeBatch.png', nrow=8, normalize=True)
        print(sample_batch.shape)
        print(sample_batch.max(), sample_batch.min())
        print('over all')

        break