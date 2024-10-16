# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2024 10 13 
"""
import numpy

"""
dataset: laion-high-resolution
path: D:/Dateset/laion-high-resolution/00028/

"""
import os
from typing import List
from PIL import Image
import statistics
from tqdm import tqdm
import albumentations as A
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def print_info(img):
    print("=-" * 20)
    print(type(img))
    print(img.shape)
    print(img.dtype)
    print(img.min(), img.max())
    print("=-" * 20)

def load_image2tensor(image_path):
    bacic_trans = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        ToTensorV2(),
    ])
    image = np.array(Image.open(image_path).convert('RGB'))
    image_tensor = bacic_trans(image=image)["image"]
    return image_tensor[None]

def save_numpy(img, save_path="./result/save_numpy.png"):
    img = Image.fromarray(img)
    img.save(save_path)

def save_tensor(img, save_path="./result/save_numpy.png"):
    img = (img + 1) * 0.5
    img = img.clamp(0, 1)
    from torchvision.utils import save_image
    save_image(img, save_path)

def image_degra_demo(image_path, save_path):
    image_path = image_path.replace('\\', '/')
    image_name = image_path.split('/')[-1].split('.')[0]
    image = np.array(Image.open(image_path).convert('RGB'))
    first_deg_image, second_degra_image = image_degraduation(image)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_numpy(image, os.path.join(save_path, f"{image_name}.png"))
    save_numpy(first_deg_image, os.path.join(save_path, f"{image_name}_1st.png"))
    save_numpy(second_degra_image, os.path.join(save_path, f"{image_name}_2nd.png"))
    print_info(image)
    print_info(first_deg_image)
    print_info(second_degra_image)


def image_degraduation(image):
    h, w, c = image.shape
    h1, w1 = h // 2, w // 2
    h2, w2 = h1 // 2, w1 // 2

    first_degra_module = A.Compose([
        A.AdvancedBlur(
            blur_limit=(7, 21),
            sigma_x_limit=(0.2, 3.0),
            sigma_y_limit=(0.2, 3.0),
            beta_limit=(0.5, 4),
            p=1.0
        ),
        A.OneOf([
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_NEAREST},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_CUBIC},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_AREA},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LANCZOS4},
                p=0.2
            ),
            # A.Resize(h1, w1, interpolation=cv2.INTER_LINEAR_EXACT, p=0.25),
            # A.Resize(h1, w1, interpolation=cv2.INTER_CUBIC, p=0.25),
            # A.Resize(h1, w1, interpolation=cv2.INTER_AREA, p=0.25),
            # A.Resize(h1, w1, interpolation=cv2.INTER_LANCZOS4, p=0.25),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(var_limit=(1, 30), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 2), p=0.5),
        ], p=1.0),
        A.OneOf([
            A.ImageCompression(quality_range=(30, 95), compression_type="jpeg", p=0.5),
            A.ImageCompression(quality_range=(30, 95), compression_type="webp", p=0.5),
        ], p=1.0),
    ])
    '''
    Second degradation processes
    Blur -> Resize -> Noise -> JPEG + sinc
    '''
    second_degra_module = A.Compose([
        A.AdvancedBlur(
            blur_limit=(7, 21),
            sigma_x_limit=(0.2, 1.5),
            sigma_y_limit=(0.2, 1.5),
            beta_limit=(0.5, 4),
            p=0.8
        ),
        A.OneOf([
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_NEAREST},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_CUBIC},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_AREA},
                p=0.2
            ),
            A.Downscale(
                scale_range=(0.5, 0.75),
                interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LANCZOS4},
                p=0.2
            ),
            # A.Resize(h2, w2, interpolation=cv2.INTER_LINEAR_EXACT, p=0.25),
            # A.Resize(h2, w2, interpolation=cv2.INTER_CUBIC, p=0.25),
            # A.Resize(h2, w2, interpolation=cv2.INTER_AREA, p=0.25),
            # A.Resize(h2, w2, interpolation=cv2.INTER_LANCZOS4, p=0.25),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(var_limit=(1, 25), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 1.5), p=0.5),
        ], p=1.0),
        A.OneOf([
            A.ImageCompression(quality_range=(30, 95), compression_type="jpeg", p=0.5),
            A.ImageCompression(quality_range=(30, 95), compression_type="webp", p=0.5),
        ], p=1.0),
        A.RingingOvershoot(
            blur_limit=(9, 21),
            cutoff=(np.pi / 6, np.pi / 3),
            p=0.8
        )
    ])
    first_deg_image = first_degra_module(image=image)["image"]
    second_degra_image = second_degra_module(image=first_deg_image)["image"]
    return first_deg_image, second_degra_image



def get_metadata_link() -> List[str]:
    link_list = []
    for i in range(127):
        pack_id = str(i).zfill(5)
        links = (f"https://huggingface.co/datasets/laion/laion-high-resolution/blob/main/part-{pack_id}-45914064-d424"
                 f"-4c1c-8d96-dc8125c645fb-c000.snappy.parquet")
        link_list.append(links)
        print(links)
    return link_list


# Filter images
def filter_image_dataset(image_folder: str, width_threshold: int = 1000, height_threshold: int = 1000) -> None:
    supported_formats = (".jpeg", ".jpg", ".png", ".bmp")

    invalid_image_count = 0
    valid_image_count = 0
    width_list = []
    height_list = []

    for filename in tqdm(os.listdir(image_folder)):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(image_folder, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < width_threshold or height < height_threshold:
                        invalid_image_count += 1
                        os.remove(file_path)
                        print(f"Deleted {filename} due to low resolution.")
                    else:
                        valid_image_count += 1
                        width_list.append(width)
                        height_list.append(height)

            except (IOError, OSError):
                invalid_image_count += 1
                os.remove(file_path)
                print(f"Deleted {filename} due to being an invalid image.")
    print("=-" * 25)
    print(f"A total of {invalid_image_count} invalid images were cleaned up")
    print(f"A total of {valid_image_count} valid images")
    height_mean = statistics.mean(height_list)
    width_mean = statistics.mean(width_list)
    print(f"The average height of all valid images is {height_mean}")
    print(f"The average width of all valid images is {width_mean}")
    print("=-" * 25)


if __name__ == "__main__":
    path = "C:/Users/WJQpe/Downloads/pixiv/108067863_p0.jpg"
    img_t = load_image2tensor(path)
    print_info(img_t)


