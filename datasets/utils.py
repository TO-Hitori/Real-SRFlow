"""
dataset: laion-high-resolution

path: D:/Dateset/laion-high-resolution/00028/


"""
import os
from typing import List
from PIL import Image
import statistics
from tqdm import tqdm


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
    dataset_directory = "D:/Dateset/laion-high-resolution/00005/"
    filter_image_dataset(dataset_directory)
    # get_metadata_link()
