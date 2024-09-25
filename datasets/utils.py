
# Filter images
import os
from PIL import Image

def clean_image_dataset(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < 1000 or height < 1000:
                        os.remove(file_path)
                        print(f"Deleted {filename} due to low resolution.")
            except (IOError, OSError):
                os.remove(file_path)
                print(f"Deleted {filename} due to being an invalid image.")

# 使用示例
dataset_directory = "path/to/your/dataset"
clean_image_dataset(dataset_directory)
