import json
from dataclasses import dataclass
from PIL import Image

import pandas as pd

train_datafile, test_datafile, val_datafile = "data/100k/train.csv", "data/100k/val.csv", "data/100k/test.csv"

image_data = {}
annotation_data = {}
category_idx = {}

df = pd.read_csv(val_datafile) 
label_files = df["labels"].tolist()

@dataclass
class ImagesList:
    id: int
    width: int
    height: int
    file_name: str
    date_captured: str

@dataclass
class AnnotationsList:
    id: int
    image_id: int
    category_id: int
    segmentation: list
    bbox: list

def get_image_data(image_filename):
    image_path = image_filename + ".jpg"
    with Image.open(image_path) as img:
        # Extract EXIF data
        exif_data = img._getexif()
        width, height = img.size

        if exif_data is None:
            exif_data = "1900-01-01 00:00:00"

        return width, height, exif_data
            

unique_image_idx = 0
unique_category_idx = 0
for file in label_files:
    absolute_unique_filename = file.split(".")[0]
    unique_filename = absolute_unique_filename.split("/")[-1] 
    if unique_filename not in image_data:
        image_data = get_image_data(absolute_unique_filename)
        image_metadata = ImagesList(id=unique_image_idx, width=image_data[0], height=image_data[1], file_name=f"{unique_filename}.jpg", date_captured=image_data[2])
        image_data[unique_filename] = image_metadata
        unique_image_idx += 1
        
    with open(file, "r") as f:
        data = json.load(f)

        # process objects within the frame
        for frame in data["frames"]:
            for object in frame["objects"]: 
                if object["category"] not in category_idx:
                    category_idx[object["category"]] = unique_category_idx
                    unique_category_idx += 1

                

print(image_data)

