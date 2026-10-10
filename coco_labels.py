import json
from dataclasses import dataclass
from PIL import Image

import pandas as pd

from mscoco_label_map import bdd_to_coco_idx, category_map

train_datafile, val_datafile, test_datafile= "data/100k/train.csv", "data/100k/val.csv", "data/100k/test.csv"

seen_images = set()
image_data = []
annotation_data = {"annotations": []}

df = pd.read_csv(val_datafile) 
label_files = df["labels"].tolist()

def get_image_metadata(image_filename):
    image_path = image_filename + ".jpg"
    with Image.open(image_path) as img:
        # Extract EXIF data
        exif = img.getexif()
        date = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        width, height = img.size

        if date:
            date = date.replace(":", "-", 2)
        else:
            date = "1900-01-01 00:00:00"

        return width, height, date

def bdd_poly2d_to_coco(poly2d_list):
    """
    Converts a BDD100K poly2d annotation list to a COCO segmentation list.
    """
    coco_segmentation = []
    
    for poly in poly2d_list:
        # 1. Skip open paths if your COCO model requires closed object masks
        if not poly.get("closed", True):
            continue
            
        vertices = poly.get("vertices", [])
        if len(vertices) < 3:
            continue  # A valid COCO polygon needs at least 3 points (6 values)
            
        # 2. Flatten the [[x1, y1], [x2, y2]] array into [x1, y1, x2, y2]
        flattened_poly = []
        for vertex in vertices:
            flattened_poly.extend([float(vertex[0]), float(vertex[1])])
            
        # 3. Explicitly close the polygon if the last point doesn't match the first
        # (Good practice for COCO parsers)
        if vertices[0] != vertices[-1]:
            flattened_poly.extend([float(vertices[0][0]), float(vertices[0][1])])
            
        coco_segmentation.append(flattened_poly)
        
    return coco_segmentation

def bbox_from_segmentation(segmentation):
    xs = [p[i] for p in segmentation for i in range(0, len(p), 2)]
    ys = [p[i] for p in segmentation for i in range(1, len(p), 2)]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

def process_image(unique_image_idx):
    # process a single image
    if unique_filename not in image_data:
        width, height, date = get_image_metadata(absolute_unique_filename)
        image_metadata = {
                            "id":unique_image_idx, 
                            "width":width, 
                            "height": height, 
                            "file_name":f"{unique_filename}.jpg", 
                            "date_captured":date
                        }
        # image_data[unique_filename] = image_metadata

    return image_metadata

def process_annotations(data, unique_annotation_idx, image_idx):
    annotations_list = []
    for frame in data["frames"]:
        for object in frame["objects"]: 
            if object["category"] in bdd_to_coco_idx:
                if object["box2d"]:
                    box2d = [object["box2d"]["x1"], object["box2d"]["y1"], object["box2d"]["x2"], object["box2d"]["y2"]] 
                    data = {
                                "id": unique_annotation_idx, 
                                "image_id": image_idx, 
                                "category_id": bdd_to_coco_idx[object["category"]],
                                "bbox": box2d
                            }
                elif object["poly2d"]:
                    box2d = bbox_from_segmentation(object["poly2d"])
                    data = {
                                "id": unique_annotation_idx, 
                                "image_id": image_idx, 
                                "category_id": bdd_to_coco_idx[object["category"]],
                                "segmentation": object["poly2d"],
                                "bbox": box2d
                            }

                annotations_list.append(data)

    return annotations_list, unique_annotation_idx

def get_categories():
    category_list = []
    for id, name in category_map.items():
        category_list.append({"id":id, "name":name})
    return {"categories": category_list}

unique_image_idx = 0
unique_annotation_idx = 0
for file in label_files:
    absolute_unique_filename = file.split(".")[0]
    unique_filename = absolute_unique_filename.split("/")[-1] 

    if unique_filename not in seen_images: # check if this image or metadata file has been seen
        curr_image= process_image(unique_image_idx)
        seen_images.add(unique_filename)
        image_data.append(curr_image)
    else: # if seen, skip
        continue
        
    with open(file, "r") as f:
        data = json.load(f)

        curr_annotations_list, unique_annotation_idx = process_annotations(data, unique_annotation_idx, unique_image_idx)
        annotation_data["annotations"].extend(curr_annotations_list)
        # category_idx, unique_category_idx = process_objects(data, unique_category_idx)

    unique_image_idx += 1


info_data = {"info": {"description": "BDD100k Dataset", 
                      "url":"https://bair.berkeley.edu/blog/2018/05/30/bdd/",
                      "version": "1.0",
                      "year": 2018,
                      "contributor": "Berkeley Artificial Intelligence Research (BAIR)",
                      "date_created": "2018/05/30"}}
license_data = {"licenses": [
        {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"}]} # no idea what this should be 

category_data = get_categories()

coco_manifest = {
    "info":info_data["info"],
    "licenses" : license_data["licenses"],
    "images" : image_data,
    "annotations" : annotation_data["annotations"],
    "categories" : category_data["categories"]
}


with open("data/bdd_coco_labels.json", "w", encoding="utf-8") as file:
    json.dump(coco_manifest, file)