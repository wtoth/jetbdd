import json
import os
import sys

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRNano, RFDETRMedium
from tqdm import tqdm

GT_PATH = "data/bdd_coco_labels.json"
IMAGE_DIR = "data/100k/val" 
PRED_PATH = "data/predictions.json"
ATTR_PATH = "data/BDD_attributes.json"

coco_gt = COCO(GT_PATH)
# print(coco_gt)
# sys.exit()

# model = RFDETRNano(device="mps")
model = RFDETRMedium(device="mps")

results = []
bdd_attributes = {}
for img_id in tqdm(coco_gt.getImgIds()):
    info = coco_gt.loadImgs(img_id)[0]
    bdd_metadata = info["file_name"].split(".")[0] + ".json" 
    with open(os.path.join(IMAGE_DIR, bdd_metadata), "r") as f:
        data = json.load(f)
        bdd_attributes[img_id] = {"weather": data["attributes"]["weather"], 
                                    "scene": data["attributes"]["scene"], 
                                    "timeofday": data["attributes"]["timeofday"]}

    image = Image.open(os.path.join(IMAGE_DIR, info["file_name"])).convert("RGB")

    detections = model.predict(image, threshold=0.01)

    for (x1, y1, x2, y2), score, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        results.append({
            "image_id": img_id,
            "category_id": int(class_id),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })

with open(PRED_PATH, "w") as f:
    json.dump(results, f)

with open(ATTR_PATH, "w") as f:
    json.dump(bdd_attributes, f)

# the role of this file is to just present headline data
coco_dt = coco_gt.loadRes(PRED_PATH)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()