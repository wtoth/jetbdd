import argparse
import json
import os
import pickle

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRNano, RFDETRMedium
from rfdetr.datasets.coco import filter_parent_categories, annotated_category_ids
from tqdm import tqdm

# Run on default Nano: uv run evals/eval_script.py   
# Run on Medium: uv run evals/eval_script.py --model medium  
# Run on Custom weights: uv run evals/eval_script.py --ckpt evals/checkpoint_best_total.pth 

GT_PATH = "data/100k/valid/_annotations.coco.json"
TRAIN_ANN_PATH = "data/100k/train/_annotations.coco.json"
IMAGE_DIR = "data/100k/valid"
PRED_PATH = "data/predictions.json"
ATTR_PATH = "data/BDD_attributes.json"
EVAL_PATH = "data/eval_all.pkl"

MODELS = {"nano": RFDETRNano, "medium": RFDETRMedium}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODELS, default="nano")
parser.add_argument("--ckpt", default=None, help="Fine-tuned checkpoint. Omit to evaluate the stock COCO weights.")
parser.add_argument("--device", default="mps")
args = parser.parse_args()

if args.ckpt:
    # Fine-tuned: rfdetr remapped the dataset's category ids to 0..N-1 at train time, so map them back.
    model = MODELS[args.model](device=args.device, pretrain_weights=args.ckpt, trust_checkpoint=True)
    with open(TRAIN_ANN_PATH) as f:
        train_anns = json.load(f)
    kept = filter_parent_categories(train_anns["categories"], annotated_category_ids(train_anns))
    label2cat = {label: int(cat["id"]) for label, cat in enumerate(kept)}
    to_cat_id = lambda class_id: label2cat[class_id]
else:
    # Stock COCO weights: class_id is already the COCO category_id.
    model = MODELS[args.model](device=args.device)
    to_cat_id = lambda class_id: class_id

coco_gt = COCO(GT_PATH)

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
            "category_id": to_cat_id(int(class_id)),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })

with open(PRED_PATH, "w") as f:
    json.dump(results, f)

with open(ATTR_PATH, "w") as f:
    json.dump(bdd_attributes, f)

coco_dt = coco_gt.loadRes(PRED_PATH)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

with open(EVAL_PATH, "wb") as f:
    pickle.dump({
        "params": coco_eval.params,
        "evalImgs": coco_eval.evalImgs,
        "attrs": {int(k): v for k, v in bdd_attributes.items()},
    }, f)