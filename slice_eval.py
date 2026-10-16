import json
import os
import argparse

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRNano, RFDETRMedium
from tqdm import tqdm

GT_PATH = "data/bdd_coco_labels.json"
IMAGE_DIR = "data/100k/val" 
PRED_PATH = "data/predictions.json"

coco_gt = COCO(GT_PATH)
# model = RFDETRNano(device="mps")
model = RFDETRMedium(device="mps")


def eval_script(weather: str|None = None, scene: str|None = None, timeofday: str|None = None,):
    results = []
    for img_id in tqdm(coco_gt.getImgIds()):
        info = coco_gt.loadImgs(img_id)[0]
        image = Image.open(os.path.join(IMAGE_DIR, info["file_name"])).convert("RGB")
        if weather or scene or timeofday:
            file = info["file_name"].split(".")[0] + ".json" # convert image filepath into json 
            with open(os.path.join(IMAGE_DIR, file), "r") as f:
                data = json.load(f)
                if weather and data["attributes"]["weather"] != weather:
                    continue
                if scene and data["attributes"]["scene"] != scene:
                    continue
                if timeofday and data["attributes"]["timeofday"] != timeofday:
                    continue
            
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

    coco_dt = coco_gt.loadRes(PRED_PATH)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weather", required=False, help="COCO-format ground truth JSON")
    parser.add_argument("--scene", required=False, help="COCO-format predictions JSON")
    parser.add_argument("--timeofday", required=False,
                        help="CSV with a 'labels' column of BDD label JSON paths")
    parser.add_argument("--out", help="Optional path to save full results as CSV")
    args = parser.parse_args()

    eval_script(args.weather, args.scene, args.timeofday)


if __name__ == "__main__":
    main()