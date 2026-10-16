import json
import os
import sys
import argparse

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRNano, RFDETRMedium
from tqdm import tqdm

GT_PATH = "data/bdd_coco_labels.json"
IMAGE_DIR = "data/100k/val" 
PRED_PATH = "data/predictions.json"
ATTR_PATH = "data/BDD_attributes.json"

def select_attrs(weather, scene, timeofday):
    imgIds = []
    with open(ATTR_PATH, "r") as attr_path:
        attr = json.load(attr_path)
        for key in attr:
            if weather and attr[key]["weather"] != weather:
                continue
            if scene and attr[key]["scene"] != scene:
                continue
            if timeofday and attr[key]["timeofday"] != timeofday:
                continue
            imgIds.append(int(key))
    return imgIds

def eval_script(weather, scene, timeofday):
    imgIds = select_attrs(weather, scene, timeofday)
    if not imgIds: # sanity check
        raise SystemExit("No GT images match those filters. Check spelling (e.g. 'dawn/dusk').")

    coco_gt = COCO(GT_PATH)
    coco_dt = coco_gt.loadRes(PRED_PATH)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = imgIds
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