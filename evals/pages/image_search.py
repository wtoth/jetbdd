import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

from common import FIELDS, load

GT_PATH = "data/bdd_coco_labels.json"
PRED_PATH = "data/predictions.json"
IMAGE_DIR = "data/100k/val"
MIN_SCORE = 0.05  # lowest score the viewer can show; lower predictions aren't kept in memory
GT_COLOR, PRED_COLOR = "#21c354", "#ff4b4b"


@st.cache_resource
def load_annotations():
    coco_gt = COCO(GT_PATH)
    preds = defaultdict(list)
    with open(PRED_PATH) as f:
        for d in json.load(f):
            if d["score"] >= MIN_SCORE:
                preds[d["image_id"]].append(d)
    return coco_gt, preds


def image_ap(entries, p):
    """AP@[.50:.95] for one image, same math as COCOeval.accumulate (area=all, maxDets=100)."""
    aps = []
    for e in entries:  # one entry per category
        if e is None:
            continue
        npig = np.count_nonzero(e["gtIgnore"] == 0)
        if npig == 0:
            continue
        if len(e["dtScores"]) == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-np.asarray(e["dtScores"]), kind="mergesort")
        dtm, keep = e["dtMatches"][:, order], np.logical_not(e["dtIgnore"][:, order])
        tp = np.cumsum(np.logical_and(dtm, keep), axis=1)
        fp = np.cumsum(np.logical_and(np.logical_not(dtm), keep), axis=1)
        rc = tp / npig
        pr = tp / (tp + fp + np.spacing(1))
        pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        q = np.zeros((len(p.iouThrs), len(p.recThrs)))
        for t in range(len(p.iouThrs)):
            inds = np.searchsorted(rc[t], p.recThrs, side="left")
            ok = inds < rc.shape[1]
            q[t, ok] = pr[t, inds[ok]]
        aps.append(q.mean())
    return np.mean(aps) if aps else np.nan


@st.cache_data
def image_table():
    data = load()
    coco_gt, _ = load_annotations()
    p, ev, attrs = data["params"], data["evalImgs"], data["attrs"]
    K, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
    a = p.areaRngLbl.index("all")
    rows = []
    for i, img_id in enumerate(p.imgIds):
        img_id = int(img_id)
        rows.append({
            "image_id": img_id,
            "file_name": coco_gt.imgs[img_id]["file_name"],
            "AP": image_ap([ev[k * A * I + a * I + i] for k in range(K)], p),
            **{f: attrs.get(img_id, {}).get(f) for f in FIELDS},
        })
    return pd.DataFrame(rows)


def draw(img_id, min_score):
    coco_gt, preds = load_annotations()
    image = Image.open(os.path.join(IMAGE_DIR, coco_gt.imgs[img_id]["file_name"])).convert("RGB")
    canvas = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    for ann in coco_gt.imgToAnns[img_id]:
        x, y, w, h = ann["bbox"]
        canvas.rectangle([x, y, x + w, y + h], outline=GT_COLOR, width=3)
    shown = [d for d in preds.get(img_id, []) if d["score"] >= min_score]
    for d in shown:
        x, y, w, h = d["bbox"]
        canvas.rectangle([x, y, x + w, y + h], outline=PRED_COLOR, width=2)
        name = coco_gt.cats.get(d["category_id"], {"name": str(d["category_id"])})["name"]
        label = f"{name} {d['score']:.2f}"
        pos = (x, max(y, 22))
        canvas.rectangle(canvas.textbbox(pos, label, font=font, anchor="lb"), fill=PRED_COLOR)
        canvas.text(pos, label, fill="white", font=font, anchor="lb")
    return image, len(shown)


st.session_state.setdefault("favorites", [])
st.session_state.setdefault("selected", None)


def select(img_id):
    st.session_state.selected = img_id


def toggle_favorite(img_id):
    favs = st.session_state.favorites
    if img_id in favs:
        favs.remove(img_id)
    else:
        favs.append(img_id)


def pick_from_table():
    rows = st.session_state["results_table"]["selection"]["rows"]
    if rows:
        select(int(results.iloc[rows[0]]["image_id"]))


st.title("Image search")
coco_gt, preds = load_annotations()
with st.spinner("Scoring each image..."):
    table = image_table()

# Favorites
st.sidebar.header("★ Favorites")
if not st.session_state.favorites:
    st.sidebar.caption("Nothing here yet. Open an image and click ☆ Favorite.")
for fav in st.session_state.favorites:
    st.sidebar.button(coco_gt.imgs[fav]["file_name"], key=f"fav_{fav}", on_click=select, args=(fav,),
                      type="primary" if fav == st.session_state.selected else "secondary")

# Search
with st.container(border=True):
    mask = np.ones(len(table), dtype=bool)
    for f in FIELDS:
        options = sorted(table[f].dropna().unique())
        mask &= table[f].isin(st.pills(f, options, selection_mode="multi", default=options)).to_numpy()

lowest_first = st.toggle("Lowest AP first")
results = table[mask].sort_values("AP", ascending=lowest_first, na_position="last").reset_index(drop=True)
st.caption(f"{len(results):,} images. Images with no ground truth boxes have no AP and are listed last.")
st.dataframe(
    results[["file_name", "AP", *FIELDS]],
    key="results_table", on_select=pick_from_table, selection_mode="single-row",
    hide_index=True, height=300,
    column_config={"AP": st.column_config.NumberColumn(format="%.3f")},
)

# Viewer
img_id = st.session_state.selected
if img_id is None:
    st.info("Click a row above, or a favorite in the sidebar, to view an image.")
    st.stop()

row = table.set_index("image_id").loc[img_id]
is_fav = img_id in st.session_state.favorites
title_col, fav_col = st.columns([4, 1])
title_col.subheader(row["file_name"])
fav_col.button("★ Favorite" if is_fav else "☆ Favorite", on_click=toggle_favorite, args=(img_id,),
               type="primary" if is_fav else "secondary")

min_score = st.slider("Show predictions with score ≥", MIN_SCORE, 1.0, 0.5, 0.05)
image, n_shown = draw(img_id, min_score)
st.image(image)
ap = "n/a" if pd.isna(row["AP"]) else f"{row['AP']:.3f}"
st.caption(f"AP {ap} · {row['weather']} · {row['scene']} · {row['timeofday']} · "
           f"green = ground truth ({len(coco_gt.imgToAnns[img_id])}), red = predictions ({n_shown})")