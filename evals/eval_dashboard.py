import copy
import pickle
import contextlib
import io
from collections import Counter

import pandas as pd
import streamlit as st
from pycocotools.cocoeval import COCOeval
from common import FIELDS, load

# ideal dashboard will contain at least 2 pages
# Overall results and best and worst performing categories
# # This will contain headline results for individual 
# Individual image viewer
# # Searchable image viewer. Should be able to search a category and find images that fit it and see performance on those images
# # Selected images will show their bounding boxes for review

MIN_IMAGES = 50
STAT_NAMES = [
    "AP @[.50:.95]", "AP @.50", "AP @.75", "AP small", "AP medium", "AP large",
    "AR maxDets=1", "AR maxDets=10", "AR maxDets=100", "AR small", "AR medium", "AR large",
]

@st.cache_data
def slice_eval(weather, scene, timeofday):
    data = load()
    p, attrs = data["params"], data["attrs"]

    # Positions (in p.imgIds) of images whose attributes match the selection
    keep = []
    for n, img_id in enumerate(p.imgIds):
        a = attrs.get(int(img_id), {})
        if a.get("weather") in weather and a.get("scene") in scene and a.get("timeofday") in timeofday:
            keep.append(n)
    if not keep:
        return 0, None

    # evalImgs is laid out as [category][area][image]; keep only the selected images
    I = len(p.imgIds)
    E = COCOeval(iouType="bbox")
    E.params = copy.deepcopy(p)
    E.params.imgIds = [p.imgIds[n] for n in keep]
    E._paramsEval = copy.deepcopy(E.params)
    E.evalImgs = [data["evalImgs"][b + n] for b in range(0, len(data["evalImgs"]), I) for n in keep]
    E.accumulate()
    E.summarize()
    return len(keep), E.stats

@st.cache_data
def rank_combos(min_images):
    data = load()
    ids, attrs = data["params"].imgIds, data["attrs"]
    counts = Counter(tuple(attrs[int(i)][f] for f in FIELDS) for i in ids if int(i) in attrs)
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):  # silence COCOeval's printing
        for combo, n in counts.items():
            if n < min_images:
                continue
            _, stats = slice_eval(*((v,) for v in combo))
            if stats[0] >= 0:
                rows.append({"combination": " · ".join(combo), "images": n, "AP": stats[0]})
    return pd.DataFrame(rows)


def style(df, base_ap):
    df = df.assign(**{"vs baseline": (df["AP"] - base_ap)*100}) # (df["AP"] / base_ap - 1) * 100})
    return (df.style
            .format({"images": "{:,}", "AP": "{:.3f}", "vs baseline": "{:+.1f}%"})
            .map(lambda v: f"color: {'#21c354' if v >= 0 else '#ff4b4b'}", subset=["vs baseline"]))

st.set_page_config(layout="wide")
st.title("BDD100K val results")
data = load()
ids, attrs = data["params"].imgIds, data["attrs"]

st.sidebar.header("Filter by attributes")
options, selected = {}, {}
for f in FIELDS:
    options[f] = sorted({attrs[int(i)][f] for i in ids if int(i) in attrs})
    selected[f] = st.sidebar.pills(f, options[f], selection_mode="multi", default=options[f])

_, base = slice_eval(*(tuple(options[f]) for f in FIELDS))
n_imgs, stats = slice_eval(*(tuple(selected[f]) for f in FIELDS))

if stats is None:
    st.warning("No images match this combination of attributes.")
    st.stop()

st.caption(f"{n_imgs:,} of {len(ids):,} images")
cols = st.columns(3)
for col, k in zip(cols, range(3)):
    col.metric(STAT_NAMES[k], f"{stats[k]:.3f}", delta=f"{stats[k] - base[k]:+.3f} vs all")

st.dataframe(
    pd.DataFrame({"metric": STAT_NAMES, "selection": stats, "all images": base}),
    hide_index=True,
)

st.subheader("Best and worst attribute combinations")
st.caption(f"Ranked by mAP@[.50:.95] relative to all images. Only combinations with at least {MIN_IMAGES} images.")
with st.spinner("Scoring every attribute combination..."):
    combos = rank_combos(MIN_IMAGES)

left, right = st.columns(2)
left.markdown("**Top 10**")
left.dataframe(style(combos.nlargest(10, "AP"), base[0]), hide_index=True)
right.markdown("**Bottom 10**")
right.dataframe(style(combos.nsmallest(10, "AP"), base[0]), hide_index=True)