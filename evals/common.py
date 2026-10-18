import pickle

import streamlit as st

EVAL_PATH = "data/eval_all.pkl"
FIELDS = ["weather", "scene", "timeofday"]


@st.cache_resource
def load():
    with open(EVAL_PATH, "rb") as f:
        return pickle.load(f)