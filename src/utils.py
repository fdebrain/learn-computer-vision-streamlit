import cv2
import numpy as np
import requests
import streamlit as st

SAMPLE_IMG_URL = "https://live.staticflickr.com/8476/8098572022_7d129c67ed_b.jpg"


def get_image_from_url(url: str):
    print(f"Load image from URL: {url}")
    response = requests.get(url)
    array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(array, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.session_state["img"] = img


def load_sample_img():
    print("Load sample image")
    get_image_from_url(SAMPLE_IMG_URL)


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())
