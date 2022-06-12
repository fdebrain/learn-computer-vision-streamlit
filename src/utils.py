import cv2
import numpy as np
import requests
import scipy.ndimage
import streamlit as st


def get_image_from_url(url: str):
    print(f"Load image from URL: {url}")
    response = requests.get(url)
    array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(array, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.session_state["img"] = img


def load_sample_img():
    print("Load sample image")
    get_image_from_url(st.session_state["sample_url"])


def normalize(img: np.ndarray):
    return (img - img.min()) / (img.max() - img.min())


def nms(matrix: np.ndarray, threshold: int, neighborhood_size: int):
    data_max = scipy.ndimage.maximum_filter(matrix, neighborhood_size)
    maxima = matrix == data_max

    data_min = scipy.ndimage.minimum_filter(matrix, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0
    return maxima


def find_top_k(matrix: np.ndarray, top_k: int):
    """Returns the n largest indices from a numpy array."""
    flat = matrix.flatten()
    indices = np.argpartition(flat, -top_k)[-top_k:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, matrix.shape)
