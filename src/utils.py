from pathlib import Path

import cv2
import numpy as np
import requests
import scipy.ndimage
import streamlit as st
from pytube import YouTube


def get_image_from_url(url: str):
    print(f"Load image from URL: {url}")
    response = requests.get(url)
    array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(array, flags=1)
    st.session_state["img"] = img


def load_sample_img():
    print("Load sample image")
    get_image_from_url(st.session_state["sample_url"])


def rgb_to_gray(img):
    w_r = 0.299
    w_g = 0.587
    w_b = 0.114
    return img.dot([w_r, w_g, w_b])


def smooth_image(img, kernel_size):
    return (
        img
        if kernel_size == 0
        else cv2.GaussianBlur(img, (kernel_size, kernel_size), 0).astype(np.uint8)
    )


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


def download_youtube_video(url, save_path):
    save_path = Path(save_path)
    if save_path.exists():
        st.info(f"{url} is already downloaded.")
        if not st.button("Download again?"):
            return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    video = YouTube(url)
    selected_format = video.streams.filter(file_extension="mp4").first()
    selected_format.download(save_path.parent, filename=save_path.name)
    print("Downloaded Youtube video locally")
