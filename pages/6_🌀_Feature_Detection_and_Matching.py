import numpy as np
import streamlit as st
from src.keypoints import (
    compute_keypoints,
    compute_matches,
    draw_keypoints,
    draw_matches,
    filter_matches,
)
from src.utils import color_hex_to_rgb, get_image_from_url

st.session_state["sample_url_1"] = "https://freesvg.org/img/MonaLisa.png"

st.session_state[
    "sample_url_2"
] = "https://live.staticflickr.com/703/32391645983_311037f6fd_b.jpg"


st.title("Feature Detection and Matching")


@st.cache
def load_sample_imgs():
    print("Load sample image")
    st.session_state["img1"] = get_image_from_url(st.session_state["sample_url_1"])
    st.session_state["img2"] = get_image_from_url(st.session_state["sample_url_2"])


# Input image
if "img1" not in st.session_state:
    st.session_state["img1"] = None

if "img2" not in st.session_state:
    st.session_state["img2"] = None

if url_1 := st.text_input("Enter an image URL"):
    st.session_state["img1"] = get_image_from_url(url_1)
if url_2 := st.text_input("Enter another image URL"):
    st.session_state["img2"] = get_image_from_url(url_2)

st.markdown("**OR**")
st.button(label="Try sample images", on_click=load_sample_imgs)


images_are_loaded = (
    st.session_state["img1"] is not None and st.session_state["img2"] is not None
)
if images_are_loaded:
    img1 = st.session_state["img1"].copy()
    img2 = st.session_state["img2"].copy()

    # Display images
    st.header("Original image")
    st.text(f"First image shape: {img1.shape}")
    st.text(f"Second image shape: {img2.shape}")
    st.image(st.session_state["img1"])
    st.image(st.session_state["img2"])

    # Extract keypoints
    st.header("Keypoints detection")
    st.info(
        "We use ORB, a feature detector based on FAST keypoint detector and BRIEF descriptor."
    )
    n_points = st.slider(
        label="Number of keypoints",
        min_value=10,
        max_value=2000,
        value=500,
        step=10,
    )
    keypoint1, descriptor1 = compute_keypoints(img1, n_points)
    st.text(f"Detected {len(keypoint1)} keypoints")
    keypoint2, descriptor2 = compute_keypoints(img2, n_points)
    st.text(f"Detected {len(keypoint2)} keypoints")

    # Draw keypoints
    kp_size = st.slider(label="Keypoint size", min_value=1, max_value=10, value=1)
    kp_color = st.color_picker("Keypoint color", value="#00FF00", key="cp1")
    img1 = draw_keypoints(
        img1,
        keypoint1,
        color=color_hex_to_rgb(kp_color),
        size=kp_size,
    )
    img2 = draw_keypoints(
        img2,
        keypoint2,
        color=color_hex_to_rgb(kp_color),
        size=kp_size,
    )
    st.image(img1)
    st.image(img2)

    # Match keypoint descriptors
    st.header("Keypoints matching")
    alpha = st.slider(label="Alpha", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    matches, distances = compute_matches(descriptor1, descriptor2)

    # Filter matches by distance
    distance_threshold = st.slider(
        label="Maximal distance between matched keypoint",
        min_value=min(distances),
        max_value=max(distances),
        value=int(np.quantile(distances, 0.10)),
    )
    matches = filter_matches(matches, distance_threshold)
    st.text(f"Matched {len(matches)} pairs of keypoints")

    # Draw matched keypoints
    kp_color_pick = st.color_picker("Keypoint color", value="#00FF00", key="cp2")
    match_color_pick = st.color_picker("Match color", value="#07BB07", key="cp3")
    matched_images = draw_matches(
        st.session_state["img1"],
        keypoint1,
        st.session_state["img2"],
        keypoint2,
        matches,
        alpha,
        color_hex_to_rgb(kp_color_pick),
        color_hex_to_rgb(match_color_pick),
    )
    st.image(matched_images)


st.header("References")
st.write(
    "[Analytics Vidhya - Feature Detection, Description and Matching of Images using OpenCV](https://www.analyticsvidhya.com/blog/2021/06/feature-detection-description-and-matching-of-images-using-opencv/)"
)
