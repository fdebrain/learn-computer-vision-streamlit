import cv2
import numpy as np
import streamlit as st
from src.hough_transform import compute_hough_lines, overlay_hough_lines
from src.utils import get_image_from_url, load_sample_img, normalize

st.session_state[
    "sample_url"
] = "https://live.staticflickr.com/8476/8098572022_7d129c67ed_b.jpg"


def smooth_image(img, kernel_size):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (
        gray
        if kernel_size == 0
        else cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    )


st.title("Detecting Lines with Hough Transform")

# Input image
if "img" not in st.session_state:
    st.session_state["img"] = None

if url := st.text_input("Enter an image URL"):
    get_image_from_url(url)
st.markdown("**OR**")
st.button(label="Try a sample image", on_click=load_sample_img)

if st.session_state["img"] is not None:
    # Load image
    st.header("Original image")
    img = st.session_state["img"].copy()
    st.text(f"Image shape: {img.shape}")
    st.image(st.session_state["img"])

    # Smoothing
    st.header("Gaussian Blur")
    kernel_size = st.slider("Kernel size", 1, 50, value=1, step=2)
    img_smoothed = smooth_image(img, kernel_size)
    st.text(f"Image shape: {img_smoothed.shape}")
    st.image(img_smoothed)

    # Canny Edge Detection https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    st.header("Canny Edge Detection")
    threshold_edge_lower = st.slider("Lower", 0, 500, value=100)
    threshold_edge_upper = st.slider("Upper", 0, 500, value=200)
    aperture_size = st.slider("Aperture size", 3, 7, value=3, step=2)
    edges = cv2.Canny(
        img_smoothed,
        threshold_edge_lower,
        threshold_edge_upper,
        apertureSize=aperture_size,
    )
    st.text(f"Image shape: {edges.shape}")
    st.image(edges)

    # Apply Hough Transform https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
    st.header("Hough Line Transform")
    rho_res = st.slider("Rho Resolution [px]", 1, 10, value=1)
    theta_res_slider = st.slider("Theta Resolution [°]", 1, 10, value=1)
    theta_res = theta_res_slider * np.pi / 180.0  # resolution in range [0, 180°]
    threshold_hough = st.slider("Threshold (NMS)", 1, 1000, value=100, step=5)
    neighborhood_size = st.slider("Neighborhood Size (NMS)", 1, 100, value=20)

    lines, accumulator = compute_hough_lines(
        edges,
        rho_res,
        theta_res,
        threshold_hough,
        neighborhood_size,
    )

    if show_accumulator := st.checkbox("Display accumulator"):
        st.text(f"Accumulator shape: {accumulator.shape}")
        st.image(normalize(accumulator))

    # Display detected lines
    line_thickness = rho_res = st.slider("Line thickness [px]", 1, 10, value=2)
    final_img = overlay_hough_lines(img, lines, line_thickness)
    st.text(final_img.shape)
    st.image(final_img)
    st.text(f"Found {len(lines)} lines")

    st.header("References")
    st.write(
        "[Towards Data Science - Lines Detection with Hough Transform](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549)"
    )
    st.write(
        "[Alyssa Quek - Understanding Hough Transform With Python](https://alyssaq.github.io/2014/understanding-hough-transform/)"
    )
    st.write(
        "[Moonbooks - Implementing a simple python code to detect straight lines using Hough transform](https://moonbooks.org/Articles/Implementing-a-simple-python-code-to-detect-straight-lines-using-Hough-transform/)"
    )
