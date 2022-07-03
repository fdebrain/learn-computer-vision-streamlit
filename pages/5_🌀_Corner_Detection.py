import numpy as np
import pandas as pd
import streamlit as st
from src.corner_detection import (
    compute_harris_response_matrix,
    compute_spatial_derivative,
    overlay_detected_corners,
)
from src.utils import get_image_from_url, load_sample_img, nms, normalize, rgb_to_gray

st.session_state[
    "sample_url"
] = "https://live.staticflickr.com/8476/8098572022_7d129c67ed_b.jpg"


st.title("Detecting Corners using the Harris Operator")

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

    # Convert to Grayscale
    st.header("Grayscale image")
    st.latex(
        """I_{grayscale} = \sum_{C \in {R, G, B}} w_C \cdot I_{color} \\\ 
        \\text{where } w_R = 0.299 \\text{ | } w_G = 0.587 \\text{ | } w_B = 0.114"""
    )
    img_gray = rgb_to_gray(img)
    st.text(f"Image shape: {img_gray.shape}")
    st.image(normalize(img_gray))

    # Calculate spatial derivative Ix and Iy
    st.header("Spatial Derivative")
    st.latex(
        """I_x = \\begin{bmatrix} -1 & 0 & 1\\\\ -2 & 0 & 2\\\\ -1 & 0 & 1 \end{bmatrix} \\ast I \\\\
           I_y = \\begin{bmatrix}  1 & 2 & 1\\\\ 0 & 0 & 0\\\\ -1 & -2 & -1 \end{bmatrix} \\ast I
        """
    )

    Ix, Iy = compute_spatial_derivative(img_gray)
    st.text(f"Ix shape: {Ix.shape}")
    st.image(normalize(Ix))
    st.text(f"Iy shape: {Iy.shape}")
    st.image(normalize(Iy))

    # Compute Harris Response
    st.header("Harris Response")
    st.latex(
        """R = \det({M}) - k \cdot \mathrm{tr}(M)^2 \\\ 
        \\text{where } M = \\begin{bmatrix} I_x^2 & I_x I_y\\\\ I_yI_x & I_y^2 \end{bmatrix}"""
    )
    st.info(
        """By looking at each element in the Harris Response Matrix R, we can interpret the nature of each pixel: \n
• $R_{x,y}$ << 0: (x,y) is an edge \n
• $R_{x,y}$ = 0: (x,y) is part of a flat surface \n
• $R_{x,y}$ >> 0: (x,y) is a corner"""
    )
    k = st.slider(label="k", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    sigma = st.slider(
        label="sigma (smoothing)", min_value=1, max_value=9, value=1, step=2
    )
    harris_response = (
        100 * normalize(compute_harris_response_matrix(Ix, Iy, k, sigma)) - 50
    )

    # Display filtered Harris Response
    thresh = st.slider(label="Threshold", min_value=-50, max_value=50, value=-15)
    if invert_filter := st.checkbox(label="Invert threshold selection"):
        harris_response[harris_response > thresh] = -50
    else:
        harris_response[harris_response < thresh] = -50
    st.image(normalize(harris_response))

    # Find coordinates of corner candidates
    st.header("Display Corner Candidates")
    neighborhood_size = st.slider(
        label="Minimal distance between two detected corners [px]",
        min_value=1,
        max_value=20,
        value=10,
    )
    y, x = np.nonzero(
        nms(harris_response, threshold=0, neighborhood_size=neighborhood_size)
    )

    # Filter false positive along the border
    min_border = st.slider(
        label="Minimal distance from border [px]",
        min_value=0,
        max_value=20,
        value=3,
    )

    corners = [
        [px, py]
        for px, py in zip(x, y)
        if min_border < px < img.shape[1] - min_border
        and min_border < py < img.shape[0] - min_border
    ]

    # Display detected corners
    overlay_detected_corners(img, corners)
    st.image(img)
    st.text(f"Detected {len(corners)} corners.")

    # Display corners coordinates
    df = pd.DataFrame(corners, columns=["x [px]", "y [px]"])
    df = df.astype(np.int16)
    df = df.sort_values(by=["x [px]"], ascending=True).reset_index(drop=True)
    st.dataframe(df, width=300)


st.header("References")
st.write(
    "[Github - Process of Harris Corner Detection Algorithm](https://github.com/muthuspark/ml_research/blob/master/Process%20of%20Harris%20Corner%20Detection%20Algorithm.ipynb)"
)
st.write(
    "[Wikipedia - Harris Corner Detector](https://en.wikipedia.org/wiki/Harris_corner_detector)"
)
