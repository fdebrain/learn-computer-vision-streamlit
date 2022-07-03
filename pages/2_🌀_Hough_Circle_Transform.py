import cv2
import numpy as np
import pandas as pd
import streamlit as st
from src.hough_transform import compute_hough_circles, overlay_hough_circles
from src.utils import get_image_from_url, load_sample_img, normalize, smooth_image

st.session_state[
    "sample_url"
] = "https://payload.cargocollective.com/1/8/272451/4087297/Four-Circles-white_6.jpg"


st.title("Detecting Circles with Hough Transform")

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
    st.image(normalize(img_smoothed))

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

    # ROI
    x_min = st.slider("x_min", min_value=0, max_value=img.shape[1], value=0)
    x_max = st.slider("x_max", min_value=0, max_value=img.shape[1], value=img.shape[1])
    y_min = st.slider("y_min", min_value=0, max_value=img.shape[0], value=0)
    y_max = st.slider("y_max", min_value=0, max_value=img.shape[0], value=img.shape[0])

    mask = np.ones_like(edges, dtype="bool")
    mask[:, :x_min] = 0
    mask[:y_min, :] = 0
    mask[:, x_max:] = 0
    mask[y_max:, :] = 0

    edges_roi = edges * mask
    st.image(edges_roi)

    # Hough Circle Transform
    r_min = st.slider("r_min", 1, max(img.shape), value=90, step=1)
    r_max = st.slider("r_max", 1, max(img.shape), value=215, step=1)
    threshold = st.slider("Threshold", 0, 500, value=150, step=1)
    neighborhood_size = st.slider("Neighborhood Size (NMS)", 1, 200, value=10, step=5)
    theta_res = st.slider("Theta res (°)", 1, 20, value=3)
    pixel_res = st.slider("Pixel res (px)", 1, 20, value=3)
    circles = compute_hough_circles(
        edges_roi,
        r_min=r_min,
        r_max=r_max,
        pixel_res=pixel_res,
        theta_res=theta_res * np.pi / 180,
        threshold=threshold,
        neighborhood_size=neighborhood_size,
    )

    # Display detected circle
    line_thickness = st.slider("Line thickness [px]", 1, 10, value=2)
    final_img = overlay_hough_circles(img, circles, line_thickness)
    st.text(final_img.shape)
    st.image(final_img)
    st.text(f"Found {len(circles)} circles")

    # Display equation
    st.latex("(x-a)^2 + (y-b)^2 = r^2")

    # Display circle parameters
    df = pd.DataFrame(circles, columns=["a [px]", "b [px]", "r [px]", "Supports"])
    df = df.astype(np.int16)
    df = df.sort_values(by=["Supports"], ascending=False).reset_index(drop=True)
    st.dataframe(df, width=500)

    st.header("References")
    st.write(
        "[Wikipedia - Circle Hough Transform](https://en.wikipedia.org/wiki/Circle_Hough_Transform)"
    )
    st.write(
        "[OpenCV - Hough Circle Transform](https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html)"
    )
