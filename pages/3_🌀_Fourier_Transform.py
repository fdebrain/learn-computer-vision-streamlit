import numpy as np
import streamlit as st
from src.fourier_transform import compute_fft, compute_inverse_fft
from src.utils import get_image_from_url, load_sample_img, normalize, rgb_to_gray

st.session_state[
    "sample_url"
] = "https://live.staticflickr.com/8476/8098572022_7d129c67ed_b.jpg"


st.title("Filtering an image with Fourier Transform")

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

    # Compute 2D Fourier Transform
    st.header("2D Fourier Transform")
    gray = rgb_to_gray(img)
    fshift, magnitude_spectrum = compute_fft(gray)
    st.text(f"Image shape: {magnitude_spectrum.shape}")
    st.text(f"{magnitude_spectrum.min()} | {magnitude_spectrum.max()}")
    st.image(normalize(magnitude_spectrum), channels="gray")

    # Select ROI in Fourier space
    st.header("2D Fourier Transform - Region of Interest")
    x_middle = magnitude_spectrum.shape[1] // 2
    y_middle = magnitude_spectrum.shape[0] // 2
    x_max = magnitude_spectrum.shape[1]
    y_max = magnitude_spectrum.shape[0]
    neighbor_size_x = st.slider(
        "neighbor_size_x", min_value=0, max_value=x_max // 2, value=10
    )
    neighbor_size_y = st.slider(
        "neighbor_size_y", min_value=0, max_value=y_max // 2, value=10
    )
    x_min, x_max = x_middle - neighbor_size_x, x_middle + neighbor_size_x
    y_min, y_max = y_middle - neighbor_size_y, y_middle + neighbor_size_y

    # Display region of interest in Fourier space
    mask = np.ones_like(magnitude_spectrum, dtype="bool")
    mask[:, :x_min] = 0
    mask[:y_min, :] = 0
    mask[:, x_max:] = 0
    mask[y_max:, :] = 0

    if st.checkbox("Inverse mask"):
        mask = ~mask

    magnitude_spectrum_roi = mask * magnitude_spectrum
    fshift_roi = mask * fshift
    st.image(normalize(magnitude_spectrum_roi))

    # Compute reconstructed image
    st.header("Image Reconstruction")
    reconstructed_img = compute_inverse_fft(fshift_roi)
    st.text(reconstructed_img.shape)
    st.image(normalize(reconstructed_img))

    st.info(
        "By inverting the mask, you can end up with an edge detector (similar to a Sobel operator)."
    )

    st.text("TODO: Adapt to RGB")

    st.header("References")
    st.write(
        "[Image Processing with Python - Application of Fourier Transformation](https://towardsdatascience.com/image-processing-with-python-application-of-fourier-transformation-5a8584dc175b)"
    )
    st.write(
        "[Image Transforms in OpenCV - Fourier Transform](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)"
    )
