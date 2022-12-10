import numpy as np
import streamlit as st

from src.utils import get_image_from_url, init_session_state, load_sample_img, normalize

st.title("Singular Value Decomposition on images")

# Session state
if st.session_state.get("page_name") != "svd":
    init_session_state(
        state={
            "page_name": "svd",
            "img": None,
            "sample_url": "https://imgs.search.brave.com/gaAg9umlNCGPfMuHWLs8zvRqJFL-ulC8p0BqOMrwxp4/rs:fit:1200:1197:1/g:ce/aHR0cHM6Ly9pbWcu/cmF3cGl4ZWwuY29t/L3MzZnMtcHJpdmF0/ZS9yYXdwaXhlbF9p/bWFnZXMvd2Vic2l0/ZV9jb250ZW50L3Bk/MzYtMS1nc2ZjXzIw/MTcxMjA4X2FyY2hp/dmVfZTAwMjA3NC5q/cGc_Ymc9dHJhbnNw/YXJlbnQmY29uPTMm/Y3M9c3JnYiZkcHI9/MSZmbT1qcGcmaXhs/aWI9cGhwLTMuMS4w/JnE9ODAmdXNtPTE1/JnZpYj0zJnc9MTMw/MCZzPTRiODU3OWI5/NzliMDFkM2Y0YjY2/NjUyYWI1NWU2OTc4",
        }
    )

# Load image
if url := st.text_input("Enter an image URL"):
    get_image_from_url(url)
st.markdown("**OR**")
st.button(label="Try a sample image", on_click=load_sample_img)

if st.session_state["img"] is not None:
    # Load image
    img = st.session_state["img"].copy()

    # Convert to float
    img = normalize(img)

    # SVD Transform
    st.header("SVD transform")
    rank = st.slider("SVD rank", 0, 250, value=50)
    img_svd = np.zeros_like(img)
    for channel in range(3):
        U, S, V = np.linalg.svd(img[:, :, channel])
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]
        img_svd[:, :, channel] = U @ np.diag(S) @ V

    # Show original and result
    col1, col2 = st.columns(2)
    col1.image(img)
    col2.image(normalize(img_svd))
    st.text(f"Image shape: {img.shape}")

    # Compute ratio of info needed compared to original
    height, width, _ = img.shape
    info_ratio = (height * rank + rank + rank * width) / (height * width)
    st.info(
        f"SVD image contains {100 * info_ratio:.1f}% the information of the original image."
    )

    st.header("References")
    st.write(
        "[Youtube (Avra) - Singular Value Decomposition Web App](https://www.youtube.com/watch?v=J2jBTFovWH8&ab_channel=Avra)"
    )
