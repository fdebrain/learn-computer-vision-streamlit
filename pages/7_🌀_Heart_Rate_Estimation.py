import contextlib
from pathlib import Path

import streamlit as st

from src.heart_rate_estimation import (
    VideoProcessorHREstimation,
    create_local_video_stream,
    create_webcam_stream,
)
from src.utils import download_youtube_video

SAVE_DIR = Path(__file__).parent.parent / "data"
SAVE_PATH = SAVE_DIR / "video.mp4"
MEDIA_SOURCES = ["Webcam", "Youtube"]

st.title("Heart Rate Estimation using Eulerian Magnification")

# Select stream source
selected_source = st.radio(
    label="Select a media source to stream",
    options=tuple(MEDIA_SOURCES),
    key="radio",
)

# Play video stream
if selected_source == "Webcam":
    ctx = create_webcam_stream(processor=VideoProcessorHREstimation)
elif url := st.text_input(
    "Enter an image URL",
    value="https://www.youtube.com/watch?v=rYrdiQckGhw&ab_channel=GoExperimental",
):
    with st.spinner(text="Downloading..."):
        download_youtube_video(url, SAVE_PATH)

    ctx = create_local_video_stream(
        processor=VideoProcessorHREstimation,
        filepath=str(SAVE_PATH),
    )

# Use interactive widgets
with contextlib.suppress(Exception):
    if ctx.video_processor:
        ctx.video_processor.flip = st.checkbox("Flip stream", value=True)
        ctx.video_processor.alpha = st.slider(
            "Alpha",
            min_value=0,
            max_value=500,
            step=10,
            value=30,
        )

st.header("References")
st.write(
    "[Blog Post by whitphx - Developing web-based real-time video/audio processing apps quickly with Streamlit](https://www.whitphx.info/posts/20211231-streamlit-webrtc-video-app-tutorial/)"
)
st.write(
    "[Eulerian Magnification - https://github.com/brycedrennan/eulerian-magnification/)"
)
