import contextlib
from pathlib import Path

import streamlit as st

from src.live_edge_detection import (
    VideoProcessorEdgeDetection,
    create_local_video_stream,
    create_webcam_stream,
)
from src.utils import download_youtube_video

SAVE_DIR = Path(__file__).parent.parent / "data"
SAVE_PATH = SAVE_DIR / "video.mp4"
MEDIA_SOURCES = ["Webcam", "Youtube"]

st.title("Live Edge Detection")

# Select stream source
selected_source = st.radio(
    label="Select a media source to stream",
    options=tuple(MEDIA_SOURCES),
    key="radio",
)

# Play video stream
if selected_source == "Webcam":
    ctx = create_webcam_stream(processor=VideoProcessorEdgeDetection)
elif url := st.text_input(
    "Enter an image URL",
    value="https://www.youtube.com/watch?v=rYrdiQckGhw&ab_channel=GoExperimental",
):
    with st.spinner(text="Downloading..."):
        download_youtube_video(url, SAVE_PATH)

    ctx = create_local_video_stream(
        processor=VideoProcessorEdgeDetection,
        filepath=str(SAVE_PATH),
    )

# Use interactive widgets
with contextlib.suppress(Exception):
    if ctx.video_processor:
        ctx.video_processor.flip = st.checkbox("Flip stream")
        ctx.video_processor.canny = st.checkbox("Apply edge detection")
        ctx.video_processor.threshold1 = st.slider(
            "Threshold1",
            min_value=0,
            max_value=1000,
            step=1,
            value=50,
        )
        ctx.video_processor.threshold2 = st.slider(
            "Threshold2",
            min_value=0,
            max_value=1000,
            step=1,
            value=100,
        )
st.header("References")
st.write(
    "[Blog Post by whitphx - Developing web-based real-time video/audio processing apps quickly with Streamlit](https://www.whitphx.info/posts/20211231-streamlit-webrtc-video-app-tutorial/)"
)
st.write(
    "[Github - whitphx/streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc/blob/0b143be13ec71c3339e4a1917aab8d4fd7dd333f/app.py#L415)"
)
