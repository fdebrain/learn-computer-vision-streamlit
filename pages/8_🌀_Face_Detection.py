import contextlib
from pathlib import Path

import streamlit as st

from src.face_detection import VideoProcessorFaceDetection
from src.utils import (
    create_local_video_stream,
    create_webcam_stream,
    download_youtube_video,
)

SAVE_DIR = Path(__file__).parent.parent / "data"
SAVE_PATH = SAVE_DIR / "video.mp4"
MEDIA_SOURCES = ["Webcam", "Youtube"]

st.title("Face detection")
st.info("Note: This implementation requires a face to work.")


# Select stream source
selected_source = st.radio(
    label="Select a media source to stream",
    options=tuple(MEDIA_SOURCES),
    key="radio",
)

# Play video stream
if selected_source == "Webcam":
    ctx = create_webcam_stream(processor=VideoProcessorFaceDetection)
elif url := st.text_input(
    "Enter an image URL",
    value="https://www.youtube.com/watch?v=rYrdiQckGhw&ab_channel=GoExperimental",
):
    with st.spinner(text="Downloading..."):
        download_youtube_video(url, SAVE_PATH)

    ctx = create_local_video_stream(
        processor=VideoProcessorFaceDetection,
        filepath=str(SAVE_PATH),
    )

# Use interactive widgets
with contextlib.suppress(Exception):
    if ctx.video_processor:
        ctx.video_processor.flip = st.checkbox("Flip stream", value=True)
        ctx.video_processor.threshold = st.slider(
            "Detection threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.8,
        )

st.header("References")
st.write(
    "[Blog Post by whitphx - Developing web-based real-time video/audio processing apps quickly with Streamlit](https://www.whitphx.info/posts/20211231-streamlit-webrtc-video-app-tutorial/)"
)
st.write(
    "[Github - whitphx/streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc/blob/0b143be13ec71c3339e4a1917aab8d4fd7dd333f/app.py#L415)"
)
