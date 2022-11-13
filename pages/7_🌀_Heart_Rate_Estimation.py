import contextlib
from pathlib import Path

import pandas as pd
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
st.info("Note: This implementation requires a face to work.")

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
    status_indicator = st.empty()

# Use interactive widgets
with contextlib.suppress(Exception):
    if ctx.video_processor:
        ctx.video_processor.flip = st.checkbox("Flip stream", value=True)
        ctx.video_processor.level = st.slider(
            "Gaussian Pyramid level",
            min_value=1,
            max_value=5,
            step=1,
            value=2,
        )
        ctx.video_processor.alpha = st.slider(
            "Alpha",
            min_value=0,
            max_value=300,
            step=5,
            value=50,
        )
        ctx.video_processor.f_min = st.slider(
            "Low frequency cut [Hz]",
            min_value=0.0,
            max_value=4.0,
            step=0.1,
            value=0.8,
        )
        ctx.video_processor.f_max = st.slider(
            "High frequency cut [Hz]",
            min_value=0.0,
            max_value=4.0,
            step=0.1,
            value=2.0,
        )

        # TODO: Make it update every 1-2 seconds
        data = {
            "amplitude": ctx.video_processor.fft_mean,
            "f": 60 * ctx.video_processor.freqs,
        }
        df = pd.DataFrame().from_dict(data)
        st.line_chart(data, x="f", y="amplitude", width=200, height=200)

st.header("References")
st.write(
    "[Paper - Eulerian Video Magnification for Revealing Subtle Changes in the World](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://people.csail.mit.edu/mrub/papers/vidmag.pdf)"
)
st.write("[Website - Video Magnification](https://people.csail.mit.edu/mrub/vidmag/)")
st.write(
    "[Eulerian Magnification - https://github.com/brycedrennan/eulerian-magnification/)"
)


# Alpha = 50 | Level = 4 or 6 | f_low = 0.83 | f_high = 1 | chrome_attenuation = 1
# Change color space: rgb2ntsc (YIQ space) -> cv2.COLOR_RGB2YCrCb
