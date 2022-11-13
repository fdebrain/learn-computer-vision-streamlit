import contextlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.heart_rate_estimation import VideoProcessorHREstimation
from src.utils import create_local_video_stream, create_webcam_stream

SAVE_DIR = Path(__file__).parent.parent / "data"
SAVE_PATH = SAVE_DIR / "face.mp4"
MEDIA_SOURCES = ["Webcam", "Sample"]

st.title("Heart Rate Estimation using Eulerian Magnification")
st.info("Note: WIP")
# TODO: Improve filtering -> suspecting some artifacts due to ideal filter

# Select stream source
selected_source = st.radio(
    label="Select a media source to stream",
    options=tuple(MEDIA_SOURCES),
    key="radio",
)

# Play video stream
if selected_source == "Webcam":
    ctx = create_webcam_stream(processor=VideoProcessorHREstimation)
else:
    ctx = create_local_video_stream(
        processor=VideoProcessorHREstimation,
        filepath=str(SAVE_PATH),
    )

# Use interactive widgets
with contextlib.suppress(Exception):
    if ctx.video_processor:
        ctx.video_processor.flip = st.checkbox("Flip stream", value=True)
        ctx.video_processor.crop = st.checkbox("Crop", value=selected_source == "Webcam")
        ctx.video_processor.level = st.slider(
            "Gaussian Pyramid level",
            min_value=1,
            max_value=5,
            step=1,
            value=3,
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
st.write("[Website - Video Magnification](https://people.csail.mit.edu/mrub/vidmag/)")
st.write(
    "[Github - Eulerian Magnification](https://github.com/brycedrennan/eulerian-magnification/)"
)
