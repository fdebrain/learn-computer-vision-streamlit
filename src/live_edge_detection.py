from pathlib import Path

import av
import cv2
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
SAVE_PATH = Path(__file__).parent.parent / "data/video.mp4"


class VideoProcessor:
    def __init__(self) -> None:
        self.threshold1 = 50
        self.threshold2 = 100
        self.flip = False
        self.canny = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.canny:
            img = cv2.Canny(img, self.threshold1, self.threshold2)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.flip:
            img = cv2.flip(img, 1)  # Left-Right flip

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def create_local_file_player():
    return MediaPlayer(str(SAVE_PATH))


def create_webcam_stream():
    return webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def create_local_video_stream():
    return webrtc_streamer(
        key="youtube",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        mode=WebRtcMode.RECVONLY,
        player_factory=create_local_file_player,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )
