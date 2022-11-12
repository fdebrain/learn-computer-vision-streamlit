import time
from functools import partial
from typing import Dict, List, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                ]
            }
        ]
    }
)
VIDEO_HEIGHT = 540
VIDEO_WIDTH = 900
VIDEO_FRAME_RATE = 5
VIDEO_CHANNELS = 3


def relative_to_absolute_box(box: Dict[str, float], shape: List[int]) -> Dict[str, int]:
    """Convert a box dictionnary from relative values (0 to 1) to absolute values (pixels).

    Required keys: xmin, ymin, height, width.
    """
    assert len(shape) >= 2, "Shape should contain at least 2 elements."
    assert set(box.keys()) == {
        "xmin",
        "ymin",
        "height",
        "width",
    }, "Box is missing one of the required keys"

    img_height = shape[0]
    img_width = shape[1]
    ymin_factor = 0.7
    height_factor = 1.2
    return {
        "xmin": int(box["xmin"] * img_width),
        "ymin": int(box["ymin"] * ymin_factor * img_height),
        "height": int(box["height"] * height_factor * img_height),
        "width": int(box["width"] * img_width),
    }


def box_to_points(box: Dict[str, float], shape: List[int]) -> Tuple[Tuple[int, int]]:
    """Compute top-left and bottom-right points given a relative box."""
    face_box = relative_to_absolute_box(box, shape)
    point1 = (face_box["xmin"], face_box["ymin"])
    point2 = (
        face_box["xmin"] + face_box["width"],
        face_box["ymin"] + face_box["height"],
    )
    return point1, point2


def get_face_box(results) -> Dict[str, float]:
    """Extract relative bounding box from MediaPipe Face Detection result."""
    return {
        coord: getattr(
            results.detections[0].location_data.relative_bounding_box,
            coord,
        )
        for coord in ["xmin", "ymin", "height", "width"]
    }


def draw_face_box(
    img: np.ndarray,
    point1: Tuple[int, int],
    point2: Tuple[int, int],
) -> None:
    """Draw a rectangle in the image given top-left and bottom-right points."""
    color = (255, 255, 255)
    thickness = 1
    lineType = cv2.LINE_AA
    cv2.rectangle(img, point1, point2, color, thickness, lineType)


def compute_gauss_pyramid_level(img: np.ndarray, level: int) -> np.ndarray:
    """Compute the image at the i_th level of a Gauss pyramid.

    A Gauss pyramid is constructed by iterative subsampling and blurring operations.
    """
    pyramid = [img.copy()]  # Level 0 = initial image
    pyramid.extend(cv2.pyrDown(pyramid[-1]) for _ in range(level))
    return pyramid[-1]


def reconstructFrame(img: np.ndarray, levels: int, shape: Tuple[int]) -> np.ndarray:
    filteredFrame = img.copy()
    for _ in range(1, levels + 1):
        filteredFrame = cv2.pyrUp(filteredFrame)
    return filteredFrame[: shape[0], : shape[1]]


def extract_heart_rate(fft: np.ndarray, freqs: np.ndarray) -> int:
    # Compute average FFT amplitude for each slice (~frequency)
    fft_mean = [np.real(fft_slice).mean() for fft_slice in fft]

    # Compute frequency corresponding with highest average FFT amplitude
    hz = freqs[np.argmax(fft_mean)]
    return int(60.0 * hz)


class VideoProcessorHREstimation:
    def __init__(self) -> None:
        # Video parameters
        self.video_height = VIDEO_HEIGHT
        self.video_width = VIDEO_WIDTH
        self.video_frame_rate = VIDEO_FRAME_RATE
        self.video_channels = VIDEO_CHANNELS
        self.flip = True

        # Face crop parameters
        self.face_detection = mp.solutions.face_detection.FaceDetection
        self.crop_size = self.video_height // 5

        # Gaussian Pyramid buffer & parameters
        # TODO: Setter
        self.level = 2  # TODO: Add slider
        self.firstFrame = np.zeros((self.crop_size, self.crop_size, self.video_channels))
        self.firstGauss = compute_gauss_pyramid_level(self.firstFrame, self.level)
        self.buffer_size = 150  # TODO: Add slider
        self.buffer_index = 0
        self.videoGauss = np.zeros(
            (
                self.buffer_size,
                self.firstGauss.shape[0],
                self.firstGauss.shape[1],
                self.video_channels,
            )
        )

        # Frequencies lookup table & bandpass filter parameters
        self.f_min = 1.0  # 60 bpm
        self.f_max = 4.0  # 240 bpm

        # Heart rate buffer & parameters
        self.heart_rate_buffer_size = 15
        self.heart_rate_buffer = np.zeros((self.heart_rate_buffer_size), dtype=np.uint8)
        self.heart_rate_buffer_index = 0
        self.heart_rate_every_n_frames = 10
        self.ind = 0

        # Reconstruction
        self.fft_mean = None
        self.alpha = 50

        self.tic = 0
        self.toc = 0

    def recv(self, frame):
        # Check frame rate (5Hz)
        self.toc = time.time()
        delta = self.toc - self.tic
        print(f"Took {(delta):.2f}s ({(1/delta):.1f}Hz)")
        self.tic = self.toc

        # Fetch frame
        img = frame.to_ndarray(format="bgr24")
        self.video_height = img.shape[0]
        self.video_width = img.shape[1]

        if self.flip:
            img = cv2.flip(img, 1)  # Left-Right flip

        # Detect face
        with self.face_detection(
            model_selection=0,
            min_detection_confidence=0.7,
        ) as face_detection:
            results = face_detection.process(img)

        # Fetch box parameters of the first face
        if results.detections:
            face_box = get_face_box(results)
            point1, point2 = box_to_points(face_box, img.shape)

            # Crop face
            xmin, ymin = point1
            xmax, ymax = point2
            img_crop = img[ymin:ymax, xmin:xmax]
            img_crop_resized = cv2.resize(
                img_crop,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_AREA,
            )

            # Fill buffer with Gaussian Pyramid level of face crop
            self.videoGauss[self.buffer_index] = compute_gauss_pyramid_level(
                img_crop_resized,
                self.level,
            )

            # Compute temporal FFT (capture pixel-intensity modulation frequency)
            fft = np.fft.fft(self.videoGauss, axis=0)

            # Apply bandpass filter (region of interest lies around 60 bpm = 1 Hz)
            self.freqs = np.fft.fftfreq(self.buffer_size, d=1 / self.video_frame_rate)
            mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)
            fft[~mask] = 0

            # Estimate heart rate every heart_rate_every_n_frames
            if self.buffer_index % self.heart_rate_every_n_frames == 0:
                self.ind += 1
                bpm = extract_heart_rate(fft, self.freqs)

                # Add to heart rate buffer
                self.heart_rate_buffer[self.heart_rate_buffer_index] = bpm
                self.heart_rate_buffer_index = (
                    self.heart_rate_buffer_index + 1
                ) % self.heart_rate_buffer_size

                # TODO: Visualize FFT signal (spacial mean of last slide)
                self.fft_mean = np.real(fft).reshape(self.buffer_size, -1).mean(axis=-1)
                print(self.heart_rate_buffer)

            # Compute inverse FFT and amplify it
            filtered = np.real(np.fft.ifft(fft, axis=0))
            filtered *= self.alpha

            # Reconstruct latest image (filtered in frequency space)
            filtered_crop = reconstructFrame(
                filtered[self.buffer_index],
                self.level,
                shape=img_crop_resized.shape,
            )
            outputFrame = img_crop_resized + filtered_crop
            outputFrame = cv2.convertScaleAbs(outputFrame)

            # Display crop image in top-left area + show detected face
            draw_face_box(img, point1, point2)
            img[: self.crop_size, : self.crop_size] = img_crop_resized
            img[self.crop_size : 2 * self.crop_size, : self.crop_size] = outputFrame

            # Display pulse
            if self.ind >= self.heart_rate_buffer_size:
                message = f"Pulse: {self.heart_rate_buffer.mean():.1f} bpm"
            else:
                message = (
                    f"Pulse: {self.heart_rate_buffer[self.ind - 1]} bpm (noisy estimate)"
                )
            cv2.putText(
                img,
                message,
                (10, self.video_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(0, 255, 0),
                lineType=1,
            )

            # Increment circular buffer
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def create_local_file_player(filepath: str):
    return MediaPlayer(filepath)


def create_webcam_stream(processor):
    return webrtc_streamer(
        key="webcam",
        video_processor_factory=processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {  # Increments until reaching desired resolution
                "width": {"min": VIDEO_WIDTH, "max": VIDEO_WIDTH},
                "height": {"min": VIDEO_HEIGHT, "max": VIDEO_HEIGHT},
                "frameRate": {"min": VIDEO_FRAME_RATE, "max": VIDEO_FRAME_RATE},
            },
            "audio": False,
        },
        async_processing=True,
    )


def create_local_video_stream(processor, filepath: str):
    return webrtc_streamer(
        key="youtube",
        video_processor_factory=processor,
        rtc_configuration=RTC_CONFIGURATION,
        mode=WebRtcMode.RECVONLY,
        player_factory=partial(create_local_file_player, filepath=filepath),
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )
