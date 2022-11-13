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
VIDEO_FRAME_RATE = 10
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


def compute_pyramid_level(
    img: np.ndarray,
    level: int,
    shape: Tuple[int] = None,
) -> np.ndarray:
    """Compute the image at the i_th level of a Gaussian pyramid.

    A Gaussian pyramid is constructed by iterative subsampling and blurring operations.
    """
    pyramid = [img.copy()]  # Level 0 = initial image
    pyramid.extend(cv2.pyrDown(pyramid[-1]) for _ in range(level))

    if shape:
        return cv2.resize(pyramid[-1], (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    else:
        return pyramid[-1]


def reconstruct(img: np.ndarray, level: int, shape: Tuple[int]) -> np.ndarray:
    filtered = img.copy()
    for _ in range(level + 1):
        filtered = cv2.pyrUp(filtered)
    # return filtered[: shape[0], : shape[1]]
    return cv2.resize(filtered, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def extract_heart_rate(fft: np.ndarray, freqs: np.ndarray) -> int:
    # Compute average FFT amplitude for each slice (~frequency)
    fft_mean = [np.abs(fft_slice).mean() for fft_slice in fft]

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
        self.crop_size = self.video_width // 5

        # Sequence buffer parameters
        self._level = 2
        self.buffer_size = self.video_frame_rate * 10  # Buffer of 10s
        self.buffer_index = 0
        self.sequence_crop = None

        # Temporal bandpass filter parameters
        self.freqs = np.fft.fftfreq(self.buffer_size, d=1 / self.video_frame_rate)
        self._f_min = 0.8  # 48 bpm
        self._f_max = 2.0  # 120 bpm
        self.mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)

        # Heart rate buffer parameters
        self.heart_rate_every_n_frames = self.video_frame_rate  # 1s
        self.heart_rate_buffer_size = 10
        self.heart_rate_buffer = np.zeros((self.heart_rate_buffer_size), dtype=np.uint8)
        self.heart_rate_buffer_index = 0

        self.ind = 0

        # Magnification parameters
        self.fft_mean = None
        self.alpha = 50

        # Timing
        self.tic = 0
        self.toc = 0

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, val: int):
        print(f"Update level: {val}")
        self._level = val
        self.buffer_index = 0
        self.sequence_crop = None
        self.ind = 0

    @property
    def f_min(self):
        return self._f_min

    @property
    def f_max(self):
        return self._f_max

    @f_min.setter
    def f_min(self, val: float):
        print(f"Update f_min: {val}")
        self._f_min = val
        self.mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)

    @f_max.setter
    def f_max(self, val: float):
        print(f"Update f_max: {val}")
        self._f_max = val
        self.mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Check the frame rate
        self.toc = time.time()
        delta = self.toc - self.tic
        print(f"Took {(delta):.2f}s ({(1/delta):.1f}Hz)")
        self.tic = self.toc

        # Fetch a new frame
        img = frame.to_ndarray(format="bgr24")  # Type: uint8
        self.video_height = img.shape[0]
        self.video_width = img.shape[1]
        self.crop_size = self.video_width // 5

        # Left-Right flip
        if self.flip:
            img = cv2.flip(img, 1)

        # Detect face
        with self.face_detection(
            model_selection=0,
            min_detection_confidence=0.5,
        ) as face_detection:
            results = face_detection.process(img)

        if results.detections:
            # Fetch box parameters of the first face
            face_box = get_face_box(results)
            point1, point2 = box_to_points(face_box, img.shape)

            # Crop face
            xmin, ymin = point1
            xmax, ymax = point2
            img_crop_bgr = img[ymin:ymax, xmin:xmax]

            # Change color space (BGR to YCrCb)
            img_crop = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2YCrCb)

            # Initialize sequence buffer
            if self.sequence_crop is None:
                # Extract desired level in Gaussian pyramid (downsampled & blurred)
                img_crop_level = compute_pyramid_level(img_crop, self.level)
                self.sequence_crop = img_crop_level * np.ones(
                    (
                        self.buffer_size,
                        img_crop_level.shape[0],
                        img_crop_level.shape[1],
                        self.video_channels,
                    )
                )
            else:
                # Extract desired level in Gaussian pyramid (downsampled & blurred)
                img_crop_level = compute_pyramid_level(
                    img_crop, self.level, self.sequence_crop.shape[1:3]
                )

            # Fill buffer with extracted gaussian pyramid level of face crop
            self.sequence_crop[self.buffer_index] = img_crop_level

            # Compute temporal FFT (capture pixel-intensity modulation frequency)
            fft = np.fft.fft(self.sequence_crop, axis=0)
            fft[~self.mask] = 0  # Apply ideal temporal bandpass filter

            # Estimate heart rate every heart_rate_every_n_frames
            if self.buffer_index % self.heart_rate_every_n_frames == 0:
                if self.ind < self.heart_rate_buffer_size:
                    self.ind += 1

                # Compute heart rate estimate from filtered FFT
                self.heart_rate_buffer[self.heart_rate_buffer_index] = extract_heart_rate(
                    fft, self.freqs
                )

                # Increment circular heart rate buffer
                self.heart_rate_buffer_index = (
                    self.heart_rate_buffer_index + 1
                ) % self.heart_rate_buffer_size

                # TODO: Visualize FFT signal (spacial mean of last slide)
                self.fft_mean = np.real(fft).reshape(self.buffer_size, -1).mean(axis=-1)
                print(self.heart_rate_buffer)

            def reconstruct(img1, img2, crop_size):
                img_raw = cv2.resize(
                    img1,
                    (crop_size, crop_size),
                    interpolation=cv2.INTER_AREA,
                )
                img_filt = cv2.resize(
                    img2,
                    (crop_size, crop_size),
                    interpolation=cv2.INTER_AREA,
                )
                img_magnified = img_raw + img_filt
                return cv2.convertScaleAbs(img_magnified)

            # Compute inverse FFT to filtered sequence and amplify it
            sequence_crop_filtered = np.real(np.fft.ifft(fft, axis=0))
            img_crop_filtered = self.alpha * sequence_crop_filtered[self.buffer_index]

            # Reconstruct magnified version of current image crop
            img_crop_magnified = reconstruct(
                img_crop,
                img_crop_filtered,
                self.crop_size,
            )

            # Convert back to RGB color space
            img_crop_magnified_bgr = cv2.cvtColor(
                img_crop_magnified,
                cv2.COLOR_YCrCb2BGR,
            )

            # Show detected face + resize and display crops in top-left area
            draw_face_box(img, point1, point2)
            img_crop_bgr = cv2.resize(
                img_crop_bgr,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_AREA,
            )
            img_crop_magnified_bgr = cv2.resize(
                img_crop_magnified_bgr,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_AREA,
            )
            img[: self.crop_size, : self.crop_size] = img_crop_bgr
            img[
                self.crop_size : 2 * self.crop_size, : self.crop_size
            ] = img_crop_magnified_bgr

            # Display pulse
            if self.ind >= self.heart_rate_buffer_size:
                mean_pulse = self.heart_rate_buffer.mean()
                message = f"Pulse: {mean_pulse:.1f} bpm"
            else:
                mean_pulse = self.heart_rate_buffer[: self.ind - 1].mean()
                message = f"Pulse: {mean_pulse:.1f} bpm (noisy estimate)"
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

            # Increment circular sequence buffer
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
