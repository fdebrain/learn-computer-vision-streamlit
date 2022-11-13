import time
from typing import Tuple

import av
import cv2
import numpy as np

from src.utils import VIDEO_CHANNELS, VIDEO_FRAME_RATE, VIDEO_HEIGHT, VIDEO_WIDTH


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


def combine(img1: np.ndarray, img2: np.ndarray, crop_size: int):
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


def extract_heart_rate(fft: np.ndarray, freqs: np.ndarray) -> int:
    # Compute average FFT amplitude for each slice (~frequency)
    fft_mean = [np.abs(fft_slice).mean() for fft_slice in fft]

    # Compute frequency corresponding with highest average FFT amplitude
    # TODO: Filter outliers
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

        # Crop parameters
        self.crop = False

        # Sequence buffer parameters
        self._level = 4
        self.resolution_bpm = 3
        # self.buffer_size = self.video_frame_rate * 10  # Buffer of 10s
        self.buffer_size = int((60 * self.video_frame_rate) / self.resolution_bpm)
        self.buffer_index = 0
        self.sequence_crop = None

        # Temporal bandpass filter parameters
        # Resolution in bpm = 60 * video_frame_rate / buffer_size
        self._f_min = 0.8  # 48 bpm
        self._f_max = 2.0  # 120 bpm
        self.freqs = np.fft.fftfreq(self.buffer_size, d=1 / self.video_frame_rate)
        self.mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)

        # Heart rate buffer parameters
        self.heart_rate_every_n_frames = 1 * self.video_frame_rate  # 1s
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
        self.tic = self.toc

        # Fetch a new frame
        img = frame.to_ndarray(format="bgr24")  # Type: uint8
        self.video_height = img.shape[0]
        self.video_width = img.shape[1]
        self.crop_size = self.video_width // 5
        self.crop_left = self.video_width // 2 - self.crop_size
        self.crop_right = self.video_width // 2 + self.crop_size
        self.crop_top = self.video_height // 2 - self.crop_size
        self.crop_bottom = self.video_height // 2 + self.crop_size

        # Left-Right flip
        if self.flip:
            img = cv2.flip(img, 1)

        # Center crop
        if self.crop:
            img_crop_bgr = img[
                self.crop_top : self.crop_bottom, self.crop_left : self.crop_right
            ]
        else:
            img_crop_bgr = img

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
                ),
                dtype=np.uint8,
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
            self.fft_mean = np.abs(fft).reshape(self.buffer_size, -1).mean(axis=-1)
            print(self.heart_rate_buffer)

            # Adapt frame rate and associated frequency table
            self.video_frame_rate = int(1 / delta)
            print(f"Frame rate: ({self.video_frame_rate}Hz)")
            self.freqs = np.fft.fftfreq(self.buffer_size, d=delta)
            self.mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)

        # Compute inverse FFT to filtered sequence and amplify it
        sequence_crop_filtered = np.real(np.fft.ifft(fft, axis=0))
        img_crop_filtered = self.alpha * sequence_crop_filtered[self.buffer_index]

        # Reconstruct magnified version of current image crop
        img_crop_magnified = combine(
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
        # draw_face_box(img, point1, point2)
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
            mean_pulse = self.heart_rate_buffer[self.heart_rate_buffer > 0].mean()
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
