import time
from typing import Dict, List, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np

from src.utils import VIDEO_CHANNELS, VIDEO_FRAME_RATE, VIDEO_HEIGHT, VIDEO_WIDTH


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
    start_factor = 0.5
    end_factor = 1.2
    return {
        "xmin": int(box["xmin"] * img_width),
        "ymin": int(start_factor * box["ymin"] * img_height),
        "height": int(end_factor * box["height"] * img_height),
        "width": int(end_factor * box["width"] * img_width),
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


class VideoProcessorFaceDetection:
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
        self.detection_threshold = 0.8

        # Timing
        self.tic = 0
        self.toc = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Check the frame rate
        self.toc = time.time()
        delta = self.toc - self.tic
        print(f"Took {(delta):.2f}s ({(1/delta):.1f}Hz)")
        self.tic = self.toc

        # Fetch a new frame
        img = frame.to_ndarray(format="bgr24")  # Type: uint8
        self.video_height = img.shape[0]  # WebRTC can change resolution during feed
        self.video_width = img.shape[1]
        self.crop_size = self.video_width // 5

        # Left-Right flip
        if self.flip:
            img = cv2.flip(img, 1)

        # Detect face
        with self.face_detection(
            model_selection=0,
            min_detection_confidence=self.detection_threshold,
        ) as face_detection:
            results = face_detection.process(img)

        if results.detections:
            # Fetch box parameters of the first face
            face_box = get_face_box(results)
            point1, point2 = box_to_points(face_box, img.shape)

            # Crop face
            xmin, ymin = point1
            xmax, ymax = point2
            img_crop = img[ymin:ymax, xmin:xmax]

            # Show detected face + resize and display crops in top-left area
            try:
                draw_face_box(img, point1, point2)
                img_crop = cv2.resize(
                    img_crop,
                    (self.crop_size, self.crop_size),
                    interpolation=cv2.INTER_AREA,
                )
                img[: self.crop_size, : self.crop_size] = img_crop
            except Exception as e:
                print(e)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
