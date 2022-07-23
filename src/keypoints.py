from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st


@st.cache(hash_funcs={cv2.KeyPoint: hash}, allow_output_mutation=True)
def compute_keypoints(img: np.ndarray, n_points=1000):
    orb = cv2.ORB_create(nfeatures=n_points)
    kp = orb.detect(img, None)
    kp, desc = orb.compute(img, kp)
    return kp, desc


def draw_keypoints(
    img: np.ndarray,
    keypoints: cv2.KeyPoint,
    color: Tuple[int, int, int],
    size: int,
):
    for kp in keypoints:
        (x, y) = kp.pt
        cv2.circle(
            img,
            (int(x), int(y)),
            radius=size,
            color=color,
            thickness=-1,
        )
    return img


def compute_matches(descriptor1: np.ndarray, descriptor2: np.ndarray):
    # Compute Hamming distance between the descriptors across images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    distances = [int(x.distance) for x in matches]
    return matches, distances


def filter_matches(matches: List[cv2.DMatch], threshold: int):
    """Filter matched keypoints by distance thresholding."""
    matches = list(filter(lambda x: x.distance < threshold, matches))
    return matches


def draw_matches(
    img1: np.ndarray,
    keypoints1: cv2.KeyPoint,
    img2: np.ndarray,
    keypoints2: cv2.KeyPoint,
    matches: List[cv2.DMatch],
    alpha: float,
    keypoint_color: Tuple[int, int, int],
    match_color: Tuple[int, int, int],
):
    # Create montage image of the two input images
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    montage = np.zeros((max([height1, height2]), width1 + width2, 3), dtype="uint8")
    montage[:height1, :width1] = img1
    montage[:height2, width1:] = img2

    # Keypoints drawings (before overlaying on montage image)
    drawing = np.zeros((max([height1, height2]), width1 + width2, 3), dtype="uint8")

    # Draw matched keypoints
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Extract keypoints coordinates
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw keypoints on each image
        cv2.circle(
            drawing,
            (int(x1), int(y1)),
            radius=2,
            color=keypoint_color,
            thickness=-1,
        )
        cv2.circle(
            drawing,
            (int(x2) + width1, int(y2)),
            radius=2,
            color=keypoint_color,
            thickness=-1,
        )

        # Draw a line between the matched keypoints
        cv2.line(
            drawing,
            (int(x1), int(y1)),
            (int(x2) + width1, int(y2)),
            color=match_color,
            thickness=1,
        )

    mask = drawing.nonzero()[:2]  # Extract pixel coordinates of the drawing
    montage[mask] = (1 - alpha) * montage[mask] + alpha * drawing[mask]
    return montage
