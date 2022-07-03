import cv2
import numpy as np
import scipy.signal


def compute_spatial_derivative(img: np.ndarray):
    kernel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )
    kernel_y = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]
    )
    Ix = scipy.signal.convolve2d(img, kernel_x, mode="same")
    Iy = scipy.signal.convolve2d(img, kernel_y, mode="same")
    return Ix, Iy


def compute_harris_response_matrix(
    Ix: np.ndarray,
    Iy: np.ndarray,
    k: float,
    sigma: int,
):
    Ixx = scipy.ndimage.gaussian_filter(Ix**2, sigma)
    Iyy = scipy.ndimage.gaussian_filter(Iy**2, sigma)
    Ixy = scipy.ndimage.gaussian_filter(Ix * Iy, sigma)
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    return det - k * trace**2


def overlay_detected_corners(img: np.ndarray, corners):
    for px, py in corners:
        cv2.circle(img, (px, py), radius=1, color=(255, 0, 0), thickness=5)
