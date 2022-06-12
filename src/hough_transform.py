import time
from typing import Tuple

import cv2
import numpy as np

from src.utils import nms


def compute_hough_lines(
    edges: np.ndarray,
    rho_res: float,
    theta_res: float,
    threshold: float,
    neighborhood_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute support lines accumulator from a binary image of edges."""
    tic = time.time()
    width, height = edges.shape
    diag_len = np.sqrt(width**2 + height**2)
    eps = 1e-5

    # Pre-compute values as lookup tables
    rho_bins = np.arange(-diag_len, diag_len + eps, rho_res)
    theta_bins = np.arange(0, np.pi + eps, theta_res)
    cos_t = np.cos(theta_bins)
    sin_t = np.sin(theta_bins)

    # Extract coordinates of all edge pixels
    y_vals, x_vals = np.nonzero(edges)  # (row_mat, col_mat) = (y_img, x_img)

    # Compute polar parameters of candidate lines in a vectorized fashion (huge speed-up)
    theta_vals = np.tile(theta_bins, reps=len(x_vals))
    rho_vals = np.concatenate(
        [x_vals[i] * cos_t + y_vals[i] * sin_t for i in range(len(x_vals))]
    )

    # Compute number of supports for a given line (polar parameters)
    accumulator, _, _ = np.histogram2d(
        rho_vals,
        theta_vals,
        bins=[rho_bins, theta_bins],
    )

    # Merge similar lines
    maxima = nms(accumulator, threshold, neighborhood_size)
    lines = [
        [accumulator[rho_idx, theta_idx], rho_bins[rho_idx], theta_bins[theta_idx]]
        for rho_idx, theta_idx in zip(*np.nonzero(maxima))
    ]

    print(f"Computing Hough Lines Transform took {(time.time() - tic):.1f}s")

    return lines, accumulator


def overlay_hough_lines(
    img: np.ndarray,
    lines: np.ndarray,
    line_thickness: int = 5,
) -> np.ndarray:
    """Plot detected lines on top of an image."""
    eps = 1e-5
    for count, rho, theta in lines:
        # Intersection point P0
        x0 = int(rho * np.cos(theta))
        y0 = int(rho * np.sin(theta))

        # Point at horizont y -> -inf
        y1 = y0 - img.shape[1]
        x1 = int((rho - y1 * np.sin(theta)) / (np.cos(theta) + eps))

        # Point at horizont y -> +inf
        y2 = y0 + img.shape[1]
        x2 = int((rho - y2 * np.sin(theta)) / (np.cos(theta) + eps))

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)
        print(
            f"Rho: {int(rho)} | Theta: {np.rad2deg(theta):.0f}° | Supports: {int(count)}"
        )
    return img


def compute_hough_circles(
    edges: np.ndarray,
    r_min: int,
    r_max: int,
    pixel_res: int,
    theta_res: int,
    threshold: int,
    neighborhood_size: int,
):
    tic = time.time()

    # Pre-compute values as lookup tables
    radius = np.arange(r_min, r_max, pixel_res, dtype=np.uint16)
    thetas = np.arange(0, 2 * np.pi, theta_res, dtype=np.float32)
    cos_t = np.cos(thetas, dtype=np.float32)
    sin_t = np.sin(thetas, dtype=np.float32)
    a_bins = np.arange(-r_max, edges.shape[1] + r_max, pixel_res, dtype=np.int16)
    b_bins = np.arange(-r_max, edges.shape[0] + r_max, pixel_res, dtype=np.int16)

    # Extract coordinates of all edge pixels
    y_vals, x_vals = np.nonzero(edges)  # (row_mat, col_mat) = (y_img, x_img)

    # Compute candidate circles
    candidates = []
    for r in radius:
        # Compute centers of candidate circles
        a_vals = np.concatenate([x - r * cos_t for x in x_vals]).astype(np.int16)
        b_vals = np.concatenate([y - r * sin_t for y in y_vals]).astype(np.int16)

        # Compute number of supports for a given circle radius
        accumulator, _, _ = np.histogram2d(a_vals, b_vals, bins=[a_bins, b_bins])
        accumulator = accumulator.astype(np.uint16)

        # Merge circles with similar centers (avoid memory overload)
        maxima = nms(accumulator, threshold=threshold, neighborhood_size=5)
        a_idxs, b_idxs = np.nonzero(maxima)
        counts = accumulator[a_idxs, b_idxs]

        # Store candidate counts and circle parameters
        candidates.extend(
            list(
                zip(
                    counts,
                    a_bins[a_idxs],
                    b_bins[b_idxs],
                    [r] * len(a_idxs),
                )
            )
        )

    candidates = np.array(candidates, dtype=np.int16)
    if len(candidates) == 0:
        return []

    # Repeat candidates by counts
    all_candidates = np.concatenate(
        [np.tile(circle, (count, 1)) for count, *circle in candidates]
    )

    # Compute number of supports for a given center and radius
    accumulator, _ = np.histogramdd(all_candidates, bins=[a_bins, b_bins, radius])
    accumulator = accumulator.astype(np.uint16)

    # Merge similar circles (center and radius)
    maxima = nms(accumulator, 1, neighborhood_size)
    circles = [
        [accumulator[a_idx, b_idx, r_idx], a_bins[a_idx], b_bins[b_idx], radius[r_idx]]
        for a_idx, b_idx, r_idx in zip(*np.nonzero(maxima))
    ]

    print(f"Computing Hough Circle Transform took {(time.time() - tic):.1f}s")
    return np.array(circles)


def overlay_hough_circles(
    img: np.ndarray,
    circles: np.ndarray,
    line_thickness: int,
) -> np.ndarray:
    """Plot detected circles on top of an image."""
    for count, xc, yc, r in circles:
        cv2.circle(
            img,
            [xc, yc],
            r,
            (255, 0, 0),
            line_thickness,
        )
        print(f"Center: ({xc}, {yc}) |  Radius: {r} | Supports: {count}")
    return img
