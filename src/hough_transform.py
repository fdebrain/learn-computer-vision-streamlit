import time

import cv2
import numpy as np
import scipy.ndimage
import streamlit as st


@st.cache()
def compute_hough_lines_accumulator(
    img_bin: np.ndarray,
    rho_res: float,
    theta_res: float,
) -> np.ndarray:
    """Compute support lines accumulator from a binary image of edges."""
    tic = time.time()
    width = img_bin.shape[0]
    height = img_bin.shape[1]
    diag_len = np.sqrt(width**2 + height**2)
    eps = 1e-5

    # Compute lookup tables
    thetas = np.arange(0, np.pi + eps, theta_res)
    rhos = np.arange(-diag_len, diag_len + eps, rho_res)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Compute support line parameters (rho, theta) for each edge point
    y_vals, x_vals = np.nonzero(img_bin)  # (row_mat, col_mat) = (y_img, x_img)
    theta_vals = np.tile(thetas, reps=len(x_vals))  # Repeat vector for each point
    rho_vals = np.concatenate(
        [x_vals[i] * cos_t + y_vals[i] * sin_t for i in range(len(x_vals))]
    )

    # Compute accumulator matrix (support line counts)
    accumulator, _, _ = np.histogram2d(theta_vals, rho_vals, bins=[thetas, rhos])
    accumulator = accumulator.T
    print(f"Computing Hough Lines Transform took {(time.time() - tic):.1f}s")

    return accumulator


def compute_hough_lines_list(
    accumulator: np.ndarray,
    rho_res: float,
    theta_res: float,
    threshold: float,
    neighborhood_size: int = 20,
) -> np.ndarray:
    """Compute the most relevant line parameters from the support lines accumulator."""
    lines = []

    # Non-Maxima-Suppression - Aggregate close maxima together
    data_max = scipy.ndimage.maximum_filter(accumulator, neighborhood_size)
    maxima = accumulator == data_max

    data_min = scipy.ndimage.minimum_filter(accumulator, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0

    # Compute remaining line parameters
    for rho_idx, theta_idx in zip(*np.nonzero(maxima)):
        rho = (rho_idx - accumulator.shape[0] // 2) * rho_res
        theta = theta_idx * theta_res
        lines.append([rho, theta])
    return np.array(lines)


def compute_hough_lines_overlay(
    img: np.ndarray,
    lines: np.ndarray,
    line_thickness: int = 5,
) -> np.ndarray:
    """Plot detected lines on top of an image."""
    eps = 1e-5
    for rho, theta in lines:
        # Intersection point P0
        x0 = int(rho * np.cos(theta))
        y0 = int(rho * np.sin(theta))

        # Point at horizont y -> -inf
        y1 = y0 - img.shape[1]
        x1 = int((rho - y1 * np.sin(theta)) / (np.cos(theta) + eps))

        # Point at horizont y -> +inf
        y2 = y0 + img.shape[1]
        x2 = int((rho - y2 * np.sin(theta)) / (np.cos(theta) + eps))

        # print(f"Rho: {rho} | Theta: {np.rad2deg(theta):.0f} | P0: ({x0}, {y0})")
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)
    return img
