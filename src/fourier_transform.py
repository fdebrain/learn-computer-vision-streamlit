import cv2
import numpy as np


def compute_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)  # Compute DFT
    fshift = np.fft.fftshift(f).astype(np.complex64)  # Center
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-3)
    return fshift, magnitude_spectrum


def compute_inverse_fft(spectrum):
    reconstructed_img = np.abs(np.fft.ifft2(spectrum)).astype(np.uint8)
    return reconstructed_img
