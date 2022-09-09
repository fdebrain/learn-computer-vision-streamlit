import numpy as np


def compute_fft(img):
    f = np.fft.fft2(img)  # Compute DFT
    fshift = np.fft.fftshift(f).astype(np.complex64)  # Center
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-3)
    return fshift, magnitude_spectrum


def compute_inverse_fft(spectrum):
    reconstructed_img = np.abs(np.fft.ifft2(spectrum)).astype(np.uint8)
    return reconstructed_img
