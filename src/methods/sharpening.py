import cv2
import numpy as np


def normalized_unsharp_masking(img: np.ndarray) -> np.ndarray:
    """
    Based on Ancuti et al. 2018 "Normalized Unsharp Masking":\n
    sharpened = (initial + normalize(initial −  Gaussian-filtered)) / 2,
    :param img: A uint8, [0, 255] image
    :return: A uint8, [0, 255] image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    gaussian_filtered = cv2.GaussianBlur(img, (3, 3), 0)

    diff = cv2.subtract(img, gaussian_filtered)

    diff_min = np.min(diff)
    diff_max = np.max(diff)
    diff_stretched = (((diff - diff_min) / (diff_max - diff_min)) * 255).astype(np.uint8)

    result = cv2.add(img, diff_stretched) / 2.0

    return result.astype(np.uint8)


def normalized_unsharp_masking_matlab(img: np.ndarray, sigma=20, N=30, gain=1.0) -> np.ndarray:
    """
    Based on fergaletto's MATLAB interpretation of Normalized Unsharp Masking.
    :param gain: Strength of sharpening
    :param N: Number of iterations of Gaussian filtering
    :param sigma: Sigma of the Gaussian filter
    :param img: A uint8, [0, 255] image
    :return: A uint8, [0, 255] image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img = img.astype(np.float32) / 255.0

    Igauss = img.copy()

    for _ in range(N):
        Igauss = cv2.GaussianBlur(Igauss, (0, 0), sigma)
        Igauss = np.minimum(img, Igauss)

    Norm = img - gain * Igauss

    Norm_eq = np.zeros_like(Norm)
    for n in range(3):
        Norm_eq[:, :, n] = cv2.equalizeHist((Norm[:, :, n] * 255).astype(np.uint8)) / 255.0

    Isharp = (img + Norm_eq) / 2
    return (Isharp * 255).astype(np.uint8)
