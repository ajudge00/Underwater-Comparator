import enum

import cv2
import numpy as np


class WeightMapMethods(enum.Enum):
    LAPLACIAN = 0,
    SATURATION = 1,
    SALIENCY = 2,
    SALIENCY2 = 3


def get_weight_map(img: np.ndarray, method: WeightMapMethods) -> np.ndarray:
    if method == WeightMapMethods.LAPLACIAN:
        return laplacian_contrast_weight(img)
    elif method == WeightMapMethods.SATURATION:
        return saturation_weight(img)
    elif method == WeightMapMethods.SALIENCY:
        return saliency_weight(img)


def laplacian_contrast_weight(img: np.ndarray) -> np.ndarray:
    """
    Calculates the absolute value of a Laplacian filter on the luminance channel.
    :param img: A uint8, [0, 255] image
    :return: A uint8, [0, 255] grayscale image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    # Színtér konverzió helyett egyszerűbb, ha grayscale az egész
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian előtt érdemes simítani kicsit, hogy a zaj ne befolyásolja az eredményt.
    laplacian = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0, sigmaY=0)
    laplacian = cv2.Laplacian(laplacian, cv2.CV_8U, ksize=3)

    # Az Ancuti2018 cikk valószínűleg előjeles értékeket akar abszolút értékezni.
    # Az opencv Laplacian függvénye ezt nem nagyon akarja, ezért skálázzuk külön
    # a 8U-t 8S-sé, majd úgy jöhet az abszolút érték.
    laplacian = cv2.normalize(laplacian, None, -128, 127, cv2.NORM_MINMAX, cv2.CV_8S)
    W_laplacian_contrast = cv2.convertScaleAbs(laplacian)

    return W_laplacian_contrast.astype(np.uint8)


def saturation_weight(img: np.ndarray) -> np.ndarray:
    """
    Computed as the deviation (for every pixel location) between
    the R,G and B color channels and the luminance L.
    W_Sat = sqrt( (1/3) * [(R − L) ^ 2 + (G − L) ^ 2 + (B − L) ^ 2] )
    :param img: A uint8, [0, 255] image
    :return: A uint8, [0, 255] grayscale image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L = (lab_img.astype(np.float32) / 255.0)[:, :, 0]
    B, G, R = cv2.split(img.astype(np.float32))

    W_Sat = np.sqrt((1 / 3.0) * ((R - L) ** 2 + (G - L) ** 2 + (B - L) ** 2))
    W_Sat = cv2.normalize(W_Sat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return 255 - W_Sat


def saliency_weight(img: np.ndarray) -> np.ndarray:
    """
    Based on Achanta et al.'s Saliency Detector C++ implementation
    :param img: A uint8, [0, 255] image
    :return: A uint8, [0, 255] grayscale image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)

    l_mean, a_mean, b_mean = np.mean(l), np.mean(a), np.mean(b)

    kernel_size = (5, 5)
    l_blurred = cv2.GaussianBlur(l, kernel_size, 0)
    a_blurred = cv2.GaussianBlur(a, kernel_size, 0)
    b_blurred = cv2.GaussianBlur(b, kernel_size, 0)

    W_sal = ((l_blurred - l_mean) ** 2 +
             (a_blurred - a_mean) ** 2 +
             (b_blurred - b_mean) ** 2)

    W_sal = cv2.normalize(W_sal, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return W_sal


def normalize_weight_maps(
        lap1: np.ndarray, sat1: np.ndarray, sal1: np.ndarray,
        lap2: np.ndarray, sat2: np.ndarray, sal2: np.ndarray,
        reg_term=0.1) -> tuple[np.ndarray, np.ndarray]:
    lap1 = lap1.astype(np.float32) / 255.0
    sal1 = sal1.astype(np.float32) / 255.0
    sat1 = sat1.astype(np.float32) / 255.0
    lap2 = lap2.astype(np.float32) / 255.0
    sat2 = sat2.astype(np.float32) / 255.0
    sal2 = sal2.astype(np.float32) / 255.0

    denom = lap1 + sal1 + sat1 + lap2 + sal2 + sat2 + 2 * reg_term
    W_Normalized1 = (lap1 + sal1 + sat1 + reg_term) / denom
    W_Normalized2 = (lap2 + sal2 + sat2 + reg_term) / denom

    W_Normalized1 = cv2.normalize(W_Normalized1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    W_Normalized2 = cv2.normalize(W_Normalized2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return W_Normalized1, W_Normalized2


def interesting_saliency(img: np.ndarray) -> np.ndarray:
    # this is rly fckin cool but doesnt work properly
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    mean_image_feature_vector = img_lab.mean(axis=(0, 1))

    binomial_kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    binomial_kernel_1d /= binomial_kernel_1d.sum()
    img_blurred = cv2.sepFilter2D(img_lab, -1, binomial_kernel_1d, binomial_kernel_1d)

    diff = cv2.subtract(img_blurred, mean_image_feature_vector)
    W_Sal = np.sqrt(np.sum(diff ** 2, axis=2))

    return cv2.normalize(W_Sal.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
