import cv2
import numpy as np


def create_gamma_lut(gamma: float) -> np.ndarray[np.uint8]:
    lut = np.arange(0, 256, 1, np.float32)
    lut = lut / 255.0
    lut = lut ** gamma
    lut = np.uint8(lut * 255.0)

    return lut


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Conventional gamma correction using a LUT generated from the given gamma value.
    :param img: A uint8, [0, 255] image
    :param gamma: A float value between 0.0 and 5.0
    :return: A uint8, [0, 255] image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    res = cv2.LUT(img, create_gamma_lut(gamma))
    return res


def clahe_with_lab(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization
    :param img: A uint8, [0, 255] bgr image
    :param clip_limit: Float between 1.0 and 40.0
    :param tile_grid_size:
    :return:
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahed = clahe.apply(img_lab[:, :, 0])

    res = cv2.merge((clahed, img_lab[:, :, 1], img_lab[:, :, 2]))
    res = cv2.cvtColor(res, cv2.COLOR_LAB2BGR)

    return res


def histogram_linearization(img: np.ndarray, r: float = 1.0) -> np.ndarray:
    """
    Based on "A Novel Approach for Contrast Enhancement Based on Histogram Equalization" by Yeganeh et al.
    :param img: A uint8 [0, 255] bgr image
    :param r: A float between 0.0 and 1.0. If =1.0, the process will be equivalent to histogram equalization
    :return: A uint8 [0, 255] bgr image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    img_l = img[:, :, 0]

    hist, bins = np.histogram(img_l.flatten(), 256, (0, 256))
    pdf = hist / np.sum(hist)

    P_max = np.max(pdf)

    pdf_new = np.where(pdf > 0, (pdf / P_max) ** r * P_max, 0)

    cdf_new = np.cumsum(pdf_new)
    cdf_new = (cdf_new - cdf_new.min()) * 255 / (cdf_new.max() - cdf_new.min())
    cdf_new = np.round(cdf_new).astype(np.uint8)

    img_new = cdf_new[img_l]
    img_new = cv2.merge([img_new, img[:, :, 1], img[:, :, 2]])

    return cv2.cvtColor(img_new, cv2.COLOR_LAB2BGR)


def pix_dis(histogram, pixel_count: int):
    L = histogram.shape[0]

    res = 0
    for i in range(0, L - 2):
        for j in range(i + 1, L - 1):
            res += int(histogram[i]) * int(histogram[j - i])

    res /= (pixel_count * (pixel_count - 1))
    return res


def histogram_linearization_auto(img: np.ndarray) -> (np.ndarray, float):
    """
    Based on "A Novel Approach for Contrast Enhancement Based on Histogram Equalization" by Yeganeh et al.
    :param img: A uint8 [0, 255] BGR image.
    :return: A uint8 [0, 255] BGR image with enhanced contrast.
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_l = img[:, :, 0]

    hist, bins = np.histogram(img_l.flatten(), 256, (0, 256))
    pdf = hist / np.sum(hist)

    P_max = np.max(pdf)

    max_pix_dis = -1000000
    max_pix_dis_r = -1

    for r in np.linspace(0.1, 1.0, 10):
        pdf_new = np.where((pdf > 0) & (pdf < P_max), (pdf / P_max) ** r * P_max, pdf)

        hist_new = (pdf_new * np.sum(hist)).astype(np.uint32)
        curr_pix_dis = pix_dis(hist_new, img.shape[0] * img.shape[1])

        if curr_pix_dis > max_pix_dis:
            max_pix_dis = curr_pix_dis
            max_pix_dis_r = r

    print(f"Best r: {max_pix_dis_r}, Max PixDis: {max_pix_dis}")

    pdf_new = np.where((pdf > 0) & (pdf < P_max), (pdf / P_max) ** max_pix_dis_r * P_max, pdf)
    cdf_new = np.cumsum(pdf_new)
    cdf_new = (cdf_new - cdf_new.min()) * 255 / (cdf_new.max() - cdf_new.min())
    cdf_new = np.round(cdf_new).astype(np.uint8)

    img_new = cdf_new[img_l]
    img_new = cv2.merge([img_new, img[:, :, 1], img[:, :, 2]])

    return cv2.cvtColor(img_new, cv2.COLOR_LAB2BGR), max_pix_dis_r
