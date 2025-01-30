import cv2
import numpy as np


def get_uciqe(img: np.ndarray) -> float:
    """
    Underwater Colour Image Quality Evaluation based on M. Yang and A. Sowmya's paper.\n
    UCIQE = c1 * σ_c + c2 * con_l + c3 * μs\n
    where:
     - σ_c is the standard deviation of chroma,
     - con_l is the contrast of luminance (difference between the top 1% and bottom 1% of luminance values),
     - μ_s is the average saturation,
     - c1=0.4680c1=0.4680, c2=0.2745c2=0.2745, and c3=0.2576c3=0.2576
    :param img: A BGR uint8 [0-255] image
    :return: a float between 0 and 1
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    c1, c2, c3 = [0.4680, 0.2745, 0.2576]
    img_lum = img_lab[..., 0] / 255
    img_a = img_lab[..., 1] / 255
    img_b = img_lab[..., 2] / 255

    chroma = np.sqrt(np.square(img_a) + np.square(img_b))

    saturation = chroma / np.sqrt(np.square(chroma) + np.square(img_lum))
    saturation_avg = np.mean(saturation)

    chroma_avg = np.mean(chroma)

    chroma_var = np.sqrt(np.mean(abs(1 - np.square(chroma_avg / chroma))))

    nbins = 256 if img_lum.dtype == np.uint8 else 65536

    hist, bins = np.histogram(img_lum, nbins)
    # cumulative distribution function
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    quality_val = c1 * chroma_var + c2 * con_lum + c3 * saturation_avg
    return quality_val
