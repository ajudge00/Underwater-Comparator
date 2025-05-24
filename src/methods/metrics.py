import cv2
import numpy as np

from src.methods.dehazing import dark_channel, estimate_atmospheric_light, estimate_transmission


def get_uciqe(img: np.ndarray) -> float:
    """
    Underwater Colour Image Quality Evaluation based on M. Yang and A. Sowmya's paper.\n
    UCIQE = c1 * σ_c + c2 * con_l + c3 * μs\n
    where:
     - σ_c is the standard deviation of chroma,
     - con_l is the contrast of luminance (difference between the top 1% and bottom 1% of luminance values),
     - μ_s is the average saturation,
     - c1=0.4680, c2=0.2745, and c3=0.2576
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


def get_histogram_spread(histogram):
    """
    From the article Performance Metrics for Image Contrast by Tripathi et al.
    :param histogram:
    :return:
    """
    cumulative_hist = histogram.cumsum()

    # kiszamoljuk az elso es harmadik kvartilist
    # (az osszes elofordulas osszegenek negyedet es haromnegyedet)
    max_cumulative_value = cumulative_hist.max()
    quartile1_value = 0.25 * max_cumulative_value
    quartile3_value = 0.75 * max_cumulative_value

    # majd megkeressuk, hogy melyik intenzitasok erik el a ket elofordulast
    quartile1_bin = np.argmax(cumulative_hist >= quartile1_value)
    quartile3_bin = np.argmax(cumulative_hist >= quartile3_value)

    quartile_distance = quartile3_bin - quartile1_bin
    histogram_spread = quartile_distance / 255

    return histogram_spread


def get_pix_dis(histogram, pixel_count: int):
    L = histogram.shape[0]

    res = 0
    for i in range(0, L - 2):
        for j in range(i + 1, L - 1):
            res += int(histogram[i]) * int(histogram[j]) * (j - i)

    res /= (pixel_count * (pixel_count - 1))
    return res


def get_dcp(img: np.ndarray):
    img_float = img.astype(np.float64) / 255.0

    dcp = dark_channel(img_float)
    atm_light = estimate_atmospheric_light(img_float, dcp)
    transmission = estimate_transmission(img_float, atm_light)

    return transmission
