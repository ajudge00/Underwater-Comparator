import cv2
import numpy as np


def dark_channel(img: np.ndarray, patch_size: int = 9):
    """
    Dark Channel Prior calculation based on He et al.
    :param img: A float32 or float64 [0.0, 1.0] bgr image.
    :param patch_size: Kernel size for the erosion.
    :return: Dark channel of the image
    """
    assert (img.dtype == np.float64 or img.dtype == np.float32) and img.ndim == 3 and np.max(img) <= 1.0

    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel_ = cv2.erode(min_channel, kernel)

    return dark_channel_


def estimate_atmospheric_light(
        img: np.ndarray,
        dcp: np.ndarray,
        use_max: bool = False,
        use_fixed_pixels: bool = False,
        fixed_num_pixels: int = 1000,
        top_percent: float = 0.1) -> [np.ndarray]:
    """
    We take the indices of the [top_percent]/top [fixed_num_pixels] of the dark channel
    and choose the brightest/mean of the original image's corresponding pixels.
    (brightest means highest sum(r, g, b) value)

    :param img: The original image (3-dim float64 0.0-1.0)
    :param dcp: The dark channel of the original image (1-dim float64 0.0-1.0)
    :param use_max: If True, use the point of maximum value for atm. light. Else, the mean (a la He et al.).
    :param use_fixed_pixels: If True, use fixed pixels instead of percentage.
    :param fixed_num_pixels: Number of fixed pixels to use.
    :param top_percent: What top percentage of the dark channel to use.
    :return: The 3-dim pixel representing the atmospheric light
    """
    assert img.dtype == np.float64 and img.ndim == 3 and np.max(img) <= 1.0
    assert dcp.dtype == np.float64 and dcp.ndim == 2 and np.max(dcp) <= 1.0

    # We flatten the dark channel to a row of 1-dim values,
    # and the original image to a row of 3-dim vectors.
    flat_dark = dcp.ravel()
    flat_image = img.reshape((-1, 3))

    # We sort-desc the dark channel by brightness
    # and take the indices of the first [fixed_num_pixels]/[top_percent] of pixels.
    sorted_indices = np.argsort(flat_dark)[::-1]

    if use_fixed_pixels:
        num_brightest = min(fixed_num_pixels, len(flat_dark))
    else:
        num_brightest = int(len(flat_dark) * top_percent)

    brightest_indices = sorted_indices[:num_brightest]
    brightest_pixels = flat_image[brightest_indices]

    if use_max:
        # # We choose the pixel that has the highest sum(r, g, b) value in the original image
        max_idx = np.argmax(np.sum(brightest_pixels, axis=1))
        atmospheric_light = brightest_pixels[max_idx]
    else:
        atmospheric_light = np.mean(brightest_pixels, axis=0)

    return np.array([atmospheric_light])


def estimate_transmission(img: np.ndarray, atm_light: np.ndarray, median_ksize: int = 5, omega: float = 0.95):
    """

    :param img:
    :param atm_light:
    :param median_ksize:
    :param omega:
    :return:
    """
    assert img.dtype == np.float64 and img.ndim == 3 and np.max(img) <= 1.0
    assert atm_light.dtype == np.float64 and atm_light.ndim == 2 and np.max(atm_light) <= 1.0

    img_normalized = np.empty(img.shape, np.float64)

    for i in range(3):
        img_normalized[:, :, i] = img[:, :, i].astype(np.float64) / atm_light[0, i]

    # img_normalized = np.clip(img_normalized, 0.0, 1.0)
    img_normalized = cv2.normalize(img_normalized, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    transmission = np.ones_like(img_normalized[:, :, 0], dtype=np.float64)

    for c in range(3):
        img_norm_c = (img_normalized[:, :, c] * 255.0).astype(np.uint8)
        transmission_c = cv2.medianBlur(img_norm_c, median_ksize)
        transmission_c = transmission_c.astype(np.float64) / 255.0
        transmission = np.minimum(transmission, transmission_c) if c > 0 else transmission_c

    return 1 - omega * transmission


def recover_radiance(img: np.ndarray,
                     transmission: np.ndarray,
                     atm_light: np.ndarray):
    """
    Recovers the dehazed image from the original, its transmission map and its atmospheric light value.
    :param img:
    :param transmission:
    :param atm_light:
    :return:
    """
    assert img.dtype == np.float64 and img.ndim == 3 and np.max(img) <= 1.0
    assert transmission.dtype == np.float64 and transmission.ndim == 2 and np.max(transmission) <= 1.0
    assert atm_light.dtype == np.float64 and atm_light.ndim == 2 and np.max(atm_light) <= 1.0

    res = np.empty(img.shape, img.dtype)

    for i in range(3):
        res[:, :, i] = (img[:, :, i] - atm_light[0, i]) / np.maximum(transmission, 0.1) + atm_light[0, i]

    return np.clip(res, 0, 1)


def he_TransmissionEstimate(im, A, sz):
    """
    He et al.'s transmission estimation function.
    :param im:
    :param A:
    :param sz:
    :return:
    """
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * he_DarkChannel(im3, sz)
    return transmission


def he_Guidedfilter(im, p, r, eps):
    """
    He et al.'s guided filtering function.
    :param im:
    :param p:
    :param r:
    :param eps:
    :return:
    """
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def he_TransmissionRefine(im, et):
    """
    He et al.'s function for refining the transmission map.
    :param im:
    :param et:
    :return:
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = he_Guidedfilter(gray, et, r, eps)

    return t


def he_DarkChannel(im, sz):
    """
    He et al.'s Dark Channel Prior function.
    :param im:
    :param sz:
    :return:
    """
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark
