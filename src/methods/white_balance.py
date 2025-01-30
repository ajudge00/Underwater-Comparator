import enum

import cv2
import numpy as np


class WhiteBalanceMethod(enum.Enum):
    GRAY_WORLD = 0
    IQBAL_GRAY_WORLD = 1
    WHITE_PATCH = 2
    SIMPLEST_LIMARE = 3


class CompChannel(enum.Enum):
    COMP_RED = 0
    COMP_GREEN = 1
    COMP_BLUE = 2


def apply_white_balance(white_balance_method: int, img: np.ndarray) -> np.ndarray:
    if white_balance_method == WhiteBalanceMethod.GRAY_WORLD.value:
        return gray_world(img)
    elif white_balance_method == WhiteBalanceMethod.IQBAL_GRAY_WORLD.value:
        return iqbal_gray_world(img)
    elif white_balance_method == WhiteBalanceMethod.WHITE_PATCH.value:
        return white_patch(img)
    elif white_balance_method == WhiteBalanceMethod.SIMPLEST_LIMARE.value:
        return simplest_color_balance(img)


def ancuti2018_precomp(channel: CompChannel, img: np.ndarray, alpha=1.0) -> np.ndarray:
    """
    I_rc(x) = I_r(x) + α * (¯I_g − ¯I_r) * (1 − I_r(x)) * I_g(x)\n
    I_bc(x) = I_b(x) + α * (¯I_g − ¯I_b) * (1 − I_b(x)) * I_g(x)

    :param channel: The channel to compensate for (COMP_RED or COMP_BLUE)
    :param img: A uint8, [0, 255] image
    :param alpha: The strength of the compensation
    :return: A uint8, [0, 255] image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img_f32 = img.astype(np.float32) / 255.0

    avg_b = np.mean(img_f32[:, :, 0])
    avg_g = np.mean(img_f32[:, :, 1])
    avg_r = np.mean(img_f32[:, :, 2])

    res = img_f32.copy()

    if channel == CompChannel.COMP_RED:
        res[:, :, 2] = img_f32[:, :, 2] + alpha * (avg_g - avg_r) * (1 - img_f32[:, :, 2]) * img_f32[:, :, 1]
    elif channel == CompChannel.COMP_BLUE:
        res[:, :, 0] = img_f32[:, :, 0] + alpha * (avg_g - avg_b) * (1 - img_f32[:, :, 0]) * img_f32[:, :, 1]

    res = (res * 255).astype(np.uint8)

    return res


def gray_world(img: np.ndarray) -> np.ndarray:
    """
    Balances the image based on the Gray World Assumption.
    Adjusts each channel so that the average intensity of all channels matches a neutral gray.

    :param img: A uint8, [0, 255] BGR image
    :return: A uint8, [0, 255] color-balanced BGR image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    overall_avg = (avg_b + avg_g + avg_r) / 3.0

    alpha_b = overall_avg / avg_b
    alpha_g = overall_avg / avg_g
    alpha_r = overall_avg / avg_r

    res = img.copy().astype(np.float32)

    res[:, :, 0] = cv2.multiply(alpha_b, res[:, :, 0])
    res[:, :, 1] = cv2.multiply(alpha_g, res[:, :, 1])
    res[:, :, 2] = cv2.multiply(alpha_r, res[:, :, 2])

    res = np.clip(res, 0, 255).astype(np.uint8)

    return res


def iqbal_gray_world(img: np.ndarray) -> np.ndarray:
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    alpha = avg_b / avg_r
    beta = avg_b / avg_g

    res = img.copy()

    res[:, :, 2] = cv2.multiply(alpha, res[:, :, 2])
    res[:, :, 1] = cv2.multiply(beta, res[:, :, 1])

    return res


def white_patch(img: np.ndarray, percentile=98) -> np.ndarray:
    """
    White Patch algorithm that aims to correct color casts in an image
    by scaling the color channels so that the brightest pixels in each channel become white
    https://python.plainenglish.io/introduction-to-image-processing-with-python-bb39c83366a4

    :param img: A uint8, [0, 255] image
    :param percentile: Percentile value to consider as channel maximum
    :return: A uint8, [0, 255] image
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1

    img_f32 = img.astype(np.float32) / 255.0
    white_patch_image = (img_f32 / np.percentile(img_f32,
                                                 percentile,
                                                 axis=(0, 1))).clip(0, 1)

    white_patch_image = (white_patch_image * 255).astype(np.uint8)
    return white_patch_image


def simplest_color_balance(img: np.ndarray, s1=1, s2=1) -> np.ndarray:
    """
    The highest values of R, G, B observed in the image must correspond
    to white, and the lowest values to obscurity
    :param img:
    :param s1:
    :param s2:
    :return:
    """
    img_f32 = img.astype(np.float32)

    for channel in range(3):
        flat = img_f32[:, :, channel].flatten()
        flat_sorted = np.sort(flat)

        N = len(flat_sorted)
        Vmin = flat_sorted[int(N * s1 / 100)]
        Vmax = flat_sorted[int(N * (1 - s2 / 100)) - 1]

        img_f32[:, :, channel] = np.clip(img_f32[:, :, channel], Vmin, Vmax)
        img_f32[:, :, channel] = (img_f32[:, :, channel] - Vmin) * 255 / (Vmax - Vmin)

    img_balanced = np.clip(img_f32, 0, 255).astype(np.uint8)
    return img_balanced


def gray_edge(img: np.ndarray) -> np.ndarray:
    pass
