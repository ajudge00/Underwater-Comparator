import enum

import cv2
import numpy as np

from src.logger import print_params
from src.methods.dehazing import dark_channel, estimate_atmospheric_light, estimate_transmission, recover_radiance, \
    he_TransmissionEstimate, he_TransmissionRefine
from src.methods.white_balance import apply_white_balance

YANG2011_STEPS = ["Original", "Dark Channel Prior", "Atmospheric Light", "Transmission Map",
                  "After Dehazing", "After Colour Correction"]


class TransmissionRefineType(enum.Enum):
    MEDIAN_FILTER = 0
    GUIDED_FILTER = 1


def Yang2011(img: np.ndarray,
             dcp_patch_size: int = 15,
             atm_light_use_fixed_pixels: bool = False,
             atm_light_fixed_num_pixels: int = 1000,
             atm_light_top_percent: float = 0.1,
             atm_light_use_max: bool = False,
             transmission_refine_type: int = 0,
             median_ksize: int = 5,
             wb_method: int = 1) -> [np.ndarray]:
    """
    My implementation of Yang et al.'s "Low Complexity Underwater Image Enhancement Based on Dark Channel Prior".
    :param img: A uint8 [0, 255] bgr image.
    :param dcp_patch_size: Kernel size for the erosion in DCP
    :param atm_light_use_fixed_pixels: If True, use fixed pixels instead of percentage for atmospheric light.
    :param atm_light_fixed_num_pixels: Number of pixels for atmospheric light.
    :param atm_light_top_percent: Percentage of pixels for atmospheric light.
    :param atm_light_use_max: If True, use maximum pixels for atmospheric light, otherwise uses mean.
    :param transmission_refine_type: Index of transmission refinement type (median or guided filter).
    :param median_ksize: Kernel size for median filter.
    :param wb_method: Index of the white balance method.
    :return: All the steps and the result.
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1
    print_params()

    img_float = img.astype(np.float64) / 255.0

    dcp = dark_channel(img_float, dcp_patch_size)
    atm_light = estimate_atmospheric_light(img_float, dcp, atm_light_use_max, atm_light_use_fixed_pixels,
                                           atm_light_fixed_num_pixels, atm_light_top_percent)

    if transmission_refine_type == TransmissionRefineType.MEDIAN_FILTER.value:
        transmission = estimate_transmission(img_float, atm_light, median_ksize)
        transmission = np.maximum(transmission, 0.1)
    elif transmission_refine_type == TransmissionRefineType.GUIDED_FILTER.value:
        transmission = he_TransmissionEstimate(img_float, atm_light, dcp_patch_size)
        transmission = he_TransmissionRefine(img, transmission)
        transmission = np.clip(transmission, 0, 1)

    dehazed = recover_radiance(img_float, transmission, atm_light)
    colour_corr = apply_white_balance(wb_method, (dehazed * 255).astype(np.uint8))

    # Make the single pixel atm_light into a full image.
    atm_light_display = np.tile(atm_light, (colour_corr.shape[0], colour_corr.shape[1], 1))
    text = f"BGR: {atm_light[0] * 255}"
    text_color = 1 - atm_light[0]
    cv2.putText(atm_light_display, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return [img, dcp, atm_light_display, transmission, dehazed, colour_corr]
