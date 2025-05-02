import cv2
import numpy as np

from src.logger import print_params
from src.methods.contrast import gamma_correction, clahe_with_lab, histogram_linearization
from src.methods.fusion import apply_fusion
from src.methods.sharpening import normalized_unsharp_masking
from src.methods.weight_maps import get_weight_map, WeightMapMethods, normalize_weight_maps
from src.methods.white_balance import ancuti2018_precomp, CompChannel, apply_white_balance

MOHANSIMON2020_STEPS = ["Original", "After Pre-comp. (R)", "After Pre-comp. (R+B)",
                        "After White Balancing", "After Gamma Correction", "After Sharpening",
                        "After CLAHE (Gamma)", "After Hist. Linearization (Sharpened)",
                        "Laplacian Contrast Weight (CLAHE)", "Laplacian Contrast Weight (HistLin)",
                        "Saturation Weight (CLAHE)", "Saturation Weight (HistLin)",
                        "Saliency Weight (CLAHE)", "Saliency Weight (HistLin)",
                        "Normalized Weight Map (CLAHE)", "Normalized Weight Map (HistLin)",
                        "Fused Result"]


def MohanSimon2020(img: np.ndarray,
                   do_precomp: bool = True,
                   precomp_red: float = 1.0,
                   precomp_blue: float = 0.0,
                   gamma: float = 2.0,
                   wb_method: int = 0,
                   clahe_clip_limit: float = 3.0,
                   clahe_tile_grid_size: int = 8,
                   fusion_method: int = 1,
                   msf_levels: int = 3) -> [np.ndarray]:
    """
    My implementation of Mohan and Simon's "Underwater Image Enhancement
    based on Histogram Manipulation and Multiscale Fusion".
    Works well enough.
    :param img: A uint8 [0, 255] bgr image.
    :param do_precomp: Whether to apply the precompensation before the white balance step.
    :param precomp_red: Alpha value for the red precompensation.
    :param precomp_blue: Alpha value for the blue precompensation.
    :param gamma: Self-explanatory.
    :param wb_method: Index of white balance method to use.
    :param clahe_clip_limit:
    :param clahe_tile_grid_size:
    :param fusion_method: Index of fusion method to use.
    :param msf_levels: Number of levels in the pyramids for Multiscale Fusion.
    :return: All the steps and the result.
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and np.max(img) > 1
    print_params()

    # WHITE BALANCE
    original = img.copy()
    white_balanced = img.copy()
    precomped_r = None
    precomped_rb = None

    if do_precomp:
        precomped_r = ancuti2018_precomp(CompChannel.COMP_RED, img, precomp_red)
        precomped_rb = ancuti2018_precomp(CompChannel.COMP_BLUE, precomped_r, precomp_blue)
        white_balanced = precomped_rb.copy()

    white_balanced = apply_white_balance(wb_method, white_balanced)

    # GAMMA CORRECTION
    input1 = gamma_correction(white_balanced, gamma)

    # SHARPENING
    input2 = normalized_unsharp_masking(white_balanced)

    # CLAHE and HISTOGRAM LINEARIZATION
    clahed = clahe_with_lab(input1, clahe_clip_limit, (clahe_tile_grid_size, clahe_tile_grid_size))
    hist_lind, best_r = histogram_linearization(input2)

    # WEIGHT MAPS
    clahed_lcw = get_weight_map(clahed, WeightMapMethods.LAPLACIAN)
    hist_lind_lcw = get_weight_map(hist_lind, WeightMapMethods.LAPLACIAN)
    clahed_satw = get_weight_map(clahed, WeightMapMethods.SATURATION)
    hist_lind_satw = get_weight_map(hist_lind, WeightMapMethods.SATURATION)
    clahed_salw = get_weight_map(clahed, WeightMapMethods.SALIENCY)
    hist_lind_salw = get_weight_map(hist_lind, WeightMapMethods.SALIENCY)

    # NORMALIZE WEIGHT MAPS
    clahed_normw, hist_lind_normw = normalize_weight_maps(
        clahed_lcw, clahed_satw, clahed_salw,
        hist_lind_lcw, hist_lind_satw, hist_lind_salw,
    )

    # FUSION
    fusioned = apply_fusion(fusion_method, clahed, hist_lind, clahed_normw, hist_lind_normw, levels=msf_levels)

    # text for best r on histlin
    text = f"Best r value: {best_r}"
    cv2.putText(hist_lind, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return [original, precomped_r, precomped_rb,
            white_balanced, input1, input2,
            clahed, hist_lind,
            clahed_lcw, hist_lind_lcw,
            clahed_satw, hist_lind_satw,
            clahed_salw, hist_lind_salw,
            clahed_normw, hist_lind_normw,
            fusioned]
