import numpy as np

from src.logger import print_params
from src.methods.contrast import gamma_correction
from src.methods.fusion import apply_fusion
from src.methods.sharpening import normalized_unsharp_masking
from src.methods.weight_maps import get_weight_map, WeightMapMethods, normalize_weight_maps
from src.methods.white_balance import ancuti2018_precomp, CompChannel, apply_white_balance

ANCUTI2018_STEPS = ["Original", "After Pre-comp. (R)", "After Pre-comp. (R+B)",
                    "After White Balancing", "After Gamma Correction", "After Sharpening",
                    "Laplacian Contrast Weight (Gamma)", "Laplacian Contrast Weight (Sharpened)",
                    "Saturation Weight (Gamma)", "Saturation Weight (Sharpened)",
                    "Saliency Weight (Gamma)", "Saliency Weight (Sharpened)",
                    "Normalized Weight Map (Gamma)", "Normalized Weight Map (Sharpened)",
                    "Fused Result"]


def Ancuti2018(img: np.ndarray,
               do_precomp: bool = True,
               precomp_red: float = 1.0,
               precomp_blue: float = 0.0,
               gamma: float = 2.0,
               wb_method: int = 0,
               fusion_method: int = 1,
               msf_levels: int = 3) -> [np.ndarray]:
    """
    My implementation of Ancuti et al.'s "Color Balance and Fusion for Underwater Image Enhancement".
    Doesn't work perfectly, the sharpening looks nothing like in the paper, even though it should do what's written.
    The saliency looks like the C++ impl., but not like in the paper.
    :param img: A uint8 [0, 255] bgr image.
    :param do_precomp: Whether to apply the precompensation before the white balance step.
    :param precomp_red: Alpha value for the red precompensation.
    :param precomp_blue: Alpha value for the blue precompensation.
    :param gamma: Self-explanatory.
    :param wb_method: Index of white balance method to use.
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

    # WEIGHT MAPS
    input1_lcw = get_weight_map(input1, WeightMapMethods.LAPLACIAN)
    input2_lcw = get_weight_map(input2, WeightMapMethods.LAPLACIAN)
    input1_satw = get_weight_map(input1, WeightMapMethods.SATURATION)
    input2_satw = get_weight_map(input2, WeightMapMethods.SATURATION)
    input1_salw = get_weight_map(input1, WeightMapMethods.SALIENCY)
    input2_salw = get_weight_map(input2, WeightMapMethods.SALIENCY)

    # NORMALIZE WEIGHT MAPS
    input1_normw, input2_normw = normalize_weight_maps(
        input1_lcw, input1_satw, input1_salw,
        input2_lcw, input2_satw, input2_salw,
    )

    # FUSION
    fusioned = apply_fusion(fusion_method, input1, input2, input1_normw, input2_normw, levels=msf_levels)

    return [original, precomped_r, precomped_rb,
            white_balanced, input1, input2,
            input1_lcw, input2_lcw,
            input1_satw, input2_satw,
            input1_salw, input2_salw,
            input1_normw, input2_normw,
            fusioned]
