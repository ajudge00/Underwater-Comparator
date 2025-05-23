import enum
import cv2
import numpy as np


class FusionMethod(enum.Enum):
    NAIVE = 0
    MULTI_SCALE = 1


def laplacian_to_imshow_equivalent(laplacian):
    """Convert Laplacian pyramid level to uint8 exactly like imshow would"""
    # 1. Normalize to [-1, 1] range while preserving zero
    normalized = laplacian / (np.maximum(np.abs(laplacian.max()),
                                         np.abs(laplacian.min())) + 1e-6)

    # 2. Shift to [0,1] range (like imshow does internally)
    normalized = (normalized + 1.0) / 2.0

    # 3. Convert to 8-bit (0-255)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def apply_fusion(fusion_method: int,
                 input1: np.ndarray, input2: np.ndarray,
                 weight1: np.ndarray, weight2: np.ndarray,
                 levels: int = 3) -> np.ndarray:
    """
    Fusion of two images and their respective normalized weights.
    :param fusion_method: FusionType.NAIVE or FusionType.MULTI_SCALE
    :param input1: A uint8, [0, 255] image
    :param input2: A uint8, [0, 255] image
    :param weight1: A uint8, [0, 255] normalized weight map
    :param weight2: A uint8, [0, 255] normalized weight map
    :param levels: The number of levels of the pyramids in MULTI_SCALE fusion.
    :return: A uint8, [0, 255] image
    """
    assert input1.dtype == np.uint8 and input1.ndim == 3 and np.max(input1) > 1
    assert input2.dtype == np.uint8 and input2.ndim == 3 and np.max(input2) > 1
    assert weight1.dtype == np.uint8 and weight1.ndim == 2 and np.max(weight1) > 1
    assert weight2.dtype == np.uint8 and weight2.ndim == 2 and np.max(weight2) > 1

    input1 = input1.astype(np.float32) / 255.0
    input2 = input2.astype(np.float32) / 255.0
    weight1 = cv2.cvtColor(weight1, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    weight2 = cv2.cvtColor(weight2, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    if fusion_method == FusionMethod.NAIVE.value:
        return naive_fusion(input1, input2, weight1, weight2)
    elif fusion_method == FusionMethod.MULTI_SCALE.value:
        return multi_scale_fusion(input1, input2, weight1, weight2, levels)


def naive_fusion(input1: np.ndarray, input2: np.ndarray,
                 weight1: np.ndarray, weight2: np.ndarray) -> np.ndarray:
    """
    Naive approach to fusion.\n
    R(x) = Σ[k=1 to K] W̄_k(x) * I_k(x)
    :param input1: A float32, [0.0, 1.0] image
    :param input2: A float32, [0.0, 1.0] image
    :param weight1: A float32, [0.0, 1.0] normalized weight map
    :param weight2: A float32, [0.0, 1.0] normalized weight map
    :return: A uint8, [0, 255] image
    """
    op1 = cv2.multiply(input1, weight1, dtype=cv2.CV_32F)
    op2 = cv2.multiply(input2, weight2, dtype=cv2.CV_32F)

    naive_fused = cv2.add(op1, op2, dtype=cv2.CV_32F)

    return (naive_fused * 255).astype(np.uint8)


def multi_scale_fusion(input1: np.ndarray, input2: np.ndarray,
                       weight1: np.ndarray, weight2: np.ndarray, levels=3) -> np.ndarray:
    """
    Multi-Scale Fusion using Gaussian/Laplacian pyramids.
    Rl(x) = Σ[k=1 to K] Gl{W̄k(x)} * Ll{Ik(x)}
    :param input1: A float32, [0.0, 1.0] image
    :param input2: A float32, [0.0, 1.0] image
    :param weight1: A float32, [0.0, 1.0] normalized weight map
    :param weight2: A float32, [0.0, 1.0] normalized weight map
    :param levels: The number of levels of the pyramids.
    :return: A uint8, [0, 255] image
    """
    weight1_pyr = gaussian_pyramid(weight1, levels)
    weight2_pyr = gaussian_pyramid(weight2, levels)

    input1_pyr = laplacian_pyramid(input1, levels)
    input2_pyr = laplacian_pyramid(input2, levels)

    # print(weight1_pyr[0].dtype, weight1_pyr[0].shape, np.min(weight1_pyr[0]), np.max(weight1_pyr[0]))
    # print(input1_pyr[0].dtype, input1_pyr[0].shape, np.min(input1_pyr[0]), np.max(input1_pyr[0]))

    # for level in range(levels):
    #     cv2.imwrite(f'test/pyr/gaussian_sharp_pyr_{level}.jpg', (weight2_pyr[level] * 255).astype(np.uint8))
        # cv2.imwrite(f'test/pyr/laplacian_pyr_{level}.jpg', laplacian_to_imshow_equivalent(input1_pyr[level]))
        # cv2.imwrite(f'gauss_pyr_l{level}.jpg', (weight1_pyr[level] * 255).astype(np.uint8))
        # cv2.imshow(f'lap_pyr_l{level}', input2_pyr[level])

    # print(np.min(input1_pyr[0]), np.max(input1_pyr[0]))
    # print(np.min(input2_pyr[0]), np.max(input2_pyr[0]))

    fused_pyramid = []
    for l1, l2, w1, w2 in zip(input1_pyr, input2_pyr, weight1_pyr, weight2_pyr):
        fused = w1 * l1 + w2 * l2
        fused_pyramid.append(fused)

    image = fused_pyramid[-1]
    for i in range(len(fused_pyramid) - 2, -1, -1):
        size = (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, fused_pyramid[i])

    return (image * 255).astype(np.uint8)


def gaussian_pyramid(img: np.ndarray, levels: int) -> [np.ndarray]:
    pyr = [img]

    for i in range(1, levels):
        img = cv2.pyrDown(img)
        pyr.append(img)

    return pyr


def laplacian_pyramid(img: np.ndarray, levels: int) -> [np.ndarray]:
    g_pyr = gaussian_pyramid(img, levels)
    l_pyr = []

    for i in range(levels - 1):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        l_pyr.append(g_pyr[i] - cv2.pyrUp(g_pyr[i + 1], dstsize=size))

    l_pyr.append(g_pyr[-1])
    return l_pyr
