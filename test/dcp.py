from datetime import datetime

import cv2
import numpy as np
from src.methods.dehazing import *


def my_dcp(img_float, save):
    dcp = dark_channel(img_float, PATCH_SIZE)
    cv2.imshow("dcp", dcp)

    atm_light, location = estimate_atmospheric_light(
        img_float, dcp, USE_FIXED_PIXELS, FIXED_PIXELS_CNT, ATM_LIGHT_TOP_PERCENT, True
    )
    print(atm_light, location)

    transmission = estimate_transmission(img_float, atm_light, MEDIAN_KSIZE)
    transmission = np.maximum(transmission, 0.1)
    cv2.imshow("transmission", transmission)

    dehaze = recover_radiance(img_float, transmission, atm_light)
    cv2.imshow("dehaze", dehaze)

    if save:
        now = datetime.now()
        now_format = now.strftime("%Y_%m_%d_%H_%M_%S")
        # print("saving image")
        # print(now_format)
        cv2.imwrite(f"transmission_{now_format}.jpg", (transmission * 255).astype(np.uint8))
        cv2.imwrite(f"dehaze_{now_format}.jpg", (dehaze * 255).astype(np.uint8))


def he_et_al_dcp(img_float):
    dcp = he_DarkChannel(img_float, PATCH_SIZE)
    cv2.imshow("he_dcp", dcp)

    atm_light = he_AtmLight(img_float, dcp)
    print("he:", atm_light)

    transmission = he_TransmissionEstimate(img_float, atm_light, PATCH_SIZE)
    transmission = he_TransmissionRefine((img_float * 255).astype(np.uint8), transmission)
    cv2.imshow("he_transmission", transmission)

    dehaze = he_Recover(img_float, transmission, atm_light, 0.1)
    cv2.imshow("he_dehaze", dehaze)


PATCH_SIZE = 9
USE_FIXED_PIXELS = False
FIXED_PIXELS_CNT = 1000
ATM_LIGHT_TOP_PERCENT = 0.1
ATM_LIGHT_USE_MAX = True
TRANSMISSION_REFINE_TYPE = 0
MEDIAN_KSIZE = 5

original = cv2.imread('../images/diver.jpg', cv2.IMREAD_COLOR)
original_float = original.astype(np.float64) / 255.0
cv2.imshow("original", original_float)

my_dcp(original_float, True)
# he_et_al_dcp(original_float)

cv2.waitKey(0)
cv2.destroyAllWindows()
