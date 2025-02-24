import cv2

from src.methods.metrics import *
from src.mohansimon2020 import MohanSimon2020

def main(image):
    original = cv2.imread(image, cv2.IMREAD_COLOR)
    MohanSimon2020(original)

    cv2.waitKey(0)


if __name__ == '__main__':
    main('images/dcp.jpg')
