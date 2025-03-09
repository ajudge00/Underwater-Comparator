import cv2
from matplotlib import pyplot as plt

from src.methods.contrast import tester_histogram_linearization_auto, histogram_linearization_auto


def main(image):
    original = cv2.imread(image, cv2.IMREAD_COLOR)
    histlin = histogram_linearization_auto(original)

    cv2.imshow('original', original)
    cv2.imshow('histlin', histlin)
    cv2.waitKey(0)


if __name__ == '__main__':
    main('images/lenna_low_contrast.jpg')
