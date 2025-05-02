import cv2
from matplotlib import pyplot as plt

from src.methods.contrast import histogram_linearization


def main(image):
    original = cv2.imread(image, cv2.IMREAD_COLOR)


    cv2.waitKey(0)


if __name__ == '__main__':
    main('images/bell.jpg')
