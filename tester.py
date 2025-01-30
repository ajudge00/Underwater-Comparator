from src.methods.metrics import *


def main(image):
    original = cv2.imread(image, cv2.IMREAD_COLOR)

    cv2.waitKey(0)


if __name__ == '__main__':
    main('images/diver01.jpg')
