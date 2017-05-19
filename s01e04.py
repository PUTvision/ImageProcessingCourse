import cv2
import numpy as np
import random


def nothing(value):
    pass


def ex_0():
    img_original = cv2.imread("_data/s01e04/not_bad.jpg", cv2.IMREAD_COLOR)
    img_original = cv2.resize(img_original, None, fx=0.25, fy=0.25)
    img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i in range(100):
        colours.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    cv2.namedWindow("img")
    cv2.createTrackbar("thresh", "img", 52, 255, nothing)

    key = ord("a")
    while key != ord("q"):
        t = cv2.getTrackbarPos('thresh', 'img')

        _, img_thresh = cv2.threshold(img_grayscale, t, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        img_morph = cv2.dilate(img_thresh, kernel, iterations=4)
        img_morph = cv2.erode(img_morph, kernel, iterations=6)

        img_contours, contours, hierarchy = \
            cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        img_with_contours = img_original.copy()

        for i in range(len(contours)):
            if i < len(colours):
                cv2.drawContours(img_with_contours, contours, i, colours[i], 3)

        cv2.imshow("img", img_with_contours)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def ex_1():
    print("Function not implemented yet!")


if __name__ == "__main__":
    ex_0()
    # ex_1()