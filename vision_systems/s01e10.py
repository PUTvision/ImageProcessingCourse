import os

import cv2
import numpy as np

# https://docs.google.com/document/d/16-TXEYUkAdvEHEYKYLm71N7pYEFILh6WGOEmxeX79tI/edit?usp=sharing


def empty_callback(_):
    pass


def ex_1():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('background_image')
    cv2.createTrackbar('threshold', 'background_image', 20, 255, empty_callback)

    background_image = np.zeros((200, 1000), dtype=np.uint8)
    current_image = np.zeros((100, 1000), dtype=np.uint8)
    foreground_image = np.zeros((100, 1000), dtype=np.uint8)

    background_image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

    backSub = cv2.createBackgroundSubtractorMOG2()

    key = ord('p')
    while key != ord('q'):
        _, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # if key == ord('a'):
        #     background_image = img
        # elif key == ord('x'):
        current_image = img

        if current_image.shape == background_image.shape:
            foreground_image = cv2.absdiff(current_image, background_image)
            kernel = np.ones((3, 3), np.uint8)
            foreground_image = cv2.erode(foreground_image, kernel, iterations=1)
            foreground_image = cv2.dilate(foreground_image, kernel, iterations=1)

            threshold = cv2.getTrackbarPos('threshold', 'background_image')
            _, foreground_image = cv2.threshold(foreground_image, threshold, 255, cv2.THRESH_BINARY)

            background_image[background_image > current_image] -= 1
            background_image[background_image < current_image] += 1

            fgMask = backSub.apply(current_image)
            bgMask = backSub.getBackgroundImage()

        cv2.imshow('background_image', background_image)
        cv2.imshow('current_image', current_image)
        cv2.imshow('foreground_image', foreground_image)
        cv2.imshow('fgMask', fgMask)
        cv2.imshow('bgMask', bgMask)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ex_1()
