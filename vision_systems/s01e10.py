# solution for very slow application start when opening camera
# more details: https://github.com/opencv/opencv/issues/17687
import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2
import numpy as np

import time


def ex_1():
    def empty(_):
        pass

    cv2.namedWindow('current_frame')
    cv2.namedWindow('background')
    cv2.namedWindow('foreground')

    cv2.createTrackbar('threshold', 'current_frame', 20, 255, empty)

    cap = cv2.VideoCapture(0)
    # alternative solution to the one with setting the env variable is to use DSHOW,
    # however, it might reduce the achievable fps
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    img_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    img_current = np.copy(img_gray)
    img_background = np.copy(img_gray)
    img_foreground = np.copy(img_gray)

    backSub = cv2.createBackgroundSubtractorMOG2()
    # backSub = cv2.createBackgroundSubtractorKNN()

    key = ord(' ')
    while key != ord('q'):
        _, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgMask = backSub.apply(frame)

        # background update
        # if key == ord('a'):
        # img_background = np.copy(img_current)
        img_background[img_background < img_current] += 1
        img_background[img_background > img_current] -= 1

        # elif key == ord('x'):
        img_current = np.copy(img_gray)

        img_diff = cv2.absdiff(img_background, img_current)
        kernel = np.ones((5, 5), np.uint8)
        img_closed = cv2.morphologyEx(img_diff, cv2.MORPH_OPEN, kernel)

        t = cv2.getTrackbarPos('threshold', 'current_frame')
        _, img_thresholded = cv2.threshold(img_closed, t, 255, cv2.THRESH_BINARY)

        cv2.imshow('current_frame', img_current)
        cv2.imshow('background', img_background)
        cv2.imshow('foreground', img_thresholded)
        cv2.imshow('fgMask', fgMask)

        key = cv2.waitKey(20)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Hello lab 10!')
    ex_1()
