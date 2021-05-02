import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def empty_callback(value):
    pass


def ex_1():
    print('ex1')

    img: np.ndarray = cv2.imread('../_data/sw_s01e06/not_bad.jpg', cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img, None, fx=0.25, fy=0.25)
    img_grayscale = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('img')
    cv2.createTrackbar('threshold', 'img', 0, 255, empty_callback)
    kernel_size = 3

    key = ord('a')
    while key != ord('q'):
        threshold = cv2.getTrackbarPos('threshold', 'img')
        _, img_after_threshold = cv2.threshold(img_grayscale, threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # img_after_morphological = cv2.erode(img_after_threshold, kernel, iterations=1)
        img_after_morphological: np.ndarray = cv2.dilate(img_after_threshold, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(img_after_morphological, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = img_resize.copy()

        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 255)
        ]
        for cnt, color in zip(contours, colors):
            moments = cv2.moments(cnt)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            cv2.circle(img_with_contours, (cx, cy), 2, color)
            cv2.drawContours(img_with_contours, [cnt], 0, color)

        cv2.imshow('img_resize', img_resize)
        cv2.imshow('img_after_threshold', img_after_threshold)
        cv2.imshow('img_after_morphological', img_after_morphological)
        cv2.imshow('img_with_contours', img_with_contours)
        key = cv2.waitKey(50)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def ex_2():
    haystack: np.ndarray = cv2.resize(cv2.imread('../_data/sw_s01e06/haystack.png'), None, fx=0.5, fy=0.5)
    needle: np.ndarray = cv2.resize(cv2.imread('../_data/sw_s01e06/needle.png'), None, fx=0.5, fy=0.5)

    cv2.namedWindow('img')
    cv2.createTrackbar('method', 'img', 0, 5, empty_callback)

    haystack_grayscale = cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY)
    needle_grayscale = cv2.cvtColor(needle, cv2.COLOR_BGR2GRAY)
    w, h = needle_grayscale.shape[::-1]

    key = ord('a')
    while key != ord('q'):
        method = cv2.getTrackbarPos('method', 'img')
        print(method)
        res = cv2.matchTemplate(haystack_grayscale, needle_grayscale, method)
        print(f'max: {np.max(res)}, min: {np.min(res)}')
        res = (res - np.min(res)) / (np.max(res) - np.min(res))
        print(f'max: {np.max(res)}, min: {np.min(res)}')
        loc = np.where(res >= 0.95)
        img_with_template = haystack.copy()
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_with_template, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        cv2.imshow('res', res)
        cv2.imshow('haystack', haystack)
        cv2.imshow('needle', needle)
        cv2.imshow('img_with_template', img_with_template)
        cv2.waitKey(0)


def main():
    # ex_1()
    ex_2()


if __name__ == '__main__':
    main()
