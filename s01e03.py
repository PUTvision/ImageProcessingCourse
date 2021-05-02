import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def empty_callback(value):
    pass


def ex_1():
    def load_image_and_filer(image_filename):
        img = cv2.imread(image_filename, cv2.IMREAD_COLOR)

        cv2.namedWindow('img')
        cv2.createTrackbar('kernel_size', 'img', 0, 50, empty_callback)

        key = ord('a')
        while key != ord('q'):
            kernel_size = 1 + 2*cv2.getTrackbarPos('kernel_size', 'img')  # 1, 3, 5, 7, 9, 11

            img_after_blur = cv2.blur(img, (kernel_size, kernel_size))
            img_after_gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            img_after_median = cv2.medianBlur(img, kernel_size)

            cv2.imshow('img', img)
            cv2.imshow('img_after_blur', img_after_blur)
            cv2.imshow('img_after_gaussian', img_after_gaussian)
            cv2.imshow('img_after_median', img_after_median)
            key = cv2.waitKey(10)

        cv2.destroyAllWindows()

    load_image_and_filer('./_data/s01e03/lenna_noise.bmp')
    load_image_and_filer('./_data/s01e03/lenna_salt_and_pepper.bmp')


def ex_2():
    # element =
    # [ 0 1 0
    #   1 0 1
    #   0 1 0 ]
    img = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('img')
    cv2.createTrackbar('threshold', 'img', 0, 255, empty_callback)
    cv2.createTrackbar('kernel_size', 'img', 1, 10, empty_callback)
    cv2.createTrackbar('0: erosion, 1: dilation', 'img', 0, 1, empty_callback)

    key = ord('a')
    while key != ord('q'):
        threshold = cv2.getTrackbarPos('threshold', 'img')
        kernel_size = 1 + 2 * cv2.getTrackbarPos('kernel_size', 'img')
        operation_type = cv2.getTrackbarPos('0: erosion, 1: dilation', 'img')

        _, img_after_threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation_type == 0:
            img_after_morphological = cv2.erode(img_after_threshold, kernel, iterations=1)
        else:
            img_after_morphological = cv2.dilate(img_after_threshold, kernel, iterations=1)

        cv2.imshow('img', img)
        cv2.imshow('img_after_threshold', img_after_threshold)
        cv2.imshow('img_after_morphological', img_after_morphological)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def ex_3():
    img: np.ndarray = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_COLOR)

    if len(img.shape) == 2:
        print('grayscale')
    elif len(img.shape) == 3:
        print('color')
    else:
        raise ValueError('Too many channels')

    cpy = np.array(img)

    start_time = time.time()

    for i in range(0, 1):
        window_size = 3  # 3x3
        kernel_size = window_size // 2
        for i in range(kernel_size, cpy.shape[0] - kernel_size):
            for j in range(kernel_size, cpy.shape[1] - kernel_size):
                tmp = np.full((1, 3), 0, dtype=np.float64)
                # tmp = 0
                for k in range(-kernel_size, 1+kernel_size):
                    for l in range(-kernel_size, 1+kernel_size):
                        tmp += img[i + k, j + l]
                cpy[i, j] = np.round(tmp / (window_size*window_size))

    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    for i in range(0, 1000):
        img_blurred = cv2.blur(img, (3, 3))
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    kernel = np.full((3, 3), 1/9, dtype=np.float32)
    for i in range(0, 1000):
        img_filter2d = cv2.filter2D(img, -1, kernel=kernel)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(np.array_equal(img_blurred, img_filter2d))
    print(np.array_equal(img_blurred[1:-1, 1:-1], cpy[1:-1, 1:-1]))
    print(np.array_equal(cpy[1:-1, 1:-1], img_filter2d[1:-1, 1:-1]))

    key = ord('a')
    while key != ord('q'):

        cv2.imshow('img', img)
        cv2.imshow('cpy', cpy)
        cv2.imshow('img_blurred', img_blurred)
        cv2.imshow('img_filter2d', img_filter2d)
        # cv2.imshow('img_own_blur', img_after_threshold)
        # cv2.imshow('img_after_morphological', img_after_morphological)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def main():
    # ex_1()
    # ex_2()
    ex_3()


if __name__ == '__main__':
    main()
