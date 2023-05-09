import time

import cv2
import numpy as np


def ex_1() -> None:
    def print_trackbar_value(v: int):
        print(f'trackbar value: {v}')

    # create a black image, a window
    img = np.zeros((300, 512, 3), dtype=np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 20, 255, print_trackbar_value)
    cv2.createTrackbar('G', 'image', 0, 255, print_trackbar_value)
    cv2.createTrackbar('B', 'image', 0, 255, print_trackbar_value)

    # create switch for ON/OFF functionality
    switch_trackbar_name = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch_trackbar_name, 'image', 1, 1, print_trackbar_value)

    while True:
        cv2.imshow('image', img)

        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch_trackbar_name, 'image')

        if s == 0:
            # assign zeros to all pixels
            img[:] = 0
        else:
            # assign the same BGR color to all pixels
            img[:] = [b, g, r]

    # closes all windows (usually optional as the script ends anyway)
    cv2.destroyAllWindows()


def ex_image_addition():
    img_from_file = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    img_add_cv = cv2.add(img_from_file, 40)
    img_add_float = img_from_file.astype(np.float32) + 40
    img_add = img_from_file + 40

    cv2.imshow('img_from_file', img_from_file)
    cv2.imshow('img_add_cv', img_add_cv)
    cv2.imshow('img_add', img_add)
    cv2.imshow('img_add_float', img_add_float/(255+40))
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def empty_callback(value):
    pass


def ex_3_1():
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

    load_image_and_filer('../_data/s01e03/lenna_noise.bmp')
    load_image_and_filer('../_data/s01e03/lenna_salt_and_pepper.bmp')

def ex_3_3():
    img: np.ndarray = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_COLOR)

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
                # tmp = np.full((1, 3), 0, dtype=np.float64)
                tmp = 0
                for k in range(-kernel_size, 1+kernel_size):
                    for l in range(-kernel_size, 1+kernel_size):
                        tmp += img[i + k, j + l]
                        # tmp += img[i + k, j + l]
                # cpy[i, j] = np.round(tmp / (window_size*window_size))
                cpy[i, j] = np.round(tmp / (window_size*window_size))

    print("for loop --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    for i in range(0, 1000):
        img_blurred = cv2.blur(img, (3, 3))
    print("cv2.blur --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    kernel = np.full((3, 3), 1/9, dtype=np.float32)
    for i in range(0, 1000):
        img_filter2d = cv2.filter2D(img, -1, kernel=kernel)
    print("cv2.filter2d --- %s seconds ---" % (time.time() - start_time))

    print(f'blur - filter2d same: {np.array_equal(img_blurred, img_filter2d)}')
    print(f'blur - for loop same: {np.array_equal(img_blurred[1:-1, 1:-1], cpy[1:-1, 1:-1])}')
    print(f'for loop - filter2d same: {np.array_equal(cpy[1:-1, 1:-1], img_filter2d[1:-1, 1:-1])}')

    key = ord('a')
    while key != ord('q'):

        cv2.imshow('img', img)
        cv2.imshow('cpy', cpy)
        cv2.imshow('img_blurred', img_blurred)
        cv2.imshow('img_filter2d', img_filter2d)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ex_1()
    # ex_image_addition()
    ex_3_3()
