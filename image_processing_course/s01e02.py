import cv2
import numpy as np


def ex_0():
    def print_trackbar_value(x):
        print(f'Trackbar reporting for duty with value: {x}')

    # Create a black image, a window
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('img')

    # create trackbars for color change
    cv2.createTrackbar('R', 'img', 0, 255, print_trackbar_value)
    cv2.createTrackbar('G', 'img', 0, 255, print_trackbar_value)
    cv2.createTrackbar('B', 'img', 0, 255, print_trackbar_value)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'img', 0, 1, print_trackbar_value)

    key = ord('a')
    while key != ord('q'):
        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'img')
        g = cv2.getTrackbarPos('G', 'img')
        b = cv2.getTrackbarPos('B', 'img')
        s = cv2.getTrackbarPos(switch, 'img')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

        cv2.imshow('img', img)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def do_nothing(x):
    pass


def ex_1():
    def convert_value_to_threshold_mode(value: int):
        if value == 0:
            return cv2.THRESH_BINARY
        elif value == 1:
            return cv2.THRESH_BINARY_INV
        elif value == 2:
            return cv2.THRESH_TRUNC
        elif value == 3:
            return cv2.THRESH_TOZERO
        elif value == 4:
            return cv2.THRESH_TOZERO_INV
        else:
            raise ValueError('Wrong value provided. It should be in 0-4 range for threshold')

    img_from_file = cv2.imread('./../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('img')
    cv2.createTrackbar('threshold', 'img', 0, 255, do_nothing)
    cv2.createTrackbar('mode', 'img', 0, 4, do_nothing)

    key = ord('a')

    while key != ord('q'):
        t = cv2.getTrackbarPos('threshold', 'img')
        mode = convert_value_to_threshold_mode(cv2.getTrackbarPos('mode', 'img'))

        _, img_thresholded = cv2.threshold(img_from_file, t, 255, mode)

        cv2.imshow('img', img_thresholded)
        key = cv2.waitKey(50)
    cv2.destroyAllWindows()


def ex_2():
    def convert_value_to_resize_mode(value: int):
        if value == 0:
            return cv2.INTER_LINEAR
        elif value == 1:
            return cv2.INTER_NEAREST
        elif value == 2:
            return cv2.INTER_AREA
        elif value == 3:
            return cv2.INTER_LANCZOS4
        else:
            raise ValueError('Wrong value provided. It should be in 0-3 range for resize')

    img_from_file = cv2.imread('./../_data/s01e02/qr.jpg', cv2.IMREAD_GRAYSCALE)
    fx = 2.75
    fy = 2.75

    cv2.namedWindow('img')
    cv2.createTrackbar('mode', 'img', 0, 3, do_nothing)

    key = ord('a')

    while key != ord('q'):
        mode = convert_value_to_resize_mode(cv2.getTrackbarPos('mode', 'img'))

        img_resized = cv2.resize(img_from_file, (0, 0), fx=fx, fy=fy, interpolation=mode)

        cv2.imshow('img', img_resized)
        key = cv2.waitKey(50)
    cv2.destroyAllWindows()

    cv2.imshow('img_from_file', img_from_file)

    img_scaled_1 = cv2.resize(img_from_file, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    img_scaled_2 = cv2.resize(img_from_file, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    img_scaled_3 = cv2.resize(img_from_file, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    img_scaled_4 = cv2.resize(img_from_file, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('INTER_LINEAR', img_scaled_1)
    cv2.imshow('INTER_NEAREST', img_scaled_2)
    cv2.imshow('INTER_AREA', img_scaled_3)
    cv2.imshow('INTER_LANCZOS4', img_scaled_4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_3():
    img_from_file = cv2.imread('./../_data/no_idea.jpg', cv2.IMREAD_COLOR)
    img_logo = cv2.imread('./../_data/s01e02/LOGO_PUT_VISION_LAB_MAIN.png', cv2.IMREAD_COLOR)
    img_logo = cv2.resize(img_logo, (img_from_file.shape[1], img_from_file.shape[0]))

    cv2.namedWindow('img')
    cv2.createTrackbar('alpha', 'img', 0, 100, do_nothing)

    key = ord('a')

    while key != ord('q'):
        alpha = cv2.getTrackbarPos('alpha', 'img') / 100.0

        img_blended = cv2.addWeighted(img_from_file, alpha, img_logo, 1 - alpha, 0)

        cv2.imshow('img', img_blended)
        key = cv2.waitKey(50)
    cv2.destroyAllWindows()


def ex_homework():
    img_color = cv2.imread("./../_data/no_idea.jpg", cv2.IMREAD_COLOR)
    img_negative_rgb = 255 - img_color
    img_negative_grayscale = 255 - cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_negative_rgb', img_negative_rgb)
    cv2.imshow('img_negative_grayscale', img_negative_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ex_0()
    ex_1()
    ex_2()
    ex_3()
    ex_homework()
