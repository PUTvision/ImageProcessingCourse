import cv2
import numpy as np


def ex_1():
    img_left = cv2.imread('_data/sw_s01e11/Rocks1/view1.png', cv2.IMREAD_COLOR)
    img_right = cv2.imread('_data/sw_s01e11/Rocks1/view5.png', cv2.IMREAD_COLOR)

    cv2.imshow('img_right', img_right)
    cv2.imshow('img_left', img_left)
    cv2.waitKey(0)

    b_left, g_left, r_left = cv2.split(img_left)
    b_right, g_right, r_right = cv2.split(img_right)

    img_merged = cv2.merge((b_right, g_right, r_left))
    cv2.imshow('img_merged', img_merged)
    cv2.waitKey(0)


def ex_2():
    # GREYSCALE
    img_left = cv2.imread('_data/sw_s01e11/Rocks1/view1.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('_data/sw_s01e11/Rocks1/view5.png', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('img_right', img_right)
    cv2.imshow('img_left', img_left)
    cv2.waitKey(0)

    img_merged = cv2.merge((img_left, img_right, img_right))
    cv2.imshow('img_merged', img_merged)
    cv2.waitKey(0)


def ex_3():
    img_left = cv2.imread('_data/sw_s01e11/Rocks1/view1.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('_data/sw_s01e11/Rocks1/view5.png', cv2.IMREAD_GRAYSCALE)

    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    disparity = stereo.compute(img_left, img_right)

    print(f'dtype: {disparity.dtype}')
    print(f'min: {np.min(disparity)}')
    print(f'max: {np.max(disparity)}')

    # disparity_normalized = cv2.normalize(
    #     disparity.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    # ).astype(np.float32)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print(f'dtype: {disparity_normalized.dtype}')
    print(f'min: {np.min(disparity_normalized)}')
    print(f'max: {np.max(disparity_normalized)}')

    cv2.imshow('img_right', img_right)
    cv2.imshow('img_left', img_left)
    cv2.imshow('disparity_normalized ', disparity_normalized)
    cv2.waitKey(0)

    # optional visualization with matplotlib
    # from matplotlib import pyplot as plt
    # plt.imshow(disparity, 'gray')
    # plt.show()

    img_disparity_left = cv2.imread('_data/sw_s01e11/Rocks1/disp1.png', cv2.IMREAD_GRAYSCALE)
    B = 160
    f = 3740
    Z = B*f / disparity[200, 200]
    print(f'disparity[200, 200]: {disparity[200, 200]}')
    print(f'Z: {Z}')
    Z_normalized = B*f / disparity_normalized[200, 200]
    print(f'disparity_normalized[200, 200]: {disparity_normalized[200, 200]}')
    print(f'Z_normalized: {Z_normalized}')
    Z_disp1 = B*f / (img_disparity_left[200, 200]+274)
    print(f'img_disparity_left[200, 200]: {img_disparity_left[200, 200]}')
    print(f'Z_disp1: {Z_disp1}')


if __name__ == '__main__':
    ex_1()
    ex_2()
    ex_3()
