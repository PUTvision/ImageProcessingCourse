import cv2
import numpy as np

import time


def ex_1():
    def empty(_):
        pass

    cv2.namedWindow('current_image')
    cv2.namedWindow('background_image')
    cv2.namedWindow('foreground_image')

    cv2.createTrackbar('threshold', 'current_image', 0, 255, empty)

    cap = cv2.VideoCapture(0)

    # current_image = np.zeros((500, 500), np.uint8)
    # background_image = np.zeros((100, 100), np.uint8)
    # foreground_image = np.zeros((100, 100), np.uint8)

    img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

    current_image = np.copy(img)
    background_image = np.copy(img)
    foreground_image = np.copy(img)

    backSub = cv2.createBackgroundSubtractorMOG2()

    key = ord(' ')
    while key != ord('q'):
        _, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if key == ord('a'):
            background_image = np.copy(img_gray)
        if key == ord('x'):
            current_image = np.copy(img_gray)

        if current_image.shape == background_image.shape:
            img_diff = cv2.absdiff(current_image, background_image)

            kernel = np.ones((5, 5), np.uint8)
            img_closed = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)

            threshold = cv2.getTrackbarPos('threshold', 'current_image')
            _, img_thresholded = cv2.threshold(img_closed, threshold, 255, cv2.THRESH_BINARY)

            foreground_image = img_thresholded

            start = time.time()
            background_image = np.where(background_image > current_image, background_image + 1, background_image)
            background_image = np.where(background_image < current_image, background_image - 1, background_image)
            end = time.time()
            print(f'np.where: {end - start}')

            start = time.time()
            background_image[background_image > current_image] += 1
            background_image[background_image < current_image] -= 1
            end = time.time()
            print(f'[]: {end - start}')


        fgMask = backSub.apply(frame)

        cv2.imshow('current_image', current_image)
        cv2.imshow('foreground_image', foreground_image)
        cv2.imshow('background_image', background_image)
        cv2.imshow('fgMask', fgMask)

        key = cv2.waitKey(30)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Hello lab 10!')
    ex_1()
