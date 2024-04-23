import cv2
import numpy as np


def _gradient_operation(
        img: np.ndarray,
        kernel_x,
        kernel_y: np.ndarray,
        divider: int
):
    pass


def ex_1():
    img = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    # 255 0 0
    # 255 0 0
    # 128 0 0
    # 638
    # -> 255

    # 0 0 128
    # 0 0 64
    # 0 0 255
    # -447
    # -> 0

    # Prewitta
    kernel_prewitt_x = np.array(
        [[1, 0, -1],
         [1, 0, -1],
         [1, 0, -1]],
        np.int8
    )
    kernel_prewitt_y = np.array(
        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],
        np.int8
    )

    # [[  1.   7.   5.  -2.  -4.]
    #  [  2.  11.   6.  -5.  -6.]
    #  [  9.  13.   0. -10.  -7.]
    #  [ 12.   7.  -1. -11. -10.]
    #  [ 10.   2.  -2.  -8.  -8.]]

    # [[ 1  7  5  0  0]
    #  [ 2 11  6  0  0]
    #  [ 9 13  0  0  0]
    #  [12  7  0  0  0]
    #  [10  2  0  0  0]]

    print(f'kernel_prewitt_x: \n{kernel_prewitt_x}')
    print(f'kernel_prewitt_y: \n{kernel_prewitt_y}')

    img_prewitt_x_int = cv2.filter2D(img, -1, kernel=kernel_prewitt_x)
    img_prewitt_x = cv2.filter2D(img, cv2.CV_32F, kernel=kernel_prewitt_x) / 3
    img_prewitt_y = cv2.filter2D(img, cv2.CV_32F, kernel=kernel_prewitt_y) / 3

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print(f'{img_prewitt_x[100:105, 100:105]}')
    print(f'{abs(img_prewitt_x[100:105, 100:105])}')
    print(f'{img_prewitt_x[100:105, 100:105].astype(np.uint8)}')
    print(f'{abs(img_prewitt_x[100:105, 100:105]).astype(np.uint8)}')
    print(f'{img_prewitt_x_int[100:105, 100:105]}')

    img_gradient = cv2.sqrt(cv2.pow(img_prewitt_x, 2) + cv2.pow(img_prewitt_y, 2))

    cv2.imshow('img', img)
    cv2.imshow('img_prewitt_x_int', img_prewitt_x_int)
    cv2.imshow('abs(img_prewitt_x)', abs(img_prewitt_x).astype(np.uint8))
    cv2.imshow('abs(img_prewitt_y)', abs(img_prewitt_y).astype(np.uint8))
    cv2.imshow('img_gradient', img_gradient.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def empty_callback(value):
    pass


def ex_2():
    img = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('canny')
    cv2.createTrackbar('th1', 'canny', 0, 255, empty_callback)
    cv2.createTrackbar('th2', 'canny', 0, 255, empty_callback)

    key = ord('a')
    while key != ord('q'):
        th1 = cv2.getTrackbarPos('th1', 'canny')
        th2 = cv2.getTrackbarPos('th2', 'canny')
        edges = cv2.Canny(img, th1, th2)

        cv2.imshow('img', img)
        cv2.imshow('canny', edges)
        key = cv2.waitKey(10)
    cv2.destroyAllWindows()


def ex_3():
    img_original: np.ndarray = cv2.imread('../_data/sw_s01e05/shapes.jpg')

    img = img_original.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1.5, np.pi / 180, 200)
    d = 2000
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + d * (-b))
        y1 = int(y0 + d * (a))
        x2 = int(x0 - d * (-b))
        y2 = int(y0 - d * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    img = img_original.copy()

    lines_p = cv2.HoughLinesP(edges, 1.5, np.pi / 180, 100)
    for line in lines_p:
        print(line)
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def ex_4():
    img_original: np.ndarray = cv2.imread('../_data/sw_s01e05/shapes.jpg')
    img = img_original.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    gray = cv2.medianBlur(gray, 5)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                              param1=200, param2=100, minRadius=10, maxRadius=100)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():
    print('Hello ex05!')
    ex_1()
    ex_2()
    ex_3()
    ex_4()


if __name__ == '__main__':
    main()
