import cv2
import numpy as np

# https://docs.google.com/document/d/11gjO980jxlltTpR_JJ4SQ_DFSq0Bipb7NMH-j9LH6Cs/edit?usp=sharing


def _gradient_operator(
        img: np.ndarray,
        kernel_x: np.ndarray,
        kernel_y: np.ndarray,
        division_value: int,
        window_name_prefix: str
):
    img_gradient_x = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    img_gradient_y = cv2.filter2D(img, cv2.CV_32F, kernel_y)

    img_gradient = cv2.sqrt(pow(img_gradient_x / division_value, 2) + pow(img_gradient_y / division_value, 2))

    cv2.imshow(f'{window_name_prefix}_x', (abs(img_gradient_x) / division_value).astype(np.uint8))
    cv2.imshow(f'{window_name_prefix}_x_no_abs', (img_gradient_x / division_value).astype(np.uint8))
    cv2.imshow(f'{window_name_prefix}_y', (abs(img_gradient_y) / division_value).astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow(f'{window_name_prefix}', img_gradient.astype(np.uint8))
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def ex_1():
    img_grayscale = cv2.imread('_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

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
    print(kernel_prewitt_x)
    print(kernel_prewitt_y)

    kernel_sobel_x = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        np.int8
    )
    kernel_sobel_y = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        np.int8
    )

    _gradient_operator(img_grayscale, kernel_prewitt_x, kernel_prewitt_y, 3, 'img_prewitt')
    _gradient_operator(img_grayscale, kernel_sobel_x, kernel_sobel_y, 4, 'img_sobel')
    cv2.destroyAllWindows()


def ex_2():
    def trackbar_callback(x):
        pass

    window_name = 'Canny'

    img_grayscale = cv2.imread('_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow(window_name)
    cv2.createTrackbar('threshold1', window_name, 0, 255, trackbar_callback)
    cv2.createTrackbar('threshold2', window_name, 0, 255, trackbar_callback)

    key = ord('a')
    while key != ord('q'):
        threshold1 = cv2.getTrackbarPos('threshold1', window_name)
        threshold2 = cv2.getTrackbarPos('threshold2', window_name)
        img_canny = cv2.Canny(img_grayscale, threshold1, threshold2)

        cv2.imshow(window_name, img_canny)
        cv2.waitKey(100)
    cv2.destroyAllWindows()


def ex_3():
    img = cv2.imread('_data/sw_s01e05/shapes.jpg', cv2.IMREAD_COLOR)
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_grayscale, 50, 150, apertureSize=3)

    img_hough_lines = np.copy(img)
    hough_lines = cv2.HoughLines(edges, 1.5, np.pi / 180, 200)
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_hough_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('hough_lines', img_hough_lines)
    cv2.waitKey(0)

    img_hough_lines_p = np.copy(img)
    hough_lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
    for line in hough_lines_p:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_hough_lines_p, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('hough_lines_p', img_hough_lines_p)
    cv2.waitKey(0)

    img_hough_circles = np.copy(img)
    circles = cv2.HoughCircles(img_grayscale, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=10, maxRadius=100)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img_hough_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img_hough_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('hough_circles', img_hough_circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
