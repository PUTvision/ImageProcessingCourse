import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(value):
    pass


def ex_0():

    def load_image_and_filer(image_filename):
        img = cv2.imread(image_filename, cv2.IMREAD_COLOR)

        cv2.namedWindow("img")
        cv2.createTrackbar("kernel_size", "img", 0, 10, nothing)

        key = ord("a")
        while key != ord("q"):
            kernel_size = 1 + 2*cv2.getTrackbarPos("kernel_size", "img")

            img_after_blur = cv2.blur(img, (kernel_size, kernel_size))
            img_after_gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            img_after_median = cv2.medianBlur(img, kernel_size)

            cv2.imshow("img", img)
            cv2.imshow("img_after_blur", img_after_blur)
            cv2.imshow("img_after_gaussian", img_after_gaussian)
            cv2.imshow("img_after_median", img_after_median)
            key = cv2.waitKey(10)

        cv2.destroyAllWindows()

    load_image_and_filer('../_data/s01e03/lenna_noise.bmp')
    load_image_and_filer('../_data/s01e03/lenna_salt_and_pepper.bmp')


def ex_1():
    img = cv2.imread("./../_data/no_idea.jpg", cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow("img")
    cv2.createTrackbar("threshold", "img", 0, 255, nothing)
    cv2.createTrackbar("kernel_size", "img", 1, 10, nothing)
    cv2.createTrackbar("0: erosion, 1: dilation", "img", 0, 1, nothing)

    key = ord("a")
    while key != ord("q"):
        threshold = cv2.getTrackbarPos("threshold", "img")
        kernel_size = 1 + 2 * cv2.getTrackbarPos("kernel_size", "img")
        operation_type = cv2.getTrackbarPos("0: erosion, 1: dilation", "img")

        _, img_after_threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation_type == 0:
            img_after_morphological = cv2.erode(img_after_threshold, kernel, iterations=1)
        else:
            img_after_morphological = cv2.dilate(img_after_threshold, kernel, iterations=1)

        cv2.imshow("img", img)
        cv2.imshow("img_after_threshold", img_after_threshold)
        cv2.imshow("img_after_morphological", img_after_morphological)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def ex_2_new():
    img = cv2.imread("./../_data/no_idea.jpg", cv2.IMREAD_GRAYSCALE)

    img_blurred_blur = cv2.blur(img, (3, 3), borderType=cv2.BORDER_REFLECT101)
    kernel = np.full((3, 3), 1/9, np.float)
    img_blurred_filter2d = cv2.filter2D(img, cv2.CV_8U, kernel, borderType=cv2.BORDER_REFLECT101)
    print(np.array_equal(img_blurred_blur, img_blurred_filter2d))

    # shape returns different number of values based on the image type
    height, width = img.shape
    print(f'height={height}, width={width}')
    for y in range(0, height):
        for x in range(0, width):
            if x % 3 == 0:
                img[y, x] = 255

    cv2.imshow('img', img)
    cv2.waitKey(0)

    img = cv2.imread("./../_data/no_idea.jpg", cv2.IMREAD_COLOR)

    # shape returns different number of values based on the image type
    height, width, channels = img.shape
    print(f'height={height}, width={width}')
    for y in range(0, height):
        for x in range(0, width):
            for c in range(0, 2):
                img[y, x, c] = 255

    cv2.imshow('img', img)
    cv2.waitKey(0)


def ex_2():
    img = cv2.imread("./../_data/no_idea.jpg", cv2.IMREAD_COLOR)

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 255])
        plt.plot(hist, color=col)
        plt.xlim([0, 255])
    plt.show()

    img = cv2.imread("./../_data/no_idea.jpg", cv2.IMREAD_GRAYSCALE)
    plt.hist(img.ravel(), 256, [0, 255])

    img_after_hist_equalization = cv2.equalizeHist(img)
    plt.hist(img_after_hist_equalization.ravel(), 256, [0, 255])
    #plt.show()

    img_stacked = np.hstack((img, img_after_hist_equalization))  # stacking images side-by-side
    cv2.imshow("img_stacked", img_stacked)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


clicked_points = []


def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_points.clear()


def ex_3():
    img = cv2.imread("./../_data/s01e03/road.jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', get_points)

    dst_points = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    key = ord("a")
    while key != ord("q"):
        img_with_points = img.copy()

        for i, p in enumerate(clicked_points):
            cv2.circle(img_with_points, p, 10, colours[i], -1)

        if len(clicked_points) == 4:
            src_points = np.ones_like(dst_points)
            for i in range(4):
                src_points[i] = list(clicked_points[i])

            perspective_transform = cv2.getPerspectiveTransform(src_points, dst_points)
            print(f'perspective_transform={perspective_transform}')
            img_dst = cv2.warpPerspective(img, perspective_transform, (500, 500))
            cv2.imshow('img_dst', img_dst)
            clicked_points.clear()

        cv2.imshow('img', img_with_points)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ex_0()
    ex_1()
    ex_2_new()
    ex_2()
    ex_3()
