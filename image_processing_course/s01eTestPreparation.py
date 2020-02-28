import cv2
import numpy as np


def ex_0():
    def nothing(x):
        pass

    cv2.namedWindow("img")

    cv2.createTrackbar('H_min', 'img', 0, 255, nothing)
    cv2.createTrackbar('H_max', 'img', 255, 255, nothing)
    cv2.createTrackbar('S_min', 'img', 0, 255, nothing)
    cv2.createTrackbar('S_max', 'img', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'img', 0, 255, nothing)
    cv2.createTrackbar('V_max', 'img', 255, 255, nothing)

    img1 = cv2.imread("./_data/s01eTestPreparation/znak1.png")
    print(img1.shape)
    img2 = cv2.resize(cv2.imread("./_data/s01eTestPreparation/znak2.png"), (img1.shape[1], img1.shape[0]))
    print(img2.shape)

    img = np.hstack((img1, img2))

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    key = ord('a')
    while key != ord('q'):
        h_min = cv2.getTrackbarPos('H_min', 'img')
        h_max = cv2.getTrackbarPos('H_max', 'img')
        s_min = cv2.getTrackbarPos('S_min', 'img')
        s_max = cv2.getTrackbarPos('S_max', 'img')
        v_min = cv2.getTrackbarPos('V_min', 'img')
        v_max = cv2.getTrackbarPos('V_max', 'img')

        img_in_range = cv2.inRange(img_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        img_res = cv2.bitwise_and(img, img, mask=img_in_range)

        cv2.imshow("img", img_res)
        key = cv2.waitKey(50)


def ex_1():
    img_original = cv2.imread("./_data/s01eTestPreparation/obiekty.png")
    img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_grayscale = cv2.bitwise_not(img_grayscale)

    kernel = np.ones((5, 5), np.uint8)
    img_grayscale = cv2.morphologyEx(img_grayscale, cv2.MORPH_OPEN, kernel)

    img_contours, contours, hierarchy = \
        cv2.findContours(img_grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img_with_contours = img_original.copy()

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        print(area)
        if area > 5000:
            cv2.drawContours(img_with_contours, contours, i, (255, 0, 0), 3)
        else:
            cv2.drawContours(img_with_contours, contours, i, (0, 0, 255), 3)

    cv2.imshow("img", img_with_contours)
    key = cv2.waitKey(0)


clicked_points = []


def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_points.clear()


def ex_2():
    img = cv2.imread("_data/no_idea.jpg")

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', get_points)

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    key = ord("a")
    while key != ord("q"):
        img_with_points = img.copy()

        for i, p in enumerate(clicked_points):
            cv2.circle(img, p, 10, colours[i], -1)

        if len(clicked_points) == 2:
            img_roi = img[
                      clicked_points[0][1]:clicked_points[1][1],
                      clicked_points[0][0]:clicked_points[1][0],
                      ]

            img[
            clicked_points[0][1]:clicked_points[1][1],
            clicked_points[0][0]:clicked_points[1][0],
            ] = cv2.bitwise_not(img_roi)

            clicked_points.clear()

        cv2.imshow('img', img_with_points)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ex_0()
    ex_1()
    ex_2()
