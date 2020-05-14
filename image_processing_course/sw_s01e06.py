import cv2
import numpy as np


def empty_callback(_):
    pass


def contours():
    img = cv2.imread('../_data/sw_s01e06/not_bad.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=0.25, fy=0.25)

    cv2.namedWindow('img')
    cv2.createTrackbar('threshold', 'img', 50, 255, empty_callback)

    cv2.imshow('img', img)

    while True:
        threshold = cv2.getTrackbarPos('threshold', 'img')
        img_copy = img.copy()
        img_copy[img < threshold] = 255
        img_copy[img >= threshold] = 0

        structuring_elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img_copy = cv2.dilate(img_copy, structuring_elem)
        img_copy = cv2.erode(img_copy, structuring_elem)

        contours, hierarchy = cv2.findContours(img_copy, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_copy, contours, -1, (255, 0, 0))

        src_points = []
        for contour in contours:
            for point in contour:
                cv2.circle(img_copy, tuple(point[0]), 3, (0, 0, 255))

            moments = cv2.moments(contour)
            src_points.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))

        src_points = np.float32(src_points)
        dst_points = np.float32([(500, 500), (0, 500), (500, 0), (0, 0)])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img_transformed = cv2.warpPerspective(img, matrix, (500, 500))

        cv2.imshow('img', img_transformed)
        cv2.imshow('contours', img_copy)
        cv2.waitKey(50)


def matching():
    haystack = cv2.imread('../_data/sw_s01e06/haystack.png')
    needle = cv2.imread('../_data/sw_s01e06/needle.png')

    w, h = needle.shape[:2]

    result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(haystack, top_left, bottom_right, (0, 0, 255), 2)

    cv2.imshow('result', result)
    cv2.imshow('haystack', haystack)
    cv2.waitKey()


if __name__ == '__main__':
    contours()
    matching()
