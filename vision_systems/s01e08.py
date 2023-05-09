import cv2
import numpy as np

# https://docs.google.com/document/d/1fOtqWj7KoP8b7kANQ7D7mci-_iLo1Ti6ryhVJVEWfLo/edit?usp=sharing


def empty_callback(_):
    pass


def ex_1():
    img_from_file = cv2.imread('../_data/s01e08/tomatoes_and_apples.jpg', cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_from_file, cv2.COLOR_BGR2HSV)

    window_detection_name = 'img'

    cv2.namedWindow(window_detection_name)
    cv2.createTrackbar('low_H', window_detection_name, 0, 255, empty_callback)
    cv2.createTrackbar('high_H', window_detection_name, 0, 255, empty_callback)
    cv2.createTrackbar('low_S', window_detection_name, 0, 255, empty_callback)
    cv2.createTrackbar('high_S', window_detection_name, 0, 255, empty_callback)
    cv2.createTrackbar('low_V', window_detection_name, 0, 255, empty_callback)
    cv2.createTrackbar('high_V', window_detection_name, 0, 255, empty_callback)

    key = ord('a')
    while key != ord('q'):
        low_H = cv2.getTrackbarPos('low_H', window_detection_name)
        high_H = cv2.getTrackbarPos('high_H', window_detection_name)
        low_S = cv2.getTrackbarPos('low_S', window_detection_name)
        high_S = cv2.getTrackbarPos('high_S', window_detection_name)
        low_V = cv2.getTrackbarPos('low_V', window_detection_name)
        high_V = cv2.getTrackbarPos('high_V', window_detection_name)

        img_inrange = cv2.inRange(img_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # img_h, img_s, img_v = cv2.split(img_hsv)

        # cv2.imshow('img_h', img_h)
        # cv2.imshow('img_s', img_s)
        # cv2.imshow('img_v', img_v)

        # cv2.imshow('img_hsv', img_hsv)
        cv2.imshow(window_detection_name, img_inrange)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def ex_2():
    img_from_file = cv2.imread('../_data/s01e08/cars.png', cv2.IMREAD_COLOR)
    img_with_clicks = img_from_file.copy()
    print(img_from_file.shape)
    print(img_from_file.dtype)

    markers = np.zeros(img_from_file.shape[:2], dtype=np.int32)
    print(markers.shape)
    print(markers.dtype)
    label_number = 1

    def on_cliok(event, x, y, flags, userdata):
        nonlocal markers, label_number, img_with_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            markers[y, x] = label_number
            img_with_clicks = cv2.circle(img_with_clicks, (x, y), 5, (0, 0, 255))
        if event == cv2.EVENT_MBUTTONDOWN:
            label_number += 1

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', on_cliok)

    key = ord('a')
    while key != ord('q'):
        cv2.imshow('img', img_with_clicks)
        key = cv2.waitKey(50)

        if key == ord(' '):
            print(markers[100:110, 100:110])
            markers = cv2.watershed(img_from_file, markers)
            print(markers[100:110, 100:110])

            segmented_img = np.zeros_like(img_from_file, dtype=np.uint8)
            segmented_img[markers == -1] = (0, 0, 255)

            for i in range(1, label_number):
                segmented_img[markers == i] = np.random.rand(3)*255

            cv2.imshow('segmented_img', segmented_img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ex_1()
    ex_2()
