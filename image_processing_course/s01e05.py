import numpy as np
import cv2


def ex_0():
    img = cv2.imread('./../_data/no_idea.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=False)
    keypoints = fast.detect(img_gray)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

    cv2.imshow('point feature detector', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_1():
    img1 = cv2.imread('./../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    # cap = cv2.VideoCapture(0)
    #
    # key = ord('a')
    # while key != ord('q'):
    #     _, frame = cap.read()
    #     cv2.imshow("img", frame)
    #     key = cv2.waitKey(50)
    # img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img1 = cv2.imread('./../_data/1b.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)
    # img1 = cv2.imread("_data/object.png", cv2.IMREAD_GRAYSCALE)

    # key = ord('a')
    # while key != ord('q'):
    #     _, frame = cap.read()
    #     cv2.imshow("img", frame)
    #     key = cv2.waitKey(50)
    # img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread('./../_data/2b.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, None, fx=0.25, fy=0.25)

    #detector = cv2.AKAZE_create()
    detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
    # descriptor = cv2.ORB_create()
    descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    _, des1 = descriptor.compute(img1, kp1)
    _, des2 = descriptor.compute(img2, kp2)

    #kp1, des1 = detector_descriptor.detectAndCompute(img1, None)
    #kp2, des2 = detector_descriptor.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:200], None)

    cv2.imwrite('result.png', img3)
    cv2.imshow("matches", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_2():
    face_cascade = cv2.CascadeClassifier('./../_data/s01e05/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./../_data/s01e05/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    key = ord('a')
    while key != ord('q'):
        _, img = cap.read()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("img", img)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def ex_3():
    def nothing(x):
        pass

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('img')
    cv2.createTrackbar('low', 'img', 50, 255, nothing)
    cv2.createTrackbar('high', 'img', 200, 255, nothing)

    key = ord('a')
    while key != ord('q'):
        _, img = cap.read()

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_h, img_s, img_v = cv2.split(img_hsv)

        img_analysed = img_s

        threshold_low = cv2.getTrackbarPos('low', 'img')
        threshold_high = cv2.getTrackbarPos('high', 'img')

        _, img_after_low = cv2.threshold(img_analysed, threshold_low, 255, cv2.THRESH_BINARY_INV)
        _, img_after_high = cv2.threshold(img_analysed, threshold_high, 255, cv2.THRESH_BINARY)

        img_after_both = cv2.bitwise_and(img_after_low, img_after_high)

        img_in_range = cv2.inRange(img_hsv, (0, 140, 160), (255, 190, 200))

        res = cv2.bitwise_and(img, img, mask=img_in_range)

        cv2.imshow('img', img_analysed)
        cv2.imshow('img_after_low', img_after_low)
        cv2.imshow('img_after_high', img_after_high)
        cv2.imshow('img_after_both', img_after_both)
        cv2.imshow("in_range", res)
        key = cv2.waitKey(50)


if __name__ == "__main__":
    ex_0()
    ex_1()
    ex_2()
    ex_3()
