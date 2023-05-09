import cv2
import numpy as np

# https://docs.google.com/document/d/19zTs02TQg0PMktduHuiFHk_yWkAL1CDFPFSqkeCFLJg/edit?usp=sharing


def ex_1():
    img: np.ndarray = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fast: cv2.FastFeatureDetector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
    print(f'fast nonmaxsuppression status: {fast.getNonmaxSuppression()}')

    keypoints_fast = fast.detect(img)
    print(f'len(keypoints): {len(keypoints_fast)}')
    img_with_fast_keypoints = img.copy()
    cv2.drawKeypoints(img, keypoints_fast, img_with_fast_keypoints)

    orb: cv2.ORB = cv2.ORB_create()

    keypoints_orb = orb.detect(img)
    print(f'len(keypoints): {len(keypoints_orb)}')
    img_with_orb_keypoints = img.copy()
    cv2.drawKeypoints(img, keypoints_orb, img_with_orb_keypoints)

    cv2.imshow('img', img)
    cv2.imshow('img_with_fast_keypoints', img_with_fast_keypoints)
    cv2.imshow('img_with_orb_keypoints', img_with_orb_keypoints)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def ex_2():
    img_1: np.ndarray = cv2.imread('../_data/1b.jpg', cv2.IMREAD_COLOR)
    img_1: np.ndarray = cv2.resize(img_1, None, fx=0.2, fy=0.2)
    img_2: np.ndarray = cv2.imread('../_data/2b.jpg', cv2.IMREAD_COLOR)
    img_2: np.ndarray = cv2.resize(img_2, None, fx=0.2, fy=0.2)

    fast: cv2.FastFeatureDetector = cv2.FastFeatureDetector_create(threshold=100, nonmaxSuppression=True)
    orb: cv2.ORB = cv2.ORB_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    detector = fast # orb
    descriptor = orb # orb

    keypoints_1 = detector.detect(img_1)
    keypoints_2 = detector.detect(img_2)

    keypoints_1, descriptors_1 = descriptor.compute(img_1, keypoints_1)
    keypoints_2, descriptors_2 = descriptor.compute(img_2, keypoints_2)

    matcher: cv2.BFMatcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches_1_to_2 = matcher.match(descriptors_1, descriptors_2)
    matches_2_to_1 = matcher.match(descriptors_2, descriptors_1)

    img_with_matches_1_to_2 = None
    img_with_matches_1_to_2 = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches_1_to_2, img_with_matches_1_to_2)
    img_with_matches_2_to_1 = None
    img_with_matches_2_to_1 = cv2.drawMatches(img_2, keypoints_2, img_1, keypoints_1, matches_2_to_1, img_with_matches_2_to_1)

    cv2.imshow('img_with_matches_1_to_2', img_with_matches_1_to_2)
    cv2.imshow('img_with_matches_2_to_1', img_with_matches_2_to_1)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ex_1()
    ex_2()
