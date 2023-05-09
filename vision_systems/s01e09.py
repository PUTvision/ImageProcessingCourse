import os

import cv2
import numpy as np

# https://docs.google.com/document/d/16-TXEYUkAdvEHEYKYLm71N7pYEFILh6WGOEmxeX79tI/edit?usp=sharing


def ex_1():
    pattern_size = (8, 5)
    image_size = None
    number_of_images_to_use = 100

    object_points = []
    for i in range(0, pattern_size[1]):
        for j in range(0, pattern_size[0]):
            print(f'i, j: {i}, {j}')
            object_points.append([j, i, 0])
    object_points = np.array(object_points, dtype=np.float32)

    image_points_from_images = []
    object_points_form_images = []

    for filename in os.listdir('../_data/sw_s01e09/')[:number_of_images_to_use]:
        print(filename)
        if filename.endswith('.bmp'):

            img_from_file = cv2.imread(f'../_data/sw_s01e09/{filename}', cv2.IMREAD_COLOR)
            image_size = img_from_file.shape

            pattern_found, corners = cv2.findChessboardCorners(img_from_file, pattern_size)

            if pattern_found:
                print(f'corners[0]: {corners[0]}')
                corners = cv2.cornerSubPix(
                    cv2.cvtColor(img_from_file, cv2.COLOR_BGR2GRAY),
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                )
                print(f'corners[0] after cornerSubPix: {corners[0]}')

                image_points_from_images.append(corners)
                object_points_form_images.append(object_points)

            img_with_corners = cv2.drawChessboardCorners(img_from_file.copy(), pattern_size, corners, pattern_found)
            cv2.imshow('img_with_corners', img_with_corners)
            _ = cv2.waitKey(100)

    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_form_images, image_points_from_images, image_size[:2], None, None)
    print(f'retval: {retval}')

    for filename in os.listdir('../_data/sw_s01e09/')[:number_of_images_to_use]:
        print(filename)
        if filename.endswith('.bmp'):
            img_from_file = cv2.imread(f'../_data/sw_s01e09/{filename}', cv2.IMREAD_COLOR)
            img_undistorted = cv2.undistort(img_from_file, camera_matrix, dist_coeffs)
            cv2.imshow('img_undistorted', img_undistorted)
            _ = cv2.waitKey(0)

    # cv2.imshow('img', img_from_file)
    # _ = cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ex_1()
