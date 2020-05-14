import cv2
import numpy as np


def task_1():
    image = cv2.imread('./../_data/sw_s01e09/img_21130751_0005.bmp')

    flag_found, corners = cv2.findChessboardCorners(image, (8, 5))
    print(corners[0])

    if flag_found:
        print('Corners found, refining their positions')
        corners = cv2.cornerSubPix(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
        print(corners[0])

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        object_points = np.zeros((8 * 5, 3), np.float32)
        print(object_points[1])
        print(object_points.shape)
        object_points[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
        print(object_points)
        print(object_points.shape)

        object_points_for = []#np.zeros_like(object_points)
        for i in range(0, 5):
            for j in range(0, 8):
                # print(object_points_for[i, j])
                # object_points_for[i, j] = [i, j, 0]
                object_points_for.append([j, i, 0])
        object_points_for = [np.array(object_points_for, dtype=np.float32)]

        print(object_points[0:3])
        print(object_points_for[0:3])

        image_points = [corners]
        # image_points.append(corners)
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_for, image_points, image.shape[:2], None, None)

        fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(
            camera_matrix, image.shape[:2], 7.2, 5.4
        )
        print(fovx)
        print(fovy)
        print(focalLength)

    else:
        print('Corners not found')

    image_with_corners = cv2.drawChessboardCorners(image, (8, 5), corners, flag_found)
    cv2.imshow('image_with_corners', image_with_corners)
    cv2.waitKey(0)


if __name__ == '__main__':
    task_1()
