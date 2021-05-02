import cv2
import numpy as np
from matplotlib import pyplot as plt


# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 5. CV_CAP_PROP_FPS Frame rate.
# 6. CV_CAP_PROP_FOURCC 4-character code of codec.
# 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

def ex_camera():
    cap = cv2.VideoCapture(0)
    print(f'CAP_PROP_FRAME_WIDTH: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print(f'CAP_PROP_BRIGHTNESS: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}')
    print(f'CAP_PROP_EXPOSURE: {cap.get(cv2.CAP_PROP_EXPOSURE)}')
    print(f'CAP_PROP_BUFFERSIZE: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}')

    cv2.waitKey(0)


def empty_callback(value):
    print(f'Trackbar reporting for duty with value: {value}')
    pass


def ex_1():
    # create a black image, a window
    img = np.zeros((300, 512, 3), dtype=np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('G', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('B', 'image', 0, 255, empty_callback)

    # create switch for ON/OFF functionality
    switch_trackbar_name = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch_trackbar_name, 'image', 0, 1, empty_callback)

    while True:
        cv2.imshow('image', img)
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch_trackbar_name, 'image')

        if s == 0:
            # assign zeros to all pixels
            img[:] = 0
        else:
            # assign the same BGR color to all pixels
            img[:] = [b, g, r]


    # closes all windows (usually optional as the script ends anyway)
    cv2.destroyAllWindows()


def convert_to_threshold_mode(mode: int) -> int:
    if mode == 0:
        return cv2.THRESH_BINARY
    elif mode == 1:
        return cv2.THRESH_BINARY_INV
    elif mode == 2:
        return cv2.THRESH_TRUNC
    elif mode == 3:
        return cv2.THRESH_TOZERO
    else:
        raise ValueError('Not supported thresholding mode!')


def todo_1():
    img_from_file = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('img')
    cv2.createTrackbar('threshold', 'img', 0, 255, empty_callback)
    cv2.createTrackbar('mode', 'img', 0, 5, empty_callback)

    key = ord('a')

    while key != ord('q'):
        t = cv2.getTrackbarPos('threshold', 'img')
        mode = cv2.getTrackbarPos('mode', 'img')

        _, img_thresholded = cv2.threshold(img_from_file, t, 255, convert_to_threshold_mode(mode))

        cv2.imshow('img', img_thresholded)
        cv2.waitKey(50)
    cv2.destroyAllWindows()


def todo_2():
    img_from_file = cv2.imread('./_data/s01e02/qr.jpg', cv2.IMREAD_GRAYSCALE)

    s = 2.75

    cv2.namedWindow('img')
    cv2.createTrackbar('threshold', 'img', 0, 255, empty_callback)
    cv2.createTrackbar('mode', 'img', 0, 5, empty_callback)

    img_linear = cv2.resize(img_from_file, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    img_nearest = cv2.resize(img_from_file, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    img_area = cv2.resize(img_from_file, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img_lanczos = cv2.resize(img_from_file, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)

    titles = ['img_from_file', 'img_linear', 'img_nearest', 'img_area', 'img_lanczos']
    images = [img_from_file, img_linear, img_nearest, img_area, img_lanczos]
    for i in range(5):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    for t, i in zip(titles, images):
        cv2.imshow(t, i)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def ex_3():
    img_from_file = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    img_add_cv = cv2.add(img_from_file, 40)#img_from_file *4
    img_add_float = img_from_file.astype(np.float32) + 40
    img_add = img_from_file + 40

    cv2.imshow('img_from_file', img_from_file)
    cv2.imshow('img_add_cv', img_add_cv)
    cv2.imshow('img_add', img_add)
    cv2.imshow('img_add_float', img_add_float/(255+40))
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def todo_3():
    img_from_file = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_COLOR)
    img_logo = cv2.imread('./_data/s01e02/LOGO_PUT_VISION_LAB_MAIN.png', cv2.IMREAD_COLOR)
    img_logo = cv2.resize(img_logo, dsize=(img_from_file.shape[1], img_from_file.shape[0]))

    cv2.namedWindow('img')
    cv2.createTrackbar('alpha', 'img', 0, 100, empty_callback)

    key = ord('a')

    while key != ord('q'):
        alpha = cv2.getTrackbarPos('alpha', 'img') / 100.0

        img_blended = cv2.addWeighted(img_from_file, alpha, img_logo, 1 - alpha, 0)

        cv2.imshow('img', img_blended)
        key = cv2.waitKey(50)
    cv2.destroyAllWindows()


def process_image(a: int, img_b: np.ndarray) -> np.ndarray:
    cap = cv2.VideoCapture(0)
    img: np.ndarray = cv2.imread()

    return img + img_b


def homework_2():
    img = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_COLOR)

    img_my = process_image(5, img)
    img_my.shape()

    imgneg = 255-img
    img_neg_2 = cv2.bitwise_not(img)
    while True:
        cv2.imshow('image', img)
        cv2.imshow('image_negative', imgneg)
        cv2.imshow('img_neg_2', img_neg_2)
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break


def main():
    # ex_camera()
    # ex_1()
    # todo_1()
    # todo_2()
    # todo_3()
    homework_2()

if __name__ == '__main__':
    main()
