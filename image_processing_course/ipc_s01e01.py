import cv2
from matplotlib import pyplot as plt


def ex_0():
    cap = cv2.VideoCapture(0)  # open the default camera

    key = ord('a')
    while key != ord('q'):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame comes here
        # Convert RGB image to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        # Detect edges on the blurred image
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)

        # Display the result of our processing
        cv2.imshow('result', img_edges)
        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(30)

    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


def ex_1():
    # TODO 1
    img_from_file = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img_from_file', img_from_file)
    cv2.waitKey(0)
    cv2.imwrite('../_data/no_idea_grayscale.png', img_from_file)
    cv2.destroyAllWindows()


def ex_2():
    img_color = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_COLOR)
    img_grayscale = cv2.imread('../_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)

    # TODO 2
    print(f'Color image parameters: {img_color.shape}')
    print(f'Grayscale image parameters: {img_grayscale.shape}')

    print(f'Pixel (220, 270) value - color: {img_color[220, 270]} , grayscale: {img_grayscale[220, 270]}')

    cv2.imshow('img_color', img_color)
    cv2.imshow('img_grayscale', img_grayscale)
    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO 3
    head = img_color[10:120, 250:420]
    img_with_head = img_color.copy()
    img_with_head[60:170, 50:220] = head

    cv2.imshow('head', head)
    cv2.imshow('img_with_head', img_with_head)
    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO 4
    cv2.imshow('img_color', img_color)
    cv2.waitKey(1)
    plt.imshow(img_color)
    plt.show()
    cv2.destroyAllWindows()

    # TODO 5
    img_bgr = cv2.imread('../_data/s01e01/AdditiveColor.png', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.5, fy=0.5)

    b, g, r = cv2.split(img_bgr)

    cv2.imshow('img_bgr', img_bgr)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)

    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_3():
    # TODO 6
    cap = cv2.VideoCapture(0)

    key = ord(' ')
    # NOTE(MF): thanks to the way key variable is initialized we do not need to load one frame before the loop
    while key != ord('q'):
        if key == ord(' '):
            _, frame = cap.read()
            cv2.imshow('video_frame', frame)
        key = cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

    # TODO 7
    cap = cv2.VideoCapture('../_data/s01e01/Wildlife.mp4')

    key = ord(' ')
    ret = True
    while key != ord('q') and ret:
        if key == ord(' '):
            ret, frame = cap.read()
            if ret:
                cv2.imshow('video_frame', frame)
            else:
                print('End of video file!')
        key = cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ex_0()
    ex_1()
    ex_2()
    ex_3()
