import cv2
import numpy as np


img_todo_1 = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_GRAYSCALE)


def mouse_callback_todo_1(event, x, y, flags, userdata):
    global img_todo_1
    if event == cv2.EVENT_LBUTTONDOWN:
        # img = cv2.rectangle(img, (x-50, y-50), (x+50, y+50), (255, 0, 0), thickness=5, lineType=cv2.LINE_8)
        img = cv2.resize(img_todo_1, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = cv2.circle(img_todo_1, (x, y), 50, (0, 0, 255), thickness=5, lineType=cv2.LINE_4)
    elif event == cv2.EVENT_MBUTTONDOWN:
        img = cv2.circle(img_todo_1, (x, y), 50, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)


def todo_1():
    global img_todo_1
    img_todo_1 = cv2.imread('./_data/no_idea.jpg', cv2.IMREAD_COLOR)

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse_callback_todo_1)

    key = ord('a')
    while key != ord('q'):
        cv2.imshow('img', img_todo_1)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


img_todo_2 = None
pts_1_todo_2 = []


def mouse_callback_todo_2(event, x, y, flags, userdata):
    global pts_1_todo_2, img_todo_2
    if event == cv2.EVENT_LBUTTONDOWN:
        img_todo_2 = cv2. circle(img_todo_2, (x, y), 5, (0, 0, 255), thickness=cv2.FILLED)
        pts_1_todo_2.append((x, y))


def todo_2():
    global pts_1_todo_2, img_todo_2
    img_todo_2 = cv2.imread('./_data/s01e03/road.jpg')
    img_todo_2 = cv2.resize(img_todo_2, dsize=None, fx=0.5, fy=0.5)
    img_todo_2_original = img_todo_2.copy()

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse_callback_todo_2)

    pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    key = ord('a')
    while key != ord('q'):
        if len(pts_1_todo_2) >= 4:
            pts_1_np = np.asarray(pts_1_todo_2, dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts_1_np, pts_2)
            dst = cv2.warpPerspective(img_todo_2, M, (300, 300))
            cv2.imshow('dst', dst)
            img_todo_2 = img_todo_2_original.copy()
            pts_1_todo_2.clear()

        cv2.imshow('img', img_todo_2)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


img_gallery_todo_3 = None
pts_2_todo_3 = []


def mouse_callback_todo_3(event, x, y, flags, userdata):
    global pts_2_todo_3, img_gallery_todo_3
    if event == cv2.EVENT_LBUTTONDOWN:
        img_gallery_todo_3 = cv2. circle(img_gallery_todo_3, (x, y), 5, (0, 0, 255), thickness=cv2.FILLED)
        pts_2_todo_3.append((x, y))


def homework_3():
    global pts_2_todo_3, img_gallery_todo_3
    img_gallery_todo_3 = cv2.imread('./_data/s01e04/gallery.png')
    img_gallery_todo_3 = cv2.resize(img_gallery_todo_3, dsize=None, fx=0.5, fy=0.5)
    img_gallery_todo_3_original = img_gallery_todo_3.copy()

    img_pug = cv2.imread('./_data/s01e04/pug.png')
    rows, cols, ch = img_pug.shape

    # img_mask = np.ones(shape=(img_gallery_todo_3.shape[0], img_gallery_todo_3.shape[1]), dtype=np.uint8)*255
    img_mask = np.ones_like(img_pug)*255

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse_callback_todo_3)

    pts_1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    key = ord('a')
    while key != ord('q'):
        if len(pts_2_todo_3) >= 4:
            pts_2_todo_3_np = np.asarray(pts_2_todo_3, dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts_1, pts_2_todo_3_np)
            dst = cv2.warpPerspective(img_pug, M, (img_gallery_todo_3.shape[1], img_gallery_todo_3.shape[0]))
            dst_mask = cv2.warpPerspective(img_mask, M, (img_gallery_todo_3.shape[1], img_gallery_todo_3.shape[0]))
            bitwise = cv2.bitwise_and(img_gallery_todo_3, cv2.bitwise_not(dst_mask))
            added = cv2.add(bitwise, dst)
            cv2.imshow('dst', dst)
            cv2.imshow('dst_mask', dst_mask)
            cv2.imshow('bitwise_and', bitwise)
            cv2.imshow('added', added)
            img_gallery_todo_3 = img_gallery_todo_3_original.copy()
            pts_2_todo_3.clear()

        cv2.imshow('img', img_gallery_todo_3)
        key = cv2.waitKey(50)

    cv2.destroyAllWindows()


def main():
    # todo_1()
    # todo_2()
    homework_3()


if __name__ == '__main__':
    main()
