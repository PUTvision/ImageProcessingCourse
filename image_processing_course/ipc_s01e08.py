import cv2
import numpy as np


def task_1():
    def empty(_):
        pass

    image = cv2.imread('./../_data/s01e08/tomatoes_and_apples.jpg')
    image = cv2.resize(image, (image.shape[0] // 2, image.shape[1] // 2), interpolation=cv2.INTER_AREA)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('image')
    cv2.imshow('image', image)

    cv2.createTrackbar('low H', 'image', 0, 180, empty)
    cv2.createTrackbar('high H', 'image', 0, 180, empty)
    cv2.createTrackbar('low S', 'image', 0, 255, empty)
    cv2.createTrackbar('high S', 'image', 0, 255, empty)
    cv2.createTrackbar('low V', 'image', 0, 255, empty)
    cv2.createTrackbar('high V', 'image', 0, 255, empty)

    key = ord('a')
    while key != ord('q'):
        low_h = cv2.getTrackbarPos('low H', 'image')
        high_h = cv2.getTrackbarPos('high H', 'image')
        low_s = cv2.getTrackbarPos('low S', 'image')
        high_s = cv2.getTrackbarPos('high S', 'image')
        low_v = cv2.getTrackbarPos('low V', 'image')
        high_v = cv2.getTrackbarPos('high V', 'image')
        # H: 25-50, S: 80-255, V: 0;255
        threshold = cv2.inRange(image_hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))
        cv2.imshow('threshold', threshold)
        key = cv2.waitKey(10)

    cv2.destroyAllWindows()


def task_2():
    colour_image = cv2.imread('./../_data/s01e08/cars.png')
    image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', image)
    _, thresholded = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('thresholded', thresholded)

    # Sure background area
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opened', opened)

    sure_background = cv2.dilate(opened, kernel, iterations=2)
    cv2.imshow('sure_background', sure_background)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresholded, cv2.DIST_L2, 5)
    print(np.unique(dist_transform))
    _, sure_foreground = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_foreground = sure_foreground.astype(np.uint8)
    cv2.imshow('sure_foreground', sure_foreground)

    unknown = cv2.subtract(sure_background, sure_foreground)
    cv2.imshow('unknown', unknown)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.imshow('markers', (markers*20).astype(np.uint8))
    print(np.unique(markers))

    markers = cv2.watershed(colour_image, markers)
    colour_image[markers == -1] = [255, 0, 0]

    labels = np.unique(markers)[2:]
    for label in labels:
        colour_image[markers == label] = np.random.rand(3) * 255

    cv2.imshow('colour_image', colour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task_2_alternative():
    colour_image = cv2.imread('./../_data/s01e08/cars.png')
    # colour_image = cv2.imread('./../_data/s01e08/tumor.jpg')
    
    mask = np.zeros_like(cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY), dtype=np.int32)
    image_with_clicks = colour_image.copy()
    label = 2  # 1 - background

    def on_click(event, x, y, flags, param):
        nonlocal mask, image_with_clicks, label

        if event == cv2.EVENT_LBUTTONDOWN:
            mask[y, x] = label
            cv2.circle(image_with_clicks, (x, y), 5, (255, 0, 0), thickness=2)
        if event == cv2.EVENT_MBUTTONDOWN:
            print(f'label: {label}')
            label += 1
        if event == cv2.EVENT_RBUTTONDOWN:
            mask[y, x] = 1
            cv2.circle(image_with_clicks, (x, y), 5, (255, 0, 255), thickness=2)

    cv2.namedWindow('image_with_clicks')
    cv2.setMouseCallback('image_with_clicks', on_click)

    segmented_image = colour_image.copy()

    key = ord('a')
    while key != ord('q'):
        if key == ord(' '):
            markers = cv2.watershed(colour_image, mask)
            print(markers.shape)
            segmented_image[markers == -1] = [255, 0, 0]

            labels = np.unique(markers)[2:]
            for label in labels:
                segmented_image[markers == label] = np.random.rand(3) * 255

        cv2.imshow('image_with_clicks', image_with_clicks)
        cv2.imshow('segmented_image', segmented_image)
        key = cv2.waitKey(10)

    cv2.destroyAllWindows()


def task_3():
    drawing_rectangle = False
    rectangle_start = None
    rectangle_end = None

    def on_click(event, x, y, flags, param):
        nonlocal drawing_rectangle
        nonlocal rectangle_start
        nonlocal rectangle_end

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_rectangle = True
            rectangle_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing_rectangle:
            rectangle_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            rectangle_end = (x, y)
            drawing_rectangle = False

    cv2.namedWindow('input')
    cv2.setMouseCallback('input', on_click)

    image = cv2.imread('./../_data/s01e08/tumor.jpg', cv2.IMREAD_COLOR)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    key = ord('a')
    while key != ord('q'):
        drawn_image = image.copy()
        if rectangle_start is not None and rectangle_end is not None:
            cv2.rectangle(drawn_image, rectangle_start, rectangle_end, (255, 0, 0))

        cv2.imshow('input', drawn_image)
        key = cv2.waitKey(10)
        if key == ord(' '):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            print(rectangle_start)
            print(rectangle_end)
            print((*rectangle_start, rectangle_end[0] - rectangle_start[0], rectangle_end[1] - rectangle_start[1]))

            # mark just the area around tumor to get it segmented properly
            cv2.grabCut(
                image,
                mask,
                (*rectangle_start, rectangle_end[0] - rectangle_start[0], rectangle_end[1] - rectangle_start[1]),
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT
            )
            print(np.unique(mask))
            print(cv2.GC_PR_FGD)
            segmented_image = image.copy()
            segmented_image[mask == cv2.GC_PR_FGD] = [0, 255, 0]
            # segmented_image[mask == cv2.GC_PR_BGD] = [0, 0, 255]
            # segmented_image[mask == 0] = [255, 0, 255]
            cv2.imshow('segmented', segmented_image)


if __name__ == '__main__':
    task_1()
    task_2()
    task_2_alternative()
    task_3()
