from depthai_sdk import OakCamera
import cv2


flag_save_image = False
img_counter = 0


def save_image(packet):
    global flag_save_image, img_counter
    # print(f'{packet}')
    if flag_save_image:
        cv2.imwrite(f'frame_{img_counter:03}.png', packet.frame)
        img_counter = img_counter + 1
        flag_save_image = False


with OakCamera() as oak:
    color = oak.create_camera('color')
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    oak.visualize([color, right], fps=True)
    oak.callback(color, callback=save_image)
    oak.start(blocking=False)

    while oak.running():
        key = oak.poll()

        if key == ord('i'):
            color.control.exposure_time_down()
        elif key == ord('o'):
            color.control.exposure_time_up()
        elif key == ord('k'):
            color.control.sensitivity_down()
        elif key == ord('l'):
            color.control.sensitivity_up()
        elif key == ord('z'):
            flag_save_image = True
        elif key == ord('x'):
            flag_save_image = False

        elif key == ord('e'):  # Switch to auto exposure
            color.control.send_controls({'exposure': {'auto': True}})
