from depthai_sdk import OakCamera
import cv2
import pathlib


flag_save_image = False
flag_save_image_depth = False
img_counter = 0
img_counter_depth = 0

dir = 'pieczarki_dwie_strony_2_setup_2'


def save_image(packet):
    global flag_save_image, img_counter
    # print(f'{packet}')
    if flag_save_image:
        cv2.imwrite(f'.//{dir}//frame_{img_counter:03}.png', packet.frame)
        img_counter = img_counter + 1
        flag_save_image = False


def save_image_depth(packet):
    global flag_save_image_depth, img_counter_depth
    # print(f'{packet}')
    if flag_save_image_depth:
        cv2.imwrite(f'.//{dir}//depth_{img_counter_depth:03}.png', packet.frame)
        img_counter_depth = img_counter_depth + 1
        flag_save_image_depth = False


directory_path = pathlib.Path(f'.//{dir}//')


with OakCamera() as oak:
    if not directory_path.is_dir():
        directory_path.mkdir()

    color = oak.create_camera('color')
    # left = oak.create_camera('left')
    # right = oak.create_camera('right')
    stereo = oak.create_stereo('800p', fps=30, encode='h264')
    oak.visualize([color, stereo], fps=True)
    oak.callback(color, callback=save_image)
    oak.callback(stereo, callback=save_image_depth)
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
            flag_save_image_depth = True
        elif key == ord('x'):
            flag_save_image = False
            flag_save_image_depth = False

        elif key == ord('e'):  # Switch to auto exposure
            color.control.send_controls({'exposure': {'auto': True}})
