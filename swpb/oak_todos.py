

def todo_1():
    # run depthai-viewer
    pass


def todo_2():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:

        print(oak.device)

        color = oak.create_camera('color')
        left = oak.create_camera('left')
        right = oak.create_camera('right')
        stereo = oak.create_stereo(left=left, right=right)
        oak.show_graph()

        visualizer = oak.visualize([color, left, right, stereo.out.depth], fps=True, scale=2 / 3, visualizer='opencv')
        print(visualizer.fps)
        oak.start(blocking=True)


def todo_3():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color')
        oak.visualize(color, fps=True, scale=2 / 3)
        oak.start()

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
            elif key == ord('a'):
                color.control.send_controls(
                    {
                        'isp': {
                            'brightness': 5  # setBrightness(), -10..10
                        }
                    }
                )
            elif key == ord('s'):
                color.control.send_controls(
                    {
                        'isp': {
                            'brightness': -5  # setBrightness(), -10..10
                        }
                    }
                )

            elif key == ord('e'):  # Switch to auto exposure
                color.control.send_controls({'exposure': {'auto': True}})


def todo_stereo_preview():
    import cv2

    from depthai_sdk import OakCamera
    from depthai_sdk.components.stereo_component import WLSLevel
    from depthai_sdk.visualize.configs import StereoColor

    with OakCamera() as oak:
        stereo = oak.stereo('800p', fps=30)

        # Configure postprocessing (done on host)
        # stereo.config_postprocessing(colorize=StereoColor.RGBD, colormap=cv2.COLORMAP_MAGMA)
        # stereo.config_wls(wls_level=WLSLevel.MEDIUM)  # WLS filtering, use for smoother results

        oak.visualize([stereo.out.depth], fps=True)
        oak.start(blocking=True)


def todo_stereo_pcl():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.camera('color')
        stereo = oak.create_stereo()
        stereo.config_stereo(align=color)
        pcl = oak.create_pointcloud(stereo=stereo, colorize=color)
        oak.visualize(pcl, visualizer='depthai-viewer')
        oak.start(blocking=True)


def todo_record():
    from depthai_sdk import OakCamera, RecordType

    with OakCamera() as oak:
        left = oak.camera('left', resolution='480p', fps=10, encode=True)
        right = oak.camera('right', resolution='480p', fps=10, encode='MJPEG')

        # Sync & save all (encoded) streams
        oak.record([left.out.encoded, right.out.encoded], './record')
        oak.start()
        frames = 0
        while oak.running():
            if frames > 500:
                break
            else:
                frames += 1
            oak.poll()


def todo_ros():
    from depthai_sdk import OakCamera, RecordType

    with OakCamera() as oak:
        color = oak.create_camera('color', encode='jpeg', fps=30)
        stereo = oak.create_stereo()
        stereo.config_stereo(align=color)

        # DB3 / ROSBAG. ROSBAG doesn't require having ROS installed, while DB3 does.
        record_components = [color.out.encoded, stereo.out.depth]
        oak.record(record_components, 'record', record_type=RecordType.ROSBAG)

        # Visualize only color stream
        oak.visualize(color.out.encoded)
        oak.start(blocking=True)


def todo_mobilenet():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color', encode='mjpeg', fps=10)

        nn = oak.create_nn('mobilenet-ssd', color, spatial=True)  # spatial flag indicates that we want to get spatial data

        oak.visualize([nn.out.encoded], fps=True)  # Display encoded output
        oak.start(blocking=True)


def todo_acc():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        imu = oak.create_imu()
        imu.config_imu(report_rate=400, batch_report_threshold=5)
        # DepthAI viewer should open, and IMU data can be viewed on the right-side panel,
        # under "Stats" tab (right of the "Device Settings" tab).
        oak.visualize(imu.out.main)
        oak.start(blocking=True)


if __name__ == '__main__':
    # todo_1()
    # todo_2()
    # todo_3()
    # todo_stereo_preview()
    # todo_stereo_pcl()
    # todo_record()
    todo_ros()
    # todo_mobilenet()
    # todo_X()
