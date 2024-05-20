


def todo_1_1():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('face-detection-retail-0004', color)
        oak.visualize([nn.out.main, nn.out.passthrough], scale=2 / 3, fps=True)
        oak.start(blocking=True)


def todo_1_2():
    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('vehicle-detection-0202', color)
        oak.visualize([nn, nn.out.passthrough], fps=True)
        # oak.visualize(nn, scale=2 / 3, fps=True)
        oak.start(blocking=True)


def todo_2():
    import cv2

    from depthai_sdk import OakCamera
    from depthai_sdk.classes import DetectionPacket
    from depthai_sdk.visualize.visualizer_helper import FramePosition, VisualizerHelper

    def callback(packet: DetectionPacket):
        visualizer = packet.visualizer
        print('Detections:', packet.img_detections.detections)
        VisualizerHelper.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
        frame = visualizer.draw(packet.frame)
        visualizer.add_text(f'{frame.shape}')
        visualizer.add_mask(img_logo, 0.1)
        cv2.imshow('Visualizer', frame)

    img_logo = cv2.imread('./../_data/s01e02/LOGO_PUT_VISION_LAB_MAIN.png', cv2.IMREAD_COLOR)
    img_logo = cv2.resize(img_logo, (1280, 720))

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color)

        visualizer = oak.visualize([nn], fps=True, callback=callback)
        visualizer.add_mask(img_logo, 0.9)
        oak.start(blocking=True)


def todo_roboblow():
    from depthai_sdk import OakCamera

    # Download & deploy a model from Roboflow universe:
    # # https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters/dataset/6

    with OakCamera() as oak:
        color = oak.create_camera('color')
        model_config = {
            'source': 'roboflow',  # Specify that we are downloading the model from Roboflow
            # 'model': 'plant-village-xvgmc/3', nie działa
            # 'model': 'american-sign-language-letters/6',
            # https://universe.roboflow.com/segmentation-qpzkk/deer-detection-scnrh/model/2
            # 'model': 'deer-detection-scnrh/2',
            # 'model': 'basketball-ls818/15',
            # 'model': 'human-dataset-v2/6', nie działa
            # https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/model/14
            'model': 'rock-paper-scissors-sxsw/14',
            'key': 'Fn24Fm2XgVxxT7zUNo1s'  # Fake API key, replace with your own!
        }
        nn = oak.create_nn(model_config, color)
        oak.visualize(nn, fps=True)
        oak.start(blocking=True)


def todo_22():
    import cv2
    import numpy as np

    from depthai_sdk import OakCamera
    from depthai_sdk.classes import TwoStagePacket
    from depthai_sdk.visualize.configs import TextPosition

    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

    def callback(packet: TwoStagePacket):
        visualizer = packet.visualizer

        for det, rec in zip(packet.detections, packet.nnData):
            emotion_results = np.array(rec.getFirstLayerFp16())
            emotion_name = emotions[np.argmax(emotion_results)]

            visualizer.add_text(emotion_name,
                                bbox=packet.bbox.get_relative_bbox(det.bbox),
                                position=TextPosition.BOTTOM_RIGHT)

        visualizer.draw(packet.frame)
        cv2.imshow(packet.name, packet.frame)

    with OakCamera() as oak:
        color = oak.create_camera('color')
        det = oak.create_nn('face-detection-retail-0004', color)
        # Passthrough is enabled for debugging purposes
        det.config_nn(resize_mode='crop')

        emotion_nn = oak.create_nn('emotions-recognition-retail-0003', input=det)
        # emotion_nn.config_multistage_nn(show_cropped_frames=True) # For debugging

        # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
        # to the callback function (where it will be displayed)
        oak.visualize(emotion_nn, callback=callback, fps=True)
        oak.visualize(det.out.passthrough)
        # oak.show_graph() # Show pipeline graph, no need for now
        oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)


def todo_33():
    import cv2

    from depthai_sdk import OakCamera
    from depthai_sdk.classes import DetectionPacket
    from depthai_sdk.visualize.configs import TextPosition

    def callback(packet: DetectionPacket):
        visualizer = packet.visualizer
        num = len(packet.img_detections.detections)
        print('New msgs! Number of people detected:', num)

        visualizer.add_text(f"Number of people: {num}", position=TextPosition.TOP_MID)
        visualizer.draw(packet.frame)
        cv2.imshow(f'frame {packet.name}', packet.frame)

    with OakCamera(replay='people-images-01') as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('person-detection-retail-0013', color)
        oak.replay.set_fps(0.5)

        oak.visualize(nn, callback=callback)
        # oak.show_graph()
        oak.start(blocking=True)


def todo_yolo():
    from depthai_sdk import OakCamera, ArgsParser
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='model/yolo.json', type=str)
    args = ArgsParser.parseArgs(parser)

    with OakCamera(args=args) as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
        oak.visualize(nn, fps=True, scale=2 / 3)
        oak.visualize(nn.out.passthrough, fps=True)
        oak.start(blocking=True)


def todo_semantic_segmentation():
    from depthai_sdk import OakCamera
    import numpy as np
    from depthai import NNData

    from depthai_sdk import OakCamera
    from depthai_sdk.classes import Detections, SemanticSegmentation

    def decode(nn_data: NNData):
        layer = nn_data.getFirstLayerFp16()
        results = np.array(layer).reshape((1, 1, -1, 7))
        dets = Detections(nn_data)

        for result in results[0][0]:
            if result[2] > 0.5:
                dets.add(result[1], result[2], result[3:])

        return dets

    def callback(packet: DetectionPacket, visualizer: Visualizer):
        detections: Detections = packet.img_detections
        ...

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('_deeplabv3_person', color)
        oak.visualize([color], scale=2 / 3, fps=True)
        oak.start(blocking=True)


if __name__ == '__main__':
    # todo_1_1()
    # todo_1_2()
    # todo_2()
    # todo_3()
    todo_roboblow()
    # todo_yolo()
    # todo_semantic_segmentation()
