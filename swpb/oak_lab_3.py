import cv2

from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import TrackerPacket
import depthai as dai


def callback(packet: TrackerPacket):
    for obj_id, tracklets in packet.tracklets.items():
        if len(tracklets) != 0:
            tracklet = tracklets[-1]
        if tracklet.speed is not None:
            print(f'Speed for object {obj_id}: {tracklet.speed:.02f} m/s, {tracklet.speed_kmph:.02f} km/h, {tracklet.speed_mph:.02f} mph')

    frame = packet.visualizer.draw(packet.decode())
    cv2.imshow('Speed estimation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    stereo = oak.create_stereo('800p')
    stereo.config_stereo(subpixel=False, lr_check=True)

    nn = oak.create_nn('yolov8n_coco_640x352', color, spatial=stereo, tracker=True)
    nn.config_tracker(calculate_speed=True)
    nn.config_spatial(
        bb_scale_factor=0.5,  # Scaling bounding box before averaging the depth in that ROI
        lower_threshold=300,  # Discard depth points below 30cm
        upper_threshold=10000,  # Discard depth pints above 10m
        # Average depth points before calculating X and Y spatial coordinates:
        calc_algo=dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    )

    visualizer = oak.visualize([nn.out.tracker], callback=callback, fps=True)
    visualizer.tracking(show_speed=True).text(auto_scale=True)
    # oak.visualize(nn.out.main, fps=True)
    oak.visualize([nn.out.passthrough, nn.out.spatials])
    oak.start(blocking=True)

    oak.start(blocking=True)