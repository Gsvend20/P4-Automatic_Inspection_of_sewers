import cv2
import numpy as np


class VideoLoader:
    def __init__(self):
        self._color_capture = cv2.VideoCapture()
        self._depth_capture = cv2.VideoCapture()
        self._ir_capture = cv2.VideoCapture()

    def load_videos(self, path):
        self._color_capture.open(path + '_bgr.avi')
        self._depth_capture.open(path + '_depth.avi')
        self._ir_capture.open(path + '_ir.avi')

        if not self._color_capture.isOpened():
            print('Could not open color video (wrong path?)')

        if not self._depth_capture.isOpened():
            print('Could not open depth video (wrong path?)')

        if not self._ir_capture.isOpened():
            print('Could not open ir video (wrong path?)')

    def get_next_frames(self):
        # Return variable to inform of successful import of frames
        successful_read = 1

        # Fetch color frame
        ret, color_frame = self._color_capture.read()
        if not ret:
            print('Could not get color frame (end of video?)')
            successful_read = 0

        # Fetch depth frame and merge its channels to form 16-bit image
        ret, depth_frame = self._depth_capture.read()
        if not ret:
            print('Could not get depth frame (end of video?)')
            successful_read = 0
        else:
            depth_hi_bytes, depth_lo_bytes, empty = cv2.split(depth_frame)
            depth_frame = depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)

        # Fetch ir frame and merge its channels to form 16-bit image
        ret, ir_frame = self._ir_capture.read()
        if not ret:
            print('Could not get ir frame (end of video?)')
            successful_read = 0
        else:
            ir_hi_bytes, ir_lo_bytes, empty = cv2.split(ir_frame)
            ir_frame = ir_lo_bytes.astype('uint16') + np.left_shift(ir_hi_bytes.astype('uint16'), 8)

        return color_frame, depth_frame, ir_frame, successful_read
