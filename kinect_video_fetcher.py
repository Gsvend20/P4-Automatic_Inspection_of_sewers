import numpy as np
import cv2
import time
from os.path import exists

# import pykinect2.PyKinectRuntime
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import kinect_python_functions as mapper

kinect_runtime = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)

# Debug mode
debug_mode = False


class KinectFrameHandler:
    def __init__(self, kinect_desired_fps):
        # Frame sizes (Not rescaling!)
        self._color_frame_size = (1080, 1920)
        self._depth_frame_size = (424, 512)

        # Start with empty frames until filled by fetched frames (allows showing before first frame received)
        self.color_frame = np.zeros(self._color_frame_size)
        self.depth_frame = np.zeros(self._depth_frame_size)
        self.rgbd_frame = np.zeros(self._color_frame_size)

        # Frame-rate timing for getting data from Kinect
        self.kinect_fps_limit = kinect_desired_fps
        self._start_time = time.time()
        self._old_time = 0
        self._time_index = 0
        self._fps_max, self._fps_min = 0, 100

        # Video codec (works for 8 bit avi) compression mangles depth data
        self._color_frame_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        # Lossless video codec (works for 8 bit avi)
        self._depth_frame_codec = cv2.VideoWriter_fourcc(*"LAGS")

        # Declare video writers
        self._video_color = cv2.VideoWriter()
        self._video_depth = cv2.VideoWriter()
        self._video_rgbd = cv2.VideoWriter()

    # Function to fetch one frame from each feed, frames are saved in class variables
    def fetch_frames(self, kinect_runtime_object):
        while True:
            # TODO: check if elapsed_time needs to be a class wide variable
            elapsed_time = time.time() - self._start_time

            # Limit fps
            if elapsed_time > self._time_index / self.kinect_fps_limit:

                if debug_mode:
                    if self._time_index > 10:  # Only for high time index try evaluating FPS or else divide by 0 errors
                        try:
                            fps = 1 / (elapsed_time - self._old_time)
                            print(fps)
                            if fps > self._fps_max:
                                self._fps_max = fps
                            if fps < self._fps_min:
                                self._fps_min = fps
                        except ZeroDivisionError:
                            print("Divide by zero error")
                            pass

                self._old_time = elapsed_time

                self._time_index += 1

                # Fetch the latest frames from kinect source
                self.color_frame = kinect_runtime_object.get_last_color_frame()
                self.depth_frame = kinect_runtime_object.get_last_depth_frame()
                self.rgbd_frame = mapper.record_rgbd(kinect_runtime_object)

                break

        return

    # Function to transform data from kinect to viewable frames
    def format_frames(self):
        # Re-slice color frame to remove superfluous value at every 4th index
        color_frame = np.reshape(self.color_frame, (2073600, 4))
        color_frame = color_frame[:, 0:3]

        # Extract color frame into bgr channels and reformat to 1080 by 1920 image of bit depth 8
        color_frame_b = color_frame[:, 0]
        color_frame_b = np.reshape(color_frame_b, self._color_frame_size)
        color_frame_g = color_frame[:, 1]
        color_frame_g = np.reshape(color_frame_g, self._color_frame_size)
        color_frame_r = color_frame[:, 2]
        color_frame_r = np.reshape(color_frame_r, self._color_frame_size)
        color_frame = cv2.merge([color_frame_b, color_frame_g, color_frame_r])
        color_frame = cv2.rotate(color_frame, cv2.ROTATE_90_CLOCKWISE)
        self.color_frame = cv2.flip(color_frame, 1)

        # Reformat the depth frame format to be a 1080 by 1920 image of bit depth 16
        depth_frame = np.reshape(self.depth_frame, self._depth_frame_size)
        depth_frame = depth_frame.astype(np.uint16)
        depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_90_CLOCKWISE)
        self.depth_frame = cv2.flip(depth_frame, 1)

        # Reformat the ir frame format to be a 424 by 512 image of bit depth 16
        rgbd_frame = np.reshape(self.rgbd_frame[:,:,1], self._color_frame_size)
        rgbd_frame = rgbd_frame.astype(np.uint16)
        rgbd_frame = cv2.rotate(rgbd_frame, cv2.ROTATE_90_CLOCKWISE)
        self.rgbd_frame = cv2.flip(rgbd_frame, 1)

        return color_frame, depth_frame, rgbd_frame

    # Function to initialise video writers
    def start_saving(self, save_path):
        length = 'test_'
        string_end = ['_bgr', '_depth', '_aligned']

        # Generate new number for each file
        for i in range(1, 1000):
            used_index = 0
            for j in range(0, 3):
                if exists(save_path + length + str(i) + string_end[j] + '.avi'):
                    used_index += 1
                    break
            if used_index == 0:
                file_number = str(i)
                break
            if i == 1000:
                exit('Too many videos in folder')

        # Initialise video writers
        self._video_color.open(save_path + length + file_number + string_end[0] + '.avi',
                               self._color_frame_codec,
                               float(self.kinect_fps_limit),
                               self._color_frame_size)
        self._video_depth.open(save_path + length + file_number + string_end[1] + '.avi',
                               self._depth_frame_codec,
                               float(self.kinect_fps_limit),
                               self._depth_frame_size)
        self._video_rgbd.open(save_path + length + file_number + string_end[2] + '.avi',
                            self._depth_frame_codec,
                            float(self.kinect_fps_limit),
                            self._color_frame_size)

        return

    # Function to save current frame to video stream
    def save_frames(self):
        # Save color frame
        self._video_color.write(self.color_frame)

        # Prepare depth frame for encoding
        depth_hi_bytes = np.right_shift(self.depth_frame, 8).astype('uint8')
        depth_lo_bytes = self.depth_frame.astype('uint8')
        split_depth_frame = cv2.merge([depth_hi_bytes, depth_lo_bytes, np.zeros_like(depth_hi_bytes)])

        # NOTICE: To unpack this into original frame do as follows
        # depth_hi_bytes, depth_lo_bytes, empty = cv2.split(split_depth_frame)
        # depth_frame = depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)

        # Save depth frame
        self._video_depth.write(split_depth_frame)

        # Prepare depth frame for encoding
        rgbd_hi_bytes = np.right_shift(self.rgbd_frame, 8).astype('uint8')
        rgbd_lo_bytes = self.rgbd_frame.astype('uint8')
        split_rgbd_frame = cv2.merge([rgbd_hi_bytes, rgbd_lo_bytes, np.zeros_like(rgbd_hi_bytes)])

        # NOTICE: To unpack this into original frame do as follows
        # ir_hi_bytes, ir_lo_bytes, empty = cv2.split(split_ir_frame)
        # ir_frame = ir_lo_bytes.astype('uint16') + np.left_shift(ir_hi_bytes.astype('uint16'), 8)

        # Save ir frame
        self._video_rgbd.write(split_rgbd_frame)

        return

    # Function to stop video saving
    def stop_saving(self):
        self._video_color.release()
        self._video_depth.release()
        self._video_rgbd.release()

        return

    # Function for showing one frame from each current video feed
    def show_frames(self):
        # Resizing image for better preview
        color_frame = cv2.resize(self.color_frame, (int(self._color_frame_size[0]/2), int(self._color_frame_size[1]/2)))

        # Show colour frames as they are recorded
        cv2.imshow('KINECT color channel', color_frame)

        # Scaling depth data for easier visualisation
        depth_segmentation_value = 256  # maximum value for each channel

        # scale depth frame to fit within 3 channels of bit depth 8
        depth_frame = self.depth_frame / 8192 * 3 * depth_segmentation_value

        # segment depth image into 3 color channels for better visualisation
        depth_frame_b = np.where(depth_frame > 2 * depth_segmentation_value - 1, cv2.subtract(depth_frame, 2 * depth_segmentation_value), np.zeros_like(depth_frame))
        depth_frame = np.where(depth_frame > 2 * depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
        depth_frame_g = np.where(depth_frame > depth_segmentation_value - 1, cv2.subtract(depth_frame, depth_segmentation_value), np.zeros_like(depth_frame))
        depth_frame_r = np.where(depth_frame > depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
        depth_frame_color = cv2.merge([depth_frame_b, depth_frame_g, depth_frame_r])
        depth_frame_color = depth_frame_color.astype(np.uint8)

        # Show depth frames as they are recorded
        cv2.imshow('KINECT depth channel', depth_frame_color)


        # scale depth frame to fit within 3 channels of bit depth 8
        rgbd_frame = self.rgbd_frame / 8192 * 3 * depth_segmentation_value

        # segment depth image into 3 color channels for better visualisation
        rgbd_frame_b = np.where(rgbd_frame > 2 * depth_segmentation_value - 1,
                                 cv2.subtract(rgbd_frame, 2 * depth_segmentation_value), np.zeros_like(rgbd_frame))
        rgbd_frame = np.where(rgbd_frame > 2 * depth_segmentation_value - 1, np.zeros_like(rgbd_frame), rgbd_frame)
        rgbd_frame_g = np.where(rgbd_frame > depth_segmentation_value - 1,
                                 cv2.subtract(rgbd_frame, depth_segmentation_value), np.zeros_like(rgbd_frame))
        rgbd_frame_r = np.where(rgbd_frame > depth_segmentation_value - 1, np.zeros_like(rgbd_frame), rgbd_frame)
        rgbd_frame_color = cv2.merge([rgbd_frame_b, rgbd_frame_g, rgbd_frame_r])
        rgbd_frame_color = rgbd_frame_color.astype(np.uint8)

        # Show aligned frames as they are recorded
        cv2.imshow('KINECT aligned image', cv2.resize(rgbd_frame_color, self._depth_frame_size))

        return


if __name__ == "__main__":
    # Select path for video saving
    path = 'sewer recordings/'

    # Initialise frame handling object
    kinect = KinectFrameHandler(10)  # parameter is frames/second

    # Initialise video writers
    kinect.start_saving(path)

    # Main recording loop
    while True:
        if kinect_runtime.has_new_color_frame() and kinect_runtime.has_new_depth_frame():
            kinect.fetch_frames(kinect_runtime)
            kinect.format_frames()
            kinect.show_frames()
            kinect.save_frames()

        # End program if the q key is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            kinect.stop_saving()
            break
        if key == ord('p'):
            cv2.waitKey(25)
            cv2.waitKey(0)
