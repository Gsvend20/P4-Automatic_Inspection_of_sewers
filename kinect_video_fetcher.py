import numpy as np
import cv2
import time
import datetime
from os.path import exists
from os import remove

# import pykinect2.PyKinectRuntime
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

kinect_runtime = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)

# Chose operating mode (options: Read, Save)
operating_mode = 'Save'

# Chose how many frames per second should be recorded
operating_fps = 30

# Scaling factors for depth and ir pixel values
# depth_value_scale = 3*256/8192  # 8191 is maximum depth pixel value and each value maps to 1mm
# ir_value_scale = 256/65536  # 65535 is maximum value

# Debug mode
debug_mode = False


class KinectFrameHandler:
    def __init__(self, kinect_desired_fps):
        # Frame sizes (Not rescaling!)
        self._color_frame_size = (1080, 1920)
        self._depth_frame_size = (424, 512)
        self._ir_frame_size = (424, 512)

        # Start with empty frames until filled by fetched frames (allows showing before first frame received)
        self.color_frame = np.zeros(self._color_frame_size)
        self.depth_frame = np.zeros(self._depth_frame_size)
        self.ir_frame = np.zeros(self._ir_frame_size)

        # Frame-rate timing for getting data from Kinect
        self.kinect_fps_limit = kinect_desired_fps
        self._start_time = time.time()
        self._old_time = 0
        self._time_index = 0
        self._fps_max, self._fps_min = 0, 100

        # Video codec (works for 8 bit)
        self._frame_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        # Initialise video writers
        self._video_color = cv2.VideoWriter()
        self._video_depth = cv2.VideoWriter()
        self._video_ir = cv2.VideoWriter()

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
                self.ir_frame = kinect_runtime_object.get_last_infrared_frame()
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
        self.color_frame = cv2.merge([color_frame_b, color_frame_g, color_frame_r])

        # Reformat the depth frame format to be a 424 by 512 image of bit depth 16
        depth_frame = np.reshape(self.depth_frame, self._depth_frame_size)
        self.depth_frame = depth_frame.astype(np.uint16)

        # Reformat the ir frame format to be a 424 by 512 image of bit depth 16
        ir_frame = np.reshape(self.ir_frame, self._ir_frame_size)
        self.ir_frame = ir_frame.astype(np.uint16)  # TODO: is this even possible? or necessary?

        # TODO: Rotate all frames
        return color_frame, depth_frame, ir_frame

    # Function to initialise video writers
    def start_saving(self, save_path):
        # Initialise video writers
        self._video_color.open(save_path + '_bgr.avi',
                               self._frame_codec,
                               float(self.kinect_fps_limit),
                               (1920, 1080))
        self._video_depth.open(save_path + '_depth.avi',
                               self._frame_codec,
                               float(self.kinect_fps_limit),
                               (512, 424))
        self._video_ir.open(save_path + '_ir.avi',
                            self._frame_codec,
                            float(self.kinect_fps_limit),
                            (512, 424))
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
        ir_hi_bytes = np.right_shift(self.ir_frame, 8).astype('uint8')
        ir_lo_bytes = self.ir_frame.astype('uint8')
        split_ir_frame = cv2.merge([ir_hi_bytes, ir_lo_bytes, np.zeros_like(ir_hi_bytes)])

        # NOTICE: To unpack this into original frame do as follows
        # depth_hi_bytes, depth_lo_bytes, empty = cv2.split(split_depth_frame)
        # depth_frame = depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)

        # Save ir frame
        self._video_ir.write(split_ir_frame)
        return

    # Function to stop video saving
    def stop_saving(self):
        self._video_color.release()
        self._video_depth.release()
        self._video_ir.release()
        return

    # Function for showing one frame from each current video feed
    def show_frames(self):
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

        # Show colour frames as they are recorded
        cv2.imshow('KINECT color channel', self.color_frame)

        # Show depth frames as they are recorded
        cv2.imshow('KINECT depth channel', depth_frame_color)

        # Show depth frames as they are recorded
        cv2.imshow('KINECT ir channel', self.ir_frame)
        return


def read_frames(desired_fps):
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)

    # Frame sizes (Not rescaling!)
    color_frame_size = (1080, 1920)
    depth_frame_size = (424, 512)
    ir_frame_size = (424, 512)

    # Framerate timing for getting information from Kinect
    start_time = time.time()
    old_time = 0
    i = 0
    fps_max = 0
    fps_min = 100

    # Actual recording loop, exit by pressing escape to close the pop-up window
    while True:
        if kinect.has_new_depth_frame() and kinect.has_new_color_frame():
            elapsed_time = time.time() - start_time

            # Limit fps
            if elapsed_time > i / desired_fps:

                if debug_mode:
                    # Only for high i try evalutaing FPS or else you get some divide by 0 errors
                    if i > 10:
                        try:
                            fps = 1 / (elapsed_time - old_time)
                            print(fps)
                            if fps > fps_max:
                                fps_max = fps
                            if fps < fps_min:
                                fps_min = fps
                        except ZeroDivisionError:
                            print("Divide by zero error")
                            pass

                old_time = elapsed_time

                # Read kinect colour and depth data
                depthframe = kinect.get_last_depth_frame()
                colourframe = kinect.get_last_color_frame()
                irframe = kinect.get_last_infrared_frame()

                # Reformat the other depth frame format for it to be displayed on screen
                depthframe = np.reshape(depthframe, depth_frame_size)
                depthframe = depthframe.astype(np.uint16)
                # depthframe = depthframe * depth_value_scale

                # Segment depth image into
                # depth_segmentation_value = int(depth_value_scale * 8192 / 3)
                # depthframeB = np.where(depthframe > 2 * depth_segmentation_value - 1, cv2.subtract(depthframe, 2 * depth_segmentation_value), np.zeros_like(depthframe))
                # depthframe = np.where(depthframe > 2 * depth_segmentation_value - 1, np.zeros_like(depthframe), depthframe)
                # depthframeG = np.where(depthframe > depth_segmentation_value - 1, cv2.subtract(depthframe, depth_segmentation_value), np.zeros_like(depthframe))
                # depthframeR = np.where(depthframe > depth_segmentation_value - 1, np.zeros_like(depthframe), depthframe)
                # depthframe = cv2.merge([depthframeB, depthframeG, depthframeR])
                # depthframe = depthframe.astype(np.uint8)

                # Reshape ir data to frame format
                irframe = np.reshape(irframe, ir_frame_size)
                # irframe = irframe * ir_value_scale
                irframe = irframe.astype(np.uint16)

                # Reslice to remove every 4th colour value, which is superfluous
                colourframe = np.reshape(colourframe, (2073600, 4))
                colourframe = colourframe[:, 0:3]

                # extract then combine the RBG data
                colourframeR = colourframe[:, 0]
                colourframeR = np.reshape(colourframeR, color_frame_size)
                colourframeG = colourframe[:, 1]
                colourframeG = np.reshape(colourframeG, color_frame_size)
                colourframeB = colourframe[:, 2]
                colourframeB = np.reshape(colourframeB, color_frame_size)
                framefullcolour = cv2.merge([colourframeR, colourframeG, colourframeB])

                # Show colour frames as they are recorded
                cv2.imshow('Recording KINECT Video Stream COLOUR', framefullcolour)

                # Show depth frames as they are recorded
                cv2.imshow('Recording KINECT Video Stream DEPTH', depthframe)

                # Show depth frames as they are recorded
                cv2.imshow('Recording KINECT Video Stream IR', irframe)

                i = i + 1

        # End recording if the q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

    return


def save_frames(file_name, desired_fps):
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)

    # Frame sizes (Not rescaling!)
    color_frame_size = (1080, 1920)
    depth_frame_size = (424, 512)
    ir_frame_size = (424, 512)

    # Initialise video writers
    video_bgr = cv2.VideoWriter(save_location + 'bgr_' + file_name, frame_codec, float(desired_fps), (1920, 1080))
    video_depth = cv2.VideoWriter(save_location + 'depth_' + file_name, frame_codec, float(desired_fps), (512, 424),
                                  False)
    video_ir = cv2.VideoWriter(save_location + 'ir_' + file_name, frame_codec, float(desired_fps), (512, 424), False)

    # Framerate timing for getting information from Kinect
    start_time = time.time()
    old_time = 0
    i = 0
    fps_max = 0
    fps_min = 100

    # Actual recording loop, exit by pressing escape to close the pop-up window
    while True:
        if kinect.has_new_depth_frame() and kinect.has_new_color_frame():
            elapsed_time = time.time() - start_time

            # Limit fps
            if elapsed_time > i / desired_fps:

                if debug_mode:
                    # Only for high i try evalutaing FPS or else you get some divide by 0 errors
                    if i > 10:
                        try:
                            fps = 1 / (elapsed_time - old_time)
                            print(fps)
                            if fps > fps_max:
                                fps_max = fps
                            if fps < fps_min:
                                fps_min = fps
                        except ZeroDivisionError:
                            print("Divide by zero error")
                            pass

                old_time = elapsed_time

                # read kinect colour and depth data
                depthframe = kinect.get_last_depth_frame()
                colourframe = kinect.get_last_color_frame()
                irframe = kinect.get_last_infrared_frame()

                # reformat the other depth frame format for it to be displayed on screen
                depthframe = np.reshape(depthframe, depth_frame_size)
                # depthframe = depthframe.astype(np.uint16)
                # depthframe = depthframe * depth_value_scale

                # Segment depth image into
                # depth_segmentation_value = int(depth_value_scale * 8192 / 3)
                # depthframeB = np.where(depthframe > 2 * depth_segmentation_value - 1, cv2.subtract(depthframe, 2 * depth_segmentation_value), np.zeros_like(depthframe))
                # depthframe = np.where(depthframe > 2 * depth_segmentation_value - 1, np.zeros_like(depthframe), depthframe)
                # depthframeG = np.where(depthframe > depth_segmentation_value - 1, cv2.subtract(depthframe, depth_segmentation_value), np.zeros_like(depthframe))
                # depthframeR = np.where(depthframe > depth_segmentation_value - 1, np.zeros_like(depthframe), depthframe)
                # depthframe = cv2.merge([depthframeB, depthframeG, depthframeR])
                # depthframe = depthframe.astype(np.uint8)

                # Reshape ir data to frame format
                irframe = np.reshape(irframe, ir_frame_size)
                # irframe = irframe * ir_value_scale
                # irframe = irframe.astype(np.uint16)

                # Reslice to remove every 4th colour value, which is superfluous
                colourframe = np.reshape(colourframe, (2073600, 4))
                colourframe = colourframe[:, 0:3]

                # extract then combine the RBG data
                colourframeR = colourframe[:, 0]
                colourframeR = np.reshape(colourframeR, color_frame_size)
                colourframeG = colourframe[:, 1]
                colourframeG = np.reshape(colourframeG, color_frame_size)
                colourframeB = colourframe[:, 2]
                colourframeB = np.reshape(colourframeB, color_frame_size)
                framefullcolour = cv2.merge([colourframeR, colourframeG, colourframeB])

                # Show depth frames as they are recorded
                cv2.imshow('Recording KINECT Video Stream DEPTH', depthframe)

                # Show colour frames as they are recorded
                cv2.imshow('Recording KINECT Video Stream COLOUR', framefullcolour)

                # Show depth frames as they are recorded
                cv2.imshow('Recording KINECT Video Stream IR', irframe)

                # Save frames to file
                video_bgr.write(framefullcolour)
                video_depth.write(depthframe)
                video_ir.write(irframe)
                if debug_mode:
                    print('frame ' + str(i) + ' saved')

                i = i + 1

        # End recording if the q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    video_bgr.release()
    video_depth.release()
    video_ir.release()

    return


if __name__ == "__main__":
    # Select path for video saving
    path = 'sewer recordings/'

    # Initialise frame handling object
    kinect = KinectFrameHandler(30)
    kinect.start_saving(path)

    # Main recording loop
    while True:
        if kinect_runtime.has_new_color_frame() and kinect_runtime.has_new_depth_frame():
            kinect.fetch_frames(kinect_runtime)
            kinect.format_frames()
            kinect.show_frames()
            kinect.save_frames()

        # End program if the q key is pressed
        if cv2.waitKey(1) == ord('q'):
            kinect.stop_saving()
            break
