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

# Video saving parameters
file_extension = ".avi"
save_location = "video_dump/"
# working video encoder
frame_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

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

        # Declare video writers
        self.video_bgr = cv2.VideoWriter
        self.video_depth = cv2.VideoWriter
        self.video_ir = cv2.VideoWriter

        # Frame-rate timing for getting data from Kinect
        self.kinect_fps_limit = kinect_desired_fps
        self._start_time = time.time()
        self._old_time = 0
        self._time_index = 0
        self._fps_max, self._fps_min = 0, 100

    # Function to fetch one frame from each feed, frames are saved in class variables
    def fetch_frames(self, kinect_runtime_object):
        while True:
            # TODO: check if elapsed_time needs to be a class wide variable
            elapsed_time = time.time() - self._start_time

            # Limit fps
            if elapsed_time > self._time_index / self.kinect_fps_limit:

                if debug_mode:
                    if self._time_index > 10:  # Only for high time index try evalutaing FPS or else divide by 0 errors
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

    # Function to start up video saving
    def open_save_location(self, save_path, frame_codec):
        # Initialise video writers
        self.video_bgr.open(save_path + '_bgr', frame_codec, float(self.kinect_fps_limit), (1920, 1080))
        self.video_depth.open(save_path + '_depth', frame_codec, float(self.kinect_fps_limit), (512, 424))
        self.video_ir.open(save_path + '_ir', frame_codec, float(self.kinect_fps_limit), (512, 424), False)
        return

    # Function to stop video saving
    def release_save_location(self):
        self.video_bgr.release()
        self.video_depth.release()
        self.video_ir.release()
        return

    def save_frames(self):

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
    #kinect_runtime = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
    kinect = KinectFrameHandler(30)  # Initialise class

    while True:
        if kinect_runtime.has_new_color_frame() and kinect_runtime.has_new_depth_frame():
            kinect.fetch_frames(kinect_runtime)
            kinect.format_frames()
            kinect.show_frames()

        if cv2.waitKey(1) == ord('p'):
            break

        #if operating_mode == 'Read':
        #    # Read and show frames from Kinect
        #    read_frames(operating_fps)
        #    exit(0)
        #
        #if operating_mode == 'Save':
        #    # Read, show and save frames from Kinect
        #    current_date = datetime.datetime.now()
        #    if not debug_mode:
        #        custom_name = input("Enter a file name: ")
        #        full_file_name = custom_name + "." + str(current_date.month) + "." + str(current_date.day) + "." + str(
        #            current_date.hour) + "." + str(current_date.minute) + file_extension
        #    else:
        #        full_file_name = 'debug' + file_extension
        #        if exists('bgr_' + full_file_name):
        #            remove('bgr_' + full_file_name)
        #            print('removed old test bgr file')
        #        if exists('depth_' + full_file_name):
        #            remove('depth_' + full_file_name)
        #            print('removed old test depth file')
        #
        #    save_frames(full_file_name, operating_fps)
        #
        #    # End program if the p key is pressed

