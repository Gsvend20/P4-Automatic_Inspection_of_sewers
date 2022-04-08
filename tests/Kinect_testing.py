import numpy as np
import cv2
import time
import datetime
from os.path import exists
from os import remove

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Chose operating mode (options: Read, Save)
operating_mode = 'Save'

# Chose how many frames per second should be recorded
operating_fps = 30

# Scaling factors for depth and ir pixel values
depth_value_scale = 256/8192  # 8191 is maximum depth pixel value and each value maps to 1mm
ir_value_scale = 256/65536 #65535 is maximum value

# Video saving parameters
file_extension = ".avi"
save_location = "video_dump/"
frame_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

# Debug mode
debug_mode = False


def read_frames(desired_fps):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)

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
                depthframe = depthframe * depth_value_scale

                # Reshape ir data to frame format
                irframe = np.reshape(irframe, ir_frame_size)

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
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)

    # Frame sizes (Not rescaling!)
    color_frame_size = (1080, 1920)
    depth_frame_size = (424, 512)
    ir_frame_size = (424, 512)

    # Initialise video writers
    video_bgr = cv2.VideoWriter(save_location+'bgr_'+file_name, frame_codec, float(desired_fps), (1920, 1080))
    video_depth = cv2.VideoWriter(save_location+'depth_'+file_name, frame_codec, float(desired_fps), (512, 424), False)
    video_ir = cv2.VideoWriter(save_location+'ir_' + file_name, frame_codec, float(desired_fps), (512, 424), False)

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
                depthframe = depthframe * depth_value_scale
                depthframe = depthframe.astype(np.uint8)

                # Reshape ir data to frame format
                irframe = np.reshape(irframe, ir_frame_size)
                irframe = irframe * ir_value_scale
                irframe = irframe.astype(np.uint8)

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
    while True:
        if operating_mode == 'Read':
            # Read and show frames from Kinect
            read_frames(operating_fps)

        if operating_mode == 'Save':
            # Read, show and save frames from Kinect
            current_date = datetime.datetime.now()
            if not debug_mode:
                custom_name = input("Enter a file name: ")
                full_file_name = custom_name+"."+str(current_date.month)+"."+str(current_date.day)+"."+str(current_date.hour)+"."+str(current_date.minute)+file_extension
            else:
                full_file_name = 'debug'+file_extension
                if exists('bgr_' + full_file_name):
                    remove('bgr_' + full_file_name)
                    print('removed old test bgr file')
                if exists('depth_' + full_file_name):
                    remove('depth_' + full_file_name)
                    print('removed old test depth file')

            save_frames(full_file_name, operating_fps)

            # End program if the p key is pressed
            if cv2.waitKey(1) == ord('p'):
                break
