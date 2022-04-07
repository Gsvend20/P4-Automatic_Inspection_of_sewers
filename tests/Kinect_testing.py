import numpy as np
import cv2
import time
import datetime

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Chose operating mode (options: Read, Save)
operating_mode = 'Save'

# Chose how many frames per second should be recorded
operating_fps = 10

# Video saving parameters
frame_codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

# Debug mode (Shows actual fps)
debug_mode = False

# Frame sizes (Not rescaling!)
color_frame_size = (1080, 1920)
depth_frame_size = (424, 512)


def read_frames(desired_fps):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
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

                # reformat the other depth frame format for it to be displayed on screen
                depthframe = depthframe.astype(np.uint8)
                depthframe = np.reshape(depthframe, depth_frame_size)
                depthframe = cv2.cvtColor(depthframe, cv2.COLOR_GRAY2RGB)

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

                i = i + 1

        # End recording if the q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

    return


def save_frames(file_name, desired_fps):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
    video_bgr = cv2.VideoWriter('bgr'+file_name+'.avi', frame_codec, desired_fps, color_frame_size, True)
    video_depth = cv2.VideoWriter('depth'+file_name+'.avi', frame_codec, desired_fps, depth_frame_size, False)
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

                # reformat the other depth frame format for it to be displayed on screen
                depthframe = depthframe.astype(np.uint8)
                depthframe = np.reshape(depthframe, depth_frame_size)
                depthframe = cv2.cvtColor(depthframe, cv2.COLOR_GRAY2RGB)

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

                # Save frames to file
                video_bgr.write(framefullcolour)
                video_depth.write(depthframe)

                i = i + 1

        # End recording if the q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":

    if operating_mode == 'Read':
        # Read and show frames from Kinect
        read_frames(operating_fps)

    if operating_mode == 'Save':
        # Read, show and save frames from Kinect
        current_date = datetime.datetime.now()
        custom_name = input("Enter a file name: ")
        full_file_name = custom_name+"."+str(current_date.month)+"."+str(current_date.day)+"."+str(current_date.hour)+"."+str(current_date.minute)
        save_frames(full_file_name, operating_fps)
