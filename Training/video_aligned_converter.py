import cv2
import numpy as np
from os import path
from os import makedirs
from Functions import kinect_python_functions as map_kin
from pykinect2 import PyKinectRuntime
import glob
import time

"""
    This Requires a connection to the Kinect V2, since the _mapper class cannot be accessed without!!!
                        To run the program 
    1: Make sure what you are working with has depth in the name, ex "video_1_depth.avi"
    2: change the "folder" variable to fit with what folders you want searched
    3: Set the "VIDEO" variable to 0 if you are working with images, 1 if videos
    4: Results get the same name as the original, but name changed to aligned, ex "video_1_aligned.avi"
"""

# DEBUGER
DEBUG = 0


class VideoWriter:
    def __init__(self):
        # Frame sizes (Not rescaling!)
        self._frame_size = (1080, 1920)

        # Lossless video codec (works for 8 bit avi)
        self._frame_codec = cv2.VideoWriter_fourcc(*"LAGS")

        # Declare video writers
        self._video = cv2.VideoWriter()

    # Function to initialise video writers
    def start_saving(self, save_path):
        # Initialise video writers
        print(save_path)
        self._video.open(save_path,
                               self._frame_codec,
                               float(30),
                               self._frame_size, True)
        return

    # Function to save current frame to video stream
    def save_frames(self, frame):
        # Prepare depth frame for encoding
        rgbd_hi_bytes = np.right_shift(frame, 8).astype('uint8')
        rgbd_lo_bytes = frame.astype('uint8')
        split_rgbd_frame = cv2.merge([rgbd_hi_bytes[:, :, 0], rgbd_lo_bytes[:, :, 0], np.zeros_like(rgbd_hi_bytes[:, :, 0])])

        # Save frame
        self._video.write(split_rgbd_frame)
        return

    # This is just for displaying the results
    def show_in_color(self, depth_frame):
        depth_segmentation_value = 256  # maximum value for each channel

        # scale depth frame to fit within 3 channels of bit depth 8
        depth_frame = depth_frame / 8192 * 3 * depth_segmentation_value

        # segment depth image into 3 color channels for better visualisation
        depth_frame_b = np.where(depth_frame > 2 * depth_segmentation_value - 1,
                                 cv2.subtract(depth_frame, 2 * depth_segmentation_value), np.zeros_like(depth_frame))
        depth_frame = np.where(depth_frame > 2 * depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
        depth_frame_g = np.where(depth_frame > depth_segmentation_value - 1,
                                 cv2.subtract(depth_frame, depth_segmentation_value), np.zeros_like(depth_frame))
        depth_frame_r = np.where(depth_frame > depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
        depth_frame_color = cv2.merge([depth_frame_b[:, :, 0], depth_frame_g[:, :, 0], depth_frame_r[:, :, 0]])
        depth_frame_color = depth_frame_color.astype(np.uint8)
        cv2.imshow('KINECT aligned image',
                   cv2.resize(cv2.rotate(depth_frame_color, cv2.ROTATE_90_CLOCKWISE), (int(1080 / 2), int(1960 / 2))))
        return



# A function for reversing the 16bit to 8bit changes
def convert_to_16(img):
    if img.shape[2] == 3:
      depth_hi_bytes, depth_lo_bytes, empty = cv2.split(img)
      return depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)
    else:
      print('Image ' + ' is not 3 channel')
      return img

def save_img(path, image):
    rgbd_hi_bytes = np.right_shift(image, 8).astype('uint8')
    rgbd_lo_bytes = image.astype('uint8')
    split_rgbd_image = cv2.merge([rgbd_hi_bytes[:, :, 0], rgbd_lo_bytes[:, :, 0], np.zeros_like(rgbd_hi_bytes[:, :, 0])])
    cv2.imwrite(path,split_rgbd_image)


# Change the folder path accordingly
#folder_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'
folder_path = r'C:\Users\Muku\Documents\P4-Automatic_Inspection_of_sewers\Training\assbrnstest'
# Set to 0 if it's pictures 1 if it's videos
VIDEO = 1


if VIDEO:
    listof_depth = glob.glob(folder_path+'/**/*depth.avi', recursive= True)
    if len(listof_depth) == 0:
        listof_depth = glob.glob(folder_path + '/*depth.avi')
else:
    listof_depth = glob.glob(folder_path + '/**/*_depth_*.png', recursive=True)
    if len(listof_depth) == 0:
        listof_depth = glob.glob(folder_path + '/*_depth_*.png')


# Init the Pykinect class here for optimisation
kinect = PyKinectRuntime.PyKinectRuntime(1)
# The delay here is so that no pictures are run through without the kinect running
time.sleep(10)

# loops through all the videos
for i in range(len(listof_depth)):

    if VIDEO:
        vid = VideoWriter()
        # Find the specific video and start reading the recording
        depth_src = cv2.VideoCapture(listof_depth[i])
    else:
        img_src = cv2.imread(listof_depth[i])

    file_name = listof_depth[i].replace("depth", "aligned")

    # Just to make sure we don't ruin anything
    if path.exists(file_name):
        print("'"+str(i+1)+"'_aligned.avi) already exists\nContinue anyways?\n y/n")
        x = input()
        if x != "y" or x != "yes":
            break
    # Create a folder if it's not there
    if not path.exists(path.dirname(file_name)):
        makedirs(path.dirname(file_name))



    if VIDEO:  # This is if it's a video
        # Start the recording
        # Initialise video writers
        vid.start_saving(file_name)
        print('Saving video '+str(i+1)+'/'+str(len(listof_depth)))
        while True:
            ret, depth_frame = depth_src.read()
            # If ret = 0 there are no more frames, so we break
            if not ret:
                break
            # The Frames read are all bit shifted and rotated, so we reverse that
            depth = cv2.rotate(convert_to_16(depth_frame), cv2.ROTATE_90_COUNTERCLOCKWISE)
            # To convert the depth image the kinect _mapper class is used
            depth_in_color = map_kin.convert_to_depthCamera(kinect, depth)
            # Save the results, but rotated again top match the bgr image
            vid.save_frames(cv2.rotate(depth_in_color, cv2.ROTATE_90_CLOCKWISE))


            # Show aligned frames as they are recorded
            if DEBUG:
                img = vid.show_in_color(depth_in_color)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    break

    else:  # This is if it's an image
        print('Saving picture ' + str(i + 1) + '/' + str(len(listof_depth)))
        # The Frames read are all bit shifted and rotated, so we reverse that
        depth = cv2.rotate(convert_to_16(img_src), cv2.ROTATE_90_COUNTERCLOCKWISE)
        # To convert the depth image the kinect _mapper class is used
        depth_in_color = map_kin.convert_to_depthCamera(kinect, depth)
        # Save the results, but rotated again top match the bgr image
        save_img(file_name, cv2.rotate(depth_in_color, cv2.ROTATE_90_CLOCKWISE))




