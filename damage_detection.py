#from Functions.Featurespace import Classifier
#from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os


# debug show effect of video handling
def show_frames(bgr, depth):
    imf.resize_image(bgr, 'color_frame', 0.5)

    # Scaling depth data for easier visualisation
    depth_segmentation_value = 256  # maximum value for each channel

    # scale depth frame to fit within 3 channels of bit depth 8
    depth_frame = depth / 8192 * 3 * depth_segmentation_value

    # segment depth image into 3 color channels for better visualisation
    depth_frame_b = np.where(depth_frame > 2 * depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, 2 * depth_segmentation_value),
                             np.zeros_like(depth_frame))
    depth_frame = np.where(depth_frame > 2 * depth_segmentation_value - 1,
                           np.zeros_like(depth_frame),
                           depth_frame)
    depth_frame_g = np.where(depth_frame > depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame_r = np.where(depth_frame > depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
    depth_frame_color = cv2.merge([depth_frame_b, depth_frame_g, depth_frame_r])
    depth_frame_color = depth_frame_color.astype(np.uint8)

    imf.resize_image(depth_frame_color, 'depth_frame', 0.5)
    return


# Function to return the circular area of interest of the image
def aoi_mask():
    return


# Path to video folders
folders_path = 'Test/Class 4'

# Load saved video streams
vid_bgr = cv2.VideoCapture()
vid_bgrd = cv2.VideoCapture()

#features = FeatureSpace()

vid_bgr.open(f"{folders_path}/110_1_bgr.avi")
vid_bgrd.open(f"{folders_path}/110_1_aligned.avi")

while vid_bgr.isOpened():
    # Fetch next frame
    ret, frame_bgr = vid_bgr.read()
    if not ret:
        break
    ret, frame_bgrd_8bit = vid_bgrd.read()
    if not ret:
        break
    # Convert depthmap to 16 bit depth
    frame_bgrd = imf.convert_to_16(frame_bgrd_8bit)

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bgr_mask = imf.threshold_values(frame_hsv)

    show_frames(bgr_mask, frame_bgrd)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
