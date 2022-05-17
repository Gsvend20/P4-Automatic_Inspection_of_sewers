from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os
import glob

# Init the classifier
c = Classifier()

# Retrieve the trained data
c.get_classifier()


# Path to video folders
folders_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\Videos'

# Load saved video streams
bgr_src = cv2.VideoCapture()
depth_src = cv2.VideoCapture()

# These are all the upper and lower bounds for the thresholding
base_up = [255, 255, 255]
base_low = [70, 37, 30]

blue_up = [124, 119, 148]
blue_low = [84, 37, 61]

scr_up = [129, 103, 59]
scr_low = [70, 21, 32]

roe_up = [107, 114, 255]
roe_low = [72, 28, 150]

# Find the categories in the folder
category_names = os.listdir(folders_path)

# Just in case we are only going through a single folder
if '.avi' in category_names[0]:
    category_names = [folders_path.split('\\')[-1]]
    folders_path = folders_path.split(f'\\{category_names[0]}')[0]

# Run through the folders one by one
for category in category_names:
    # Find every video in the folder
    list_depth = glob.glob(f'{folders_path}/{category}/**/*aligned*.avi', recursive=True)
    for depth_path in list_depth:

        class_level = depth_path.split('Class')[1][1]  # Find the classes in the folder

        depth_src = cv2.VideoCapture(depth_path)
        bgr_src = cv2.VideoCapture(depth_path.replace('aligned', 'bgr'))

        # Init the adaptive threshold for depth images
        depth_masker = imf.AdaptiveGRDepthMasker(3, (1500, 1600), (3000, 40000))

        # Run through the video and wait for input
        while True:
            ret, frame_bgr = bgr_src.read()
            if not ret:  # Break if there are no frames left
                break
            ret, frame_depth_8bit = depth_src.read()
            if not ret:
                break
            frame_depth = imf.convert_to_16(frame_depth_8bit)  # Convert the depth data back into readable data

            # Blur the image to remove the worst MJPG artifacts
            blur = cv2.medianBlur(frame_bgr, 13)

            # Convert the image into desired Color spaces
            frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            # Adaptive thresholding
            # Generate area of interest from pipe depth data, by finding the end of the pipe
            aoi_end = cv2.inRange(frame_depth, int(np.max(frame_depth) - 100), int(np.max(frame_depth)))
            # Then the front of the pipe is extracted
            aoi_pipe = cv2.inRange(frame_depth, 600, int(np.max(frame_depth) - 100))
            cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Make a mask out of both
            pipe_mask = np.zeros_like(frame_depth).astype('uint8')
            pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
            bg_mask = cv2.subtract(pipe_mask, aoi_end)
            bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
            # use the mask to generate an area of interest in the depth data
            fg_d_frame = cv2.bitwise_and(frame_depth, frame_depth, mask=bg_mask)
            # start the adaptive thresholding
            depth_masker.add_image(fg_d_frame)

            # Wait for frames to detect flaws in
            # TODO: CLASSIFY THE CLASSES AND CATEGORIES TOGETHER
