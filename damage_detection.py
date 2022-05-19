from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os
import glob

# If the video should be displayed
TESTING = 0

# Init the classifier
c = Classifier()

# Retrieve the trained data
c.get_classifier()


# Path to video folders
folders_path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Test data'

# Load saved video streams
bgr_src = cv2.VideoCapture()
depth_src = cv2.VideoCapture()

# Create a folder for saving results
save_path = f'/Test Results'
if not os.path.exists(os.getcwd() + save_path):
    os.makedirs(os.getcwd() + save_path)
writer = open(f".{save_path}/test_result.txt", "w")
writer.close() # Just clearing out the text file

# These are all the upper and lower bounds for the thresholding
base_up = [255, 255, 255]
base_low = [70, 37, 30]

blue_up = [124, 132, 150]
blue_low = [84, 37, 61]

scr_up = [129, 103, 59]
scr_low = [70, 21, 32]

# Find the categories in the folder
category_names = os.listdir(folders_path)

# Just in case we are only going through a single folder
if '.avi' in category_names[0]:
    category_names = [folders_path.split('\\')[-1]]
    folders_path = folders_path.split(f'\\{category_names[0]}')[0]

# Define how many frames are checked for damages
frame_counter = 0   # Curr frame counter
max_frames = 5   # Max frames


# Run through the folders one by one
for category in category_names:
    # Find every video in the folder
    list_depth = glob.glob(f'{folders_path}/{category}/**/*aligned*.avi', recursive=True)
    for depth_path in list_depth:
        success_counter = 0 # For determining the category
        fp_counter = 0  # Count the amount of false positives

        class_level = depth_path.split('Class')[1][1]  # Find the classes in the folder

        depth_src = cv2.VideoCapture(depth_path)
        bgr_src = cv2.VideoCapture(depth_path.replace('aligned', 'bgr'))

        # Init the adaptive threshold for depth images
        depth_masker = imf.AdaptiveGRDepthMasker(3, (1500, 1600), (3000, 40000))

        succ_prob = []  # array for saving the BLOBS according to their probability
        false_prob = []
        # Run through the video and wait for input
        while True:
            ret, frame_bgr = bgr_src.read()
            if not ret:  # Break if there are no frames left
                break
            ret, frame_depth_8bit = depth_src.read()
            if not ret:
                break
            frame_depth = imf.convert_to_16(frame_depth_8bit)  # Convert the depth data back into readable data
            frame_counter += 1
            # Blur the image to remove the worst MJPG artifacts
            blur = cv2.GaussianBlur(frame_bgr, (13, 13), cv2.BORDER_DEFAULT)

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

            # TESTING
            if TESTING:
                imf.resize_image(frame_bgr, 'source', 0.4)
                cv2.waitKey(1)
            # Wait for frames to detect flaws in
            # TODO: CLASSIFY THE CLASSES AND CATEGORIES TOGETHER
            if frame_counter >= max_frames:
                #  Start creating the BLOBS and classify them
                depth_mask = depth_masker.return_masks()  # Make the adaptive thresholder return BLOBS of interest

                mask1 = cv2.inRange(frame_hsi, np.asarray(base_low),
                                    np.asarray(base_up))  # Threshold around highlights
                mask2 = cv2.inRange(frame_hsi, np.asarray(blue_low),
                                    np.asarray(blue_up))  # Remove blue, due to the piece of cloth
                mask3 = cv2.inRange(frame_hsi, np.asarray(scr_low),
                                    np.asarray(scr_up))  # Remove blue, due to scratches

                hsi_thresh = cv2.add(mask1, depth_mask)
                hsi_thresh = cv2.subtract(hsi_thresh, mask2)
                hsi_thresh = cv2.subtract(hsi_thresh, mask3)

                binary = imf.open_img(hsi_thresh, 5, 5)  # By opening we remove noise and the edges that are of no interest

                # Find the contours
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if hierarchy is not None:
                    hierarchy = hierarchy[0]  # Unpacking the hierarchy
                    # Go through every contour and save them with their probability
                    for cnt, hrc in zip(contours, hierarchy):
                        if cv2.contourArea(cnt) >= 50:  # Ignore contours that are too small to care for
                            test_feature = FeatureSpace()
                            test_feature.create_features(cnt, 'Test Contour')

                            detected, probability = c.classify(np.asarray(test_feature.get_features()[0]))
                            if probability > 0.85:
                                if detected in category:
                                    success_counter += 1
                                    succ_prob.append(probability)
                                else:  # If it's a false positive save a picture of what went wrong
                                    fp_counter += 1
                                    draw_frame = frame_bgr.copy()
                                    cv2.drawContours(draw_frame, cnt, -1, (0, 0, 255), 3)
                                    folder_name = 'false postives/'+depth_path.split('/')[-1]
                                    if not os.path.exists(f'{os.getcwd()}{save_path}/{folder_name}'):
                                        os.makedirs(f'{os.getcwd()}{save_path}/{folder_name}')
                                    cv2.imwrite(f'{os.getcwd()}{save_path}/{folder_name}/{fp_counter}_fail_{category}{class_level}.png', draw_frame)
                                    false_prob.append(probability)
                    frame_counter = 0  # Reset the frame counter

        print(f'successes = {success_counter} with an average probability of {np.mean(succ_prob):.4f},\n'
              f'false positives = {fp_counter} with an average probability of {np.mean(false_prob):.4f}')
        # Open the file for saving results
        writer = open(f".{save_path}/test_result.txt", "a")
        writer.write(f'{depth_path}\n'
                     f'successes = {success_counter} with an average probability of {np.mean(succ_prob):.4f},\n'
                     f'false positives = {fp_counter} with an average probability of {np.mean(false_prob):.4f}\n')
        writer.close()
