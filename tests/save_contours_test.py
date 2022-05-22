import cv2
from Functions import imgproc_func as imf
from Functions import sewer_image_process as imp
from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
import numpy as np
import os
import glob
import pickle

# Path to video folders
folders_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Test data\Videos'


# Init the classifier
c = Classifier()
# Load the trained data
c.get_classifier()

# Load saved video streams
bgr_src = cv2.VideoCapture()
depth_src = cv2.VideoCapture()

# Find the categories in the folder
category_names = os.listdir(folders_path)

# Define how many frames are checked for damages
frame_counter = 0   # Curr frame counter
save_counter = 5
max_frames = 5   # Max frames

# For saving the video as features only
video_contours = []  # [Vid_1_name, Category, Class_level, [[F_vec_1]..[F_vec_n]] ...
                        # Vid_n_name, Category, Class_level, [[F_vec_1]..[F_vec_n]]]

# Run through the folders one by one
for category in category_names:
    print(f'Started Working on {category}')
    # Find every video in the folder
    list_depth = glob.glob(f'{folders_path}/{category}/**/*aligned*.avi', recursive=True)
    for depth_path in list_depth:

        success_counter = 0 # For determining the category
        fp_counter = 0  # Count the amount of false positives
        succ_prob = []  # array for saving the BLOBS according to their probability
        false_prob = []
        # Feature vector
        feature_vec = []  # [features]

        class_level = depth_path.split('Class')[1][1]  # Find the classes in the folder
        vid_name = depth_path.replace('aligned', 'bgr').split('\\')[-1]

        depth_src = cv2.VideoCapture(depth_path)
        bgr_src = cv2.VideoCapture(depth_path.replace('aligned', 'bgr'))

        # Put everything in the list
        video_contours.append(vid_name)
        video_contours.append(category)
        video_contours.append(class_level)

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
            frame_counter += 1
            frame_depth = imf.convert_to_16(frame_depth_8bit)  # Convert the depth data back into readable data

            imp.add_adaptive_frame(frame_depth, depth_masker)

            #imf.resize_image(frame_bgr, 'source', 0.4)
            #cv2.waitKey(1)

            # Wait for frames to detect flaws in
            if frame_counter >= save_counter:
                #  Start creating the BLOBS and classify them
                depth_mask = depth_masker.return_masks()  # Make the adaptive thresholder return BLOBS of interest

                binary = imp.get_binary(frame_bgr, depth_mask)

                # Find the contours
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


                # Go through every contour and save them with their probability
                for cnt in contours:
                    test_feature = FeatureSpace()
                    if cv2.contourArea(cnt) >= 50:  # Ignore contours that are too small to care for
                        avg_depth = imf.average_contour_depth(frame_depth, cnt)
                        test_feature.create_features(cnt, avg_depth, 'test')
                    feature_vec.append([test_feature.get_features()])
                save_counter += max_frames
        print(f'finished video {list_depth.index(depth_path)+1}/{len(list_depth)}')
        video_contours.append(feature_vec)


print(f'name of vids = {video_contours[0::4]}\n Category = {video_contours[1::4]}\n Class = {video_contours[2::4]}')

filename = input('Please enter the pkl file name\n')
parent = os.path.dirname(os.getcwd())
file_path = f'{parent}\\classifiers\\{filename}.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(video_contours, file)

for i in range(0, len(video_contours), 4):
    success_counter = 0
    fp_counter = 0
    succ_prob = []
    false_prob = []

    category = video_contours[i+1]
    class_level = video_contours[i+2]
    feature_list = video_contours[i+3]
    for n in range(0, len(feature_list), 2):
        features = feature_list[n][0]
        frame = feature_list[n+1]
        for feature in features:
            detected, probability = c.classify(np.asarray(feature))
            if probability > 0.85:
                if detected in category:
                    success_counter += 1
                    succ_prob.append(probability)
                else:  # If it's a false positive save a picture of what went wrong
                    fp_counter += 1
                    false_prob.append(probability)


    print(f'successes = {success_counter} with an average probability of {np.mean(succ_prob):.4f},\n'
    f'false positives = {fp_counter} with an average probability of {np.mean(false_prob):.4f}')

with open(file_path, 'rb') as file:
    video_contours = pickle.load(file)

for i in range(0, len(video_contours), 4):
    success_counter = 0
    fp_counter = 0
    succ_prob = []
    false_prob = []

    category = video_contours[i+1]
    class_level = video_contours[i+2]
    feature_list = video_contours[i+3]
    for n in range(len(feature_list)):
        features = feature_list[n][0]
        for feature in features:
            detected, probability = c.classify(np.asarray(feature))
            if probability > 0.85:
                if detected in category:
                    success_counter += 1
                    succ_prob.append(probability)
                else:  # If it's a false positive save a picture of what went wrong
                    fp_counter += 1
                    false_prob.append(probability)

    print(f'successes = {success_counter} with an average probability of {np.mean(succ_prob):.4f},\n'
          f'false positives = {fp_counter} with an average probability of {np.mean(false_prob):.4f}')