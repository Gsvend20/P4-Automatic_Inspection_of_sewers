import os
import cv2
import numpy as np
from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
from Functions import sewer_image_process as imp
import imageio

path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Exam videos'

video_list = os.listdir(path)

# Init classifier
c = Classifier()
# Load trained data
c.get_classifier()

# Train a new classifier
for i in range(0, len(video_list), 3):
    depth_src = cv2.VideoCapture(f'{path}/{video_list[i]}')
    bgr_src = cv2.VideoCapture(f'{path}/{video_list[i+1]}')

    # Init the adaptive threshold for depth images
    depth_masker = imf.AdaptiveGRDepthMasker(3, (1500, 1600), (3000, 40000))

    # For creating the gifs
    rgb_frames = []
    binary_frames = []

    # Run through the video and wait for input
    while True:
        ret, frame_bgr = bgr_src.read()
        if not ret:  # Break if there are no frames left
            break
        ret, frame_depth_8bit = depth_src.read()
        if not ret:
            break
        frame_depth = imf.convert_to_16(frame_depth_8bit)  # Convert the depth data back into readable data

        imp.add_adaptive_frame(frame_depth)
        depth_mask = depth_masker.return_masks()  # Make the adaptive thresholder return BLOBS of interest

        binary = imp.get_binary(frame_bgr, depth_mask)

        # Find the contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        feature_vec=[]
        contour_vec=[]
        # Go through every contour and save them with their probability
        for cnt in contours:
            test_feature = FeatureSpace()
            if cv2.contourArea(cnt) >= 50:  # Ignore contours that are too small to care for
                avg_depth = imf.average_contour_depth(frame_depth, cnt)
                test_feature.create_features(cnt, avg_depth, 'test')
                feature_vec.append(test_feature.get_features()[0])
                contour_vec.append(cnt)

        predict = c._classifier.predict(feature_vec)
        #print(f'feature: {feature_vec}')
        prob = np.max(c._classifier.predict_proba(feature_vec), axis=1)
        pass_index = np.array(prob >= 0.80)

        contour_vec = np.array(contour_vec)[pass_index]
        if len(contour_vec) > 0:
            cv2.drawContours(frame_bgr, contour_vec, -1, (10, 200, 10), -1)
            cv2.drawContours(frame_bgr, contour_vec, -1, (255, 255, 255), 2)

        # Create and show video
        imf.resize_image(frame_bgr, 'final product', 0.4)
        rgb_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        imf.resize_image(imf.depth_to_display(frame_depth), 'depth', 0.4)
        imf.resize_image(binary, 'binary', 0.4)
        binary_frames.append(binary)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    video_name = video_list[i].split('_')[1]
    print(f'Working on creating gif with\n {len(rgb_frames)} rgb frames and {len(binary_frames)} binary frames')
    with imageio.get_writer(f'{video_name}_rgb.gif', mode="I", duration=0.2) as writer:
        for idx, frame in enumerate(rgb_frames):
            print("Adding frame to GIF file: ", idx + 1)
            for i in range(3):
                writer.append_data(frame)

    with imageio.get_writer(f'{video_name}_binary.gif', mode="I", duration=0.2) as writer:
        for idx, frame in enumerate(binary_frames):
            print("Adding frame to GIF file: ", idx + 1)
            writer.append_data(frame)


