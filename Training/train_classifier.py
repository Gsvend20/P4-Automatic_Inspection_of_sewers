import cv2
from Functions import imgproc_func as imf
from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions.Featurespace import find_annodir
import numpy as np
import os
import glob

def find_largest(contours, hierarchy):
    largest_a = 0
    for n in range(len(hierarchy)):
        a = cv2.contourArea(contours[n])
        if largest_a < a:
            largest_a = a
            largest_no = n
    return contours[largest_no], hierarchy[largest_no]

# TODO: FIX FS false positives
# TODO: Get better training data

"""

The path folder should go into a folder holding every annotation category, with the folder structure like this
    ./category/class/*.png
eg. ./AF/Class 1/9/rgbMasks/*.png

"""

path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'
# Definitions used for the sklearn classifier
feature_space = []
label_list = []

# Init the classifier
c = Classifier()

# Run test area 0 = no, 1 = yes
TESTAREA = 1

class_name, anotations = find_annodir(path)
for category, img_folders in zip(class_name, anotations):
    f = FeatureSpace()
    print(f"Importing {category}")
    for img_path in img_folders:
        # get the name of the folder just after the category
        class_level = img_path.split(category+'\\')[-1].split('\\')[0]

        # read through all the pictures
        img = cv2.imread(img_path, 0)
        if img is not None and np.mean(img) > 0:
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt, hir = find_largest(contours, hierarchy[0])
            if cv2.contourArea(cnt) > 0:
                f.create_features(cnt, hir[2] != -1, f"{category}")

    for feature in f.get_features():
        feature_space.append(feature)
        label_list.append(category)

print('Training the classifier')
c.prepare_training_data(feature_space, label_list)
c.train_classifier()
print('done importing')

"""
TESTING AREA, TOO CHECK HOW WELL IT WORKS
"""

for category in class_name:
    depth_paths = glob.glob(path.replace('\\', '/') + '/' + category + '/**/*aligned*.png', recursive=True)
    for i in range(120,125):
        depth_path = depth_paths[i]
        bgr_path = depth_path.replace('aligned','bgr')

        depth_img = imf.convert_to_16(cv2.imread(depth_path))
        bgr_img = cv2.imread(bgr_path)
        draw_frame = np.zeros_like(bgr_img)

        blur = cv2.medianBlur(bgr_img, 13)

        frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

        # Generate area of interest from pipe depth data
        aoi_end = cv2.inRange(depth_img, int(np.max(depth_img) - 100), int(np.max(depth_img)))
        aoi_pipe = cv2.inRange(depth_img, 600, int(np.max(depth_img) - 100))
        cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pipe_mask = np.zeros_like(depth_img).astype('uint8')
        pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
        bg_mask = cv2.subtract(pipe_mask, aoi_end)
        bg_mask = imf.open_img(bg_mask, 21, 21)
        bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

        hls_uppervalues = [255, 255, 255]
        hls_lowervalues = [70, 37, 30]

        blue_uppervalues = [124, 119, 148]
        blue_lowervalues = [84, 37, 61]

        scr_uppervalues = [129, 94, 56]
        scr_lowervalues = [70, 16, 34]

        mask1 = cv2.inRange(frame_hsi, np.asarray(hls_lowervalues),
                            np.asarray(hls_uppervalues))  # Threshold around highlights
        mask2 = cv2.inRange(frame_hsi, np.asarray(blue_lowervalues),
                            np.asarray(blue_uppervalues))  # Remove blue, due to the piece of cloth
        mask3 = cv2.inRange(frame_hsi, np.asarray(scr_lowervalues),
                            np.asarray(scr_uppervalues))  # Remove blue, due to scratches

        hsi_thresh = cv2.subtract(mask1, mask2)
        hsi_thresh = cv2.subtract(hsi_thresh, mask3)

        bin = imf.open_img(hsi_thresh, 3, 3)

        contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if hierarchy is not None:
            hierarchy = hierarchy[0]  # [[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]
            draw_frame = bgr_img.copy()
            for cnt, hrc in zip(contours, hierarchy):
                if cv2.contourArea(cnt) >= 50:
                    mask = np.zeros(bin.shape, np.uint8)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    rgbd_aoi = cv2.bitwise_and(depth_img, depth_img, mask=mask)
                    # mask_display = imf.depth_to_display(rgbd_aoi)
                    # imf.resize_image(mask_display, 'mask', 0.5)
                    # cv2.waitKey(1)

                    test_feature = FeatureSpace()
                    test_feature.create_features(cnt, np.array(hrc[2] != -1), 'test')

                    detected, probability = c.classify(np.asarray(test_feature.get_features()[0]))
                    if probability > 0.60:
                        cv2.drawContours(draw_frame, cnt, -1, (0, 255, 0), 3)
                        print(test_feature.get_features())
                        print(detected, probability)

        imf.resize_image(bin, 'binary', 0.4)
        imf.resize_image(blur, 'blur frame', 0.4)
        imf.resize_image(draw_frame, 'results', 0.4)
        cv2.waitKey(0)