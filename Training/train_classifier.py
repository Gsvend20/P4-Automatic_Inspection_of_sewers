import cv2
from Functions import imgproc_func as imf
from Functions import sewer_image_process as imp
from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions.Featurespace import find_annodir
import numpy as np
import glob

"""

The path folder should go into a folder holding every annotation category, with the folder structure like this
    ./category/class/*.png
eg. ./AF/Class 1/9/rgbMasks/*.png

"""

path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\training_images'
# Definitions used for the sklearn classifier
feature_space = []
label_list = []

# Init the classifier
c = Classifier()

# Train the classifier and save it, if you just want to test say yes to the prompt
c.get_classifier(path)

# Run test area 0 = no, 1 = yes
TESTAREA = 1

if not TESTAREA:
    exit()

"""
TESTING AREA, TOO CHECK HOW WELL IT WORKS
"""

path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\training_images'
class_name, anotations = find_annodir(path)

for category in class_name:
    depth_paths = glob.glob(path.replace('\\', '/') + '/' + category + '/**/*aligned*.png', recursive=True)
    for i in range(120,150):
        print(f'current file {category} number {i}')
        depth_path = depth_paths[i]
        bgr_path = depth_path.replace('aligned','bgr')

        depth_img = imf.convert_to_16(cv2.imread(depth_path))
        bgr_img = cv2.imread(bgr_path)
        draw_frame = np.zeros_like(bgr_img)

        binary = imp.get_binary(bgr_img, np.zeros(depth_img.shape, dtype='uint8'), True)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if hierarchy is not None:
            hierarchy = hierarchy[0]  # [[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]
            draw_frame = bgr_img.copy()
            for cnt, hrc in zip(contours, hierarchy):
                if cv2.contourArea(cnt) >= 50:
                    mask = np.zeros(binary.shape, np.uint8)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    rgbd_aoi = cv2.bitwise_and(depth_img, depth_img, mask=mask)
                    # mask_display = imf.depth_to_display(rgbd_aoi)
                    # imf.resize_image(mask_display, 'mask', 0.5)
                    # cv2.waitKey(1)
                    avg_depth = imf.average_contour_depth(depth_img, cnt)
                    test_feature = FeatureSpace()
                    test_feature.create_features(cnt, avg_depth, 'test')

                    detected, probability = c.classify(np.asarray(test_feature.get_features()[0]))
                    if probability > 0.90:
                        cv2.drawContours(draw_frame, cnt, -1, (0, 255, 0), 3)
                        print(test_feature.get_features())
                        print(detected, probability)

        imf.resize_image(draw_frame, 'results', 0.4)
        cv2.waitKey(0)