import cv2
from Functions import imgproc_func as imf
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

path = r'C:\Users\mikip\Documents\P4-Automatic_Inspection_of_sewers\P4-Automatic_Inspection_of_sewers\Training\training_images'
# Definitions used for the sklearn classifier
feature_space = []
label_list = []

# Init the classifier
c = Classifier()

# Train the classifier and save it, if you just want to test say yes to the prompt
c.get_classifier(path)

# Run test area 0 = no, 1 = yes
TESTAREA = 1


"""
TESTING AREA, TOO CHECK HOW WELL IT WORKS
"""

path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'
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

        blur = cv2.medianBlur(bgr_img, 13)

        frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

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

        scr_uppervalues = [129, 103, 59]
        scr_lowervalues = [70, 21, 32]

        roe_uppervalues = [107, 114, 255]
        roe_lowervalues = [72, 28, 150]

        mask1 = cv2.inRange(frame_hsi, np.asarray(hls_lowervalues),
                            np.asarray(hls_uppervalues))  # Threshold around highlights
        mask2 = cv2.inRange(frame_hsi, np.asarray(blue_lowervalues),
                            np.asarray(blue_uppervalues))  # Remove blue, due to the piece of cloth
        mask3 = cv2.inRange(frame_hsi, np.asarray(scr_lowervalues),
                            np.asarray(scr_uppervalues))  # Remove blue, due to scratches
        mask4 = cv2.inRange(frame_hsv, np.asarray(roe_lowervalues),
                            np.asarray(roe_uppervalues))  # Add in some dark blue for roots

        hsi_thresh = cv2.add(mask1, mask4)
        hsi_thresh = cv2.subtract(hsi_thresh, mask2)
        hsi_thresh = cv2.subtract(hsi_thresh, mask3)

        bin = imf.open_img(hsi_thresh, 5, 5)

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
                    test_feature.create_features(cnt, 'test')

                    detected, probability = c.classify(np.asarray(test_feature.get_features()[0]))
                    if probability > 0.90:
                        cv2.drawContours(draw_frame, cnt, -1, (0, 255, 0), 3)
                        print(test_feature.get_features())
                        print(detected, probability)

        imf.resize_image(bin, 'binary', 0.4)
        imf.resize_image(blur, 'blur frame', 0.4)
        imf.resize_image(draw_frame, 'results', 0.4)
        cv2.waitKey(0)