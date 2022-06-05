import os
import glob
import cv2
from Functions import imgproc_func as imf
from Functions.Featurespace import FeatureSpace
from Functions.Featurespace import Classifier
import numpy as np


path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'
category_list = os.listdir(path)

# Init the classifier
c = Classifier()
parent = os.path.dirname(os.getcwd())

feature_space = []  # Definitions used for the sklearn classifier
label_list = []

# Train a new classifier
for category in category_list:
    class_list = os.listdir(f'{path}\\{category}')
    for class_level in class_list:
        mask_list = glob.glob(f'{path}/{category}/{class_level}/**/rgbMasks/*.png', recursive=True)
        depth_list = glob.glob(f'{path}/{category}/{class_level}/**/aligned/*.png', recursive=True)

        f = FeatureSpace()
        print(f"Importing {category} {class_level}")
        for img_path, depth_path in zip(mask_list, depth_list):
            # read through all the pictures
            img = cv2.imread(img_path, 0)
            depth = cv2.imread(depth_path, -1)
            depth = imf.convert_to_16(depth)

            if img is not None and np.mean(img) > 0:    # Incase the mask is empty
                contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt, hir = imf.find_largest_contour(contours, hierarchy[0])
                if cv2.contourArea(cnt) > 0:
                    avg_depth = imf.average_contour_depth(depth, cnt)
                    f.create_features(cnt, avg_depth, f"{category}")

        for feature in f.get_features():
            feature_space.append(feature)
            label_list.append(category)

print('conversion files the classifier')
c.prepare_training_data(feature_space, label_list)
c.train_classifier()
print('done importing')
print('saving the classifier')
c.save_trained_classifier(parent + '/classifiers/anotated_training.pkl')


path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\training_images'
category_list = os.listdir(path)

# Train a new classifier
for category in category_list:
    f = FeatureSpace()
    img_folders = glob.glob(path.replace('\\', '/') + '/' + category + '/**/*mask*.png', recursive=True)
    print(f"Importing {category}")
    for img_path in img_folders:
        # read through all the pictures
        img = cv2.imread(img_path, 0)
        depth = cv2.imread(img_path.replace('mask','aligned'), -1)
        depth = imf.convert_to_16(depth)

        if img is not None and np.mean(img) > 0:    # Incase the mask is empty
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt, hir = imf.find_largest_contour(contours, hierarchy[0])
            if cv2.contourArea(cnt) > 0:
                avg_depth = imf.average_contour_depth(depth, cnt)
                f.create_features(cnt, avg_depth, f"{category}")

    for feature in f.get_features():
        feature_space.append(feature)
        label_list.append(category)

print('conversion files the classifier')
c.prepare_training_data(feature_space, label_list)
c.train_classifier()
print('done importing')
print('saving the classifier')
c.save_trained_classifier(parent + '/classifiers/combined_training.pkl')