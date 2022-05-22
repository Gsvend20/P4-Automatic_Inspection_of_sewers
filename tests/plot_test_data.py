import numpy as np
import os
import pickle
from Functions import Featurespace as fs
from Functions import imgproc_func as imf
import glob
import cv2


NEW = 0

# Retireve test data
filename = 'features_test_data'
parent = os.path.dirname(os.getcwd())
file_path = f'{parent}\\data\\test data\\{filename}.pkl'
with open(file_path, 'rb') as file:
    feature_space = pickle.load(file)
    label_list = pickle.load(file)
    video_list = pickle.load(file)


if NEW:
    parent = os.path.dirname(os.getcwd())

    annotated_feature = []  # Definitions used for the sklearn classifier
    segmented_feature = []
    combined_feature = []

    label_ano = []
    label_seg = []
    label_comb = []

    path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\training_images'
    category_list = os.listdir(path)

    # Train a new classifier
    for category in category_list:
        f = fs.FeatureSpace()
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
            segmented_feature.append(feature)
            combined_feature.append(feature)
            label_seg.append(category)
            label_comb.append(category)

    # Retrieve the training data
    path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'
    category_list = os.listdir(path)

    # Train a new classifier
    for category in category_list:
        class_list = os.listdir(f'{path}\\{category}')
        for class_level in class_list:
            mask_list = glob.glob(f'{path}/{category}/{class_level}/**/rgbMasks/*.png', recursive=True)
            depth_list = glob.glob(f'{path}/{category}/{class_level}/**/aligned/*.png', recursive=True)

            f = fs.FeatureSpace()
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
                annotated_feature.append(feature)
                label_ano.append(category)
                combined_feature.append(feature)
                label_comb.append(category)



    file_path = f'{parent}\\data\\test data\\train_seg_features.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(segmented_feature, file)
        pickle.dump(label_seg, file)

    file_path = f'{parent}\\data\\test data\\train_ano_features.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(annotated_feature, file)
        pickle.dump(label_ano, file)

    file_path = f'{parent}\\data\\test data\\train_comb_features.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(combined_feature, file)
        pickle.dump(label_comb, file)
else:
    file_path = f'{parent}\\data\\test data\\train_seg_features.pkl'
    with open(file_path, 'rb') as file:
        segmented_feature = pickle.load(file)
        label_seg = pickle.load(file)

    file_path = f'{parent}\\data\\test data\\train_ano_features.pkl'
    with open(file_path, 'rb') as file:
        annotated_feature = pickle.load(file)
        label_ano = pickle.load(file)

    file_path = f'{parent}\\data\\test data\\train_comb_features.pkl'
    with open(file_path, 'rb') as file:
        combined_feature = pickle.load(file)
        label_comb = pickle.load(file)

categories = ['GR', 'AF', 'ROE', 'FS']

print('test set creation')
test_set = fs.create_dataset(feature_space, label_list, categories[0:2])
ano_set = fs.create_dataset(annotated_feature, label_ano, categories[0:2])
seg_set = fs.create_dataset(segmented_feature, label_seg, categories[0:2])
comb_set = fs.create_dataset(combined_feature, label_comb, categories[0:2])


fs.plot_features(test_set, ano_set)

fs.plot_features(test_set, seg_set)

fs.plot_features(test_set, comb_set)
