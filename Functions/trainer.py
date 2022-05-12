from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions.Featurespace import find_annodir
import os
import cv2
import numpy as np

featurelist = FeatureSpace()
type_list = find_annodir()

# Run through all types
for types in type_list:
    print(f"Importing {types}")
    # Run through all subtypes
    for category in os.listdir(types):
        mask_path = os.listdir(f"{types}/{category}/rgbMasks")
        # Get filenames
        for images in mask_path:
            # Load image
            if images.endswith('.png'):
                img = cv2.imread(f"{types}/{category}/rgbMasks/{images}", 0)
                if img is not None and np.mean(img) > 0:
                    cnt, hir = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2:]
                    for contours, hierarchy in zip(cnt, hir):
                        featurelist.create_features(contours, hierarchy, f"{types}_{category}")

    if types == 'ROE':
        print('Done import')
        break

path = os.getcwd()
os.chdir(path.replace('\Annotations', ''))
classifier_path = 'classifier.pkl'

clf = Classifier()
clf.prepare_training_data(featurelist.get_features(), featurelist.type)
if os.path.exists(classifier_path):
    clf.load_trained_classifier(classifier_path)
else:
    clf.train_classifier()
    clf.save_trained_classifier(classifier_path)
