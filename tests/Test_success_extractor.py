import numpy as np
import os
import pickle
from Functions.Featurespace import Classifier

# Init classifier
c = Classifier()
# Load trained data
c.get_classifier()

# Create a folder for saving results
save_path = f'/Test Results'
if not os.path.exists(os.getcwd() + save_path):
    os.makedirs(os.getcwd() + save_path)
writer = open(f".{save_path}/computed_test_result.txt", "w")
writer.close() # Just clearing out the text file

# Load the test data
parent = os.path.dirname(os.getcwd())
filename = 'features_test_data'
file_path = f'{parent}\\data\\test data\\{filename}.pkl'
with open(file_path, 'rb') as file:
    feature_space= pickle.load(file)
    label_list = pickle.load(file)
    video_list = pickle.load(file)

# Find the different videos
videos = np.unique(video_list, axis=0)

# Go through each video individually with the computed annotator
for video in videos:
    # Extract the index and video data
    vid_name, vid_cat, vid_class = video
    index = np.all(video_list == video, axis=1)
    # Find the corresponding features
    feature_vector = np.array(feature_space)[index]
    feature_names = np.array(label_list)[index]

    # Using Scikit to predict every feature
    predict = c._classifier.predict(feature_vector)
    prob = np.max(c._classifier.predict_proba(feature_vector), axis=1)

    pass_index = prob >= 0.85
    passed = np.unique(predict[pass_index], return_counts=True)
    passed_prob = prob[pass_index]
    true_p = passed[0] == vid_cat
    false_p = passed[0] != vid_cat
    if 'FS' in vid_cat:
        true_p = passed[0] == 'FS'
        false_p = passed[0] != 'FS'
    print(f'video {vid_name}\n'
          f'Category {vid_cat}, class {vid_class}'
          f'\nsuccesses = {passed[0][true_p]} amount {passed[1][true_p]}, probability {passed_prob[predict[pass_index] == vid_cat]}'
          f'\nfails = {passed[0][false_p]} amount {passed[1][false_p]}, probability {passed_prob[predict[pass_index] != vid_cat]}')

    writer = open(f".{save_path}/computed_test_result.txt", "a")
    writer.write(f'video {vid_name}\n'
                 f'Category {vid_cat}, class {vid_class}'
          f'\nsuccesses = {passed[0][true_p]} amount {passed[1][true_p]}'
          f'\nfails = {passed[0][false_p]} amount {passed[1][false_p]}\n')

# Init classifier
c = Classifier()

writer = open(f".{save_path}/segmented_test_result.txt", "w")
writer.close() # Just clearing out the text file

class_path = r'C:\Users\mikip\Documents\P4-Automatic_Inspection_of_sewers\P4-Automatic_Inspection_of_sewers\data\classifiers\annotated_training.pkl'
c._load_trained_classifier(class_path)

# Go through each video individually with the manual annotator
for video in videos:
    # Extract the index and video data
    vid_name, vid_cat, vid_class = video
    index = np.all(video_list == video, axis=1)
    # Find the corresponding features
    feature_vector = np.array(feature_space)[index]
    feature_names = np.array(label_list)[index]

    # Using Scikit to predict every feature
    predict = c._classifier.predict(feature_vector)
    prob = np.max(c._classifier.predict_proba(feature_vector), axis=1)

    pass_index = prob >= 0.80
    passed = np.unique(predict[pass_index], return_counts=True)
    passed_prob = prob[pass_index]
    true_p = passed[0] == vid_cat
    false_p = passed[0] != vid_cat
    if 'FS' in vid_cat:
        true_p = passed[0] == 'FS'
        false_p = passed[0] != 'FS'
    print(f'video {vid_name} class {vid_class}'
          f'\nsuccesses = {passed[0][true_p]} amount {passed[1][true_p]}, probability {passed_prob[predict[pass_index] == vid_cat]}'
          f'\nfails = {passed[0][false_p]} amount {passed[1][false_p]}, probability {passed_prob[predict[pass_index] != vid_cat]}')

    writer = open(f".{save_path}/segmented_test_result.txt", "a")
    writer.write(f'video {vid_name}\n'
                 f'Category {vid_cat}, class {vid_class}'
          f'\nsuccesses = {passed[0][true_p]} amount{passed[1][true_p]}'
          f'\nfails = {passed[0][false_p]} amount {passed[1][false_p]}\n')

# Init classifier
c = Classifier()

writer = open(f".{save_path}/combined_test_result.txt", "w")
writer.close()  # Just clearing out the text file

class_path = r'C:\Users\mikip\Documents\P4-Automatic_Inspection_of_sewers\P4-Automatic_Inspection_of_sewers\data\classifiers\combined_training.pkl'
c._load_trained_classifier(class_path)

# Go through each video individually with the combined annotator
for video in videos:
    # Extract the index and video data
    vid_name, vid_cat, vid_class = video
    index = np.all(video_list == video, axis=1)
    # Find the corresponding features
    feature_vector = np.array(feature_space)[index]
    feature_names = np.array(label_list)[index]

    # Using Scikit to predict every feature
    predict = c._classifier.predict(feature_vector)
    prob = np.max(c._classifier.predict_proba(feature_vector), axis=1)

    pass_index = prob >= 0.80
    passed = np.unique(predict[pass_index], return_counts=True)
    passed_prob = prob[pass_index]
    true_p = passed[0] == vid_cat
    false_p = passed[0] != vid_cat
    if 'FS' in vid_cat:
        true_p = passed[0] == 'FS'
        false_p = passed[0] != 'FS'
    print(f'video {vid_name} class {vid_class}'
          f'\nsuccesses = {passed[0][true_p]} amount {passed[1][true_p]}, probability {passed_prob[predict[pass_index] == vid_cat]}'
          f'\nfails = {passed[0][false_p]} amount {passed[1][false_p]}, probability {passed_prob[predict[pass_index] != vid_cat]}')

    writer = open(f".{save_path}/combined_test_result.txt", "a")
    writer.write(f'video {vid_name}\n'
                 f'Category {vid_cat}, class {vid_class}'
          f'\nsuccesses = {passed[0][true_p]} amount{passed[1][true_p]}'
          f'\nfails = {passed[0][false_p]} amount {passed[1][false_p]}\n')