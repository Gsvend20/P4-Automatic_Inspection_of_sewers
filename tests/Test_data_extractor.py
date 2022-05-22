from Functions.Featurespace import Classifier
import numpy as np
import os
import pickle



# Init the classifier
c = Classifier()
# Load the trained data
c.get_classifier()

filename = input('Please enter the pkl file name\n')
parent = os.path.dirname(os.getcwd())
file_path = f'{parent}\\classifiers\\{filename}.pkl'

feature_space = []  # List for the Test feature space
label_list = []  # List of names for the found feature
video_list = []  # List of the videos

with open(file_path, 'rb') as file:
    video_contours = pickle.load(file)

for i in range(0, len(video_contours), 4):
    success_counter = 0
    fp_counter = 0
    succ_prob = []
    false_prob = []

    # Unpacking the pkl file
    video_name = video_contours[i]
    category = video_contours[i+1]
    class_level = video_contours[i+2]
    feature_list = video_contours[i+3]
    print(f'Video = {video_name}\nCategory = {category}\nClass {class_level}')
    for n in range(len(feature_list)):
        features = feature_list[n][0]
        frame = int(n*5)
        for feature in features:
            detected, probability = c.classify(np.asarray(feature))
            if probability > 0.50:  # This is just to cut down in the file size
                if detected in category:
                    success_counter += 1
                    succ_prob.append(probability)

                    #  Append the feature into the feature space
                    feature_space.append(feature)
                    label_list.append(detected)
                    video_list.append([video_name, category, class_level])
                else:  # If it's a false positive save a picture of what went wrong
                    fp_counter += 1
                    false_prob.append(probability)

                    #  Append the feature into the feature space
                    feature_space.append(feature)
                    label_list.append(detected+'_false')  # Add false so we can see what was false positives
                    video_list.append([video_name, category, class_level])


    print(f'successes = {success_counter} with an average probability of {np.mean(succ_prob):.4f},\n'
    f'false positives = {fp_counter} with an average probability of {np.mean(false_prob):.4f}')

parent = os.path.dirname(os.getcwd())
file_path = f'{parent}\\classifiers\\test_feature_space.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(feature_space, file)
    pickle.dump(label_list, file)
    pickle.dump(video_list, file)

