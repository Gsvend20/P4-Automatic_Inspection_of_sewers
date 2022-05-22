from Functions.Featurespace import Classifier
import numpy as np
import os
import pickle



# Init the classifier
c = Classifier()
# Load the trained data
c.get_classifier()

# Retireve test data
filename = 'test_feature_space'
parent = os.path.dirname(os.getcwd())
file_path = f'{parent}\\classifiers\\{filename}.pkl'
with open(file_path, 'rb') as file:
    feature_space = pickle.load(file)
    label_list = pickle.load(file)
    video_list = pickle.load(file)

c.prepare_test_data(feature_space, label_list)
c.test_classifier()