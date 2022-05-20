import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Functions import imgproc_func as imf


class FeatureSpace:
    def __init__(self):
        self.type = []
        self.depth = []
        self.centerX = []
        self.centerY = []
        self.convex_ratio_perimeter = []
        self.compactness = []
        self.elongation = []
        self.ferets = []
        self.thinness = []

    def create_features(self, cnt, avg_depth, error_type):
        # saving type of data
        self.type.append(error_type)
        area = cv2.contourArea(cnt)

        self.depth.append(avg_depth/1000)  # Division by 1000 due to the that being around max length
        # Center of mass
        M = cv2.moments(cnt)
        self.centerX.append(int(M['m10'] / M['m00'])/1080*3)    # Division is to normalise according to the image
        self.centerY.append(int(M['m01'] / M['m00'])/1920*3)

        # Detect jaggedness of edges
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hullperimeter = cv2.arcLength(hull, True)
        self.convex_ratio_perimeter.append(hullperimeter / perimeter)

        # Compactness
        x, y, w, h = cv2.boundingRect(cnt)
        self.compactness.append(area / (w * h))

        # Elongation of min area rect
        (x_elon, y_elon), (width_elon, height_elon), angle = cv2.minAreaRect(cnt)
        self.elongation.append(min(width_elon, height_elon) / max(width_elon, height_elon))

        # Longest internal line and its angle
        #self.ferets_angle.append((angle+180)/360) # angle is -180/180 degress, so we make it all positive and normalise

        # TODO: check if 800 fits the ferets found
        self.ferets.append(max(width_elon, height_elon)/800) # divide by 800 to normalise

        # Thinness TODO: Check normalisation
        self.thinness.append(perimeter / area * 6)  # Multiply by 6 to bring the thinness average closer to one

    def get_features(self):
        features = []
        for i in range(0, np.shape(self.type)[0]):
            features.append([self.centerX[i],
                             self.centerY[i],
                             self.convex_ratio_perimeter[i],
                             self.depth[i],
                             self.compactness[i],
                             self.elongation[i],
                             self.ferets[i],
                             self.thinness[i]
                             ])
        #print(np.max(features), 'in', np.where(features == np.max(features)))
        return features


class Classifier:
    def __init__(self):
        self._classifier_names = ["Nearest Neighbors",
                                  "Linear SVM",
                                  "RBF SVM",
                                  "Gaussian Process",
                                  "Decision Tree",
                                  "Random Forest",
                                  "Neural Net",
                                  "AdaBoost",
                                  "Naive Bayes",
                                  "QDA"
                                  ]
        self._classifier_list = [KNeighborsClassifier(3),
                                 SVC(kernel="linear", C=0.025),
                                 SVC(gamma=2, C=1),
                                 GaussianProcessClassifier(RBF(length_scale_bounds=(1.0E-5, 1.0E+100)),
                                                           max_iter_predict=1000),
                                 DecisionTreeClassifier(max_depth=5),
                                 RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                                 MLPClassifier(alpha=1, max_iter=1000),
                                 AdaBoostClassifier(),
                                 GaussianNB(),
                                 QuadraticDiscriminantAnalysis()
                                 ]

        self._training_features = None
        self._training_labels = None
        self._test_features = None
        self._test_labels = None

        self._classifier = GaussianProcessClassifier(RBF(length_scale_bounds=(1.0E-5, 1.0E+100)), max_iter_predict=1000)

    # Scale and normalise training data to allow for model training
    def prepare_training_data(self, training_features, training_labels):
        #self._training_features = StandardScaler().fit_transform(training_features)
        self._training_features = training_features
        self._training_labels = training_labels

    # Create test data by splitting training set
    def split_training_data(self, test_ratio=0.4):
        x_tra, x_tes, y_tra, y_tes = train_test_split(self._training_features,
                                                      self._training_labels,
                                                      test_size=test_ratio,
                                                      random_state=42)
        self._training_features = x_tra
        self._training_labels = y_tra
        self._test_features = x_tes
        self._test_labels = y_tes

    # Train selected classifier
    def train_classifier(self):
        if self._training_features is not None:
            self._classifier.fit(self._training_features, self._training_labels)
        else:
            exit('no training data available!')

    # Test how well the classifier does on the test set
    def test_classifier(self):
        np.set_printoptions(precision=2)  # TODO what this do?

        # Plot non-normalized confusion matrix
        disp = ConfusionMatrixDisplay.from_estimator(self._classifier,
                                                     self._test_features,
                                                     self._test_labels,
                                                     display_labels=np.unique(np.array(self._test_labels)),
                                                     cmap=plt.cm.Blues)
        disp.ax_.set_title("Confusion matrix for classifier")
        plt.show()

    # Save trained classifiers in dump file for future use
    def save_trained_classifier(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self._classifier, file)

    # Load trained classifier from dump file for use in this program
    def _load_trained_classifier(self, file_path):
        with open(file_path, 'rb') as file:
            self._classifier = pickle.load(file)

    # Find a classifier or train a new if needed
    def get_classifier(self, training_path=None):
        cur_dir = os.getcwd()   # get current directive
        # Find the base directive

        if cur_dir.split('\\')[-1] == 'P4-Automatic_Inspection_of_sewers':
            parent = cur_dir
        else:
            # Use the parent directory
            parent = os.path.dirname(os.getcwd())

        # Load in the training data if no path is specified
        if os.path.exists(parent + '/classifiers/annotated_training.pkl') and training_path is None:
            print('Successfully loaded training data')
            self._load_trained_classifier(parent + '/classifiers/annotated_training.pkl')
            return
        elif training_path is None:
            print(f'You have no trained classifier, put it in folder {parent}/classifiers/annotated_training.pkl and try again')
            exit(1)

        # If a path was specified treat it as wanting to train new data

        feature_space = []  # Definitions used for the sklearn classifier
        label_list = []

        # Find the folders in path
        class_name, _ = find_annodir(training_path)

        # If the trained classifier does not exist recreate it
        if os.path.exists(parent + '/classifiers/annotated_training.pkl') and input(
                'Trained data exists\nUse it? y/n?') == 'y':
            # Load the trained classifier
            self._load_trained_classifier(parent + '/classifiers/annotated_training.pkl')
        else:
            # Train a new classifier
            for category in class_name:
                f = FeatureSpace()
                img_folders = glob.glob(training_path.replace('\\', '/') + '/' + category + '/**/*mask*.png', recursive=True)
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

            print('Training the classifier')
            self.prepare_training_data(feature_space, label_list)
            self.train_classifier()
            print('done importing')
            print('saving the classifier')
            self.save_trained_classifier(parent + '/classifiers/annotated_training.pkl')

    # Generate and print a list of the most suitable classifiers for selected classification
    def best_classifiers(self):
        # Check all available classifiers and return final scores
        score = []
        for cf in self._classifier_list:
            # Train classifier
            cf.fit(self._training_features, self._training_labels)

            # Score classifier
            score.append(cf.score(self._test_features, self._test_labels))

        # Make a list of the best classifiers for the dataset
        best_classifiers = []
        highest_scores = score.copy()
        highest_scores.sort(reverse=True)
        for i in range(0, len(highest_scores)):
            index = score.index(highest_scores[i])
            best_classifiers.append([self._classifier_names[index], score[index]])

        print(best_classifiers)

    def classify(self, test_data, test_labels=None):
        test_data = test_data.reshape(1, -1)
        detected = self._classifier.predict(test_data)[0]
        certainty = np.max(self._classifier.predict_proba(test_data))
        return detected, certainty



# Old function
def plot_features(dataset):
    # # Code needed to create the dataset for this function
    # index_1 = np.where(np.char.find(np.array(featurelist.type), 'ROE_70') + 1)[0]
    # index_2 = np.where(np.char.find(np.array(featurelist.type), 'ROE_150') + 1)[0]
    # index_3 = np.where(np.char.find(np.array(featurelist.type), 'ROE_300') + 1)[0]
    #
    # # Create signifiers of which category each datapoint belongs to
    # intervals = []
    # for i in index_1:
    #     intervals.append(0)
    # for i in index_2:
    #     intervals.append(1)
    #
    # # Create datasets
    # datasets1 = []
    # datasets2 = []
    # datasets3 = []
    # for i in index_1:
    #     datasets1.append([featurelist.convex_ratio_perimeter[i], featurelist.compactness[i]])
    #     datasets2.append([featurelist.elongation[i], featurelist.hierachy_Bool[i]])
    #     datasets3.append([featurelist.ferets[i], featurelist.thinness[i]])
    # for i in index_2:
    #     datasets1.append([featurelist.convex_ratio_perimeter[i], featurelist.compactness[i]])
    #     datasets2.append([featurelist.elongation[i], featurelist.hierachy_Bool[i]])
    #     datasets3.append([featurelist.ferets[i], featurelist.thinness[i]])
    # datasets = [(np.array(datasets1), np.array(intervals)),
    #             (np.array(datasets2), np.array(intervals)),
    #             (np.array(datasets3), np.array(intervals))]
    #
    # plot_features(datasets)

    h = 0.02  # step size in the mesh

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        # "Decision Tree",
        # "Random Forest",
        "Neural Net",
        # "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    figure = plt.figure(figsize=(29, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(dataset):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            print(xx.shape)
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            # Plot the training points
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    plt.tight_layout()
    plt.show()
    return


def find_annodir(path):
    folder_list = []
    class_name = os.listdir(path)
    for categories in class_name:
        folder_list.append(glob.glob(path.replace('\\','/') + '/' + categories + '/**/rgbMasks/*.png', recursive=True))
    return class_name, folder_list
#
# featurelist = FeatureSpace()
# type_list = find_annodir()
#
# # Run through all types
# for types in type_list:
#     print(f"Importing {types}")
#     # Run through all subtypes
#     for category in os.listdir(types):
#         mask_path = os.listdir(f"{types}/{category}/rgbMasks")
#         # Get filenames
#         for images in mask_path:
#             # Load image
#             if images.endswith('.png'):
#                 img = cv2.imread(f"{types}/{category}/rgbMasks/{images}", 0)
#                 if img is not None and np.mean(img) > 0:
#                     cnt, hir = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2:]
#                     featurelist.create_features(cnt, hir, f"{types}_{category}")
#     if types == 'ROE':
#         print('Done import')
#         break
#
# classifier_path = 'test.pkl'
#
# clf = Classifier()
# clf.prepare_training_data(featurelist.get_features(), featurelist.type)
# clf.split_training_data()
#
# if os.path.exists(classifier_path):
#     clf.load_trained_classifier(classifier_path)
# else:
#     clf.train_classifier()
#     clf.save_trained_classifier(classifier_path)
#
# clf.best_classifiers()
#
# clf.test_classifier()
# detected_class, detection_certainty = clf.classify(clf._test_features[0])
# print(f"Detected {detected_class} with a certainty of {detection_certainty}")


def find_height(depth_img, depth, centerY):
    p_ratio = 2.9/ 1000  # mm
    focal_length = 3.6  # mm

    camera_height = 265  # mm
    b = 1 / (1 / focal_length - 1 / depth)
    height = (camera_height * b / depth)
    pixel_height = height / p_ratio  # pixel height to the bottom of the pipe
    max_pos = int(1920/2) + int(pixel_height)

    min_pos = centerY
    point = depth_img[centerY, int(1080/2)]
    while point != 0:
        point = depth_img[min_pos, int(1080/2)]
        min_pos -= 1
    pixel_height = max_pos-min_pos
    height = pixel_height*p_ratio

    return height*depth/b
