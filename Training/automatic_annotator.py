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




path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'
# Definitions used for the sklearn classifier
feature_space = []
label_list = []

class_name, anotations = find_annodir(path)

# Init the classifier
c = Classifier()

# Find the parent directory
parent = os.path.dirname(os.getcwd())
# If the trained classifier does not exist recreate it
if os.path.exists(parent+'/classifiers/annotated_training.pkl') and input('Trained data exists\nUse it? y/n?') == 'y':
    # Load the trained classifier
    c.load_trained_classifier(parent+'/classifiers/annotated_training.pkl')
else:
    # Train a new classifier
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
    print('saving the classifier')
    c.save_trained_classifier(parent+'/classifiers/annotated_training.pkl')


"""
 This is the actual part where the BLOBS are extracted
 The path should lead to the folder containing every video with training data, folder structure should follow this:
     ./category/class/**.avi
    eg. ./AF/Class 1/horizontal/*.avi
    
 Training data will be saved in the git Training folder
 MAKE SURE TO MOVE AND SAVE THE IMAGES WHEN YOU ARE DONE, THEY WILL BE OVERWRITTEN!!!! 
"""

# Path leading to the training videos
path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\Videos'
category_names = os.listdir(path)

P = 1  # Pause input
print("Press 'P' to start/pause, 'S' to save, 'Q' for next image")

for category in category_names:
    D = 0
    listof_depth = glob.glob(f'{path}/{category}/**/*aligned*.avi', recursive=True)
    for depth_path in listof_depth:

        if D:
            break

        class_level = depth_path.split('Class')[1][1]

        depth_src = cv2.VideoCapture(depth_path)
        bgr_src = cv2.VideoCapture(depth_path.replace('aligned', 'bgr'))

        # Init the adaptive threshold for depth images
        depth_masker = imf.AdaptiveGRDepthMasker(3, (1500, 1600), (3000, 40000))

        # Start going through the videos
        ret, frame_bgr = bgr_src.read()
        if not ret:
            break
        ret, frame_depth_8bit = depth_src.read()
        if not ret:
            break
        frame_depth = imf.convert_to_16(frame_depth_8bit)


        while True:
            if not P:
                ret, frame_bgr = bgr_src.read()
                if not ret:
                    break
                ret, frame_depth_8bit = depth_src.read()
                if not ret:
                    break
                frame_depth = imf.convert_to_16(frame_depth_8bit)

            blur = cv2.medianBlur(frame_bgr, 13)

            frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            #imf.resize_image(imf.depth_to_display(frame_depth), 'aligned depth image', 0.5)
            imf.resize_image(frame_bgr, 'color image', 0.3)


            # Adaptive thresholding
            # Generate area of interest from pipe depth data
            aoi_end = cv2.inRange(frame_depth, int(np.max(frame_depth) - 100), int(np.max(frame_depth)))
            aoi_pipe = cv2.inRange(frame_depth, 600, int(np.max(frame_depth) - 100))
            cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pipe_mask = np.zeros_like(frame_depth).astype('uint8')
            pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
            bg_mask = cv2.subtract(pipe_mask, aoi_end)
            #bg_mask = imf.open_img(bg_mask, 21, 21)
            bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
            hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

            # adaptive depth
            fg_d_frame = cv2.bitwise_and(frame_depth, frame_depth, mask=bg_mask)
            depth_masker.add_image(fg_d_frame)

            # Wait for input
            key = cv2.waitKey(1)
            if key == ord('p'):
                if P == 1:
                    P = 0
                else:
                    P = 1
            if key == ord('q'):
                break
            elif key == ord('d'):
                D = 1
                break

            elif key == ord('s'):
                draw_frame = np.zeros_like(frame_bgr)

                depth_mask = depth_masker.return_masks()

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

                canny = cv2.Canny(frame_hsi[:, :, 1], 50, 255)
                canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


                hsi_thresh = cv2.add(mask1, mask4)
                hsi_thresh = cv2.add(hsi_thresh, canny)
                hsi_thresh = cv2.add(hsi_thresh, depth_mask)
                hsi_thresh = cv2.subtract(hsi_thresh, mask2)
                hsi_thresh = cv2.subtract(hsi_thresh, mask3)

                bin = imf.open_img(hsi_thresh, 5, 5)

                contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                best_fit = []
                if hierarchy is not None:
                    hierarchy = hierarchy[0]  # [[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]
                    for cnt, hrc in zip(contours, hierarchy):
                        if cv2.contourArea(cnt) >= 50:
                            test_feature = FeatureSpace()
                            test_feature.create_features(cnt, np.array(hrc[2] != -1), 'test')

                            detected, probability = c.classify(np.asarray(test_feature.get_features()[0]))
                            best_fit.append([detected, probability, cnt])
                    best_fit.sort(key=lambda x: x[1])

                print(f'results {best_fit[-1][0]}, with a {best_fit[-1][1]} chance')


                print(f'Is this correct?, yes = s, no = d, skip = q')
                num = 1
                while True:
                    draw_frame = frame_bgr.copy()
                    cv2.drawContours(draw_frame, [best_fit[-num][2]], 0, (0, 0, 255), 2)
                    imf.resize_image(draw_frame, 'results', 0.3)
                    imf.resize_image(bin, 'binary', 0.3)
                    key = cv2.waitKey(0)
                    if key == ord('s'):
                        file_id = 1
                        while os.path.exists(f'./training_images/{category}/Class {class_level}/{file_id}_{category}_{class_level}_mask.png'):
                            file_id += 1
                        save_img = np.zeros_like(bin)
                        cv2.drawContours(save_img, [best_fit[-num][2]], 0, 255, -1)
                        save_path = f'/training_images/{category}/Class {class_level}'
                        if not os.path.exists(os.getcwd() + save_path):
                            os.makedirs(os.getcwd() + save_path)
                        print(f'.{save_path}/{file_id}_{category}_{class_level}_mask.png')
                        cv2.imwrite(f'.{save_path}/{file_id}_{category}_{class_level}_mask.png', save_img)
                        imf.save_depth_img(f'.{save_path}/{file_id}_{category}_{class_level}_aligned.png', frame_depth)
                        cv2.imwrite(f'.{save_path}/{file_id}_{category}_{class_level}_bgr.png', frame_bgr)
                        print('saved')
                        imf.resize_image(np.zeros_like(frame_depth), 'results', 0.3)
                        break
                    elif key == ord('d'):
                        num += 1
                    elif key == ord('q'):
                        print('skipped')
                        imf.resize_image(np.zeros_like(frame_depth), 'results', 0.3)
                        break

