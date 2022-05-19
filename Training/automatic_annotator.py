import cv2
from Functions import imgproc_func as imf
from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
import numpy as np
import os
import glob

"""
 This is the actual part where the BLOBS are extracted
 The path should lead to the folder containing every video with training data, folder structure should follow this:
     ./category/class/**.avi
    eg. ./AF/Class 1/horizontal/*.avi

 Training data will be saved in the git Training folder
 MAKE SURE TO MOVE AND SAVE THE IMAGES WHEN YOU ARE DONE, THEY WILL NOT BE OVERWRITTEN!!!! 
"""

# Path for the annotated training data
path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'

# Init the classifier
c = Classifier()
# Load the trained data
c.get_classifier(path)


# Path leading to the training videos
path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Test data'
category_names = os.listdir(path)

P = 1  # Pause input
print("Press 'P' to start/pause, 'S' to save, 'Q' for next image")

for category in category_names:
    D = 0
    print(f'{path}/{category}/**/*aligned*.avi')
    listof_depth = glob.glob(f'{path}/{category}/**/*aligned*.avi', recursive=True)
    for depth_path in listof_depth:

        if D:
            break

        class_level = depth_path.split('Class')[1][1]  # Find the classes in the folder
        print(f'{depth_path}\n current class {class_level}')
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

        # Run through the video and wait for input
        while True:
            if not P:
                ret, frame_bgr = bgr_src.read()
                if not ret:  # Break if there are no frames left
                    break
                ret, frame_depth_8bit = depth_src.read()
                if not ret:
                    break
                frame_depth = imf.convert_to_16(frame_depth_8bit)  # Convert the depth data back into readable data

            # Begin treating the image in the same way you would detect flaws
            blur = cv2.GaussianBlur(frame_bgr, (13, 13), cv2.BORDER_DEFAULT)

            frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            imf.resize_image(frame_bgr, 'color image', 0.3)


            # Adaptive thresholding
            # Generate area of interest from pipe depth data, by finding the end of the pipe
            aoi_end = cv2.inRange(frame_depth, int(np.max(frame_depth) - 100), int(np.max(frame_depth)))
            # Then the front of the pipe is extracted
            aoi_pipe = cv2.inRange(frame_depth, 600, int(np.max(frame_depth) - 100))
            cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Make a mask out of both
            pipe_mask = np.zeros_like(frame_depth).astype('uint8')
            pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
            bg_mask = cv2.subtract(pipe_mask, aoi_end)
            bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
            # use the mask to generate an area of interest in the depth data
            fg_d_frame = cv2.bitwise_and(frame_depth, frame_depth, mask=bg_mask)
            # start the adaptive thresholding
            depth_masker.add_image(fg_d_frame)

            # Wait for input
            key = cv2.waitKey(1)
            if key == ord('p'):  # Pause key
                if P == 1:
                    P = 0
                else:
                    P = 1
            if key == ord('q'):  # Stop key
                break
            elif key == ord('d'):  # Skip class key
                D = 1
                break

            elif key == ord('s'):  # Save key
                #  Start creating the BLOBS and classify them
                draw_frame = np.zeros_like(frame_bgr)

                depth_mask = depth_masker.return_masks()  # Make the adaptive thresholder return BLOBS of interest

                # These are all the upper and lower bounds for the thresholding
                base_up = [255, 255, 255]
                base_low = [70, 37, 30]

                blue_up = [124, 140, 150]
                blue_low = [84, 37, 61]

                scr_up = [129, 103, 59]
                scr_low = [70, 21, 32]

                mask1 = cv2.inRange(frame_hsi, np.asarray(base_low),
                                    np.asarray(base_up))  # Threshold around highlights
                mask2 = cv2.inRange(frame_hsi, np.asarray(blue_low),
                                    np.asarray(blue_up))  # Remove blue, due to the piece of cloth
                mask3 = cv2.inRange(frame_hsi, np.asarray(scr_low),
                                    np.asarray(scr_up))  # Remove blue, due to scratches

                hsi_thresh = cv2.add(mask1, depth_mask)
                hsi_thresh = cv2.subtract(hsi_thresh, mask2)
                hsi_thresh = cv2.subtract(hsi_thresh, mask3)

                bin = imf.open_img(hsi_thresh, 5, 5)  # By opening we remove noise and the edges that are of no interest

                # Find the contours
                contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                best_fit = []  # array for saving the BLOBS according to their probability
                if hierarchy is not None:
                    hierarchy = hierarchy[0]  # Unpacking the hierarchy
                    # Go through every contour and save them with their probability
                    for cnt, hrc in zip(contours, hierarchy):
                        if cv2.contourArea(cnt) >= 50:  # Ignore contours that are too small to care for
                            test_feature = FeatureSpace()
                            test_feature.create_features(cnt, 'test')

                            detected, probability = c.classify(np.asarray(test_feature.get_features()[0]))
                            best_fit.append([detected, probability, cnt])
                    best_fit.sort(key=lambda x: x[1]) # Sort the found BLOBS accordingly
                if len(best_fit) > 1:  # just in case nothing is in the picture
                    print(f'results {best_fit[-1][0]}, with a {best_fit[-1][1]} chance')

                    print(f'Is this correct?, yes = s, no = d, skip = q')
                    num = 1
                    # Wait for user input
                    while True:
                        draw_frame = frame_bgr.copy()
                        cv2.drawContours(draw_frame, [best_fit[-num][2]], 0, (0, 0, 255), 2)
                        imf.resize_image(draw_frame, 'results', 0.3)
                        imf.resize_image(bin, 'binary', 0.3)
                        key = cv2.waitKey(0)
                        if key == ord('s'):  # Save the found contour key
                            file_id = 1
                            # Count the existing images and get a new file id
                            while os.path.exists(f'./training_images/{category}/Class {class_level}/{file_id}_{category}_{class_level}_mask.png'):
                                file_id += 1
                            save_img = np.zeros_like(bin)  # save frame
                            # Draw the BLOB into the save frame
                            cv2.drawContours(save_img, [best_fit[-num][2]], 0, 255, -1)

                            # Save everything into the training images folder
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
                        elif key == ord('d'):  # Go to next contour key
                            num += 1
                        elif key == ord('q'):   # skip key
                            print('skipped')
                            imf.resize_image(np.zeros_like(frame_depth), 'results', 0.3)
                            break