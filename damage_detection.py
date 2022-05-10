from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os




# Path to video folders
folders_path = 'Test'

# Load saved video streams
vid_bgr = cv2.VideoCapture()
vid_bgrd = cv2.VideoCapture()

# Trackbars for adjusting values
# imf.define_trackbar('Blur level', 'trackbars', (0, 31))
#imf.define_trackbar('high', 'trackbars', (0, 255))
#imf.define_trackbar('low', 'trackbars', (0, 255))
#imf.define_trackbar('Closing level', 'trackbars', (0, 255))
#imf.define_trackbar('Min depth', 'trackbars', (500, 1500))
imf.define_trackbar('Kernel', 'trackbars', (3, ))

# load classifier
#classifier = Classifier()
#classifier.load_trained_classifier('training_data.pkl')

while True:
    vid_bgr.open(f"{folders_path}/1_bgr.avi")
    vid_bgrd.open(f"{folders_path}/1_aligned.avi")

    while vid_bgr.isOpened():
        # Fetch next frame
        ret, frame_bgr = vid_bgr.read()
        if not ret:
            break
        ret, frame_bgrd_8bit = vid_bgrd.read()
        if not ret:
            break
        # Convert depthmap to 16 bit depth
        frame_bgrd = imf.convert_to_16(frame_bgrd_8bit)

        #frame_bgrd = cv2.GaussianBlur(frame_bgrd, (5, 5), 0)
        frame_bgr = cv2.medianBlur(frame_bgr, 5)

        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        grey_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        #
        # # Remove noise
        # kblur = imf.retrieve_trackbar('Blur level', 'trackbars', True)
        # if kblur >= 3:
        #     grey_frame = cv2.medianBlur(grey_frame, kblur)
        #
        # # Edge detection in grayscale
        # cmin = imf.retrieve_trackbar('Canny min', 'trackbars')
        # cmax = imf.retrieve_trackbar('Canny max', 'trackbars')
        # grey_frame = cv2.Canny(grey_frame, cmin, cmax)
        #
        # clev = imf.retrieve_trackbar('Closing level', 'trackbars', True)
        # if clev >= 3:
        #     grey_frame = imf.close_img(grey_frame, clev, clev)

        # Threshold viewing range in depth image to generate aoi
        #dmin = imf.retrieve_trackbar('Min depth', 'trackbars')
        #dmax = int(np.max(frame_bgrd) - 300)
        #aoi_near = cv2.inRange(frame_bgrd, dmin, dmax)

        #dmax = int(np.max(frame_bgrd) + imf.retrieve_trackbar('Max depth', 'trackbars'))

        #
        aoi_end = cv2.inRange(frame_bgrd, int(np.max(frame_bgrd) - 200), int(np.max(frame_bgrd)))
        aoi_pipe = cv2.inRange(frame_bgrd, 550, int(np.max(frame_bgrd) - 200))
        cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pipe_mask = np.zeros_like(frame_bgrd).astype('uint8')
        pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
        bg_mask = cv2.subtract(pipe_mask, aoi_end)
        bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        hsv_aoi = cv2.bitwise_and(frame_hsv, frame_hsv, mask=bg_mask)


        #Vlow = imf.retrieve_trackbar('low', 'trackbars')
        upperLimit1 = np.array([95, 255, 255])
        lowerLimit1 = np.array([40, 0, 35])  # Threshold around highlights
        #upperLimit2 = np.array([240, 255, 230])
        #lowerLimit2 = np.array([120, 0, 0])

        mask1 = cv2.inRange(hsv_aoi, lowerLimit1, upperLimit1)
        #mask2 = cv2.inRange(hsv_aoi, lowerLimit2, upperLimit2)
        thresh_hsv = mask1# + mask2

        #thresh_hsv = cv2.inRange(hsv_aoi, (50, 0, 30), (255, 255, 255))
        hsv_aoi = cv2.bitwise_and(frame_bgr, frame_bgr, mask=thresh_hsv)

        # Get edges in depth data
        depth_edges = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        depth_edges = depth_edges - np.amin(depth_edges)
        depth_edges = depth_edges * 255.0 / (np.amax(depth_edges) - np.amin(depth_edges))
        depth_edges = np.uint8(depth_edges)
        canny = cv2.Canny(depth_edges, 20, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        canny = cv2.dilate(canny, kernel)

        kernel = imf.retrieve_trackbar('Kernel', 'trackbars', odd_only=True)
        bin = thresh_hsv#cv2.add(thresh_hsv, canny)
        bin = imf.open_img(bin, kernel, kernel)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if hierarchy is not None:
            hierarchy = hierarchy[0] #[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]
            draw_frame = frame_bgr.copy()
            for cnt, hrc in zip(contours, hierarchy):
                if cv2.contourArea(cnt) >= 10:
                    mask = np.zeros(bin.shape, np.uint8)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    rgbd_aoi = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=mask)

                    cv2.drawContours(draw_frame, cnt, -1, (0, 0, 255), 2)
                    # mask_display = imf.depth_to_display(rgbd_aoi)
                    # imf.resize_image(mask_display, 'mask', 0.5)
                    # cv2.waitKey(1)

                    features = FeatureSpace()
                    features.create_features(cnt, np.array(hrc[2] != -1), 'test')

                #detected, probability = classifier.classify(features.get_features())

        depth_display = imf.depth_to_display(frame_bgrd)

        imf.resize_image(depth_display, 'aligned depth image', 0.5)
        imf.resize_image(bin, 'thresh color image', 0.5)
        imf.resize_image(draw_frame, 'results', 0.5)

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit(0)
        elif key == ord('s'):
            for i in range(10):
                frame_bgr = vid_bgr.read()
                frame_bgrd_8bit = vid_bgrd.read()
        elif key == ord('p'):
            cv2.waitKey(5)
            cv2.waitKey(0)
